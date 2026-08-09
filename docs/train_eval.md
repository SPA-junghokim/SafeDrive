# SafeDrive Training and Evaluation

The agent lives in [`navsim/agents/safedrive/`](../navsim/agents/safedrive) and is
selected through the configs in
[`navsim/planning/script/config/common/agent/`](../navsim/planning/script/config/common/agent).

There is **no default agent**: every entrypoint requires an explicit `agent=...`
override. Omitting it fails fast with `You must specify 'agent'`.

Three scripts at the repository root drive the whole pipeline. Each keeps every
setting in a single configuration block at the top — edit them in place rather
than passing environment variables.

| script | role |
| --- | --- |
| [`cache.sh`](../cache.sh) | feature cache and metric caches, run once ([details](preprocess.md)) |
| [`train.sh`](../train.sh) | phase 1, phase 2 and phase 3 training |
| [`test.sh`](../test.sh) | navtest forward pass with score weights, then PDM scoring |

## 0. Agent configs

| config | role |
| --- | --- |
| `SafeDrive_Phase1_Perception` | phase 1 — perception pretraining, no planning head |
| `SafeDrive_Phase2_Planner_FreezePerception` | phase 2 — planner on frozen perception |
| `SafeDrive_Phase3_Planner_FullTrain` | phase 3 — end-to-end (main config) |

Phase 2 and phase 3 differ only in `freeze_perception`. Both compute their safety
ground truth by rolling out the model's own trajectories through the PDM
simulator during training (`use_target_scores: True`),
which is why they need a navtrain metric cache in `safety_metric_cache_path`.

## 1. Training

```bash
bash train.sh          # phase 1 (90) -> phase 2 (5) -> phase 3 (10 epochs)
```

Each phase loads the previous phase's `last.ckpt` as weights rather than resuming:
freezing changes the number of optimizer parameter groups, so optimizer state
cannot be carried across the boundary. The optimizer and LR schedule therefore
start fresh and epochs are counted from 0 within each phase.

Phase 1 is the longest and its result ships as `ckpts/safedrive_phase1_90ep.ckpt`.
To start from the shipped weights, comment out the phase 1 block and point
`PHASE1_CKPT` at that file.

> **`+second_lidar=True` is mandatory.** Cached `lidar_feature` entries are `.pcd`
> paths that `custom_collate_fn` loads at runtime; without the flag the default
> collate hits a `PosixPath` and dies on the first batch. Keep it if you write
> your own launcher.

If you invoke the entrypoint yourself, note that `agent=` alone is not enough —
`train_test_split`, `split` and `experiment_name` are also required:

```bash
python navsim/planning/script/run_training.py \
        agent=SafeDrive_Phase3_Planner_FullTrain \
        train_test_split=navtrain split=navtrain \
        experiment_name=my_run \
        cache_path=$NAVSIM_EXP_ROOT/safedrive_train_cache \
        use_cache_without_dataset=True force_cache_computation=False \
        +second_lidar=True +ddp_find_unused_parameters=True
```

The checkpoint callback writes `last.ckpt`, which is how phase 2 hands its
weights to phase 3.

## 2. Evaluation

```bash
bash test.sh
```

Stage 1 runs a GPU forward pass over navtest and writes the selected
trajectories; stage 2 scores them with the PDM simulator and writes `result.csv`
(per-token rows plus a final `average` row) under `$NAVSIM_EXP_ROOT/<EXP>/`.

Stage 1 has to go through `run_training.py`. The code that injects
`scoring_test` and the `*_test_weight` values into the model exists only there —
the model's own `scoring_test` is hardcoded to `False` in `__init__`, so a Hydra
override cannot switch it on and weights handed to `run_evaluation_gpu.py` are
silently ignored.

## 3. Test-time score weights

Candidate trajectories are ranked by a weighted sum of the safety subscores. The
weights are inference-time only — no retraining is needed to change them. They
live in the `WEIGHTS` block of `test.sh`, one line per metric.

> **`no_EP_TC_sum_scoring` decides the formula** and the two branches do not agree:
> `True` gives `EP_w*log(EP) + TTC_w*log(TTC)`, `False` (the dataclass default)
> gives `W_w*log(EP_w*EP + TTC_w*TTC)`. The shipped weights were searched offline
> against the first form, so `test.sh` sets the flag on. Turning it off changes
> the ranking and lowers the score.
