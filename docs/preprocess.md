# SafeDrive Preprocessing

```bash
bash cache.sh
```

Run once before [`train.sh`](../train.sh). Comment out steps you do not need.

| Step | Output | Needed by |
| --- | --- | --- |
| 1 feature cache | `exp/safedrive_train_cache` | all training phases |
| 2 train metric cache | `exp/train_metric_cache_navtrain` | phases 2 and 3 |
| 3 test metric cache | `exp/metric_cache_navtest` | `test.sh` |

Sizing: the feature cache is ~4.7 MB per sample.

## Gotchas

- **Build the feature cache with the phase 3 config.** The rollout ground truth
  needs `agent_token_ids`, which phase 1 does not write.
- **`+second_lidar=True` is mandatory in every train/eval command.** The cached
  `lidar_feature` is a `.pcd` path loaded in the collate function; without the
  flag the default collate hits a `PosixPath` and dies on the first batch.
- **Keep `safety_score_mode: epdms`.** Phases 2 and 3 score their own
  trajectories with the PDM simulator during training; `epdms` yields
  `[NC, DAC, EP, TTC, comfort, DDC, TLC, LK, final]`, one column per safety head.
  `pdms` drops TLC and LK, leaving those heads untrained but still used at
  scoring time.
- Evaluation runs without a feature cache (`cache_path=null`) and reads navtest
  sensor data directly, so step 3 is all it needs.
