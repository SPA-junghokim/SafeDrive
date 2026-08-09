#!/bin/bash
# =============================================================================
#  SafeDrive - Training
#
#    phase 1 : perception pretraining, no planning head          90 epochs
#    phase 2 : perception frozen, planning + safety heads only    5 epochs
#    phase 3 : full end-to-end fine-tune (final config)          10 epochs
#
#  Each phase hands its last.ckpt to the next. Weights are loaded, not resumed:
#  freezing changes the number of optimizer param groups, so optimizer state
#  cannot be carried across a phase boundary.
#
#  Run cache.sh first -- this script reads the caches it writes.
#
#  Phase 1 is the longest and its result ships as ckpts/safedrive_phase1_90ep.ckpt.
#  To start from the shipped weights instead, comment out the phase 1 block and
#  point PHASE1_CKPT at that file.
# =============================================================================
set -e

BASE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATA_ROOT=$BASE/dataset


# ---- environment ------------------------------------------------------------
export PYTHONPATH=$BASE
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=$DATA_ROOT/maps
export OPENSCENE_DATA_ROOT=$DATA_ROOT
export NAVSIM_EXP_ROOT=$BASE/exp
export NAVSIM_DEVKIT_ROOT=$BASE/navsim
export HYDRA_FULL_ERROR=1

export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=gcc-11
export CXX=g++-11
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1


# ---- configuration ----------------------------------------------------------
CONFIG_P1=SafeDrive_Phase1_Perception
CONFIG_P2=SafeDrive_Phase2_Planner_FreezePerception
CONFIG_P3=SafeDrive_Phase3_Planner_FullTrain
EXP_P1=safedrive/phase1_perception
EXP_P2=safedrive/phase2_freeze
EXP_P3=safedrive/phase3_e2e

SPLIT=navtrain
BATCH=32                        # per GPU
EPOCHS_P1=90                    # shipped checkpoint is epoch 89 of this run
EPOCHS_P2=5
EPOCHS_P3=10

CACHE_PATH=$BASE/exp/safedrive_train_cache          # written by cache.sh
METRIC_CACHE=$BASE/exp/train_metric_cache_navtrain  # written by cache.sh

PHASE1_CKPT=$NAVSIM_EXP_ROOT/$EXP_P1/lightning_logs/checkpoints/last.ckpt
#PHASE1_CKPT=$BASE/ckpts/safedrive_phase1_90ep.ckpt  # use this to skip phase 1
PHASE2_CKPT=$NAVSIM_EXP_ROOT/$EXP_P2/lightning_logs/checkpoints/last.ckpt

COMMON=(
    train_test_split=$SPLIT
    split=$SPLIT
    cache_path=$CACHE_PATH
    use_cache_without_dataset=True
    force_cache_computation=False
    dataloader.params.batch_size=$BATCH
    dataloader.params.num_workers=4
    ++trainer.params.precision=bf16-mixed
    ++trainer.params.check_val_every_n_epoch=1
    +ddp_find_unused_parameters=True
    +second_lidar=True              # lidar_feature is a .pcd path, loaded in collate
)

# rollout ground truth, phases 2 and 3 only (phase 1 has safety scoring off)
SAFETY_GT=( ++agent.config.safety_metric_cache_path=$METRIC_CACHE )


# ---- phase 1 : perception pretraining ---------------------------------------
# Detection and BEV segmentation only: no_planning=True and scene_level_safety=False,
# so there is no planning head and no rollout ground truth to compute.
python navsim/planning/script/run_training.py \
    agent=$CONFIG_P1 \
    experiment_name=$EXP_P1 \
    trainer.params.max_epochs=$EPOCHS_P1 \
    "${COMMON[@]}"


# ---- phase 2 : frozen perception --------------------------------------------
python navsim/planning/script/run_training.py \
    agent=$CONFIG_P2 \
    experiment_name=$EXP_P2 \
    trainer.params.max_epochs=$EPOCHS_P2 \
    agent.checkpoint_path=$PHASE1_CKPT \
    "${COMMON[@]}" "${SAFETY_GT[@]}"


# ---- phase 3 : end-to-end fine-tune -----------------------------------------
python navsim/planning/script/run_training.py \
    agent=$CONFIG_P3 \
    experiment_name=$EXP_P3 \
    trainer.params.max_epochs=$EPOCHS_P3 \
    agent.checkpoint_path=$PHASE2_CKPT \
    "${COMMON[@]}" "${SAFETY_GT[@]}"

echo "done -> $NAVSIM_EXP_ROOT/$EXP_P3/lightning_logs/checkpoints"
