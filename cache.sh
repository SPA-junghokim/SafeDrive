#!/bin/bash
# =============================================================================
#  SafeDrive - Preprocessing
#
#    step 1 : feature cache      navtrain sensor data -> tensors for training
#    step 2 : train metric cache navtrain PDM caches  -> rollout safety GT
#    step 3 : test metric cache  navtest  PDM caches  -> evaluation scoring
#
#  Run this once before train.sh. Steps 1 and 2 are needed to train, step 3 only
#  to evaluate; comment out whichever you do not need.
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
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


# ---- configuration ----------------------------------------------------------
CONFIG=SafeDrive_Phase3_Planner_FullTrain   # phase 3 also caches agent_token_ids
TRAIN_SPLIT=navtrain
TEST_SPLIT=navtest
WORKERS=60                                  # ray workers for the feature cache
METRIC_WORKERS=10                           # ray workers for the metric caches

FEATURE_CACHE=$BASE/exp/safedrive_train_cache
TRAIN_METRIC_CACHE=$BASE/exp/train_metric_cache_navtrain
TEST_METRIC_CACHE=$BASE/exp/metric_cache_navtest


# ---- step 1 : feature cache (navtrain) --------------------------------------
# Use the phase 3 config: the rollout needs the agent_token_ids field, which the
# phase 1 config does not write. Per-worker BLAS threads are pinned to 1 because
# 60 workers x 8 OMP threads would oversubscribe the machine.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python navsim/planning/script/run_dataset_caching.py \
    agent=$CONFIG \
    experiment_name=caching/safedrive \
    train_test_split=$TRAIN_SPLIT \
    cache_path=$FEATURE_CACHE \
    worker.threads_per_node=$WORKERS


# ---- step 2 : train metric cache (navtrain) ---------------------------------
# Phases 2 and 3 score their own trajectories with the PDM simulator during
# training, which reads these caches through agent.config.safety_metric_cache_path.
python navsim/planning/script/run_train_metric_caching.py \
    train_test_split=$TRAIN_SPLIT \
    cache.cache_path=$TRAIN_METRIC_CACHE \
    worker.threads_per_node=$METRIC_WORKERS


# ---- step 3 : test metric cache (navtest) -----------------------------------
# Only needed for test.sh; evaluation itself runs without a feature cache.
python navsim/planning/script/run_metric_caching.py \
    train_test_split=$TEST_SPLIT \
    cache.cache_path=$TEST_METRIC_CACHE \
    worker.threads_per_node=$METRIC_WORKERS


echo "done"
echo "  feature cache      -> $FEATURE_CACHE"
echo "  train metric cache -> $TRAIN_METRIC_CACHE"
echo "  test metric cache  -> $TEST_METRIC_CACHE"
