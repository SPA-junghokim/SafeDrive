#!/bin/bash
# =============================================================================
#  SafeDrive - build the planning trajectory anchors
#
#  k-means over the ground-truth ego trajectories in the training feature cache.
#  The shipped anchors were built this way; rerun only if you change the horizon,
#  the anchor count, or the training split.
#
#  Needs the feature cache from train.sh STEP 0.
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


# ---- configuration ----------------------------------------------------------
CACHE_PATH=$BASE/exp/safedrive_train_cache        # from train.sh STEP 0
OUT=$BASE/trajectory_anchors/trajectory_anchors_256_kmeans.npy
NUM_ANCHORS=256
SEED=0


python navsim/planning/script/run_trajectory_anchor_kmeans.py \
    --cache "$CACHE_PATH" \
    --out "$OUT" \
    --num-anchors $NUM_ANCHORS \
    --seed $SEED

echo "done -> $OUT"
echo "point plan_anchor_path at it in the agent configs to use it."
