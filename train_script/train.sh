#!/usr/bin/env bash
set -euo pipefail

# Usage: ./train_script/train.sh <split_dir> <data_root_dir> [additional arguments]
# Example: ./train_script/train.sh tcga_coad360 /path/to/graphs
split_dir=${1:?Usage: train.sh <split_dir> <data_root_dir> [additional arguments]}
data_root_dir=${2:?Usage: train.sh <split_dir> <data_root_dir> [additional arguments]}
shift 2
[[ $data_root_dir = /* ]] || data_root_dir="$PWD/$data_root_dir"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} python main.py \
    --which_splits 5foldcv \
    --split_dir "$split_dir" \
    --mode graph \
    --model_type protopath_hgnn \
    --data_root_dir "$data_root_dir" \
    --num_features 1536 \
    --hidden_dim 128 \
    --max_epochs 200 \
    --results_dir ./result_new \
    "$@"
