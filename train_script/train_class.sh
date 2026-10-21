#!/usr/bin/env bash
set -euo pipefail

# Usage: ./train_script/train_class.sh <task> <data_root_dir> [additional arguments]
# Example: ./train_script/train_class.sh TCGA_COAD_stage3 /path/to/data
task=${1:?Usage: ./train_class.sh <task> <data_root_dir> [additional arguments]}
data_root_dir=${2:?Usage: ./train_class.sh <task> <data_root_dir> [additional arguments]}
shift 2
[[ $data_root_dir = /* ]] || data_root_dir="$PWD/$data_root_dir"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} python main_class.py \
    --task "$task" \
    --data_root_dir "$data_root_dir" \
    --which_splits 5foldcv \
    --model_type protopath_hgnn \
    --num_features 1536 \
    --hidden_dim 128 \
    --max_epochs 200 \
    --results_dir ./result_new \
    "$@"
