#!/usr/bin/env bash
set -euo pipefail

# Usage: bash preprocess/graph.sh <feature_dir> <wsi_dir> <output_dir> [graph options]
feature_dir=${1:?Usage: graph.sh <feature_dir> <wsi_dir> <output_dir> [graph options]}
wsi_dir=${2:?Usage: graph.sh <feature_dir> <wsi_dir> <output_dir> [graph options]}
output_dir=${3:?Usage: graph.sh <feature_dir> <wsi_dir> <output_dir> [graph options]}
shift 3
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

exec "${PYTHON:-python}" "$script_dir/graph_hovernet_pipeline.py" \
    --feature_dir "$feature_dir" --wsi_dir "$wsi_dir" --output_dir "$output_dir" "$@"
