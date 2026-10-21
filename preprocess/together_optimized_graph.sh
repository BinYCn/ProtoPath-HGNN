#!/usr/bin/env bash
set -euo pipefail

# Usage: bash preprocess/together_optimized_graph.sh <wsi_dir> <job_dir> <trident_root> [graph options]
wsi_dir=${1:?Usage: together_optimized_graph.sh <wsi_dir> <job_dir> <trident_root> [graph options]}
job_dir=${2:?Usage: together_optimized_graph.sh <wsi_dir> <job_dir> <trident_root> [graph options]}
trident_root=${3:?Usage: together_optimized_graph.sh <wsi_dir> <job_dir> <trident_root> [graph options]}
shift 3
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# TRIDENT runs from its own directory, so resolve caller-relative paths first.
[[ $wsi_dir = /* ]] || wsi_dir="$PWD/$wsi_dir"
[[ $job_dir = /* ]] || job_dir="$PWD/$job_dir"
[[ $trident_root = /* ]] || trident_root="$PWD/$trident_root"

wsi_ext=${WSI_EXT:-.tif}
gpu=${GPU:-0}
patch_encoder=${PATCH_ENCODER:-uni_v2}
mag=${MAG:-20}
patch_size=${PATCH_SIZE:-256}
overlap=${OVERLAP:-0}
coords_dir="${mag}x_${patch_size}px_${overlap}px_overlap"
feature_dir="$job_dir/$coords_dir/features_$patch_encoder"
graph_output_dir=${GRAPH_OUTPUT_DIR:-"$job_dir/${coords_dir}_graph_rrwp_inline_hovernet/features_$patch_encoder"}
[[ $graph_output_dir = /* ]] || graph_output_dir="$PWD/$graph_output_dir"

if [[ ! -f "$trident_root/run_batch_of_slides.py" ]]; then
    echo "Cannot find run_batch_of_slides.py in $trident_root" >&2
    exit 1
fi
(
    cd "$trident_root"
    "${TRIDENT_PYTHON:-${PYTHON:-python}}" run_batch_of_slides.py \
        --task all --wsi_dir "$wsi_dir" --job_dir "$job_dir" \
        --patch_encoder "$patch_encoder" --mag "$mag" --patch_size "$patch_size" \
        --overlap "$overlap" --wsi_ext "$wsi_ext" --gpu "$gpu" \
        --batch_size "${TRIDENT_BATCH_SIZE:-32}"
)

graph_cmd=("${PYTHON:-python}" "$script_dir/graph_hovernet_pipeline.py"
    --feature_dir "$feature_dir" --wsi_dir "$wsi_dir" --output_dir "$graph_output_dir"
    --wsi_ext "$wsi_ext" --device "${DEVICE:-cuda:$gpu}"
    --workers_per_device "${WORKERS_PER_DEVICE:-1}"
    --hover_patch_size "${HOVER_PATCH_SIZE:-512}"
    --hover_batch_size "${HOVER_BATCH_SIZE:-64}"
    --postproc_workers "${POSTPROC_WORKERS:-8}")
if [[ -n ${TIATOOLBOX_ROOT:-} ]]; then
    graph_cmd+=(--tiatoolbox_root "$TIATOOLBOX_ROOT")
fi
if [[ -n ${DEVICES:-} ]]; then
    read -r -a device_array <<< "$DEVICES"
    graph_cmd+=(--devices "${device_array[@]}")
fi
exec "${graph_cmd[@]}" "$@"
