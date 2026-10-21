# Preprocessing: WSI to graph

This directory provides the HoVer-Net workflow migrated from `TRIDENT`.
It reads TRIDENT patch features and coordinates from H5 files, obtains a
majority nuclear type for each patch using TIA Toolbox HoVer-Net, and saves a
spatial graph with RRWP for each slide. Patch images are read from the WSI in
memory; intermediate PNG files and HoVer-Net `.dat` files are not required.

## Dependencies

See the repository's [environment instructions](../README.md#environment)
to install `preprocess/requirements.txt` in a separate preprocessing
environment. The full TRIDENT workflow also requires the source checkout
described there.

- For feature extraction, install [TRIDENT](https://github.com/mahmoodlab/TRIDENT)
  and its dependencies, and obtain any required encoder weights and access.
  The full workflow expects a TRIDENT checkout containing
  `run_batch_of_slides.py` and an importable `trident` package.
- For graph construction, install Python, PyTorch, PyTorch Geometric,
  torch-sparse, nmslib, NumPy, h5py, joblib and tqdm. `--exact_knn` also
  requires scikit-learn. Use mutually compatible PyTorch, CUDA, PyG and
  torch-sparse versions.
- For HoVer-Net inference, install a compatible TIA Toolbox, OpenSlide and
  openslide-python, and make the chosen pretrained model weights available.
  An installed TIA Toolbox is used by default. For a source checkout, pass
  `--tiatoolbox_root /path/to/tiatoolbox` to the graph script, or set
  `TIATOOLBOX_ROOT` for the combined script.

TRIDENT, TIA Toolbox, their models and encoders have their own licenses and
access requirements; consult their respective projects before redistribution or
use. The examined TRIDENT checkout declares CC BY-NC-ND 4.0 in its project
metadata. The third-party projects and model weights are not included here.

## 1. Build graphs from existing features

Each input file must be named `<slide>.h5` and contain datasets `features`
(shape `[N, D]`) and `coords` (shape `[N, 2]`) in the same row order. Its WSI
must be directly inside the WSI directory and have the same stem.

```bash
bash preprocess/graph.sh /path/to/features /path/to/wsis /path/to/graphs \
  --wsi_ext .svs --device cuda:0 --hover_patch_size 512 \
  --hover_batch_size 64 --postproc_workers 8
```

Set `PYTHON=/path/to/python` if the graph dependencies are in a particular
Python environment. Additional arguments are passed directly to
`graph_hovernet_pipeline.py`; run `python preprocess/graph_hovernet_pipeline.py
--help` to list them. For multiple GPUs, use for example
`--devices cuda:0 cuda:1 --workers_per_device 1`.

## 2. Run TRIDENT and build graphs

```bash
GPU=0 WSI_EXT=.svs PATCH_ENCODER=uni_v2 \
  bash preprocess/together_optimized_graph.sh \
  /path/to/wsis /path/to/trident_output /path/to/TRIDENT \
  --hover_batch_size 64 --postproc_workers 8
```

The three positional arguments are WSI directory, TRIDENT output directory and
TRIDENT checkout directory. Remaining arguments configure graph construction.
`PYTHON` chooses the graph Python; `TRIDENT_PYTHON` optionally chooses a
separate Python for TRIDENT. The script runs TRIDENT with its checkout as the
working directory. Common environment variables are:

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPU` | `0` | GPU index for TRIDENT and the default graph device |
| `DEVICE` | `cuda:$GPU` | Single HoVer-Net device |
| `DEVICES` | unset | Space-separated graph devices, e.g. `cuda:0 cuda:1` |
| `WSI_EXT` | `.tif` | WSI extension for both stages |
| `PATCH_ENCODER` | `uni_v2` | TRIDENT patch encoder |
| `MAG`, `PATCH_SIZE`, `OVERLAP` | `20`, `256`, `0` | TRIDENT patch settings |
| `TRIDENT_BATCH_SIZE` | `32` | Feature extraction batch size |
| `HOVER_PATCH_SIZE`, `HOVER_BATCH_SIZE` | `512`, `64` | HoVer-Net read size and inference batch size |
| `POSTPROC_WORKERS`, `WORKERS_PER_DEVICE` | `8`, `1` | HoVer-Net CPU postprocessing and slide workers |
| `GRAPH_OUTPUT_DIR` | see below | Override the graph output directory |
| `TIATOOLBOX_ROOT` | unset | Optional TIA Toolbox source checkout |

By default graphs are written to
`<job_dir>/<mag>x_<patch_size>px_<overlap>px_overlap_graph_rrwp_inline_hovernet/features_<patch_encoder>/`.
The feature directory is `<job_dir>/<mag>x_<patch_size>px_<overlap>px_overlap/features_<patch_encoder>/`.

Coordinates are read as OpenSlide level-0 locations. `--hover_patch_size` is
the size passed to OpenSlide at `--hover_level` (default 0); choose the size
and level to match the actual WSI resolution and TRIDENT patch footprint. The
script does not automatically infer or rescale this footprint.

## Output and practical notes

Each `<slide>.pt` is a PyTorch Geometric `Data` object with `x`, `edge_index`,
`edge_attr`, `centroid`, `node_type` and `rrwp`. Edge attributes contain source
type, target type, feature distance and spatial distance; RRWP defaults to 8
steps, as expected by ProtoPath-HGNN. Select an encoder whose feature dimension
matches the training `--num_features`. The model expects node types in the
range 0–5 with the default configuration.

Outputs already present are skipped unless `--overwrite` is given. Node types
are cached in `<output_dir>/node_type_cache/` by default; use
`--node_type_cache_dir` to change the location. The cache only checks slide
name and node count: use `--overwrite` or a fresh cache after changing the WSI,
coordinates, HoVer-Net model or inference settings. `runtime_summary.csv`
records slide status and timings; any failed slide causes a nonzero exit code.

For a CPU smoke test without HoVer-Net, add `--disable_hovernet --device cpu`;
this assigns every node type 0 and is only suitable for testing or ablation.
`--exact_knn` uses exact neighbors instead of approximate HNSW. RRWP can consume
substantial memory on large graphs; reduce concurrent workers if necessary.
For the alternative workflow using precomputed HoVer-Net `.dat` files, see
[Graph_Construction/README.md](Graph_Construction/README.md).
