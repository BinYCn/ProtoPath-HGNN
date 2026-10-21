# Graph construction

`graph_new_rrwp_mp.py` builds one PyTorch Geometric graph per whole-slide image
from patch features, coordinates and cell-type results. All paths are supplied
on the command line; no dataset-specific paths are required by the script.

## Dependencies

See the repository's [environment instructions](../../README.md#environment)
for the preprocessing Python environment.

The script requires Python, PyTorch, PyTorch Geometric, `torch-sparse`,
`nmslib`, NumPy, joblib and tqdm. Install PyTorch Geometric and `torch-sparse`
versions compatible with the installed PyTorch and CUDA versions. The script was
smoke-tested with Python 3.8, PyTorch 1.12.1, PyG 2.5.3, torch-sparse 0.6.16
and nmslib 2.1.1.

## Prepare patches and HoVer-Net results

The following workflow uses TRIDENT for tissue coordinates and patch features,
OpenSlide to extract patch images, and TIA Toolbox HoVer-Net for nuclear
segmentation. Install TRIDENT and its dependencies, OpenSlide and
openslide-python, Pillow, h5py, and a compatible TIA Toolbox with access to the
`hovernet_fast-pannuke` weights. The feature and segmentation models have their
own license and access requirements. Consult the respective projects before use.

### 1. Extract tissue coordinates and features with TRIDENT

From a TRIDENT checkout, run for example:

```bash
python run_batch_of_slides.py --task all \
  --wsi_dir /path/to/wsis --job_dir /path/to/trident_output \
  --wsi_ext .svs --gpu 0 --patch_encoder uni_v2 \
  --mag 20 --patch_size 256 --overlap 0 --batch_size 32
```

For this configuration, coordinates are saved as
`<job_dir>/20x_256px_0px_overlap/patches/<slide>_patches.h5`; features and
their coordinates are saved as
`<job_dir>/20x_256px_0px_overlap/features_uni_v2/<slide>.h5`.
TRIDENT selects tissue patch coordinates here; patch image files are created
in the next step.

### 2. Extract patch images from WSIs

From this repository's root, run:

```bash
python preprocess/Graph_Construction/multi_patch_pt.py \
  --coord_dir /path/to/trident_output/20x_256px_0px_overlap/patches \
  --wsi_dir /path/to/wsis --wsi_ext .svs \
  --output_dir /path/to/patch_images \
  --patch_size 512 --level 0 --workers 4
```

The script accepts `<slide>_patches.h5` with a `coords` dataset, or
`<slide>_patches.pt` containing a coordinate tensor or array. It saves
`<output_dir>/<slide>/x_<x>_y_<y>.png`. WSI names must match the coordinate
file stems, and each WSI is processed by a separate process. Coordinates are
OpenSlide level-0 locations. `--patch_size` is the number of pixels read at
`--level`; choose both according to the slide resolution and the TRIDENT patch
footprint. For example, 512 pixels at level 0 corresponds to a 256-pixel patch
at 20x only when the level-0 slide is 40x. The script does not automatically
rescale the image. Existing files with the same names are overwritten, but
old patch files with other names are not removed. Use a fresh output directory
when the coordinate set changes.

### 3. Segment nuclei with HoVer-Net

Run the included `cell_seg.py` in the environment with TIA Toolbox installed:

```bash
python preprocess/Graph_Construction/cell_seg.py \
  --data_dir /path/to/patch_images --save_dir /path/to/hover \
  --gpu_id 0 --batch_size 64 --loader_workers 16 --postproc_workers 16
```

Each slide gets a separate output directory containing `file_map.dat` and
the corresponding result `.dat` files. The script records absolute patch
paths in the mapping. Use the same absolute patch root with `--patch_roots`
when constructing graphs. Existing slide output directories are rejected;
choose a new output directory when rerunning a slide. A failed or interrupted
slide may leave an incomplete directory; use a fresh directory for reruns.
Use `--help` on either script to see all options.

### 4. Convert TRIDENT features and coordinates for graph construction

`graph_new_rrwp_mp.py` reads PT files. Convert the feature H5 files from step 1
using the script below. It checks that their coordinate order matches the
coordinate files used in step 2:

```bash
python preprocess/Graph_Construction/convert_trident_h5.py \
  --feature_h5_dir /path/to/trident_output/20x_256px_0px_overlap/features_uni_v2 \
  --coord_h5_dir /path/to/trident_output/20x_256px_0px_overlap/patches \
  --output_dir /path/to/graph_inputs
```

Use `/path/to/graph_inputs/features`, `/path/to/graph_inputs/coordinates`,
`/path/to/hover` and `/path/to/patch_images` in the graph construction command
below. The feature rows, coordinate rows, and patch filenames must represent
the same patches. Only load `.pt` and joblib files from trusted sources.

## Input layout

For a slide named `slide_001`, the script expects:

```text
features/
  slide_001.pt                  # tensor or array, shape [N, feature_dim]
coordinates/
  slide_001_patches.pt          # tensor or array, shape [N, 2]
hover/
  slide_001/
    file_map.dat                # joblib mapping or list of (patch_path, result_stem)
patches/
  slide_001/
    x_<x>_y_<y>.png             # path used as a key in file_map.dat
```

The `features` and `coordinates` rows must correspond in the same order. The
patch key is constructed as `<patch_root>/<slide_name>/x_<x>_y_<y>.png` using
the coordinate values exactly as stored. The `--patch_roots` values must match
the path prefixes in `file_map.dat`; the patch images themselves are not read.

Each value in `file_map.dat` is a **result file stem**. The script appends
`.dat` to it and loads that path with joblib. Each resulting `.dat` file must
contain a dictionary of cell records, for example:

```python
{
    'cell_1': {'type': 1},
    'cell_2': {'type': 2},
    'cell_3': {'type': 1},
}
```

The most frequent non-negative integer cell type becomes the patch's node
type. An empty cell-type list is assigned type 0. For the default model,
node types should be in the range 0–5. A missing patch key raises an error by
default; `--missing_type zero` assigns type 0 instead. Paths stored as result
stems must be accessible from the process working directory if they are relative.

## Run

From the repository root:

```bash
python preprocess/Graph_Construction/graph_new_rrwp_mp.py \
  --feature_path /path/to/features \
  --coord_path /path/to/coordinates \
  --hover_path /path/to/hover \
  --patch_roots /path/used/in/file_map/patches \
  --save_path /path/to/output/graphs \
  --mode multi \
  --n_file_processes 4
```

If `file_map.dat` contains keys under multiple patch roots, list them in search
order, for example `--patch_roots /data/patches_a /data/patches_b`. In single
mode, files are processed sequentially and node lookups may run in parallel:

```bash
python preprocess/Graph_Construction/graph_new_rrwp_mp.py \
  --feature_path /path/to/features \
  --coord_path /path/to/coordinates \
  --hover_path /path/to/hover \
  --patch_roots /path/used/in/file_map/patches \
  --save_path /path/to/output/graphs \
  --mode single --node_processes 4
```

Run with `--help` for all options. `--radius` is the number of neighbors
requested **including the query node** (default 9, typically at most 8
outgoing edges per node). `--walk_length` defaults to 8; the current
ProtoPath-HGNN model expects eight RRWP features. In multi mode, parallelism is
across slides; nested node worker pools are not used. RRWP computation can use
substantial memory as the graph and its random-walk powers grow, so choose the
number of file workers accordingly.

## Output

Each `<slide>.pt` output is a `torch_geometric.data.Data` object with:

- `x`: patch features, shape `[N, feature_dim]`.
- `edge_index`: directed spatial k-nearest-neighbor edges, shape `[2, E]`.
- `edge_attr`: shape `[E, 4]`, columns `[source_type, target_type,
  feature_euclidean_distance, spatial_euclidean_distance]`.
- `centroid`: patch coordinates, shape `[N, 2]`.
- `node_type`: most frequent cell type per patch, shape `[N]`.
- `rrwp`: diagonal entries of `I, P, ..., P^(walk_length-1)`, shape
  `[N, walk_length]`, where `P` is the row-normalized directed adjacency.

Files are written as `<save_path>/<slide>.pt`; existing output files with the
same name are overwritten. A failed slide is reported, processing continues for
other slides, and the command exits with a nonzero status if any slide failed.
Only load `.pt` and joblib files from trusted sources.
