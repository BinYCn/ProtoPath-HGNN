# ProtoPath-HGNN

This repository contains the implementation of:

> **Prototype-Enhanced Hierarchical Graph Neural Network for Gastrointestinal Cancer Analysis in Whole Slide Images**
>
> Manuscript under review.

ProtoPath-HGNN operates on whole-slide graphs built from patch features,
spatial relationships, cell types and random-walk positional encodings. The
repository includes preprocessing workflows and training code for survival
prediction and classification.

## Environment

Use separate Python environments for training and preprocessing. To create
the training environment from the repository root:

```bash
conda create -n protopath-train python=3.10
conda activate protopath-train
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To create the preprocessing environment:

```bash
conda create -n protopath-preprocess python=3.10
conda activate protopath-preprocess
python -m pip install --upgrade pip
python -m pip install -r preprocess/requirements.txt
```

The training environment uses PyTorch 2.0.1; preprocessing uses PyTorch
2.2.2. Each requirements file specifies the corresponding PyG CUDA 11.8
extensions. TRIDENT also uses system libraries such as libvips;
install the libraries required by your slide formats separately.

Model weights and any required access must be obtained separately.
Third-party projects and pretrained models have their own licenses and access
requirements. See the preprocessing guides for details.

## Preprocessing

Two workflows are available:

1. **Direct workflow:** extract TRIDENT features and build graphs with inline
   HoVer-Net inference, or build graphs from existing feature H5 files. See
   [preprocess/README.md](preprocess/README.md).
2. **Step-by-step workflow:** extract TRIDENT coordinates and features, save
   patch images, run HoVer-Net segmentation, convert H5 files to PT, and build
   graphs. See [preprocess/Graph_Construction/README.md](preprocess/Graph_Construction/README.md).

Both workflows produce one PyTorch Geometric graph per slide. Graph files
should be located directly in the directory passed as `data_root_dir`:

```text
graphs/
  <slide_id>.pt
  ...
```

Each graph provides `x` (patch features), `edge_index` (spatial edges),
`edge_attr` (source type, target type, feature distance and spatial distance),
`centroid`, `node_type` and `rrwp`. The default model expects node types in
0–5 and eight RRWP features. Set `--num_features` to the feature dimension of
the selected patch encoder; the training scripts default to 1536.

## Survival prediction

The provided survival datasets use four discrete survival time intervals:

| Dataset | Metadata CSV                             |
| --- |------------------------------------------|
| `tcga_coad360` | `dataset_csv/tcga_coad360_all_clean.csv` |
| `tcga_stad343` | `dataset_csv/tcga_stad343_all_clean.csv` |
| `tcga_lihc359` | `dataset_csv/tcga_lihc359_all_clean.csv` |
| `tcga_paad172` | `dataset_csv/tcga_paad172_all_clean.csv` |

Survival training uses the metadata under `dataset_csv/` and split files under
`splits/<which_splits>/<split_dir>/`. The split CSVs contain `train` and `val`
columns with patient IDs. Each metadata row associates a `case_id` with a
`slide_id`, `survival_months`, and `censorship` (`0` for an observed event,
`1` for censored). A patient can have multiple slides; each corresponding
slide graph must exist in the graph directory.

The survival entry point derives the metadata filename from the first two
underscore-separated parts of the split directory name. For example,
`tcga_coad360` resolves to the supplied
`dataset_csv/tcga_coad360_all_clean.csv`. Run:

```bash
CUDA_VISIBLE_DEVICES=0 bash train_script/train.sh tcga_coad360 /path/to/graphs
```

## Classification

The classification entry point defines these tasks and label counts:

| Task | Classes | Metadata CSV |
| --- | ---: | --- |
| `TCGA_COAD_stage3` | 3 | `dataset_csv/TCGA_COAD_stage3.csv` |
| `TCGA_STAD_stage3` | 3 | `dataset_csv/TCGA_STAD_stage3.csv` |
| `TCGA_LIHC_stage2` | 2 | `dataset_csv/TCGA_LIHC_stage2.csv` |

Classification split CSVs contain `train` and `val` columns with slide IDs;
the default split directory is `splits/5foldcv/<task>/`. Metadata contains
`case_id`, `slide_id` and integer `label` columns. Each slide ID must have a
corresponding `<slide_id>.pt` graph in `data_root_dir`.

For example, run:

```bash
CUDA_VISIBLE_DEVICES=0 bash train_script/train_class.sh TCGA_COAD_stage3 /path/to/graphs
```

## Acknowledgments

We thank the authors and contributors of the following projects for making
their code and tools available:

- [MambaMIL](https://github.com/isyangshu/MambaMIL)
- [CLAM](https://github.com/mahmoodlab/clam)
- [Patch-GCN](https://github.com/mahmoodlab/Patch-GCN)
- [TRIDENT](https://github.com/mahmoodlab/TRIDENT)
- [TIA Toolbox](https://github.com/tissueimageanalytics/tiatoolbox)
