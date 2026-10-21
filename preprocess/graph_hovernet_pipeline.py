#!/usr/bin/env python3
"""Build graph PT files directly from TRIDENT H5 features with inline HoverNet.

This script replaces the slow middle of the previous workflow:

    h5 -> pt coords/features -> PNG patches -> tiatoolbox HoverNet -> graph pt

with:

    TRIDENT feature H5 -> in-memory WSI reads -> inline HoverNet node types -> graph pt

The output PyG ``Data`` object is intentionally kept compatible with
``graph_new_rrwp_mp.py``: ``x``, ``edge_index``, ``edge_attr``, ``centroid``,
``node_type`` and ``rrwp`` are all present.
"""

from __future__ import annotations
import numpy as np
import torch
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create final graph .pt files from TRIDENT H5 features, optionally with inline HoverNet.",
    )
    parser.add_argument(
        "--feature_dir",
        required=True,
        help="Directory containing TRIDENT feature H5 files, e.g. job/20x_256px_0px_overlap/features_uni_v2.",
    )
    parser.add_argument(
        "--wsi_dir",
        default=None,
        help="Directory containing source WSI files.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for final graph .pt files. Default: <feature_dir>_graph_rrwp_inline_hovernet.",
    )
    parser.add_argument(
        "--wsi_ext",
        nargs="+",
        default=[".svs", ".tif", ".tiff", ".ndpi", ".mrxs"],
        help="Allowed WSI extensions used to match feature H5 stems.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Single device for HoverNet inference. Ignored when --devices is set.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Devices for slide-level parallel HoverNet workers, e.g. --devices cuda:4 cuda:5 cuda:6 cuda:7 or --devices 4 5 6 7.",
    )
    parser.add_argument(
        "--workers_per_device",
        type=int,
        default=1,
        help="Number of long-lived slide workers per device. Increase only if one GPU has spare memory and WSI I/O is the bottleneck.",
    )
    parser.add_argument(
        "--tiatoolbox_root",
        default=None,
        help="Path to the tiatoolbox project root. Default: ./tiatoolbox if present.",
    )
    parser.add_argument(
        "--hovernet_model",
        default="hovernet_fast-pannuke",
        help="TIAtoolbox pretrained model name.",
    )
    parser.add_argument(
        "--disable_hovernet",
        action="store_true",
        help="Skip HoverNet and set all node_type values to 0. Useful for ablation/runtime comparison.",
    )
    parser.add_argument(
        "--hover_patch_size",
        type=int,
        default=512,
        help="Level-0 pixel size read for HoverNet around each TRIDENT coordinate. Use 512 for 20x 256px patches from a 40x level-0 slide.",
    )
    parser.add_argument(
        "--hover_level",
        type=int,
        default=0,
        help="OpenSlide level used for HoverNet reads.",
    )
    parser.add_argument(
        "--hover_batch_size",
        type=int,
        default=64,
        help="HoverNet GPU batch size.",
    )
    parser.add_argument(
        "--postproc_workers",
        type=int,
        default=8,
        help="CPU workers for HoverNet post-processing. Use 0 for in-process postproc.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=9,
        help="HNSW nearest-neighbor count. Matches old script default of 9 including self.",
    )
    parser.add_argument(
        "--exact_knn",
        action="store_true",
        help="Use deterministic exact spatial KNN instead of nmslib HNSW. Slower, but stable.",
    )
    parser.add_argument(
        "--rrwp_walk_length",
        type=int,
        default=8,
        help="RRWP walk length.",
    )
    parser.add_argument(
        "--node_type_cache_dir",
        default=None,
        help="Optional directory to cache/reuse per-slide node_type tensors.",
    )
    parser.add_argument(
        "--no_cache_node_types",
        action="store_true",
        help="Do not write node_type cache files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute graph files that already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N H5 files. Handy for timing tests.",
    )
    return parser.parse_args()


def normalize_device(device: str) -> str:
    if device.isdigit():
        return f"cuda:{device}"
    return device


def configure_visible_cuda(device: str) -> tuple[str, str]:
    """Map a requested physical CUDA device to logical cuda:0 for DataParallel."""
    normalized = normalize_device(device)
    if not normalized.startswith("cuda"):
        return normalized, normalized
    if ":" not in normalized:
        return "cuda:0", normalized

    physical_id = normalized.split(":", 1)[1]
    if physical_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = physical_id
        return "cuda:0", normalized
    return "cuda:0", normalized


def get_worker_devices(args: argparse.Namespace) -> list[str]:
    devices = args.devices if args.devices else [args.device]
    devices = [normalize_device(device) for device in devices]
    return [device for device in devices for _ in range(max(args.workers_per_device, 1))]


def split_round_robin(items: list[Path], num_chunks: int) -> list[list[Path]]:
    chunks: list[list[Path]] = [[] for _ in range(num_chunks)]
    for index, item in enumerate(items):
        chunks[index % num_chunks].append(item)
    return [chunk for chunk in chunks if chunk]


def ensure_tiatoolbox_import_path(tiatoolbox_root: str | None = None) -> None:
    """Prefer a real tiatoolbox package over the namespace directory shadow."""
    script_dir = Path(__file__).resolve().parent
    candidates = []
    if tiatoolbox_root:
        candidates.append(Path(tiatoolbox_root).resolve())
    candidates.append((script_dir / "tiatoolbox").resolve())

    selected_root = None
    for candidate in candidates:
        if (candidate / "tiatoolbox" / "__init__.py").exists():
            selected_root = candidate
            break

    if selected_root is None:
        return

    selected_root_str = str(selected_root)
    if selected_root_str not in sys.path:
        sys.path.insert(0, selected_root_str)
    else:
        sys.path.remove(selected_root_str)
        sys.path.insert(0, selected_root_str)

    loaded = sys.modules.get("tiatoolbox")
    if loaded is not None and (
        getattr(loaded, "__file__", None) is None or not hasattr(loaded, "rcParam")
    ):
        for module_name in list(sys.modules):
            if module_name == "tiatoolbox" or module_name.startswith("tiatoolbox."):
                sys.modules.pop(module_name, None)


def read_trident_h5(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(h5_path, "r") as handle:
        features = handle["features"][:]
        coords = handle["coords"][:]
    if features.shape[0] != coords.shape[0]:
        raise ValueError(
            f"{h5_path.name}: features and coords have different lengths "
            f"({features.shape[0]} vs {coords.shape[0]})."
        )
    return features, coords


def find_wsi_path(wsi_dir: Path, stem: str, extensions: Iterable[str]) -> Path:
    for ext in extensions:
        candidates = [
            wsi_dir / f"{stem}{ext}",
            wsi_dir / f"{stem}{ext.lower()}",
            wsi_dir / f"{stem}{ext.upper()}",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    matches = [p for p in wsi_dir.iterdir() if p.is_file() and p.stem == stem]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find WSI for stem '{stem}' in {wsi_dir}.")


def read_wsi_batch(
    slide,
    coords: np.ndarray,
    *,
    level: int,
    patch_size: int,
) -> torch.Tensor:
    import numpy as np
    import torch

    images = []
    for coord in coords:
        location = (int(coord[0]), int(coord[1]))
        image = slide.read_region(location, level, (patch_size, patch_size)).convert("RGB")
        images.append(np.asarray(image))
    return torch.as_tensor(np.stack(images, axis=0))


def majority_type_from_raw(raw_maps: list[np.ndarray], postproc_func) -> int:
    import numpy as np

    _, inst_dict = postproc_func(raw_maps)
    if not inst_dict:
        return 0

    types = [value.get("type", 0) or 0 for value in inst_dict.values()]
    if not types:
        return 0

    types_array = np.asarray(types, dtype=np.int64)
    if types_array.size == 0:
        return 0
    return int(np.bincount(types_array).argmax())


def postprocess_hovernet_batch(
    raw_outputs: tuple[np.ndarray, ...],
    postproc_func,
    executor: ProcessPoolExecutor | None,
) -> list[int]:
    num_items = raw_outputs[0].shape[0]
    raw_per_patch = [[head[idx] for head in raw_outputs] for idx in range(num_items)]

    if executor is None or num_items < 2:
        return [majority_type_from_raw(raw, postproc_func) for raw in raw_per_patch]

    return list(executor.map(majority_type_from_raw, raw_per_patch, [postproc_func] * num_items))


class HoverNetRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        ensure_tiatoolbox_import_path(args.tiatoolbox_root)

        from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
        from tiatoolbox.models.models_abc import model_to

        self.args = args
        self.device, self.requested_device = configure_visible_cuda(args.device)
        self.segmentor = NucleusInstanceSegmentor(
            pretrained_model=args.hovernet_model,
            num_loader_workers=0,
            num_postproc_workers=0,
            batch_size=args.hover_batch_size,
            auto_generate_mask=False,
            verbose=False,
        )
        self.model = model_to(self.segmentor.model, device=self.device)
        self.postproc_func = self.segmentor.model.postproc_func
        if args.postproc_workers > 0:
            import multiprocessing as mp

            self.executor = ProcessPoolExecutor(
                max_workers=args.postproc_workers,
                mp_context=mp.get_context("spawn"),
            )
        else:
            self.executor = None

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None

    def infer_node_types(
        self,
        h5_path: Path,
        wsi_path: Path,
        coords: np.ndarray,
    ) -> torch.Tensor:
        import openslide
        import torch
        from tqdm import tqdm

        node_types: list[int] = []
        slide = openslide.OpenSlide(str(wsi_path))
        try:
            for start in tqdm(
                range(0, coords.shape[0], self.args.hover_batch_size),
                desc=f"{h5_path.stem}: HoverNet",
                leave=False,
            ):
                batch_coords = coords[start : start + self.args.hover_batch_size]
                batch_images = read_wsi_batch(
                    slide,
                    batch_coords,
                    level=self.args.hover_level,
                    patch_size=self.args.hover_patch_size,
                )
                raw_outputs = self.segmentor.model.infer_batch(
                    self.model,
                    batch_images,
                    device=self.device,
                )
                node_types.extend(
                    postprocess_hovernet_batch(
                        raw_outputs,
                        self.postproc_func,
                        executor=self.executor,
                    )
                )
        finally:
            slide.close()

        return torch.as_tensor(node_types, dtype=torch.long)


def infer_hovernet_node_types(
    h5_path: Path,
    wsi_path: Path | None,
    coords: np.ndarray,
    args: argparse.Namespace,
    cache_path: Path | None,
    hovernet_runner: HoverNetRunner | None,
) -> torch.Tensor:
    import torch

    if args.disable_hovernet:
        return torch.zeros(coords.shape[0], dtype=torch.long)

    if cache_path is not None and cache_path.exists() and not args.overwrite:
        cached = torch.load(cache_path, map_location="cpu")
        if cached.shape[0] == coords.shape[0]:
            return cached.to(torch.long)
        print(f"Cache length mismatch for {h5_path.stem}; recomputing node types.")

    if hovernet_runner is None:
        raise RuntimeError("HoverNet runner was not initialized.")
    if wsi_path is None:
        raise RuntimeError("WSI path is required when HoverNet is enabled.")

    node_type_tensor = hovernet_runner.infer_node_types(h5_path, wsi_path, coords)
    if cache_path is not None and not args.no_cache_node_types:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(node_type_tensor, cache_path)
    return node_type_tensor


def build_graph(
    features: np.ndarray,
    coords: np.ndarray,
    node_type: torch.Tensor,
    *,
    radius: int,
    rrwp_walk_length: int,
    exact_knn: bool = False,
) -> GeomData:
    import numpy as np
    import torch
    from torch_geometric.data import Data as GeomData

    import nmslib

    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from preprocess.Graph_Construction.graph_new_rrwp_mp import add_lightweight_rrwp

    features_tensor = torch.as_tensor(features, dtype=torch.float)
    coords_tensor = torch.as_tensor(coords, dtype=torch.float)

    num_patches = coords.shape[0]
    if num_patches == 0:
        raise ValueError("Cannot build graph with zero patches.")

    source_nodes: list[int] = []
    target_nodes: list[int] = []
    spatial_distances: list[float] = []

    if num_patches > 1 and radius > 1:
        topn = min(radius, num_patches)
        if exact_knn:
            from sklearn.neighbors import NearestNeighbors

            nbrs = NearestNeighbors(n_neighbors=topn, metric="euclidean", algorithm="auto")
            nbrs.fit(coords)
            distances, indices = nbrs.kneighbors(coords)
            # Keep the historical edge_attr convention: old HNSW returned squared
            # L2 distances, and the graph code applied sqrt() before saving.
            distances = distances**2
            for src_idx in range(num_patches):
                valid_indices = indices[src_idx, 1:]
                valid_dists = distances[src_idx, 1:]
                source_nodes.extend([src_idx] * len(valid_indices))
                target_nodes.extend(valid_indices.tolist())
                spatial_distances.extend(valid_dists.tolist())
        else:
            hnsw = nmslib.init(space="l2", method="hnsw")
            hnsw.addDataPointBatch(coords.astype(np.float32, copy=False))
            hnsw.createIndex({"M": 16, "post": 0, "efConstruction": 400}, print_progress=False)
            hnsw.setQueryTimeParams({"ef": 90})

            for src_idx in range(num_patches):
                indices, dists = hnsw.knnQuery(coords[src_idx].astype(np.float32), k=topn)
                valid_indices = indices[1:]
                valid_dists = dists[1:]
                source_nodes.extend([src_idx] * len(valid_indices))
                target_nodes.extend(valid_indices.tolist())
                spatial_distances.extend(valid_dists.tolist())

    if source_nodes:
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        src = edge_index[0]
        dst = edge_index[1]
        feature_dist = torch.linalg.vector_norm(features_tensor[src] - features_tensor[dst], dim=1)
        spatial_dist = torch.sqrt(torch.as_tensor(spatial_distances, dtype=torch.float))
        edge_attr = torch.stack(
            [
                node_type[src].float(),
                node_type[dst].float(),
                feature_dist,
                spatial_dist,
            ],
            dim=1,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float)

    graph = GeomData(
        x=features_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        centroid=coords_tensor,
        node_type=node_type,
    )
    return add_lightweight_rrwp(graph, walk_length=rrwp_walk_length)


def write_runtime_summary(output_dir: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    summary_path = output_dir / "runtime_summary.csv"
    fieldnames = [
        "slide",
        "status",
        "num_patches",
        "hovernet_enabled",
        "feature_load_seconds",
        "hovernet_seconds",
        "graph_seconds",
        "total_seconds",
        "worker_id",
        "device",
        "output_path",
        "message",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_single_slide(
    h5_path: Path,
    args: argparse.Namespace,
    *,
    wsi_dir: Path | None,
    output_dir: Path,
    node_type_cache_dir: Path | None,
    hovernet_runner: HoverNetRunner | None,
) -> dict[str, object]:
    import torch

    slide_start = time.perf_counter()
    output_path = output_dir / f"{h5_path.stem}.pt"
    if output_path.exists() and not args.overwrite:
        return {
            "slide": h5_path.stem,
            "status": "skipped",
            "num_patches": "",
            "hovernet_enabled": not args.disable_hovernet,
            "feature_load_seconds": 0,
            "hovernet_seconds": 0,
            "graph_seconds": 0,
            "total_seconds": 0,
            "output_path": str(output_path),
            "message": "Output exists.",
        }

    try:
        load_start = time.perf_counter()
        features, coords = read_trident_h5(h5_path)
        feature_load_seconds = time.perf_counter() - load_start

        wsi_path = (
            find_wsi_path(wsi_dir, h5_path.stem, args.wsi_ext)
            if not args.disable_hovernet
            else None
        )
        cache_path = node_type_cache_dir / f"{h5_path.stem}_node_type.pt" if node_type_cache_dir else None

        hover_start = time.perf_counter()
        node_type = infer_hovernet_node_types(
            h5_path,
            wsi_path,
            coords,
            args,
            cache_path,
            hovernet_runner,
        )
        hovernet_seconds = time.perf_counter() - hover_start

        graph_start = time.perf_counter()
        graph = build_graph(
            features,
            coords,
            node_type,
            radius=args.radius,
            rrwp_walk_length=args.rrwp_walk_length,
            exact_knn=args.exact_knn,
        )
        torch.save(graph, output_path)
        graph_seconds = time.perf_counter() - graph_start

        total_seconds = time.perf_counter() - slide_start
        return {
            "slide": h5_path.stem,
            "status": "success",
            "num_patches": coords.shape[0],
            "hovernet_enabled": not args.disable_hovernet,
            "feature_load_seconds": round(feature_load_seconds, 3),
            "hovernet_seconds": round(hovernet_seconds, 3),
            "graph_seconds": round(graph_seconds, 3),
            "total_seconds": round(total_seconds, 3),
            "output_path": str(output_path),
            "message": "",
        }
    except Exception as exc:
        print(f"{h5_path.stem}: failed: {exc}", flush=True)
        return {
            "slide": h5_path.stem,
            "status": "failed",
            "num_patches": "",
            "hovernet_enabled": not args.disable_hovernet,
            "feature_load_seconds": "",
            "hovernet_seconds": "",
            "graph_seconds": "",
            "total_seconds": round(time.perf_counter() - slide_start, 3),
            "output_path": str(output_path),
            "message": str(exc),
        }


def process_slide_batch(
    worker_id: int,
    h5_paths: list[str],
    args_dict: dict[str, object],
    worker_device: str,
    wsi_dir: str | None,
    output_dir: str,
    node_type_cache_dir: str | None,
) -> list[dict[str, object]]:
    args = argparse.Namespace(**args_dict)
    args.device = worker_device
    ensure_tiatoolbox_import_path(args.tiatoolbox_root)

    resolved_wsi_dir = Path(wsi_dir) if wsi_dir is not None else None
    resolved_output_dir = Path(output_dir)
    resolved_cache_dir = Path(node_type_cache_dir) if node_type_cache_dir is not None else None

    hovernet_runner = None if args.disable_hovernet else HoverNetRunner(args)
    rows: list[dict[str, object]] = []
    try:
        for h5_path in h5_paths:
            row = process_single_slide(
                Path(h5_path),
                args,
                wsi_dir=resolved_wsi_dir,
                output_dir=resolved_output_dir,
                node_type_cache_dir=resolved_cache_dir,
                hovernet_runner=hovernet_runner,
            )
            row["worker_id"] = worker_id
            row["device"] = worker_device
            rows.append(row)
    finally:
        if hovernet_runner is not None:
            hovernet_runner.close()
    return rows


def main() -> None:
    args = parse_args()
    ensure_tiatoolbox_import_path(args.tiatoolbox_root)

    from tqdm import tqdm

    feature_dir = Path(args.feature_dir)
    if not args.disable_hovernet and args.wsi_dir is None:
        raise ValueError("--wsi_dir is required unless --disable_hovernet is set.")
    wsi_dir = Path(args.wsi_dir) if args.wsi_dir is not None else None
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"{feature_dir}_graph_rrwp_inline_hovernet")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.node_type_cache_dir:
        node_type_cache_dir = Path(args.node_type_cache_dir)
    else:
        node_type_cache_dir = output_dir / "node_type_cache"
    if args.disable_hovernet or args.no_cache_node_types:
        node_type_cache_dir = None

    h5_files = sorted(feature_dir.glob("*.h5"))
    if args.limit is not None:
        h5_files = h5_files[: args.limit]
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {feature_dir}.")

    worker_devices = get_worker_devices(args)
    summary_rows: list[dict[str, object]] = []

    if len(worker_devices) <= 1 or len(h5_files) <= 1:
        hovernet_runner = None if args.disable_hovernet else HoverNetRunner(args)
        try:
            for h5_path in tqdm(h5_files, desc="Slides"):
                row = process_single_slide(
                    h5_path,
                    args,
                    wsi_dir=wsi_dir,
                    output_dir=output_dir,
                    node_type_cache_dir=node_type_cache_dir,
                    hovernet_runner=hovernet_runner,
                )
                row["worker_id"] = 0
                row["device"] = args.device
                summary_rows.append(row)
                write_runtime_summary(output_dir, summary_rows)
        finally:
            if hovernet_runner is not None:
                hovernet_runner.close()
    else:
        import multiprocessing as mp

        chunks = split_round_robin(h5_files, len(worker_devices))
        worker_specs = [
            (worker_id, chunk, worker_devices[worker_id])
            for worker_id, chunk in enumerate(chunks)
        ]
        print(
            f"Running {len(worker_specs)} slide workers over devices: "
            f"{', '.join(device for _, _, device in worker_specs)}"
        )

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(worker_specs), mp_context=ctx) as executor:
            futures = [
                executor.submit(
                    process_slide_batch,
                    worker_id,
                    [str(path) for path in chunk],
                    vars(args),
                    device,
                    str(wsi_dir) if wsi_dir is not None else None,
                    str(output_dir),
                    str(node_type_cache_dir) if node_type_cache_dir is not None else None,
                )
                for worker_id, chunk, device in worker_specs
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Slide workers"):
                rows = future.result()
                summary_rows.extend(rows)
                summary_rows.sort(key=lambda row: str(row["slide"]))
                write_runtime_summary(output_dir, summary_rows)

    write_runtime_summary(output_dir, summary_rows)
    failed = sum(row["status"] == "failed" for row in summary_rows)
    print(f"Done. Graphs and runtime_summary.csv saved to {output_dir}")
    if failed:
        raise SystemExit(f"{failed} slide(s) failed; see runtime_summary.csv.")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()
