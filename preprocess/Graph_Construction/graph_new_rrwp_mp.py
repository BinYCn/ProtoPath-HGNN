"""Construct PyG spatial graphs with node-type labels and RRWP encodings."""

import argparse
import os
from functools import partial
from multiprocessing import Pool, cpu_count, get_context
from pathlib import Path

import joblib
import nmslib
import numpy as np
import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from tqdm import tqdm


def add_lightweight_rrwp(data, walk_length=8):
    """Add diagonal entries of I, P, ..., P^(walk_length-1) as data.rrwp.

    P is the row-normalized directed spatial adjacency. Sparse powers may
    densify for large graphs, so memory consumption still grows with graph size.
    """
    num_nodes = data.num_nodes
    adj = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(num_nodes, num_nodes))
    degree = adj.sum(dim=1)
    inverse_degree = torch.where(degree > 0, 1.0 / degree, torch.zeros_like(degree))
    transition = adj * inverse_degree.view(-1, 1)
    diagonals = [torch.ones(num_nodes, dtype=torch.float)]
    power = transition
    for step in range(1, walk_length):
        diagonals.append(power.to_dense().diagonal())
        if step < walk_length - 1:
            power = power @ transition
    data.rrwp = torch.stack(diagonals, dim=-1)
    return data


def process_single_coord(coord_idx_pair, hover_file, wsi_name, patch_roots, missing_type):
    """Look up the most frequent cell type for one patch."""
    index, coord = coord_idx_pair
    relative_path = os.path.join(wsi_name, f'x_{coord[0]}_y_{coord[1]}.png')
    for root in patch_roots:
        patch_key = os.path.join(root, relative_path)
        if patch_key not in hover_file:
            continue
        dat_path = os.fspath(hover_file[patch_key]) + '.dat'
        patch_data = joblib.load(dat_path)
        types = [cell['type'] for cell in patch_data.values() if 'type' in cell]
        if not types:
            return index, 0
        types = np.asarray(types)
        if not np.issubdtype(types.dtype, np.integer) or np.any(types < 0):
            raise ValueError(f'Invalid cell type in {dat_path}: expected non-negative integers')
        return index, int(np.bincount(types).argmax())
    if missing_type == 'zero':
        return index, 0
    raise KeyError(f'Patch not found in file_map.dat: {relative_path} (searched {patch_roots})')


def pt2graph(hover_file, features, coords, wsi_name, patch_roots, radius=9,
             walk_length=8, node_processes=1, missing_type='error'):
    """Build the original directed k-nearest-neighbor graph for one slide."""
    features = torch.as_tensor(features).detach().cpu().float().numpy()
    coords = torch.as_tensor(coords).detach().cpu().numpy()
    if features.ndim != 2 or coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError('features must have shape [N, D] and coords must have shape [N, 2]')
    if len(features) != len(coords) or len(coords) == 0:
        raise ValueError('features and coords must have the same nonzero number of rows')
    if not np.isfinite(features).all() or not np.isfinite(coords).all():
        raise ValueError('features and coords must contain finite values')

    process_coord = partial(process_single_coord, hover_file=hover_file, wsi_name=wsi_name,
                            patch_roots=patch_roots, missing_type=missing_type)
    indexed_coords = enumerate(coords)
    if node_processes > 1:
        with get_context('spawn').Pool(node_processes) as pool:
            node_results = list(pool.imap(process_coord, indexed_coords))
    else:
        node_results = list(map(process_coord, indexed_coords))
    node_types = torch.tensor([value for _, value in node_results], dtype=torch.long)

    source, target, spatial_distance = [], [], []
    if len(coords) > 1:
        index = nmslib.init(space='l2', method='hnsw')
        index.addDataPointBatch(coords.astype(np.float32))
        index.createIndex({'M': 16, 'post': 0, 'efConstruction': 400}, print_progress=False)
        index.setQueryTimeParams({'ef': 90})
        for node_id, coord in enumerate(coords):
            indices, distances = index.knnQuery(coord.astype(np.float32), k=min(radius, len(coords)))
            # The original code skips the first (self) neighbor.
            source.extend([node_id] * len(indices[1:]))
            target.extend(indices[1:].tolist())
            spatial_distance.extend(distances[1:].tolist())
    edge_index = torch.tensor([source, target], dtype=torch.long)
    feature_tensor = torch.from_numpy(features)
    if source:
        src, dst = edge_index
        feature_distance = torch.linalg.vector_norm(feature_tensor[src] - feature_tensor[dst], dim=1)
        edge_attr = torch.stack((node_types[src].float(), node_types[dst].float(), feature_distance,
                                 torch.tensor(spatial_distance, dtype=torch.float).clamp_min(0).sqrt()), dim=1)
    else:
        edge_attr = torch.empty((0, 4), dtype=torch.float)
    graph = Data(x=feature_tensor, edge_index=edge_index, edge_attr=edge_attr,
                 centroid=torch.as_tensor(coords, dtype=torch.float), node_type=node_types)
    return add_lightweight_rrwp(graph, walk_length=walk_length)


def process_single_file(filename, args, node_processes=1):
    """Load one slide and write its graph; return a status message on failure."""
    slide_name = Path(filename).stem
    try:
        features = torch.load(os.path.join(args.feature_path, filename), map_location='cpu')
        coords = torch.load(os.path.join(args.coord_path, slide_name + '_patches.pt'), map_location='cpu')
        mapping = joblib.load(os.path.join(args.hover_path, slide_name, 'file_map.dat'))
        hover_file = dict(mapping)
        graph = pt2graph(hover_file, features, coords, slide_name, args.patch_roots,
                         args.radius, args.walk_length, node_processes, args.missing_type)
        torch.save(graph, os.path.join(args.save_path, filename))
        return filename, None
    except Exception as exc:
        return filename, f'{type(exc).__name__}: {exc}'


def build_parser():
    parser = argparse.ArgumentParser(description='Construct spatial graphs with node RRWP features.')
    parser.add_argument('--feature_path', required=True, help='Directory containing <slide>.pt feature tensors')
    parser.add_argument('--coord_path', required=True, help='Directory containing <slide>_patches.pt coordinates')
    parser.add_argument('--hover_path', required=True, help='Directory containing <slide>/file_map.dat')
    parser.add_argument('--save_path', required=True, help='Output directory for PyG graphs')
    parser.add_argument('--patch_roots', nargs='+', required=True,
                        help='Patch path prefixes used in file_map.dat, searched in order')
    parser.add_argument('--radius', type=int, default=9, help='Number of nearest neighbors including self (default: 9)')
    parser.add_argument('--walk_length', type=int, default=8, help='Number of node RRWP features (default: 8)')
    parser.add_argument('--mode', choices=['multi', 'single'], default='multi',
                        help='Parallelize over files (multi) or process files sequentially (single)')
    parser.add_argument('--n_file_processes', type=int, default=min(cpu_count(), 8), help='Workers for multi mode')
    parser.add_argument('--node_processes', type=int, default=1, help='Node workers in single mode')
    parser.add_argument('--missing_type', choices=['error', 'zero'], default='error',
                        help='How to handle a patch missing from file_map.dat (default: error)')
    return parser


def main():
    args = build_parser().parse_args()
    if args.radius < 1 or args.walk_length < 1 or args.n_file_processes < 1 or args.node_processes < 1:
        raise ValueError('radius, walk_length and process counts must be positive')
    if args.mode == 'multi' and args.node_processes != 1:
        raise ValueError('node_processes must be 1 in multi mode; nested pools are not supported')
    for path in (args.feature_path, args.coord_path, args.hover_path):
        if not os.path.isdir(path):
            raise FileNotFoundError(f'Input directory does not exist: {path}')
    os.makedirs(args.save_path, exist_ok=True)
    filenames = sorted(path.name for path in Path(args.feature_path).glob('*.pt') if path.is_file())
    if not filenames:
        raise ValueError(f'No .pt feature files found in {args.feature_path}')
    process = partial(process_single_file, args=args,
                      node_processes=args.node_processes if args.mode == 'single' else 1)
    if args.mode == 'multi':
        with get_context('spawn').Pool(args.n_file_processes) as pool:
            results = list(tqdm(pool.imap(process, filenames), total=len(filenames), desc='Slides'))
    else:
        results = [process(filename) for filename in tqdm(filenames, desc='Slides')]
    failures = [(name, error) for name, error in results if error is not None]
    for name, error in failures:
        print(f'{name}: {error}')
    print(f'Completed {len(results) - len(failures)} of {len(results)} slides.')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
