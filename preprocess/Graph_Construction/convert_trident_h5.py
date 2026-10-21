#!/usr/bin/env python3
"""Convert TRIDENT feature H5 files to graph feature and coordinate tensors."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature_h5_dir', required=True, help='Directory containing <slide>.h5 feature files')
    parser.add_argument('--coord_h5_dir', required=True, help='Directory containing <slide>_patches.h5 coordinate files')
    parser.add_argument('--output_dir', required=True, help='Directory for features/ and coordinates/ PT outputs')
    args = parser.parse_args()

    feature_h5_dir = Path(args.feature_h5_dir)
    coord_h5_dir = Path(args.coord_h5_dir)
    if not feature_h5_dir.is_dir() or not coord_h5_dir.is_dir():
        parser.error('--feature_h5_dir and --coord_h5_dir must be existing directories')
    feature_paths = sorted(feature_h5_dir.glob('*.h5'))
    if not feature_paths:
        parser.error(f'No .h5 feature files found in {feature_h5_dir}')

    output_dir = Path(args.output_dir)
    feature_dir, coord_dir = output_dir / 'features', output_dir / 'coordinates'
    feature_dir.mkdir(parents=True, exist_ok=True)
    coord_dir.mkdir(parents=True, exist_ok=True)

    for feature_path in feature_paths:
        with h5py.File(feature_path, 'r') as handle:
            features, coords = handle['features'][:], handle['coords'][:]
        with h5py.File(coord_h5_dir / f'{feature_path.stem}_patches.h5', 'r') as handle:
            patch_coords = handle['coords'][:]
        if len(features) != len(coords) or not np.array_equal(coords, patch_coords):
            raise ValueError(f'Coordinate mismatch: {feature_path.name}')
        if not np.isfinite(coords).all() or not np.equal(coords, np.floor(coords)).all():
            raise ValueError(f'Coordinates must be finite integers: {feature_path.name}')
        torch.save(torch.from_numpy(features), feature_dir / f'{feature_path.stem}.pt')
        torch.save(torch.from_numpy(coords.astype(np.int64)), coord_dir / f'{feature_path.stem}_patches.pt')
        print(f'Converted {feature_path.name}')


if __name__ == '__main__':
    main()
