#!/usr/bin/env python3
"""Extract patch PNGs from WSIs using TRIDENT H5 or PT coordinate files."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def load_coords(path):
    import numpy as np

    if path.suffix.lower() == '.h5':
        import h5py
        with h5py.File(path, 'r') as handle:
            coords = handle['coords'][:]
    else:
        import torch
        coords = torch.as_tensor(torch.load(path, map_location='cpu')).cpu().numpy()
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError(f'{path}: expected finite coordinates with shape [N, 2]')
    if not np.equal(coords, np.floor(coords)).all() or (coords < 0).any():
        raise ValueError(f'{path}: coordinates must be non-negative integers')
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError(f'{path}: duplicate coordinates would overwrite patch images')
    return coords


def process_slide(coord_path, wsi_path, output_dir, patch_size, level):
    import openslide

    coords = load_coords(coord_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    slide = openslide.open_slide(str(wsi_path))
    try:
        if level >= slide.level_count:
            raise ValueError(f'{wsi_path}: level {level} is unavailable')
        for x, y in coords:
            x, y = int(x), int(y)
            patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert('RGB')
            patch.save(output_dir / f'x_{x}_y_{y}.png')
    finally:
        slide.close()
    return len(coords)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coord_dir', required=True, help='Directory of <slide>_patches.h5 or .pt coordinate files')
    parser.add_argument('--wsi_dir', required=True, help='Directory containing WSIs')
    parser.add_argument('--output_dir', required=True, help='Directory for per-slide patch folders')
    parser.add_argument('--wsi_ext', default='.svs', help='WSI extension (default: .svs; matched case-insensitively)')
    parser.add_argument('--patch_size', type=int, default=512, help='Read size in pixels at --level (default: 512)')
    parser.add_argument('--level', type=int, default=0, help='OpenSlide read level (default: 0)')
    parser.add_argument('--workers', type=int, default=4, help='Number of slide processes (default: 4)')
    args = parser.parse_args()
    if args.patch_size < 1 or args.level < 0 or args.workers < 1:
        parser.error('--patch_size and --workers must be positive; --level must be non-negative')
    coord_dir, wsi_dir, output_dir = (Path(p).resolve() for p in (args.coord_dir, args.wsi_dir, args.output_dir))
    if not coord_dir.is_dir() or not wsi_dir.is_dir():
        parser.error('--coord_dir and --wsi_dir must be existing directories')
    extension = args.wsi_ext if args.wsi_ext.startswith('.') else '.' + args.wsi_ext
    coord_paths = sorted(p for p in coord_dir.iterdir() if p.is_file() and p.name.endswith(('_patches.h5', '_patches.pt')))
    if not coord_paths:
        parser.error(f'No *_patches.h5 or *_patches.pt files found in {coord_dir}')
    wsis = {}
    for path in wsi_dir.iterdir():
        if path.is_file() and path.suffix.lower() == extension.lower():
            wsis.setdefault(path.stem, []).append(path)
    tasks = []
    failures = []
    seen = set()
    for coord_path in coord_paths:
        stem = coord_path.name.rsplit('_patches.', 1)[0]
        if stem in seen:
            failures.append(f'{stem}: multiple coordinate files found')
            continue
        seen.add(stem)
        matches = wsis.get(stem, [])
        if len(matches) != 1:
            failures.append(f'{stem}: expected one matching WSI, found {len(matches)}')
            continue
        tasks.append((coord_path, matches[0], output_dir / stem, args.patch_size, args.level))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_slide, *task): task[0] for task in tasks}
        for future in as_completed(futures):
            path = futures[future]
            try:
                print(f'{path.name}: wrote {future.result()} patches', flush=True)
            except Exception as exc:
                failures.append(f'{path.name}: {exc}')
    if failures:
        for failure in failures:
            print(f'ERROR: {failure}', flush=True)
        raise SystemExit(f'{len(failures)} slide(s) failed')


if __name__ == '__main__':
    main()
