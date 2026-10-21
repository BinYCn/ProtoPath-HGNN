#!/usr/bin/env python3
"""Run TIA Toolbox HoVer-Net on per-slide patch directories."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True, help='Directory containing per-slide patch directories')
    parser.add_argument('--save_dir', required=True, help='Output directory for per-slide HoVer-Net results')
    parser.add_argument('--gpu_id', type=int, default=0, help='Physical GPU index (default: 0)')
    parser.add_argument('--model', default='hovernet_fast-pannuke', help='TIA Toolbox pretrained model')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--loader_workers', type=int, default=16)
    parser.add_argument('--postproc_workers', type=int, default=16)
    args = parser.parse_args()
    if args.gpu_id < 0 or args.batch_size < 1 or args.loader_workers < 0 or args.postproc_workers < 0:
        parser.error('GPU index and worker counts must be non-negative; batch size must be positive')
    data_dir, save_dir = Path(args.data_dir).resolve(), Path(args.save_dir).resolve()
    if not data_dir.is_dir():
        parser.error(f'Patch directory does not exist: {data_dir}')
    slide_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if not slide_dirs:
        parser.error(f'No slide directories found in {data_dir}')
    # Set before importing TIA Toolbox or PyTorch so the selected GPU is logical cuda:0.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor

    segmentor = NucleusInstanceSegmentor(
        pretrained_model=args.model, num_loader_workers=args.loader_workers,
        num_postproc_workers=args.postproc_workers, batch_size=args.batch_size,
    )
    for slide_dir in slide_dirs:
        images = sorted(str(path.resolve()) for path in slide_dir.glob('*.png') if path.is_file())
        if not images:
            raise ValueError(f'No PNG patches found in {slide_dir}')
        output = save_dir / slide_dir.name
        if output.exists():
            raise FileExistsError(f'{output} already exists; choose a new output directory')
        print(f'{slide_dir.name}: segmenting {len(images)} patches', flush=True)
        segmentor.predict(images, save_dir=str(output), mode='tile', device='cuda:0', crash_on_exception=True)


if __name__ == '__main__':
    main()
