#!/usr/bin/env python3
"""Download a Kaggle dataset via kagglehub and create a small 100/100 image subset.

Usage:
    python scripts/download_and_sample_kaggle.py --dataset birdy654/cifake-real-and-ai-generated-synthetic-images

The script will download/unzip the dataset into /home/codespace/datasets/DAIA/raw (unless overridden),
then copy up to `--n` images per class into /home/codespace/datasets/DAIA/dummy_subset/{real,ai_generated}.
"""
import argparse
import os
import random
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Kaggle dataset slug owner/dataset')
    parser.add_argument('--n', type=int, default=100, help='Number of images per class')
    parser.add_argument('--dest', default='/home/codespace/datasets/DAIA', help='Base dataset directory')
    args = parser.parse_args()

    try:
        import kagglehub
    except Exception as e:
        print('kagglehub is not installed or failed to import:', e)
        raise

    raw_dir = Path(args.dest) / 'raw'
    subset_dir = Path(args.dest) / 'dummy_subset'
    real_out = subset_dir / 'real'
    ai_out = subset_dir / 'ai_generated'

    raw_dir.mkdir(parents=True, exist_ok=True)
    subset_dir.mkdir(parents=True, exist_ok=True)
    real_out.mkdir(parents=True, exist_ok=True)
    ai_out.mkdir(parents=True, exist_ok=True)

    print(f'Downloading dataset {args.dataset} into {raw_dir} (this may take a while)')
    # Try common kagglehub signatures; handle returned archive if needed
    path = None
    try:
        # try signature with dest_path and unzip
        path = kagglehub.dataset_download(args.dataset, dest_path=str(raw_dir), unzip=True)
    except TypeError:
        try:
            # try minimal signature
            path = kagglehub.dataset_download(args.dataset)
        except Exception as e:
            print('Download failed:', e)
            raise

    print('Download result:', path)

    # If a zip file was returned or exists in raw_dir, unzip it
    import zipfile
    # if path points to a zip, extract it
    try:
        if path and isinstance(path, str) and path.endswith('.zip'):
            print('Found zip archive, extracting...')
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall(str(raw_dir))
        else:
            # look for zip files in raw_dir and unzip the first one
            zips = list(raw_dir.glob('*.zip'))
            if zips:
                print('Found zip(s) in raw_dir, extracting first one:', zips[0])
                with zipfile.ZipFile(zips[0], 'r') as z:
                    z.extractall(str(raw_dir))
    except Exception as e:
        print('Unzip step failed:', e)

    # Walk raw_dir to find candidate image folders named like 'real' and 'ai_generated' or similar
    candidates = {'real': [], 'ai_generated': []}
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                full = Path(root) / f
                # heuristics: parent folder names containing 'real' or 'ai' or 'generated'
                parent = full.parent.name.lower()
                if 'real' in parent:
                    candidates['real'].append(full)
                elif 'ai' in parent or 'generated' in parent or 'synthetic' in parent:
                    candidates['ai_generated'].append(full)

    print(f'Found {len(candidates["real"])} real images and {len(candidates["ai_generated"])} ai-generated images (heuristic).')

    def sample_and_copy(lst, out_dir, n):
        if not lst:
            print('No candidates found for', out_dir)
            return 0
        chosen = lst if len(lst) <= n else random.sample(lst, n)
        for src in chosen:
            dst = out_dir / src.name
            shutil.copy2(src, dst)
        return len(chosen)

    r = sample_and_copy(candidates['real'], real_out, args.n)
    a = sample_and_copy(candidates['ai_generated'], ai_out, args.n)

    print(f'Copied {r} real and {a} ai images into {subset_dir}.')
    print('Done.')

if __name__ == '__main__':
    main()
