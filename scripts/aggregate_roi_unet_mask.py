#!/usr/bin/env python3
"""Aggregate per-subject ROI masks (train split) into one UNet 3rd-channel mask."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dataset import read_list


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO / "data_rois")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=REPO / "data_info" / "train.txt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "data_rois" / "roi_mask_unet.nii.gz",
    )
    parser.add_argument(
        "--mode",
        choices=("union", "mean"),
        default="union",
        help="union: any voxel positive in any train subject; mean: average then threshold 0.5",
    )
    args = parser.parse_args()

    mask_dir = args.data_root / "mask_basal_ganglia"
    mri_dir = args.data_root / "whole_MRI"
    train_ids = read_list(str(args.train_file))
    if not train_ids:
        print("FAIL: empty train list")
        return 1

    ref_affine = None
    accum = None
    count = 0
    for sid in train_ids:
        mask_path = mask_dir / f"{sid}.nii.gz"
        if not mask_path.is_file():
            print(f"WARN: missing mask for {sid}, skipping")
            continue
        img = nib.load(str(mask_path))
        data = (img.get_fdata(dtype=np.float32) > 0.5).astype(np.float32)
        if ref_affine is None:
            ref_affine = img.affine
            accum = np.zeros_like(data, dtype=np.float32)
        if data.shape != accum.shape:
            print(f"FAIL: shape mismatch for {sid}: {data.shape} vs {accum.shape}")
            return 1
        accum += data
        count += 1

    if accum is None or count == 0:
        print("FAIL: no masks aggregated")
        return 1

    if args.mode == "union":
        out = (accum > 0).astype(np.float32)
    else:
        out = (accum / count >= 0.5).astype(np.float32)

    if out.sum() == 0:
        print("FAIL: aggregated mask is empty")
        return 1

    # Use affine from a reference MRI for consistency with volumes.
    ref_mri = nib.load(str(mri_dir / f"{train_ids[0]}.nii.gz"))
    out_img = nib.Nifti1Image(out, ref_mri.affine, header=ref_mri.header.copy())
    out_img.set_data_dtype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(args.out))
    print(f"Wrote {args.out} from {count} train masks ({args.mode}), shape={out.shape}, voxels={int(out.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
