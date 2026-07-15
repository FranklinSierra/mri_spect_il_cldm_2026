#!/usr/bin/env python3
"""Preflight checks for data_rois (fixed patch shape, MRI/SPECT/mask alignment)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO / "data_rois")
    args = parser.parse_args()
    root = args.data_root
    mri_dir = root / "whole_MRI"
    spect_dir = root / "whole_Abeta"
    mask_dir = root / "mask_basal_ganglia"

    for p in (mri_dir, spect_dir, mask_dir):
        if not p.is_dir():
            print(f"FAIL: missing {p}")
            return 1

    ids = sorted(p.name.replace(".nii.gz", "") for p in mri_dir.glob("*.nii.gz"))
    if not ids:
        print(f"FAIL: no subjects in {mri_dir}")
        return 1

    shapes = set()
    mismatches = []
    for sid in ids:
        m = nib.load(str(mri_dir / f"{sid}.nii.gz")).shape
        s = nib.load(str(spect_dir / f"{sid}.nii.gz")).shape
        k = nib.load(str(mask_dir / f"{sid}.nii.gz")).shape
        shapes.add(m)
        if m != s or m != k:
            mismatches.append((sid, m, s, k))

    print(f"data_root={root}")
    print(f"n_subjects={len(ids)}")
    print(f"unique_shapes={shapes}")
    if len(shapes) != 1:
        print("FAIL: MRI shapes not identical across subjects")
        return 1
    if mismatches:
        print(f"FAIL: {len(mismatches)} MRI/SPECT/mask mismatches")
        return 1

    shape = next(iter(shapes))
    import config
    from code.volume_shapes import clear_shape_cache, get_latent_shape, get_working_volume_shape

    if os.getenv("ILCLDM_SKIP_CROP", "0") == "1" or str(root).endswith("data_rois"):
        os.environ.setdefault("ILCLDM_SKIP_CROP", "1")
        os.environ.setdefault("ILCLDM_ROI_VOLUME_SHAPE", f"{shape[0]},{shape[1]},{shape[2]}")
        clear_shape_cache()
        import importlib

        importlib.reload(config)
        from code import volume_shapes

        volume_shapes.clear_shape_cache()
        working = get_working_volume_shape()
        latent = get_latent_shape()
        print(f"working_volume_shape={working} latent_shape={latent}")
    else:
        import torch
        import torch.nn.functional as F

        t = torch.zeros(1, 1, *shape)
        t = F.max_pool3d(t, 2, 2)
        t = F.max_pool3d(t, 2, 2)
        latent = tuple(int(x) for x in t.shape[2:])
        print(f"volume_shape={shape} -> latent_shape={latent}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
