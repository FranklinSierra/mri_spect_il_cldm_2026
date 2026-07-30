#!/usr/bin/env python3
"""Encode SPECT E1 ROI volumes to h64 cache (+ optional gap1 stats). Does not train E2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--aae-checkpoint",
        default="result/exp_nm_s1_spect_rois/AAE.pth.tar",
        help="Frozen SPECT AAE checkpoint",
    )
    p.add_argument("--output-dir", default="data_rois/latent_Abeta_h64")
    p.add_argument(
        "--stats-path",
        default="data_info/latent_stats_latent_Abeta_rois_h64.json",
        help="Global mean/std over raw h64 voxels",
    )
    p.add_argument(
        "--gap1-stats-path",
        default="data_info/latent_stats_latent_Abeta_rois_h64_gap1.json",
        help="Global mean/std after channel-mean GAP → 1ch",
    )
    p.add_argument("--splits", nargs="+", default=None, help="Override split files (default train/val/test)")
    return p.parse_args()


def main():
    args = parse_args()
    # ROI profile must be active before importing config-dependent helpers.
    import os

    os.environ.setdefault("ILCLDM_SKIP_CROP", "1")
    os.environ.setdefault("ILCLDM_WHOLE_ABETA", "data_rois/whole_Abeta")
    os.environ.setdefault("ILCLDM_ROI_VOLUME_SHAPE", "105,102,102")
    os.environ.setdefault("ILCLDM_REFERENCE_NIFTI", "data_rois/whole_MRI/3102.nii.gz")

    # Avoid stdlib `code` shadowing the local package when run as a script.
    sys.modules.pop("code", None)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import config
    from code.volume_shapes import clear_shape_cache, get_latent_shape
    from dataset import OneDataset, read_list
    from model import AAE
    from torch.utils.data import DataLoader
    from utils import load_checkpoint
    import torch.optim as optim

    clear_shape_cache()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.aae_checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing SPECT AAE checkpoint: {ckpt}")

    model = AAE().to(config.device).eval()
    opt = optim.Adam(model.parameters(), lr=1e-4)
    load_checkpoint(str(ckpt), model, opt, 1e-4)

    split_files = args.splits or [config.train, config.validation, config.test]
    ids: list[str] = []
    for task in split_files:
        for image_id in read_list(task):
            if image_id not in ids:
                ids.append(str(image_id))

    # Encode via OneDataset to reuse crop/pad.
    seen = set()
    n_saved = 0
    sum_h = 0.0
    sumsq_h = 0.0
    count_h = 0
    sum_g = 0.0
    sumsq_g = 0.0
    count_g = 0
    latent_shape = get_latent_shape()

    for task in split_files:
        dataset = OneDataset(root_Abeta=config.whole_Abeta, task=task, stage="encoding")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        for volume, name in tqdm(loader, desc=f"encode_h64:{Path(task).name}"):
            subject = str(name[0]).replace(".nii.gz", "").replace(".nii", "")
            if subject in seen:
                continue
            seen.add(subject)
            # DataLoader batch: volume (B,D,H,W)
            x = volume.unsqueeze(1).to(config.device).float()
            with torch.no_grad():
                latent, h64 = model.encoder(x, return_hidden=True)
            if tuple(h64.shape[2:]) != latent_shape:
                raise ValueError(f"Unexpected h64 spatial {tuple(h64.shape)} vs latent {latent_shape}")
            if tuple(latent.shape[1:]) != (1, *latent_shape):
                raise ValueError(f"Unexpected latent shape {tuple(latent.shape)}")

            arr = h64[0].detach().cpu().numpy().astype(np.float32)
            np.save(out_dir / f"{subject}.npy", arr)
            n_saved += 1

            flat = arr.reshape(-1).astype(np.float64)
            sum_h += float(flat.sum())
            sumsq_h += float(np.square(flat).sum())
            count_h += flat.size

            gap = arr.mean(axis=0).astype(np.float64).reshape(-1)
            sum_g += float(gap.sum())
            sumsq_g += float(np.square(gap).sum())
            count_g += gap.size

    if n_saved == 0:
        raise RuntimeError("No SPECT h64 latents were written")

    mean_h = sum_h / count_h
    var_h = max(sumsq_h / count_h - mean_h * mean_h, 0.0)
    std_h = float(np.sqrt(var_h)) or 1.0

    mean_g = sum_g / count_g
    var_g = max(sumsq_g / count_g - mean_g * mean_g, 0.0)
    std_g = float(np.sqrt(var_g)) or 1.0

    stats_h = {
        "mean": mean_h,
        "std": std_h,
        "latent_dir": str(out_dir),
        "n_files": n_saved,
        "feature": "h64",
        "shape_per_file": [64, *latent_shape],
    }
    stats_g = {
        "mean": mean_g,
        "std": std_g,
        "latent_dir": str(out_dir),
        "n_files": n_saved,
        "feature": "h64_gap1",
        "shape_per_file": [1, *latent_shape],
        "note": "stats over channel-mean(h64)",
    }
    Path(args.stats_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.gap1_stats_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_h, f, indent=2)
    with open(args.gap1_stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_g, f, indent=2)

    print(f"Saved {n_saved} h64 maps → {out_dir}")
    print(f"h64 stats → {args.stats_path} mean={mean_h:.6g} std={std_h:.6g}")
    print(f"gap1 stats → {args.gap1_stats_path} mean={mean_g:.6g} std={std_g:.6g}")


if __name__ == "__main__":
    main()
