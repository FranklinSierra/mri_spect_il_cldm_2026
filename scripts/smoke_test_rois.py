#!/usr/bin/env python3
"""Smoke test for data_rois profile (shapes, mask, E2 batch forward)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# ROI profile env (must be set before importing config-backed modules).
os.environ.setdefault("ILCLDM_SKIP_CROP", "1")
os.environ.setdefault("ILCLDM_WHOLE_MRI", "data_rois/whole_MRI")
os.environ.setdefault("ILCLDM_WHOLE_ABETA", "data_rois/whole_Abeta")
os.environ.setdefault("ILCLDM_ROI_VOLUME_SHAPE", "105,102,102")
os.environ.setdefault("ILCLDM_REFERENCE_NIFTI", "data_rois/whole_MRI/3102.nii.gz")
os.environ.setdefault("ILCLDM_ROI_MASK_PATH", "data_rois/roi_mask_unet.nii.gz")
os.environ.setdefault("ILCLDM_LATENT_ABETA", "data_rois/latent_Abeta/")
os.environ.setdefault("ILCLDM_LATENT_CALIBRATION", "1")
os.environ.setdefault("ILCLDM_UNET_ROI_CHANNEL", "1")
os.environ.setdefault("ILCLDM_USE_TRAIN_LABEL", "1")
os.environ.setdefault("ILCLDM_DEVICE", "cpu")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
# Jupyter / scripts may shadow stdlib `code`; ensure package resolves to repo/code/.
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

import config
from code.roi_mask_utils import load_roi_latent_buffer, load_roi_mask_for_pipeline, load_roi_mask_volume
from code.volume_shapes import clear_shape_cache, get_latent_shape, get_volume_shape, get_working_volume_shape
from dataset import TwoDataset, get_latent_stats, resolve_nifti_path
from dataset import nifti_to_numpy
from model import MRIEncoder, build_unet
from main import Diffusion


def main() -> int:
    clear_shape_cache()
    mask_path = Path(config.ROI_MASK_PATH)
    if not mask_path.is_file():
        print(f"FAIL: run scripts/aggregate_roi_unet_mask.py first ({mask_path})")
        return 1

    vol_shape = get_volume_shape()
    working_shape = get_working_volume_shape()
    latent_shape = get_latent_shape()
    print(f"volume_shape={vol_shape} working_shape={working_shape} latent_shape={latent_shape}")

    train_ids = Path(config.train).read_text().split()
    sid = train_ids[0]
    mri = nifti_to_numpy(resolve_nifti_path(config.whole_MRI, sid))
    mask_disk = load_roi_mask_volume(mask_path)
    assert mri.shape == vol_shape, (mri.shape, vol_shape)
    assert mask_disk.shape == vol_shape, (mask_disk.shape, vol_shape)
    print(f"OK: on-disk MRI and mask shape match for {sid}")

    mask = load_roi_mask_for_pipeline(mask_path)
    assert mask.shape == working_shape, (mask.shape, working_shape)

    roi_latent = load_roi_latent_buffer(mask_path)
    assert tuple(roi_latent.shape[-3:]) == latent_shape
    print(f"OK: roi_latent buffer shape {tuple(roi_latent.shape)}")

    ds = TwoDataset(root_Abeta=config.whole_Abeta, task=config.train, stage="validation")
    mri_b, spect_b, name, label = ds[0]
    assert mri_b.shape == working_shape
    assert spect_b.shape == working_shape
    print(f"OK: TwoDataset sample MRI/SPECT shapes {mri_b.shape}")

    diffusion = Diffusion()
    unet = build_unet()
    mri_enc = MRIEncoder()
    mri_t = torch.tensor(np.expand_dims(mri_b, (0, 1)), dtype=torch.float32)
    cond = mri_enc(mri_t)
    latent_shape_t = get_latent_shape()
    lat_cal = torch.randn(1, 1, *latent_shape_t)
    t = torch.tensor([50], dtype=torch.long)
    x_t, _ = diffusion.noise_images(lat_cal, t)
    label_t = torch.tensor([float(label[0])], dtype=torch.float32)
    with torch.no_grad():
        eps = unet(x_t, cond, t, label_t, precomputed_condition=True)
    assert eps.shape == lat_cal.shape
    print(f"OK: UNet forward {tuple(eps.shape)}")

    if Path(config.latent_Abeta).is_dir() and list(Path(config.latent_Abeta).glob("*.nii*")):
        mean, std = get_latent_stats()
        print(f"OK: latent stats mean={mean:.4f} std={std:.4f}")

        ds_lat = TwoDataset(root_Abeta=config.latent_Abeta, stage="validation")
        mri_b2, latent_b, _, label2 = ds_lat[0]
        mri_t2 = torch.tensor(np.expand_dims(mri_b2, (0, 1)), dtype=torch.float32)
        lat_t = torch.tensor(np.expand_dims(latent_b, (0, 1)), dtype=torch.float32)
        lat_norm = diffusion.normalize_latent(lat_t)
        cond2 = mri_enc(mri_t2)
        t2 = torch.tensor([100], dtype=torch.long)
        x_t2, noise2 = diffusion.noise_images(lat_norm, t2)
        label2_t = torch.tensor([float(label2[0])], dtype=torch.float32)
        with torch.no_grad():
            pred = unet(x_t2, cond2, t2, label2_t, precomputed_condition=True)
        assert pred.shape == noise2.shape
        print(f"OK: E2 calibrated batch forward {tuple(pred.shape)}")
    else:
        print("SKIP: latent_Abeta not populated yet (run E1 encode after training)")

    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
