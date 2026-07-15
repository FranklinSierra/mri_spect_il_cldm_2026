"""Tests for ROI patch shape helpers (SKIP_CROP profile)."""

import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def roi_env(monkeypatch, tmp_path):
    vol = np.zeros((32, 32, 32), dtype=np.float32)
    vol[10:22, 10:22, 10:22] = 1.0
    mask_path = tmp_path / "roi_mask.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(mask_path))
    monkeypatch.setenv("ILCLDM_SKIP_CROP", "1")
    monkeypatch.setenv("ILCLDM_ROI_VOLUME_SHAPE", "32,32,32")
    monkeypatch.setenv("ILCLDM_ROI_MASK_PATH", str(mask_path))
    import config
    from code import volume_shapes

    volume_shapes.clear_shape_cache()
    importlib = __import__("importlib")
    importlib.reload(config)
    volume_shapes.clear_shape_cache()
    yield mask_path
    volume_shapes.clear_shape_cache()
    monkeypatch.delenv("ILCLDM_SKIP_CROP", raising=False)
    monkeypatch.delenv("ILCLDM_ROI_VOLUME_SHAPE", raising=False)


def test_volume_and_latent_shape_roi(roi_env):
    from code.volume_shapes import get_latent_shape, get_volume_shape

    assert get_volume_shape() == (32, 32, 32)
    assert get_latent_shape() == (8, 8, 8)


def test_crop_volume_to_raw_shape(roi_env):
    from code.volume_shapes import crop_volume_to_raw_shape, get_volume_shape, get_working_volume_shape

    working = get_working_volume_shape()
    raw = get_volume_shape()
    padded = np.zeros(working, dtype=np.float32)
    padded[: raw[0], : raw[1], : raw[2]] = 1.0
    cropped = crop_volume_to_raw_shape(padded)
    assert cropped.shape == raw
    assert cropped.sum() == padded.sum()


def test_load_roi_latent_buffer_roi_profile(roi_env):
    from code.roi_mask_utils import load_roi_latent_buffer
    from code.volume_shapes import get_latent_shape

    buf = load_roi_latent_buffer(roi_env)
    assert tuple(buf.shape[-3:]) == get_latent_shape()
