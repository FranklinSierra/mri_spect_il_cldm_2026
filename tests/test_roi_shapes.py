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


def test_two_dataset_pads_spect_like_mri_on_train(roi_env, monkeypatch, tmp_path):
    """Train-stage SPECT volumes must get the same ROI working-shape pad as MRI."""
    import config
    from code import volume_shapes
    from dataset import TwoDataset

    raw = (32, 32, 32)
    working = volume_shapes.get_working_volume_shape()
    mri_dir = tmp_path / "whole_MRI"
    spect_dir = tmp_path / "whole_Abeta"
    mri_dir.mkdir()
    spect_dir.mkdir()
    vol = np.ones(raw, dtype=np.float32)
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(mri_dir / "9999.nii.gz"))
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(spect_dir / "9999.nii.gz"))
    list_path = tmp_path / "train.txt"
    list_path.write_text("9999\n")
    info_path = tmp_path / "data_info.csv"
    info_path.write_text("ID,label\n9999,1\n")

    monkeypatch.setattr("dataset.DATA_INFO_CSV", str(info_path))

    ds = TwoDataset(
        root_MRI=str(mri_dir),
        root_Abeta=str(spect_dir),
        task=str(list_path),
        stage="train",
    )
    mri, spect, name, label = ds[0]
    assert tuple(mri.shape) == working
    assert tuple(spect.shape) == working
    assert name.endswith("9999.nii.gz")
