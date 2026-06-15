import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from code.roi_mask_utils import CROP_SHAPE, crop_roi_mask, load_roi_mask_cropped, load_roi_mask_volume


def test_crop_roi_mask_shape():
    vol = np.zeros((180, 228, 180), dtype=np.float32)
    vol[50, 100, 50] = 1.0
    cropped = crop_roi_mask(vol)
    assert cropped.shape == CROP_SHAPE
    assert cropped.sum() > 0


def test_load_roi_mask_cropped_from_nifti(tmp_path):
    vol = np.zeros((180, 228, 180), dtype=np.float32)
    vol[80, 120, 80] = 1.0
    path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(path))

    loaded = load_roi_mask_volume(path)
    assert loaded.shape == vol.shape
    mask = load_roi_mask_cropped(path)
    assert mask.shape == CROP_SHAPE
    assert mask.sum() > 0
