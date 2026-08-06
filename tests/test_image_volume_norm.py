"""IMAGE_VOLUME_NORM legacy vs v2load (MRI foreground z-score, SPECT raw)."""

import numpy as np

from dataset import (
    apply_image_volume_norm,
    infer_image_modality,
    zscore_volume,
    zscore_volume_foreground,
)


def test_infer_modality_from_path():
    assert infer_image_modality("/workspace/IL-CLDM/data/whole_MRI/3102.nii.gz") == "mri"
    assert infer_image_modality("data_rois/whole_Abeta/3102.nii.gz") == "spect"
    assert infer_image_modality("data/latent_Abeta_reproc/3102.nii.gz") is None
    assert infer_image_modality("data/latent_Abeta_reproc_v2load/x.nii.gz") is None


def test_legacy_zscore_matches_full_volume():
    rng = np.random.default_rng(0)
    vol = rng.normal(size=(8, 8, 8)).astype(np.float32)
    vol[0, 0, 0] = 0.0
    out = apply_image_volume_norm(vol, modality="mri", norm_mode="legacy")
    expected = zscore_volume(vol)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_v2load_mri_uses_foreground_stats():
    vol = np.zeros((4, 4, 4), dtype=np.float32)
    vol[1:3, 1:3, 1:3] = np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32
    )
    out = apply_image_volume_norm(vol, modality="mri", norm_mode="v2load")
    expected = zscore_volume_foreground(vol)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
    fg = out[vol > 0]
    assert abs(float(fg.mean())) < 1e-5
    assert abs(float(fg.std()) - 1.0) < 1e-5
    # Background is transformed with the same affine (not left as raw zeros).
    assert not np.allclose(out[vol == 0], 0.0)


def test_v2load_spect_is_identity():
    rng = np.random.default_rng(1)
    vol = rng.uniform(0.1, 2.0, size=(5, 5, 5)).astype(np.float32)
    out = apply_image_volume_norm(vol, modality="spect", norm_mode="v2load")
    np.testing.assert_array_equal(out, vol)


def test_v2load_unknown_modality_falls_back_to_legacy_zscore():
    rng = np.random.default_rng(2)
    vol = rng.normal(size=(3, 3, 3)).astype(np.float32)
    out = apply_image_volume_norm(vol, modality=None, norm_mode="v2load")
    np.testing.assert_allclose(out, zscore_volume(vol), rtol=1e-5, atol=1e-5)
