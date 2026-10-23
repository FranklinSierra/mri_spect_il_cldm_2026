"""Inference manifest must restore ROI profile (latent shape) from run_manifest.json."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from code.volume_shapes import clear_shape_cache, get_latent_shape
from inference import (
    apply_manifest_to_config,
    resolve_mri_encoder_checkpoint,
    resolve_unet_build_config,
)


@pytest.fixture
def legacy_config():
    saved = {
        "SKIP_CROP": config.SKIP_CROP,
        "ROI_VOLUME_SHAPE": config.ROI_VOLUME_SHAPE,
        "UNET_ROI_CHANNEL": config.UNET_ROI_CHANNEL,
        "UNET_IN_CHANNELS": config.UNET_IN_CHANNELS,
    }
    config.SKIP_CROP = False
    config.ROI_VOLUME_SHAPE = None
    config.UNET_ROI_CHANNEL = False
    config.UNET_IN_CHANNELS = 2
    clear_shape_cache()
    yield
    for key, value in saved.items():
        setattr(config, key, value)
    clear_shape_cache()


def test_apply_manifest_roi_latent_shape(legacy_config):
    manifest = apply_manifest_to_config("exp_nm_s2_frozen_calib_roi3ch_uni_rois")
    assert manifest is not None
    assert config.SKIP_CROP is True
    assert config.ROI_VOLUME_SHAPE == (105, 102, 102)
    assert config.UNET_ROI_CHANNEL is True
    assert get_latent_shape() == (27, 26, 26)

    unet_path = ROOT / "result/exp_nm_s2_frozen_calib_roi3ch_uni_rois/Unet.pth.tar"
    if unet_path.is_file():
        use_roi, in_ch = resolve_unet_build_config(
            "exp_nm_s2_frozen_calib_roi3ch_uni_rois",
            unet_path,
        )
        assert use_roi is True
        assert in_ch == 3


def test_resolve_mri_encoder_uses_manifest_config(monkeypatch):
    """MRI path must follow run_manifest, not the hardcoded exp_nm_s1_mri default."""
    exp = "exp_nm_s2_frozen_calib_roi3ch_uni_rois_reproc_x0_minmax_kl_roiw10"
    manifest_path = ROOT / "result" / exp / "run_manifest.json"
    if not manifest_path.is_file():
        pytest.skip(f"missing {manifest_path}")

    monkeypatch.delenv("ILCLDM_MRI_ENCODER_CHECKPOINT", raising=False)
    saved = getattr(config, "MRI_ENCODER_CHECKPOINT", None)
    try:
        manifest = apply_manifest_to_config(exp)
        assert manifest is not None
        expected = "result/exp_nm_s1_mri_rois_reproc_minmax_kl/phase1_mri/checkpoints/MRI_AAE.pth.tar"
        assert config.MRI_ENCODER_CHECKPOINT == expected
        resolved = resolve_mri_encoder_checkpoint(exp, "frozen", explicit_checkpoint=None)
        assert resolved == Path(expected)
    finally:
        config.MRI_ENCODER_CHECKPOINT = saved
