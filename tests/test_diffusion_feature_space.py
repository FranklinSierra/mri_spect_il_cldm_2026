"""Tests for gated E2 diffusion feature spaces (z18k / h64 / h64_gap1)."""

import os

import numpy as np
import torch


def test_diffusion_channels_and_unet_width(monkeypatch):
    monkeypatch.setenv("ILCLDM_DIFFUSION_FEATURE_SPACE", "h64")
    monkeypatch.setenv("ILCLDM_UNET_ROI_CHANNEL", "1")
    import importlib

    import config
    from code import diffusion_features

    importlib.reload(config)
    assert config.DIFFUSION_FEATURE_SPACE == "h64"
    assert config.DIFFUSION_CHANNELS == 64
    assert config.UNET_IN_CHANNELS == 129
    assert diffusion_features.unet_in_channels("h64", use_roi=True) == 129
    assert diffusion_features.diffusion_channels("h64_gap1") == 1
    assert diffusion_features.unet_in_channels("h64_gap1", use_roi=True) == 3


def test_channel_mean_gap_preserves_grid():
    from code.diffusion_features import channel_mean_to_one

    h64 = torch.randn(2, 64, 27, 26, 26)
    g = channel_mean_to_one(h64)
    assert g.shape == (2, 1, 27, 26, 26)
    assert torch.allclose(g[0, 0], h64[0].mean(dim=0))


def test_batch_latent_to_tensor_shapes():
    from code.diffusion_features import batch_latent_to_tensor

    nifti_batch = np.random.randn(2, 27, 26, 26).astype(np.float32)
    t = batch_latent_to_tensor(nifti_batch, channels=1, device="cpu")
    assert t.shape == (2, 1, 27, 26, 26)

    h64_batch = np.random.randn(2, 64, 27, 26, 26).astype(np.float32)
    t64 = batch_latent_to_tensor(h64_batch, channels=64, device="cpu")
    assert t64.shape == (2, 64, 27, 26, 26)


def test_build_unet_h64_out_channels(monkeypatch):
    monkeypatch.setenv("ILCLDM_DIFFUSION_FEATURE_SPACE", "h64")
    monkeypatch.setenv("ILCLDM_UNET_ROI_CHANNEL", "1")
    monkeypatch.setenv("ILCLDM_SKIP_CROP", "1")
    monkeypatch.setenv("ILCLDM_ROI_VOLUME_SHAPE", "105,102,102")
    import importlib

    import config
    from code import volume_shapes

    importlib.reload(config)
    volume_shapes.clear_shape_cache()
    from model import build_unet

    net = build_unet()
    first = None
    for m in net.modules():
        if isinstance(m, torch.nn.Conv3d):
            first = m
            break
    assert first is not None
    assert first.in_channels == 129
    # Smoke forward
    x = torch.zeros(1, 64, 27, 26, 26)
    cond = torch.zeros(1, 64, 27, 26, 26)
    t = torch.tensor([10])
    with torch.no_grad():
        out = net(x, cond, t, precomputed_condition=True)
    assert out.shape == (1, 64, 27, 26, 26)


def test_spect_h64_cache_exists():
    root = "data_rois/latent_Abeta_h64"
    assert os.path.isdir(root)
    files = [f for f in os.listdir(root) if f.endswith(".npy")]
    assert len(files) >= 100
    arr = np.load(os.path.join(root, files[0]))
    assert arr.shape[0] == 64
    assert os.path.isfile("data_info/latent_stats_latent_Abeta_rois_h64.json")
    assert os.path.isfile("data_info/latent_stats_latent_Abeta_rois_h64_gap1.json")


def test_registry_xm64_keys():
    import json
    from pathlib import Path

    reg = json.loads(Path("experiments/registry.json").read_text(encoding="utf-8"))
    for key in (
        "calib_roi3ch_uni_rois_xm64_z18",
        "calib_roi3ch_uni_rois_xm64_h64",
        "calib_roi3ch_uni_rois_xm64_h64_gap1",
    ):
        assert key in reg["experiments"]
    assert (
        reg["experiments"]["calib_roi3ch_uni_rois_xm64_h64"]["overrides"][
            "ILCLDM_DIFFUSION_FEATURE_SPACE"
        ]
        == "h64"
    )
