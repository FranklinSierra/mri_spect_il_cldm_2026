"""Tests for spatial multiscale FiLM (h64 map, gamma/beta per ResnetBlock)."""

from __future__ import annotations

import importlib

import pytest
import torch


@pytest.fixture(autouse=True)
def _cpu_and_reload(monkeypatch):
    monkeypatch.setenv("ILCLDM_DEVICE", "cpu")
    monkeypatch.setenv("ILCLDM_USE_FILM", "0")
    monkeypatch.setenv("ILCLDM_FILM_MODE", "off")
    monkeypatch.setenv("ILCLDM_ATTN_RES", "4")
    import config

    importlib.reload(config)
    yield
    importlib.reload(config)


def test_spatial_film_resizes_cond_to_feature_map():
    from model import SpatialFiLM

    film = SpatialFiLM(cond_channels=64, feat_channels=128)
    h = torch.randn(2, 128, 12, 14, 14)
    cond = torch.randn(2, 64, 24, 28, 28)
    out = film(h, cond)
    assert out.shape == h.shape


@pytest.mark.parametrize("spatial", [(24, 28, 28), (12, 14, 14), (6, 7, 7), (3, 3, 3)])
def test_spatial_film_output_shape_matches_h(spatial):
    from model import SpatialFiLM

    film = SpatialFiLM(64, 256)
    h = torch.randn(1, 256, *spatial)
    cond = torch.randn(1, 64, 24, 28, 28)
    out = film(h, cond)
    assert tuple(out.shape) == tuple(h.shape)


def test_unet_spatial_ms_forward(monkeypatch):
    monkeypatch.setenv("ILCLDM_FILM_MODE", "spatial_ms")
    monkeypatch.setenv("ILCLDM_FILM_COND_CHANNELS", "64")
    import config
    from model import UNet

    importlib.reload(config)
    net = UNet(in_channel=2, out_channel=1, image_size=24, attn_res=(4,), film_mode="spatial_ms")
    assert net.count_spatial_film_blocks() > 0
    x = torch.randn(1, 1, 24, 28, 28)
    cond = torch.randn(1, 1, 24, 28, 28)
    mri_film = torch.randn(1, 64, 24, 28, 28)
    t = torch.tensor([100], dtype=torch.long)
    out = net(x, cond, t, mri_film=mri_film, precomputed_condition=True)
    assert out.shape == x.shape


def test_unet_off_mode_ignores_mri_film(monkeypatch):
    monkeypatch.setenv("ILCLDM_FILM_MODE", "off")
    import config
    from model import UNet

    importlib.reload(config)
    net = UNet(in_channel=2, out_channel=1, image_size=8, attn_res=(2,), film_mode="off")
    assert net.count_spatial_film_blocks() == 0
    x = torch.randn(1, 1, 8, 8, 8)
    cond = torch.randn(1, 1, 8, 8, 8)
    t = torch.tensor([50], dtype=torch.long)
    out0 = net(x, cond, t, precomputed_condition=True)
    out1 = net(
        x, cond, t, mri_film=torch.randn(1, 64, 8, 8, 8), precomputed_condition=True
    )
    assert torch.allclose(out0, out1)


def test_encode_mri_dual_returns_z18_and_h64():
    from code.diffusion_features import encode_mri_dual
    from model import MRIEncoder

    enc = MRIEncoder()
    mri = torch.randn(2, 1, 96, 112, 114)
    z18, h64 = encode_mri_dual(enc, mri, "z18k")
    assert z18.shape[0] == 2 and z18.shape[1] == 1
    assert h64.shape[0] == 2 and h64.shape[1] == 64
    assert z18.shape[2:] == h64.shape[2:]
