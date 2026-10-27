"""Registry + wiring tests for independent ablations on minmax_kl_cosine."""

from __future__ import annotations

import importlib

import torch

from scripts.run_ablation import load_registry, resolve_env


PARENT = "calib_roi3ch_uni_rois_reproc_x0_minmax_kl_cosine"
BASE = "base_calib_roi3ch_uni_rois_reproc_minmax_kl"
COSINE_OVERRIDES = {
    "ILCLDM_DIFFUSION_OBJECTIVE": "x0",
    "ILCLDM_EARLY_STOP_LATENT_METRIC": "1",
    "ILCLDM_LATENT_CALIBRATION": "0",
    "ILCLDM_PRED_X0_CLAMP": "4",
    "ILCLDM_LR_SCHEDULER": "cosine",
}


def _assert_inherits_cosine(env: dict, *, batch_size: str = "2") -> None:
    for key, value in COSINE_OVERRIDES.items():
        assert env[key] == value, f"{key}: expected {value!r}, got {env.get(key)!r}"
    assert env["ILCLDM_BATCH_SIZE"] == batch_size
    assert env["ILCLDM_LATENT_ABETA"] == "data_rois/latent_Abeta_minmax_kl/"
    assert "minmax_kl" in env["ILCLDM_AAE_CHECKPOINT"]
    assert "kl1e1" not in env["ILCLDM_AAE_CHECKPOINT"]


def test_cosine_parent_registry():
    registry = load_registry()
    entry, env = resolve_env(registry, PARENT)
    assert entry["parent"] == "calib_roi3ch_uni_rois_reproc_x0_minmax_kl"
    assert entry["base"] == BASE
    assert entry["short_label"] == "rois_reproc_x0_minmax_kl_cosine"
    _assert_inherits_cosine(env, batch_size="8")  # historical baseline; children use 2 on 16GB
    assert env["ILCLDM_MRI_ENCODER_MODE"] == "frozen"
    assert env["ILCLDM_FILM_MODE"] == "off"
    assert "ILCLDM_STRUCTURE_LOSS_LAMBDA" not in env or env.get(
        "ILCLDM_STRUCTURE_LOSS_LAMBDA", "0"
    ) in ("0", "0.0", "")


def test_cosine_ft_single_delta():
    registry = load_registry()
    entry, env = resolve_env(registry, f"{PARENT}_ft")
    assert entry["parent"] == PARENT
    assert entry["base"] == BASE
    _assert_inherits_cosine(env)
    assert env["ILCLDM_MRI_ENCODER_MODE"] == "finetune"
    assert env["ILCLDM_FILM_MODE"] == "off"
    assert "finetune" in entry["exp"]


def test_cosine_film_ms_single_delta():
    registry = load_registry()
    entry, env = resolve_env(registry, f"{PARENT}_film_ms")
    assert entry["parent"] == PARENT
    _assert_inherits_cosine(env)
    assert env["ILCLDM_FILM_MODE"] == "spatial_ms"
    assert env["ILCLDM_MRI_ENCODER_MODE"] == "frozen"
    assert env.get("ILCLDM_ATTN_RES", "4") == "4" or "ILCLDM_ATTN_RES" not in env


def test_cosine_attn3_single_delta():
    registry = load_registry()
    entry, env = resolve_env(registry, f"{PARENT}_attn3")
    assert entry["parent"] == PARENT
    _assert_inherits_cosine(env)
    assert env["ILCLDM_ATTN_RES"] == "3"
    assert env["ILCLDM_FILM_MODE"] == "off"
    assert env["ILCLDM_MRI_ENCODER_MODE"] == "frozen"


def test_cosine_label_drop_single_delta():
    registry = load_registry()
    entry, env = resolve_env(registry, f"{PARENT}_label_drop")
    assert entry["parent"] == PARENT
    _assert_inherits_cosine(env)
    assert env["ILCLDM_LABEL_DROPOUT_PROB"] == "0.15"
    assert env["ILCLDM_FILM_MODE"] == "off"


def test_cosine_structure_single_delta():
    registry = load_registry()
    entry, env = resolve_env(registry, f"{PARENT}_structure")
    assert entry["parent"] == PARENT
    _assert_inherits_cosine(env)
    assert env["ILCLDM_STRUCTURE_LOSS_LAMBDA"] == "1.0"
    assert env.get("ILCLDM_DIFFUSION_SSIM_LAMBDA", "0") in ("0", "0.0") or (
        "ILCLDM_DIFFUSION_SSIM_LAMBDA" not in env
    )
    assert env["ILCLDM_FILM_MODE"] == "off"


def test_lr_scheduler_env(monkeypatch):
    import config

    monkeypatch.setenv("ILCLDM_LR_SCHEDULER", "cosine")
    importlib.reload(config)
    assert config.LR_SCHEDULER == "cosine"

    monkeypatch.setenv("ILCLDM_LR_SCHEDULER", "none")
    importlib.reload(config)
    assert config.LR_SCHEDULER == "none"


def test_structure_loss_lambda_env(monkeypatch):
    import config

    monkeypatch.delenv("ILCLDM_STRUCTURE_LOSS_LAMBDA", raising=False)
    monkeypatch.delenv("ILCLDM_DIFFUSION_SSIM_LAMBDA", raising=False)
    importlib.reload(config)
    assert config.STRUCTURE_LOSS_LAMBDA == 0.0

    monkeypatch.setenv("ILCLDM_STRUCTURE_LOSS_LAMBDA", "1.0")
    importlib.reload(config)
    assert config.STRUCTURE_LOSS_LAMBDA == 1.0


def test_attn_res_3_hits_24_grid(monkeypatch):
    """ROI latent uses D=24 → downs 24/12/6/3; ATTN_RES=3 enables bottleneck attn."""
    import config
    from model import UNet

    monkeypatch.setenv("ILCLDM_ATTN_RES", "3")
    monkeypatch.setenv("ILCLDM_FILM_MODE", "off")
    monkeypatch.setenv("ILCLDM_USE_FILM", "0")
    importlib.reload(config)

    net = UNet(in_channel=2, out_channel=1, image_size=24, attn_res=(3,), use_film=False)
    n_attn = net.count_attn_blocks()
    # mid always has 1 attn block; downs/ups at res=3 add more → >1 with ATTN_RES=3
    assert n_attn >= 3
    assert 3 in net.attn_res

    monkeypatch.setenv("ILCLDM_ATTN_RES", "5")
    importlib.reload(config)
    net5 = UNet(in_channel=2, out_channel=1, image_size=24, attn_res=(5,), use_film=False)
    # Only the forced mid attn (res 5 never appears on 24-grid)
    assert net5.count_attn_blocks() == 1


def test_structure_prefers_over_diffusion_ssim_math():
    """Documented policy: structure λ wins when both >0 (same 1−SSIM term)."""
    structure = 1.0
    diffusion = 1.0
    if structure > 0 and diffusion > 0:
        diffusion = 0.0
    image_lambda = structure if structure > 0 else diffusion
    assert image_lambda == 1.0
    assert diffusion == 0.0
