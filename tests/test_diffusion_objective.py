"""Diffusion objective: epsilon (default) vs clean-latent x0, and constant latent L1."""

from __future__ import annotations

import importlib

import pytest
import torch


@pytest.fixture(autouse=True)
def _cpu_device(monkeypatch):
    monkeypatch.setenv("ILCLDM_DEVICE", "cpu")
    monkeypatch.setenv("ILCLDM_PRED_X0_CLAMP", "4.0")
    monkeypatch.setenv("ILCLDM_DIFFUSION_OBJECTIVE", "epsilon")
    import config

    importlib.reload(config)
    yield
    monkeypatch.setenv("ILCLDM_DIFFUSION_OBJECTIVE", "epsilon")
    importlib.reload(config)


def _diffusion():
    from main import Diffusion

    return Diffusion()


def test_epsilon_objective_matches_noise_inversion():
    diff = _diffusion()
    t = torch.tensor([400], dtype=torch.long)
    x_t = torch.randn(1, 1, 2, 2, 2)
    eps = torch.randn_like(x_t)
    pred = diff.x0_from_model_output(x_t, t, eps)
    assert torch.allclose(pred, diff.predict_x0(x_t, t, eps))
    _x0, direction_eps = diff.ddim_pred(x_t, t, eps)
    assert torch.equal(direction_eps, eps)


def test_x0_objective_does_not_invert_epsilon(monkeypatch):
    monkeypatch.setenv("ILCLDM_DIFFUSION_OBJECTIVE", "x0")
    import config

    importlib.reload(config)
    diff = _diffusion()
    assert diff.predicts_x0()
    t = torch.tensor([500], dtype=torch.long)
    x_t = torch.randn(1, 1, 2, 2, 2)
    model_out = torch.full_like(x_t, 9.0)
    pred = diff.x0_from_model_output(x_t, t, model_out)
    assert torch.allclose(pred, torch.full_like(x_t, 4.0))
    inverted = diff.predict_x0(x_t, t, model_out)
    assert not torch.allclose(pred, inverted)
    _x0, direction_eps = diff.ddim_pred(x_t, t, model_out)
    assert not torch.allclose(direction_eps, model_out)
    recovered = diff.predict_x0(x_t, t, direction_eps)
    assert torch.allclose(recovered, pred, atol=1e-5)


def test_latent_l1_is_not_gated_by_aux_every_n():
    from main import use_constant_latent_l1

    assert use_constant_latent_l1(True, objective="epsilon", global_step=3, every_n=10)
    assert use_constant_latent_l1(True, objective="epsilon", global_step=0, every_n=10)
    assert use_constant_latent_l1(True, objective="epsilon", global_step=10, every_n=10)
    assert not use_constant_latent_l1(False, objective="epsilon", global_step=0, every_n=1)
    assert not use_constant_latent_l1(True, objective="x0", global_step=0, every_n=10)


def test_registry_roi_diagnostics_are_distinct():
    from scripts.run_ablation import load_registry, resolve_env

    registry = load_registry()
    eps_entry, eps_env = resolve_env(registry, "calib_roi3ch_uni_rois_reproc_eps_x0")
    x0_entry, x0_env = resolve_env(registry, "calib_roi3ch_uni_rois_reproc_x0")

    assert eps_entry["parent"] == "calib_roi3ch_uni_rois_reproc"
    assert x0_entry["parent"] == "calib_roi3ch_uni_rois_reproc"
    assert eps_entry["split_profile"] == "data_rois"
    assert x0_entry["split_profile"] == "data_rois"
    assert eps_entry["exp"] != x0_entry["exp"]

    assert eps_env["ILCLDM_LATENT_LOSS"] == "1"
    assert eps_env["ILCLDM_LATENT_LOSS_LAMBDA"] == "5.0"
    assert eps_env["ILCLDM_EARLY_STOP_LATENT_METRIC"] == "1"
    assert "ILCLDM_DIFFUSION_OBJECTIVE" not in eps_env

    assert x0_env["ILCLDM_DIFFUSION_OBJECTIVE"] == "x0"
    assert x0_env["ILCLDM_EARLY_STOP_LATENT_METRIC"] == "1"
    assert x0_env.get("ILCLDM_LATENT_LOSS", "0") == "0"
    assert eps_env["ILCLDM_LATENT_ABETA"] == "data_rois/latent_Abeta_reproc/"
    assert x0_env["ILCLDM_LATENT_ABETA"] == eps_env["ILCLDM_LATENT_ABETA"]


def test_manifest_restores_x0_objective():
    import config
    from utils import restore_diffusion_objective

    config.DIFFUSION_OBJECTIVE = "epsilon"
    restored = restore_diffusion_objective(
        {"ILCLDM_DIFFUSION_OBJECTIVE": "x0"},
        {"DIFFUSION_OBJECTIVE": "epsilon"},
    )
    assert restored == "x0"
    assert config.DIFFUSION_OBJECTIVE == "x0"

    config.DIFFUSION_OBJECTIVE = "epsilon"
    restored = restore_diffusion_objective({}, {"DIFFUSION_OBJECTIVE": "x0"})
    assert restored == "x0"
    assert config.DIFFUSION_OBJECTIVE == "x0"

    with pytest.raises(ValueError):
        restore_diffusion_objective({"ILCLDM_DIFFUSION_OBJECTIVE": "start_x"}, {})
