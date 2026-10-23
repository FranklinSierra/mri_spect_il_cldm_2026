"""Shared min-max image norm, AAE KL head, and the x0 minmax-kl registry entry."""

import json

import numpy as np
import torch

import config
from dataset import (
    apply_image_volume_norm,
    get_latent_stats,
    scale_to_unit_interval,
    stored_image_from_model,
    write_latent_mu_report,
)
from model import Encoder, kl_standard_normal
from scripts.run_ablation import load_registry, resolve_env


def test_scale_to_unit_interval_uses_shared_bounds():
    data = np.array([-10.0, 0.0, 10.0, 50.0], dtype=np.float32)
    out = scale_to_unit_interval(data, lo=0.0, hi=40.0)
    np.testing.assert_allclose(out, [0.0, 0.0, 0.25, 1.0], rtol=1e-5, atol=1e-5)


def test_minmax_mode_uses_saved_constants(tmp_path, monkeypatch):
    stats = tmp_path / "minmax.json"
    stats.write_text(json.dumps({"spect": {"lo": 0.0, "hi": 10.0}, "mri": {"lo": 1.0, "hi": 5.0}}))
    monkeypatch.setattr(config, "MINMAX_STATS_PATH", str(stats))
    vol = np.array([0.0, 5.0, 20.0], dtype=np.float32)
    out = apply_image_volume_norm(vol, modality="spect", norm_mode="minmax")
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0], rtol=1e-5, atol=1e-5)
    mri = apply_image_volume_norm(np.array([1.0, 3.0, 9.0], dtype=np.float32), modality="mri", norm_mode="minmax")
    np.testing.assert_allclose(mri, [0.0, 0.5, 1.0], rtol=1e-5, atol=1e-5)


def test_kl_is_zero_at_standard_normal():
    mu = torch.zeros(2, 1, 2, 2, 2)
    logvar = torch.zeros_like(mu)
    value = kl_standard_normal(mu, logvar)
    assert float(value) == 0.0


def test_kl_encoder_returns_one_channel_mu(monkeypatch):
    monkeypatch.setattr(config, "KL_LOSS", True)
    encoder = Encoder()
    try:
        volume = torch.zeros(1, 1, 8, 8, 8)
        mu = encoder(volume)
        assert mu.shape[1] == 1
        hidden, h64 = encoder(volume, return_hidden=True)
        assert hidden.shape[1] == 1
        assert h64.shape[1] == 64
        z, mu_s, logvar = encoder(volume, sample=True)
        assert z.shape == mu_s.shape == logvar.shape
        assert z.shape[1] == 1
    finally:
        monkeypatch.setattr(config, "KL_LOSS", False)


def test_calibration_off_does_not_rescale(monkeypatch):
    monkeypatch.setattr(config, "LATENT_CALIBRATION", False)
    assert get_latent_stats("data_rois/latent_Abeta_minmax_kl") == (0.0, 1.0)


def test_latent_mu_report_records_variance(tmp_path):
    import nibabel as nib

    affine = np.eye(4)
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), affine), tmp_path / "a.nii.gz")
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), affine), tmp_path / "b.nii.gz")
    report = write_latent_mu_report(tmp_path)
    assert report["n_files"] == 2
    assert abs(report["mean"] - 0.5) < 1e-6
    assert report["variance"] == report["std"] ** 2
    assert (tmp_path / "latent_mu_report.json").is_file()


def test_calibration_reads_minmax_kl_train_stats(monkeypatch):
    from dataset import _LATENT_STATS_CACHE

    monkeypatch.setattr(config, "LATENT_CALIBRATION", True)
    monkeypatch.setattr(config, "LATENT_MEAN", None)
    monkeypatch.setattr(config, "LATENT_STD", None)
    monkeypatch.setattr(
        config,
        "LATENT_STATS_PATH",
        "data_info/latent_stats_latent_Abeta_minmax_kl.json",
    )
    _LATENT_STATS_CACHE.clear()
    try:
        mean, std = get_latent_stats("data_rois/latent_Abeta_minmax_kl")
    finally:
        _LATENT_STATS_CACHE.clear()
    assert abs(mean - (-0.14870218474624203)) < 1e-12
    assert abs(std - 0.5632211290537218) < 1e-12


def test_calib_registry_uses_train_stats_and_keeps_roi_loss_off():
    registry = load_registry()
    entry, env = resolve_env(registry, "calib_roi3ch_uni_rois_reproc_x0_minmax_kl_calib")
    assert entry["parent"] == "calib_roi3ch_uni_rois_reproc_x0_minmax_kl"
    assert entry["base"] == "base_calib_roi3ch_uni_rois_reproc_minmax_kl"
    assert env["ILCLDM_LATENT_CALIBRATION"] == "1"
    assert env["ILCLDM_LATENT_STATS_PATH"] == "data_info/latent_stats_latent_Abeta_minmax_kl.json"
    assert env["ILCLDM_PRED_X0_CLAMP"] == "4"
    assert env["ILCLDM_DIFFUSION_OBJECTIVE"] == "x0"
    assert env["ILCLDM_USE_ROI_LOSS"] == "0"
    assert env["ILCLDM_LATENT_LOSS"] == "0"
    assert env["ILCLDM_EARLY_STOP_LATENT_METRIC"] == "1"


def test_roi_metrics_sum_every_volume_in_the_batch():
    from main import batch_metric_scores_roi

    rng = np.random.default_rng(0)
    target = rng.random((2, 16, 16, 16)).astype(np.float32)
    prediction = np.clip(target + rng.normal(0.0, 0.05, target.shape), 0.0, 1.0).astype(np.float32)
    mask = np.zeros((16, 16, 16), dtype=np.float32)
    mask[2:14, 2:14, 2:14] = 1.0
    total_psnr, total_ssim = batch_metric_scores_roi(target, prediction, mask)
    first_psnr, first_ssim = batch_metric_scores_roi(target[0], prediction[0], mask)
    second_psnr, second_ssim = batch_metric_scores_roi(target[1], prediction[1], mask)
    assert abs(total_psnr - (first_psnr + second_psnr)) < 1e-6
    assert abs(total_ssim - (first_ssim + second_ssim)) < 1e-6
    assert abs(total_psnr - first_psnr) > 1.0


def test_stored_image_is_dataset_range_before_psnr(monkeypatch):
    from main import batch_metric_scores, prepare_prediction_for_metrics

    monkeypatch.setattr(config, "IMAGE_VOLUME_NORM", "minmax")
    target = np.zeros((8, 8, 8), dtype=np.float32)
    target[2:6, 2:6, 2:6] = 1.0
    pred_model = target * 2.0 - 1.0
    restored = stored_image_from_model(pred_model)
    np.testing.assert_allclose(np.asarray(restored), target, atol=1e-6)
    # Public metric API must restore model-space preds (not compare [-1,1] to [0,1]).
    np.testing.assert_allclose(prepare_prediction_for_metrics(pred_model), target, atol=1e-6)
    psnr_ok, ssim_ok = batch_metric_scores(target, pred_model)
    assert psnr_ok > 30
    assert ssim_ok > 0.9
    # Double-restore would squash contrast and collapse PSNR — callers must not
    # pre-apply stored_image_from_model before batch_metric_scores.
    psnr_double, _ = batch_metric_scores(target, restored)
    assert psnr_double < psnr_ok


def test_x0_minmax_kl_registry_points_at_x0_parent():
    registry = load_registry()
    entry, env = resolve_env(registry, "calib_roi3ch_uni_rois_reproc_x0_minmax_kl")
    assert entry["parent"] == "calib_roi3ch_uni_rois_reproc_x0"
    assert entry["base"] == "base_calib_roi3ch_uni_rois_reproc_minmax_kl"
    assert entry["short_label"] == "rois_reproc_x0_minmax_kl"
    assert env["ILCLDM_DIFFUSION_OBJECTIVE"] == "x0"
    assert env["ILCLDM_LATENT_CALIBRATION"] == "0"
    assert env["ILCLDM_PRED_X0_CLAMP"] == "4"
    assert env["ILCLDM_IMAGE_VOLUME_NORM"] == "minmax"
    assert env["ILCLDM_KL_LOSS"] == "1"
    assert env["ILCLDM_LATENT_ABETA"] == "data_rois/latent_Abeta_minmax_kl/"
    assert env["ILCLDM_DIFFUSION_FEATURE_SPACE"] == "z18k"
    assert env["ILCLDM_FILM_MODE"] == "off"
    assert "minmax_kl" in env["ILCLDM_AAE_CHECKPOINT"]
    assert "minmax_kl" in env["ILCLDM_MRI_ENCODER_CHECKPOINT"]


def test_x0_minmax_kl_roiw10_r3_registry():
    registry = load_registry()
    entry, env = resolve_env(registry, "calib_roi3ch_uni_rois_reproc_x0_minmax_kl_roiw10")
    assert entry["parent"] == "calib_roi3ch_uni_rois_reproc_x0_minmax_kl"
    assert entry["base"] == "base_calib_roi3ch_uni_rois_reproc_minmax_kl"
    assert entry["short_label"] == "rois_reproc_x0_minmax_kl_roiw10"
    assert env["ILCLDM_DIFFUSION_OBJECTIVE"] == "x0"
    assert env["ILCLDM_DIFFUSION_ROI_WEIGHT"] == "10"
    assert env["ILCLDM_LATENT_ABETA"] == "data_rois/latent_Abeta_minmax_kl/"
    assert "minmax_kl" in env["ILCLDM_AAE_CHECKPOINT"]
    assert "kl1e1" not in env["ILCLDM_AAE_CHECKPOINT"]
    assert "minmax_kl" in env["ILCLDM_MRI_ENCODER_CHECKPOINT"]
    assert "kl1e1" not in env["ILCLDM_MRI_ENCODER_CHECKPOINT"]


def test_x0_minmax_kl_ssim_r4_registry():
    registry = load_registry()
    entry, env = resolve_env(registry, "calib_roi3ch_uni_rois_reproc_x0_minmax_kl_ssim")
    assert entry["parent"] == "calib_roi3ch_uni_rois_reproc_x0_minmax_kl"
    assert entry["base"] == "base_calib_roi3ch_uni_rois_reproc_minmax_kl"
    assert entry["short_label"] == "rois_reproc_x0_minmax_kl_ssim"
    assert env["ILCLDM_DIFFUSION_OBJECTIVE"] == "x0"
    assert env["ILCLDM_DIFFUSION_SSIM_LAMBDA"] == "1.0"
    assert "ILCLDM_DIFFUSION_ROI_WEIGHT" not in env or env.get("ILCLDM_DIFFUSION_ROI_WEIGHT", "0") == "0"
    assert env["ILCLDM_LATENT_ABETA"] == "data_rois/latent_Abeta_minmax_kl/"
    assert "minmax_kl" in env["ILCLDM_AAE_CHECKPOINT"]
    assert "kl1e1" not in env["ILCLDM_AAE_CHECKPOINT"]
    assert "minmax_kl" in env["ILCLDM_MRI_ENCODER_CHECKPOINT"]
    assert "kl1e1" not in env["ILCLDM_MRI_ENCODER_CHECKPOINT"]


def test_x0_minmax_kl1e1_r1c_registry():
    registry = load_registry()
    entry, env = resolve_env(registry, "calib_roi3ch_uni_rois_reproc_x0_minmax_kl1e1")
    assert entry["parent"] == "calib_roi3ch_uni_rois_reproc_x0_minmax_kl"
    assert entry["base"] == "base_calib_roi3ch_uni_rois_reproc_minmax_kl1e1"
    assert entry["short_label"] == "rois_reproc_x0_minmax_kl1e1"
    assert env["ILCLDM_DIFFUSION_OBJECTIVE"] == "x0"
    assert env["ILCLDM_KL_LAMBDA"] == "1e-1"
    assert env["ILCLDM_LATENT_ABETA"] == "data_rois/latent_Abeta_minmax_kl1e1/"
    assert env["ILCLDM_SAMPLING_T_MAX"] == "999"
    assert env["ILCLDM_GRAD_CLIP_NORM"] == "1.0"
    assert env["ILCLDM_PRED_X0_CLAMP_LO"] == "-2.6189742624759673"
    assert env["ILCLDM_PRED_X0_CLAMP_HI"] == "1.7349464029073798"
    assert "minmax_kl1e1" in env["ILCLDM_AAE_CHECKPOINT"]
    assert "minmax_kl1e1" in env["ILCLDM_MRI_ENCODER_CHECKPOINT"]
