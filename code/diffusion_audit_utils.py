"""Helpers for code/auditoria_diffusion.ipynb — quick diffusion diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from dataset import TwoDataset, get_latent_stats
from main import Diffusion, batch_metric_scores
from model import AAE, build_unet
from utils import load_checkpoint, seed_torch

REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_NAMES = {0.0: "control", 1.0: "parkinson"}


def repo_root() -> Path:
    return REPO_ROOT


def cosine_schedule(noise_steps: int, cosine_s: float = 0.008) -> tuple[np.ndarray, np.ndarray]:
    """Return beta_t and alpha_hat arrays for a cosine schedule with offset s."""
    steps = noise_steps + 1
    x = np.linspace(0, noise_steps, steps)
    alpha_hat_ext = np.cos(((x / noise_steps) + cosine_s) / (1 + cosine_s) * np.pi * 0.5) ** 2
    alpha_hat_ext = alpha_hat_ext / alpha_hat_ext[0]
    beta = 1.0 - (alpha_hat_ext[1:] / alpha_hat_ext[:-1])
    beta = np.clip(beta, 0.0001, 0.9999)
    alpha = 1.0 - beta
    alpha_hat = np.cumprod(alpha)
    return beta.astype(np.float64), alpha_hat.astype(np.float64)


def snr_at_t(alpha_hat: np.ndarray, t: int, sigma_z: float = 1.0) -> float:
    ah = float(alpha_hat[t])
    return (ah * sigma_z ** 2) / max(1.0 - ah, 1e-12)


def snr_curve(alpha_hat: np.ndarray, sigma_z: float = 1.0) -> np.ndarray:
    ah = np.asarray(alpha_hat, dtype=np.float64)
    return (ah * sigma_z ** 2) / np.maximum(1.0 - ah, 1e-12)


def amplification_factor(alpha_hat: np.ndarray, t: int) -> float:
    ah = float(alpha_hat[t])
    return float(np.sqrt((1.0 - ah) / max(ah, 1e-12)))


def first_t_snr_below(alpha_hat: np.ndarray, sigma_z: float = 1.0, threshold: float = 1.0) -> int | None:
    for t in range(1, len(alpha_hat)):
        if snr_at_t(alpha_hat, t, sigma_z) < threshold:
            return t
    return None


def calibration_modes() -> tuple[str, ...]:
    return ("raw", "scale_only", "center_only", "full")


def apply_calibration(z: torch.Tensor, mode: str, mean: float, std: float) -> torch.Tensor:
    if mode == "raw":
        return z
    if mode == "scale_only":
        return z / std
    if mode == "center_only":
        return z - mean
    if mode == "full":
        return (z - mean) / std
    raise ValueError(f"Unknown calibration mode: {mode}")


def invert_calibration(z_prime: torch.Tensor, mode: str, mean: float, std: float) -> torch.Tensor:
    if mode == "raw":
        return z_prime
    if mode == "scale_only":
        return z_prime * std
    if mode == "center_only":
        return z_prime + mean
    if mode == "full":
        return z_prime * std + mean
    raise ValueError(f"Unknown calibration mode: {mode}")


def calibration_sigma(mode: str, std: float) -> float:
    if mode in {"raw", "center_only"}:
        return std
    return 1.0


def calibration_mean(mode: str, mean: float) -> float:
    if mode in {"raw", "scale_only"}:
        return mean
    return 0.0


def latent_metrics(z_hat: torch.Tensor, z_gt: torch.Tensor) -> dict[str, float]:
    a = z_hat.detach().float().cpu().numpy().ravel()
    b = z_gt.detach().float().cpu().numpy().ravel()
    mse = float(np.mean((a - b) ** 2))
    mae = float(np.mean(np.abs(a - b)))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan")
    return {"latent_mse": mse, "latent_mae": mae, "latent_corr": corr}


def image_metrics(aae: AAE, spect_gt: torch.Tensor, z_raw: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        pred = aae.decoder(z_raw)
    psnr_v, ssim_v = batch_metric_scores(spect_gt.detach().cpu(), pred.detach().cpu())
    return {"PSNR": float(psnr_v), "SSIM": float(ssim_v)}


def volume_mid_slice(vol: np.ndarray, axis: int = 2) -> np.ndarray:
    idx = vol.shape[axis] // 2
    if axis == 0:
        return vol[idx, :, :]
    if axis == 1:
        return vol[:, idx, :]
    return vol[:, :, idx]


def tensor_mid_slice(t: torch.Tensor, axis: int = 2) -> np.ndarray:
    return volume_mid_slice(t.detach().float().cpu().numpy().squeeze(), axis=axis)


def effective_t(t: int, noise_steps: int) -> int:
    return int(min(max(t, 0), noise_steps - 1))


def forward_noisy(
    diff: Diffusion,
    z_cal: torch.Tensor,
    t: int,
    eps: torch.Tensor,
) -> torch.Tensor:
    t_eff = effective_t(t, diff.noise_steps)
    t_batch = torch.full((z_cal.shape[0],), t_eff, dtype=torch.long, device=z_cal.device)
    sqrt_ah = torch.sqrt(diff.alpha_hat[t_batch])[:, None, None, None, None]
    sqrt_om = torch.sqrt(1.0 - diff.alpha_hat[t_batch])[:, None, None, None, None]
    return sqrt_ah * z_cal + sqrt_om * eps


def read_manifest(exp: str) -> dict:
    path = REPO_ROOT / "result" / exp.strip("/") / "run_manifest.json"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_audit_models(exp: str, device: torch.device | None = None) -> dict:
    """Load AAE, UNet, MRI encoder, and Diffusion for audit notebooks."""
    from inference import (
        checkpoint_paths,
        infer_mri_encoder_mode,
        load_mri_encoder,
        resolve_mri_encoder_checkpoint,
        resolve_unet_build_config,
    )

    if device is None:
        device = torch.device(config.device)

    checkpoint_aae, checkpoint_unet = checkpoint_paths(exp)
    manifest = read_manifest(exp)
    manifest_cfg = manifest.get("config", {})

    def _resolve(p) -> Path:
        """Resolve a path against REPO_ROOT so it does not depend on the cwd
        (audit notebooks run from ``code/``)."""
        p = Path(p)
        return p if p.is_absolute() else (REPO_ROOT / p)

    # UNet always lives in the experiment directory.
    checkpoint_unet = _resolve(checkpoint_unet)

    # Resolve the AAE checkpoint robustly: prefer the manifest value (relative
    # to the repo root), then the exp dir, then whatever checkpoint_paths gave.
    aae_candidates = []
    if manifest_cfg.get("CHECKPOINT_AAE"):
        aae_candidates.append(_resolve(manifest_cfg["CHECKPOINT_AAE"]))
    aae_candidates.append(REPO_ROOT / "result" / exp.strip("/") / "AAE.pth.tar")
    aae_candidates.append(_resolve(checkpoint_aae))
    checkpoint_aae = next((c for c in aae_candidates if c.is_file()), aae_candidates[0])
    if not checkpoint_aae.is_file():
        raise FileNotFoundError(
            "No se encontró un checkpoint AAE válido para el experimento "
            f"'{exp}'. Rutas probadas: {[str(c) for c in aae_candidates]}"
        )

    mode = infer_mri_encoder_mode(exp, None)
    mri_ckpt = resolve_mri_encoder_checkpoint(exp, mode, None)
    if mri_ckpt is not None:
        mri_ckpt = _resolve(mri_ckpt)
    use_roi, in_channels = resolve_unet_build_config(exp, checkpoint_unet)

    aae = AAE().to(device).eval()
    load_checkpoint(str(checkpoint_aae), aae, None, config.learning_rate)

    unet = build_unet(in_channel=in_channels, use_roi_channel=use_roi).to(device).eval()
    load_checkpoint(str(checkpoint_unet), unet, None, config.learning_rate)

    mri_encoder, mri_ckpt_resolved = load_mri_encoder(mode, mri_ckpt, device)

    diffusion = Diffusion(cosine_s=0.008)

    use_train_label = bool(manifest_cfg.get("USE_TRAIN_LABEL", config.USE_TRAIN_LABEL))
    latent_mean, latent_std = diffusion.latent_mean, diffusion.latent_std

    meta = {
        "exp": exp,
        "checkpoint_aae": str(checkpoint_aae),
        "checkpoint_unet": str(checkpoint_unet),
        "mri_encoder_mode": mode,
        "mri_encoder_checkpoint": str(mri_ckpt_resolved) if mri_ckpt_resolved else None,
        "in_channels": in_channels,
        "use_roi_channel": use_roi,
        "use_train_label": use_train_label,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "latent_calibration": bool(manifest_cfg.get("LATENT_CALIBRATION", config.LATENT_CALIBRATION)),
        "cosine_s_train": 0.008,
    }
    return {
        "aae": aae,
        "unet": unet,
        "mri_encoder": mri_encoder,
        "diffusion": diffusion,
        "meta": meta,
    }


def subject_id_from_name(name) -> str:
    from inference import subject_id_from_name as _sid

    return _sid(name)


def pick_test_subjects(n_per_class: int = 1) -> pd.DataFrame:
    test_file = REPO_ROOT / config.test
    with open(test_file) as handle:
        test_ids = [line.strip() for line in handle if line.strip()]
    info = pd.read_csv(REPO_ROOT / "data_info" / "data_info.csv", dtype={"ID": str})
    info = info[info["ID"].isin(test_ids)].copy()
    rows = []
    for label_val, label_name in LABEL_NAMES.items():
        part = info[info["label"] == label_val].sort_values("ID")
        if part.empty:
            raise RuntimeError(f"No test subjects with label={label_val}")
        for _, row in part.head(n_per_class).iterrows():
            rows.append({"ID": row["ID"], "label": int(label_val), "label_name": label_name})
    return pd.DataFrame(rows).reset_index(drop=True)


def build_test_loader(max_subjects: int | None = None, batch_size: int = 1) -> DataLoader:
    def _resolve(p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (REPO_ROOT / p)

    dataset = TwoDataset(
        root_MRI=str(_resolve(config.whole_MRI)),
        root_Abeta=str(_resolve(config.whole_Abeta)),
        task=str(_resolve(config.test)),
        stage="test",
    )
    if max_subjects is not None:
        dataset.images = dataset.images[:max_subjects]
        dataset.length_dataset = len(dataset.images)
        dataset.len = len(dataset.images)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def mri_condition(mri_encoder, mri: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if mri_encoder is not None:
        with torch.no_grad():
            return mri_encoder(mri), True
    return mri, False


def predict_noise(
    unet,
    x_t: torch.Tensor,
    cond: torch.Tensor,
    t: torch.Tensor,
    label: torch.Tensor | None,
    precomputed: bool,
    use_label: bool,
    mri_film: torch.Tensor | None = None,
):
    if use_label and label is not None:
        return unet(
            x_t, cond, t, label,
            precomputed_condition=precomputed,
            mri_film=mri_film,
        )
    return unet(
        x_t, cond, t, label=None,
        precomputed_condition=precomputed,
        mri_film=mri_film,
    )


def one_step_denoise(
    diffusion: Diffusion,
    unet,
    z_cal: torch.Tensor,
    t_val: int,
    cond: torch.Tensor,
    label: torch.Tensor | None,
    precomputed: bool,
    use_label: bool,
    eps: torch.Tensor | None = None,
    forward_diff: Diffusion | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-step epsilon prediction in calibrated space; returns z_hat_cal, eps_true, eps_pred."""
    fwd = forward_diff or diffusion
    t = torch.full((z_cal.shape[0],), int(t_val), dtype=torch.long, device=z_cal.device)
    if eps is None:
        x_t, eps_true = fwd.noise_images(z_cal, t)
    else:
        x_t = forward_noisy(fwd, z_cal, int(t_val), eps)
        eps_true = eps
    eps_pred = predict_noise(unet, x_t, cond, t, label, precomputed, use_label)
    z_hat_cal = diffusion.x0_from_model_output(x_t, t, eps_pred)
    return z_hat_cal, eps_true, eps_pred


def ddim_sample(
    diffusion: Diffusion,
    unet,
    cond: torch.Tensor,
    label: torch.Tensor | None,
    precomputed: bool,
    use_label: bool,
    sampling_steps: int | None = None,
    eta: float | None = None,
    seed: int | None = None,
    return_trajectory: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    if seed is not None:
        seed_torch(seed)

    def model_fn(x, y, t):
        return predict_noise(unet, x, y, t, label, precomputed, use_label)

    if return_trajectory:
        return _ddim_sample_with_trajectory(
            diffusion, model_fn, cond, sampling_steps=sampling_steps, eta=eta
        )
    return diffusion.sample(
        model_fn,
        cond,
        sampling_steps=sampling_steps,
        eta=eta,
    )


def partial_ddim_sample(
    diffusion: Diffusion,
    unet,
    z_cal: torch.Tensor,
    t_start: int,
    cond: torch.Tensor,
    label: torch.Tensor | None,
    precomputed: bool,
    use_label: bool,
    sampling_steps: int = 50,
    eta: float | None = None,
    eps: torch.Tensor | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """DDIM from x_t at t_start down to t=1 (calibrated space in, raw latent out)."""
    if seed is not None:
        seed_torch(seed)
    if eta is None:
        eta = config.ddim_eta

    device = z_cal.device
    batch = z_cal.shape[0]
    t_eff = effective_t(int(t_start), diffusion.noise_steps)
    t_batch = torch.full((batch,), t_eff, dtype=torch.long, device=device)
    if eps is None:
        x, _ = diffusion.noise_images(z_cal, t_batch)
    else:
        x = forward_noisy(diffusion, z_cal, t_eff, eps)

    step_count = min(sampling_steps, t_eff)
    timesteps = torch.linspace(t_eff, 1, step_count, dtype=torch.long, device=device)
    timesteps = torch.unique_consecutive(timesteps)

    def model_fn(x_in, y, t):
        return predict_noise(unet, x_in, y, t, label, precomputed, use_label)

    with torch.no_grad():
        for idx, timestep in enumerate(timesteps):
            t = torch.full((batch,), timestep.item(), dtype=torch.long, device=device)
            model_output = model_fn(x, cond, t)
            pred_x0, predicted_noise = diffusion.ddim_pred(x, t, model_output)
            alpha_hat = diffusion.alpha_hat[t][:, None, None, None, None]
            if idx == len(timesteps) - 1:
                alpha_hat_prev = torch.ones_like(alpha_hat)
            else:
                prev_t = torch.full(
                    (batch,), timesteps[idx + 1].item(), dtype=torch.long, device=device
                )
                alpha_hat_prev = diffusion.alpha_hat[prev_t][:, None, None, None, None]
            sigma = eta * torch.sqrt(
                (1 - alpha_hat_prev) / (1 - alpha_hat) * (1 - alpha_hat / alpha_hat_prev)
            )
            direction = torch.sqrt(torch.clamp(1 - alpha_hat_prev - sigma ** 2, min=0)) * predicted_noise
            noise = sigma * torch.randn_like(x) if eta > 0 else 0
            x = torch.sqrt(alpha_hat_prev) * pred_x0 + direction + noise
    return diffusion.denormalize_latent(x)


def _ddim_sample_with_trajectory(
    diffusion: Diffusion,
    model: Callable,
    cond: torch.Tensor,
    sampling_steps: int | None = None,
    eta: float | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if sampling_steps is None:
        sampling_steps = config.sampling_steps
    if eta is None:
        eta = config.ddim_eta

    trajectory: list[torch.Tensor] = []
    with torch.no_grad():
        n = cond.shape[0]
        x = torch.randn((n, 1, 40, 48, 40), device=cond.device)
        step_count = min(sampling_steps, diffusion.noise_steps - 1)
        timesteps = torch.linspace(
            diffusion.noise_steps - 1,
            1,
            step_count,
            dtype=torch.long,
            device=cond.device,
        )
        timesteps = torch.unique_consecutive(timesteps)
        trajectory.append(x.detach().clone())
        for idx, timestep in enumerate(timesteps):
            t = torch.full((n,), timestep.item(), dtype=torch.long, device=cond.device)
            model_output = model(x, cond, t)
            pred_x0, predicted_noise = diffusion.ddim_pred(x, t, model_output)
            alpha_hat = diffusion.alpha_hat[t][:, None, None, None, None]
            if idx == len(timesteps) - 1:
                alpha_hat_prev = torch.ones_like(alpha_hat)
            else:
                prev_t = torch.full((n,), timesteps[idx + 1].item(), dtype=torch.long, device=cond.device)
                alpha_hat_prev = diffusion.alpha_hat[prev_t][:, None, None, None, None]
            sigma = eta * torch.sqrt(
                (1 - alpha_hat_prev) / (1 - alpha_hat) * (1 - alpha_hat / alpha_hat_prev)
            )
            direction = torch.sqrt(torch.clamp(1 - alpha_hat_prev - sigma ** 2, min=0)) * predicted_noise
            noise = sigma * torch.randn_like(x) if eta > 0 else 0
            x = torch.sqrt(alpha_hat_prev) * pred_x0 + direction + noise
            trajectory.append(x.detach().clone())
    return diffusion.denormalize_latent(x), trajectory


def schedule_summary_table(
    cosine_s_values: list[float],
    t_values: list[int],
    noise_steps: int = 1000,
    sigma_cal: float = 1.0,
    sigma_raw: float = 5.545,
) -> pd.DataFrame:
    rows = []
    for s in cosine_s_values:
        _, alpha_hat = cosine_schedule(noise_steps, s)
        for t in t_values:
            t_eff = effective_t(t, noise_steps)
            rows.append(
                {
                    "cosine_s": s,
                    "t": t_eff,
                    "alpha_bar": float(alpha_hat[t_eff]),
                    "noise_frac": float(np.sqrt(1.0 - alpha_hat[t_eff])),
                    "snr_cal": snr_at_t(alpha_hat, t_eff, sigma_cal),
                    "snr_raw": snr_at_t(alpha_hat, t_eff, sigma_raw),
                    "amp_factor": amplification_factor(alpha_hat, t_eff),
                    "snr_below_1_cal": first_t_snr_below(alpha_hat, sigma_cal),
                }
            )
    return pd.DataFrame(rows)
