"""Helpers for stage2_experiments_report.ipynb."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import pandas as pd

from viz_pairs import CROP_SHAPE


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REGISTRY_PATH = repo_root() / "experiments" / "registry.json"


def load_registry() -> dict:
    """Load the experiment registry (single source of truth for exp/run_id/label)."""
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _comparison_from_registry() -> dict[str, dict[str, str]]:
    registry = load_registry()
    return {
        key: {
            "exp": entry["exp"],
            "run_id": entry["run_id"],
            "label": entry["label"],
            "early_stop": entry["early_stop"],
        }
        for key, entry in registry["experiments"].items()
    }


# Experiment comparison driven by experiments/registry.json
# (frozen_v1 baseline, frozen_v2 historical, plus the ablation runs).
FROZEN_COMPARISON: dict[str, dict[str, str]] = _comparison_from_registry()

# Legacy three-way comparison (baseline / frozen / finetune).
EXPERIMENTS = {
    "baseline": "exp_nm_s2_baseline",
    "frozen": "exp_nm_s2_mri_frozen",
    "finetune": "exp_nm_s2_mri_finetune",
}

MODE_LABELS = {
    "baseline": "Baseline (raw MRI)",
    "frozen": "MRI frozen",
    "finetune": "MRI finetune",
    **{k: v["label"] for k, v in FROZEN_COMPARISON.items()},
}

LABEL_NAMES = {0.0: "control", 1.0: "parkinson"}

EarlyStopCriterion = Literal["image", "latent"]


def result_dir(exp_key: str) -> Path:
    """Legacy: exp_key in EXPERIMENTS."""
    return repo_root() / "result" / EXPERIMENTS[exp_key]


def result_dir_exp(exp_name: str) -> Path:
    return repo_root() / "result" / exp_name


def inference_dir(exp_key: str, run_id: str) -> Path:
    return result_dir(exp_key) / "inference" / run_id


def inference_dir_exp(exp_name: str, run_id: str) -> Path:
    return result_dir_exp(exp_name) / "inference" / run_id


def comparison_entry(mode_key: str) -> dict[str, str]:
    if mode_key not in FROZEN_COMPARISON:
        raise KeyError(f"Unknown mode {mode_key!r}; expected one of {list(FROZEN_COMPARISON)}")
    return FROZEN_COMPARISON[mode_key]


def available_comparison() -> dict[str, dict[str, str]]:
    """Registry entries whose training results exist on disk (skips not-yet-run ablations)."""
    return {
        key: cfg
        for key, cfg in FROZEN_COMPARISON.items()
        if (result_dir_exp(cfg["exp"]) / "validation.csv").exists()
    }


def load_run_manifest(exp_name: str) -> dict | None:
    """Load result/<exp>/run_manifest.json written at training time (None if absent)."""
    path = result_dir_exp(exp_name) / "run_manifest.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def manifest_summary_table() -> pd.DataFrame:
    """One row per registry experiment: label, exp, run_id, overrides and manifest config."""
    registry = load_registry()
    rows = []
    for key, entry in registry["experiments"].items():
        manifest = load_run_manifest(entry["exp"])
        row = {
            "mode": key,
            "label": entry["label"],
            "exp": entry["exp"],
            "run_id": entry["run_id"],
            "early_stop": entry["early_stop"],
            "overrides": ", ".join(f"{k}={v}" for k, v in entry.get("overrides", {}).items()) or "-",
            "has_manifest": manifest is not None,
        }
        if manifest is not None:
            row["git_commit"] = manifest.get("git_commit", "")[:8]
            row["timestamp"] = manifest.get("timestamp", "")
            cfg = manifest.get("config", {})
            row["latent_loss"] = cfg.get("LATENT_LOSS")
            row["latent_lambda"] = cfg.get("LATENT_LOSS_LAMBDA")
            row["high_t_prob"] = cfg.get("HIGH_T_SAMPLE_PROB")
            row["aux_every_n"] = cfg.get("AUX_IMAGE_EVERY_N")
            row["aux_lambda"] = cfg.get("AUX_IMAGE_LAMBDA")
            row["roi_loss"] = cfg.get("USE_ROI_LOSS")
            row["early_stop_latent"] = cfg.get("EARLY_STOP_LATENT_METRIC")
        rows.append(row)
    return pd.DataFrame(rows)


def load_test_ids() -> list[str]:
    path = repo_root() / "data_info" / "test.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_training_frames(exp_key: str) -> dict[str, pd.DataFrame]:
    """Legacy loader keyed by EXPERIMENTS short name."""
    return load_training_frames_for(EXPERIMENTS[exp_key])


def load_training_frames_for(exp_name: str) -> dict[str, pd.DataFrame]:
    root = result_dir_exp(exp_name)
    frames: dict[str, pd.DataFrame] = {}
    for name in (
        "loss_curve",
        "validation",
        "test",
        "validation_latent",
        "test_latent",
    ):
        path = root / f"{name}.csv"
        if path.exists():
            frames[name] = pd.read_csv(path)
    return frames


def load_inference_reconstruction(
    exp_key: str,
    run_id: str,
    mode_key: str | None = None,
) -> pd.DataFrame:
    """Load test reconstruction metrics. exp_key may be short name or full exp folder."""
    exp_name = EXPERIMENTS.get(exp_key, exp_key)
    return load_inference_reconstruction_for(exp_name, run_id, mode_key or exp_key)


def load_inference_reconstruction_for(
    exp_name: str,
    run_id: str,
    mode_key: str,
) -> pd.DataFrame:
    path = inference_dir_exp(exp_name, run_id) / "reconstruction_metrics.csv"
    df = pd.read_csv(path, dtype={"ID": str})
    test_df = df[df["split"] == "test"].copy()
    test_df["mode"] = mode_key
    test_df["label_name"] = test_df["label"].map(LABEL_NAMES)
    return test_df


def load_classification_metrics(
    exp_key: str,
    run_id: str,
    mode_key: str | None = None,
) -> pd.DataFrame:
    exp_name = EXPERIMENTS.get(exp_key, exp_key)
    return load_classification_metrics_for(exp_name, run_id, mode_key or exp_key)


def load_classification_metrics_for(exp_name: str, run_id: str, mode_key: str) -> pd.DataFrame:
    path = inference_dir_exp(exp_name, run_id) / "classification" / "classification_metrics.csv"
    df = pd.read_csv(path)
    df["mode"] = mode_key
    return df


def best_validation_epoch(
    validation: pd.DataFrame,
    *,
    criterion: EarlyStopCriterion = "image",
    validation_latent: pd.DataFrame | None = None,
) -> pd.Series:
    if criterion == "latent":
        if validation_latent is None or validation_latent.empty:
            raise ValueError("validation_latent required for latent early-stop criterion")
        idx = validation_latent["latent_MSE"].idxmin()
        return validation_latent.loc[idx]
    idx = validation["PSNR"].idxmax()
    return validation.loc[idx]


def training_checkpoint_summary(
    validation: pd.DataFrame,
    test_ckpt: pd.DataFrame,
    *,
    criterion: EarlyStopCriterion = "image",
    validation_latent: pd.DataFrame | None = None,
    test_latent: pd.DataFrame | None = None,
) -> dict:
    best_val = best_validation_epoch(
        validation,
        criterion=criterion,
        validation_latent=validation_latent,
    )
    best_epoch = int(best_val["Epoch"])
    test_row = test_ckpt[test_ckpt["Epoch"] == best_epoch]
    test_psnr = float(test_row["PSNR"].iloc[0]) if len(test_row) else float("nan")
    test_ssim = float(test_row["SSIM"].iloc[0]) if len(test_row) else float("nan")

    val_psnr = float(best_val["PSNR"]) if "PSNR" in best_val else float("nan")
    val_ssim = float(best_val["SSIM"]) if "SSIM" in best_val else float("nan")
    val_latent_mse = float(best_val["latent_MSE"]) if "latent_MSE" in best_val else float("nan")
    val_latent_mae = float(best_val["latent_MAE"]) if "latent_MAE" in best_val else float("nan")

    test_latent_mse = float("nan")
    test_latent_mae = float("nan")
    if test_latent is not None and not test_latent.empty:
        tl_row = test_latent[test_latent["Epoch"] == best_epoch]
        if len(tl_row):
            test_latent_mse = float(tl_row["latent_MSE"].iloc[0])
            test_latent_mae = float(tl_row["latent_MAE"].iloc[0])

    val_row = validation[validation["Epoch"] == best_epoch]
    val_psnr_roi = float(val_row["PSNR_roi"].iloc[0]) if len(val_row) and "PSNR_roi" in val_row else float("nan")
    val_ssim_roi = float(val_row["SSIM_roi"].iloc[0]) if len(val_row) and "SSIM_roi" in val_row else float("nan")

    return {
        "best_epoch": best_epoch,
        "early_stop_criterion": criterion,
        "val_psnr": val_psnr,
        "val_ssim": val_ssim,
        "val_latent_mse": val_latent_mse,
        "val_latent_mae": val_latent_mae,
        "val_psnr_roi": val_psnr_roi,
        "val_ssim_roi": val_ssim_roi,
        "test_psnr_at_best": test_psnr,
        "test_ssim_at_best": test_ssim,
        "test_latent_mse_at_best": test_latent_mse,
        "test_latent_mae_at_best": test_latent_mae,
        "last_epoch": int(validation["Epoch"].max()),
        "n_val_points": len(validation),
        "n_test_checkpoints": len(test_ckpt),
    }


def pick_stratified_subjects(
    recon_by_mode: dict[str, pd.DataFrame],
    *,
    percentiles: tuple[float, float, float] = (10.0, 50.0, 90.0),
) -> pd.DataFrame:
    """3 control + 3 parkinson at p10/p50/p90 of mean PSNR across experiments."""
    pivot = None
    for mode, df in recon_by_mode.items():
        part = df[["ID", "label", "PSNR"]].rename(columns={"PSNR": f"psnr_{mode}"})
        pivot = part if pivot is None else pivot.merge(part, on=["ID", "label"], how="inner")

    psnr_cols = [c for c in pivot.columns if c.startswith("psnr_")]
    pivot["psnr_mean"] = pivot[psnr_cols].mean(axis=1)

    rows = []
    tier_names = ["peor_p10", "mediana_p50", "mejor_p90"]
    for label_value, label_name in LABEL_NAMES.items():
        subset = pivot[pivot["label"] == label_value].copy()
        if subset.empty:
            continue
        for tier, pct in zip(tier_names, percentiles):
            target = np.percentile(subset["psnr_mean"], pct)
            idx = (subset["psnr_mean"] - target).abs().idxmin()
            row = subset.loc[idx].copy()
            row["tier"] = tier
            row["label_name"] = label_name
            rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def load_synthetic_volume(exp_key: str, run_id: str, subject_id: str) -> np.ndarray:
    exp_name = EXPERIMENTS.get(exp_key, exp_key)
    return load_synthetic_volume_for(exp_name, run_id, subject_id)


def load_synthetic_volume_for(exp_name: str, run_id: str, subject_id: str) -> np.ndarray:
    inf = inference_dir_exp(exp_name, run_id)
    nifti_path = inf / "synthetic_spect" / "test" / f"{subject_id}.nii.gz"
    if nifti_path.exists():
        vol = nib.load(str(nifti_path)).get_fdata(dtype=np.float32)
        return vol[0] if vol.ndim == 4 else vol

    npz_path = inf / "representations" / "test" / f"{subject_id}.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        if "synthetic_spect_flat" in data:
            flat = data["synthetic_spect_flat"]
            return flat.reshape(CROP_SHAPE)

    raise FileNotFoundError(
        f"No synthetic volume for {subject_id} in {exp_name} "
        f"(checked {nifti_path} and {npz_path})"
    )


def validate_artifacts(run_id: str) -> list[str]:
    """Legacy: single run_id for all EXPERIMENTS modes."""
    mapping = {mode: {"exp": exp, "run_id": run_id} for mode, exp in EXPERIMENTS.items()}
    return validate_comparison(mapping)


def validate_frozen_comparison() -> list[str]:
    return validate_comparison(FROZEN_COMPARISON)


def validate_comparison(comparison: dict[str, dict[str, str]]) -> list[str]:
    warnings: list[str] = []
    test_ids = set(load_test_ids())
    for mode_key, cfg in comparison.items():
        exp_name = cfg["exp"]
        run_id = cfg["run_id"]
        root = result_dir_exp(exp_name)
        for fname in ("loss_curve.csv", "validation.csv", "test.csv"):
            if not (root / fname).exists():
                warnings.append(f"Missing {root / fname}")

        for optional in ("validation_latent.csv", "test_latent.csv"):
            path = root / optional
            if mode_key == "frozen_v2" and not path.exists():
                warnings.append(f"Missing {path}")

        inf = inference_dir_exp(exp_name, run_id)
        recon_path = inf / "reconstruction_metrics.csv"
        if not recon_path.exists():
            warnings.append(f"Missing {recon_path}")
            continue

        recon = pd.read_csv(recon_path, dtype={"ID": str})
        test_recon = recon[recon["split"] == "test"]
        if len(test_recon) != len(test_ids):
            warnings.append(
                f"{exp_name} ({run_id}): expected {len(test_ids)} test rows, got {len(test_recon)}"
            )

        cls_path = inf / "classification" / "classification_metrics.csv"
        if not cls_path.exists():
            warnings.append(f"Missing {cls_path}")

    return warnings
