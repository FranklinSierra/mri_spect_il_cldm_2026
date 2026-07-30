"""Helpers for stage2_contrastive_experiments_report.ipynb (E1 contrastive + E2 xm64 wave)."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from code.stage2_report_utils import (
    inference_dir_exp,
    load_classification_metrics_for,
    load_inference_reconstruction_for,
    load_registry,
    load_test_latent_matrix_for,
    repo_root,
    result_dir_exp,
    _sanitize_training_csv,
)

EXPORT_DIR_NAME = "stage2_contrastive_wave_summary"

# Encoder MRI seleccionado para variantes E2 xm64.
E1_SELECTED_KEY = "e1_xmodal_h64"

# E1 MRI wave contrastiva (2×2 SupCon/Xmodal × z18k/h64) + baseline recon.
E1_EXPERIMENTS: dict[str, dict[str, str]] = {
    "e1_baseline": {
        "exp": "exp_nm_s1_mri_rois",
        "label": "E1 MRI ROI baseline (recon only, z18k)",
        "short_label": "mri_rois",
        "analysis_name": "embedding_analysis",
        "embeddings_name": "embeddings",
        "contrastive_features": "z18k",
        "training_mode": "recon",
        "e2_mode": "calib_roi3ch_uni_rois",
    },
    "e1_supcon_z18k": {
        "exp": "exp_nm_s1_mri_rois_supcon_z18k",
        "label": "E1 MRI SupCon z18k (bottleneck)",
        "short_label": "supcon_z18k",
        "analysis_name": "embedding_analysis",
        "embeddings_name": "embeddings",
        "contrastive_features": "z18k",
        "training_mode": "supcon",
    },
    "e1_supcon_h64": {
        "exp": "exp_nm_s1_mri_rois_supcon_h64",
        "label": "E1 MRI SupCon h64 (pre-bottleneck map)",
        "short_label": "supcon_h64",
        "analysis_name": "embedding_analysis",
        "embeddings_name": "embeddings",
        "contrastive_features": "h64",
        "training_mode": "supcon",
    },
    "e1_xmodal_z18k": {
        "exp": "exp_nm_s1_mri_rois_xmodal_z18k",
        "label": "E1 MRI xmodal z18k (InfoNCE, bottleneck)",
        "short_label": "xmodal_z18k",
        "analysis_name": "embedding_analysis",
        "embeddings_name": "embeddings",
        "contrastive_features": "z18k",
        "training_mode": "xmodal",
    },
    "e1_xmodal_h64": {
        "exp": "exp_nm_s1_mri_rois_xmodal_h64",
        "label": "E1 MRI xmodal_h64 (InfoNCE, h64 map)",
        "short_label": "xmodal_h64",
        "analysis_name": "embedding_analysis",
        "embeddings_name": "embeddings",
        "contrastive_features": "h64",
        "training_mode": "xmodal",
        "e2_modes": "xm64_z18,xm64_h64,xm64_h64_gap1",
    },
}

E1_ORDER: tuple[str, ...] = (
    "e1_baseline",
    "e1_supcon_z18k",
    "e1_supcon_h64",
    "e1_xmodal_z18k",
    "e1_xmodal_h64",
)

# E2 ROI diffusion: baseline MRI encoder vs xm64 variants (all use xmodal_h64 MRI ckpt).
E2_ORDER: tuple[str, ...] = (
    "calib_roi3ch_uni_rois",
    "calib_roi3ch_uni_rois_xm64_z18",
    "calib_roi3ch_uni_rois_xm64_h64",
    "calib_roi3ch_uni_rois_xm64_h64_gap1",
)

E1_E2_LINK_ROWS: list[dict[str, str]] = [
    {
        "e1_key": "e1_baseline",
        "e1_short": "mri_rois",
        "e1_feature": "z18k",
        "e2_mode": "calib_roi3ch_uni_rois",
        "e2_short": "calib_roi3ch_uni_rois",
        "mri_condition": "z18k bottleneck",
        "diffusion_space": "z18k",
    },
    {
        "e1_key": "e1_xmodal_h64",
        "e1_short": "xmodal_h64",
        "e1_feature": "h64",
        "e1_analysis_z18": "embedding_analysis_z18k",
        "e2_mode": "calib_roi3ch_uni_rois_xm64_z18",
        "e2_short": "xm64_z18",
        "mri_condition": "z18k bottleneck (same encoder)",
        "diffusion_space": "z18k",
    },
    {
        "e1_key": "e1_xmodal_h64",
        "e1_short": "xmodal_h64",
        "e1_feature": "h64",
        "e2_mode": "calib_roi3ch_uni_rois_xm64_h64",
        "e2_short": "xm64_h64",
        "mri_condition": "h64 map",
        "diffusion_space": "h64 (64ch)",
    },
    {
        "e1_key": "e1_xmodal_h64",
        "e1_short": "xmodal_h64",
        "e1_feature": "h64 (GAP mean)",
        "e2_mode": "calib_roi3ch_uni_rois_xm64_h64_gap1",
        "e2_short": "xm64_h64_gap1",
        "mri_condition": "h64 map",
        "diffusion_space": "h64_gap1 (channel-mean → 1ch)",
    },
]


def export_dir() -> Path:
    return repo_root() / "result" / EXPORT_DIR_NAME


def phase1_mri_root(exp_name: str) -> Path:
    return result_dir_exp(exp_name) / "phase1_mri"


def load_phase1_run_config(exp_name: str) -> dict | None:
    path = phase1_mri_root(exp_name) / "run_config.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_phase1_validation_summary(exp_name: str) -> pd.DataFrame:
    path = phase1_mri_root(exp_name) / "reconstruction" / "validation_summary.csv"
    if not path.is_file():
        return pd.DataFrame()
    return _sanitize_training_csv(pd.read_csv(path))


def analysis_dir(exp_name: str, analysis_name: str) -> Path:
    return phase1_mri_root(exp_name) / analysis_name


def load_phase1_classification(exp_name: str, analysis_name: str = "embedding_analysis") -> pd.DataFrame:
    path = analysis_dir(exp_name, analysis_name) / "classification" / "classification_metrics.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_crossmodal_retrieval(exp_name: str, analysis_name: str = "embedding_analysis") -> dict | None:
    path = analysis_dir(exp_name, analysis_name) / "classification" / "crossmodal_retrieval.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_phase1_projection_csv(
    exp_name: str,
    method: str,
    ndim: int,
    analysis_name: str = "embedding_analysis",
) -> pd.DataFrame | None:
    path = analysis_dir(exp_name, analysis_name) / "visualizations" / f"{method}_{ndim}d.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path, dtype={"ID": str})


def load_embedding_row_vector(embedding_path: str | Path) -> np.ndarray:
    path = Path(embedding_path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32).reshape(-1)
    return nib.load(str(path)).get_fdata().astype(np.float32).reshape(-1)


def load_e1_internal_test_embeddings(exp_name: str, analysis_name: str = "embedding_analysis") -> tuple[
    np.ndarray, np.ndarray, list[str]
]:
    """Internal 80/20 eval split test rows from embedding_analysis."""
    split_path = analysis_dir(exp_name, analysis_name) / "embedding_eval_split.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing {split_path}")
    split_df = pd.read_csv(split_path, dtype={"ID": str})
    test_rows = split_df[split_df["embedding_eval_split"] == "test"].copy()
    if test_rows.empty:
        raise ValueError(f"No internal test rows in {split_path}")

    features: list[np.ndarray] = []
    labels: list[int] = []
    ids: list[str] = []
    for row in test_rows.itertuples(index=False):
        features.append(load_embedding_row_vector(row.embedding_path))
        labels.append(int(row.label))
        ids.append(str(row.ID))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), ids


def available_e1_experiments() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for key, cfg in E1_EXPERIMENTS.items():
        exp = cfg["exp"]
        analysis = cfg["analysis_name"]
        cls_path = analysis_dir(exp, analysis) / "classification" / "classification_metrics.csv"
        val_path = phase1_mri_root(exp) / "reconstruction" / "validation_summary.csv"
        if cls_path.is_file() or val_path.is_file():
            out[key] = dict(cfg)
    return out


def e2_comparison_from_registry() -> dict[str, dict[str, str]]:
    registry = load_registry()["experiments"]
    out: dict[str, dict[str, str]] = {}
    for mode_key in E2_ORDER:
        if mode_key not in registry:
            continue
        entry = registry[mode_key]
        exp_name = entry["exp"]
        root = result_dir_exp(exp_name)
        if not (root / "validation.csv").is_file():
            continue
        out[mode_key] = {
            "exp": exp_name,
            "run_id": entry["run_id"],
            "label": entry["label"],
            "short_label": entry.get("short_label", mode_key),
            "early_stop": entry.get("early_stop", "image"),
            "split_profile": entry.get("split_profile", "primary"),
            "overrides": entry.get("overrides", {}),
        }
    return out


def validate_e1_artifacts(keys: list[str] | None = None) -> list[str]:
    warnings: list[str] = []
    for key in keys or E1_ORDER:
        if key not in E1_EXPERIMENTS:
            warnings.append(f"Unknown E1 key {key!r}")
            continue
        cfg = E1_EXPERIMENTS[key]
        exp = cfg["exp"]
        root = phase1_mri_root(exp)
        for rel in ("run_config.json", "reconstruction/validation_summary.csv"):
            if not (root / rel).exists():
                warnings.append(f"Missing {root / rel}")
        analysis = cfg["analysis_name"]
        cls = analysis_dir(exp, analysis) / "classification" / "classification_metrics.csv"
        if not cls.is_file():
            warnings.append(f"Missing {cls}")
    return warnings


def validate_e2_artifacts(comparison: dict[str, dict[str, str]]) -> list[str]:
    warnings: list[str] = []
    test_ids_path = repo_root() / "data_info" / "test.txt"
    test_ids = {line.strip() for line in test_ids_path.read_text().splitlines() if line.strip()}
    for mode_key, cfg in comparison.items():
        exp_name = cfg["exp"]
        run_id = cfg["run_id"]
        root = result_dir_exp(exp_name)
        for fname in ("loss_curve.csv", "validation.csv"):
            if not (root / fname).is_file():
                warnings.append(f"Missing {root / fname}")
        inf = inference_dir_exp(exp_name, run_id)
        recon_path = inf / "reconstruction_metrics.csv"
        if not recon_path.is_file():
            warnings.append(f"Missing {recon_path}")
            continue
        recon = pd.read_csv(recon_path, dtype={"ID": str})
        test_recon = recon[recon["split"] == "test"]
        if len(test_recon) != len(test_ids):
            warnings.append(
                f"{exp_name}: expected {len(test_ids)} test recon rows, got {len(test_recon)}"
            )
        cls_path = inf / "classification" / "classification_metrics.csv"
        if not cls_path.is_file():
            warnings.append(f"Missing {cls_path}")
    return warnings


def build_e1_leaderboard(e1_comparison: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Per-experiment best classifier + contrastive signals; ranks evidence for E2 selection."""
    rows: list[dict] = []
    baseline_bal = np.nan
    for key in E1_ORDER:
        if key not in e1_comparison:
            continue
        cfg = e1_comparison[key]
        cls = load_phase1_classification(cfg["exp"], cfg["analysis_name"])
        best = best_classifier_row(cls)
        val = load_phase1_validation_summary(cfg["exp"])
        min_val_ctr = np.nan
        if not val.empty and "val_contrastive_loss" in val.columns:
            ctr = val["val_contrastive_loss"].dropna()
            if len(ctr):
                min_val_ctr = float(ctr.min())
        retrieval_top1 = np.nan
        if cfg.get("training_mode") == "xmodal":
            ret = load_crossmodal_retrieval(cfg["exp"], cfg["analysis_name"])
            if ret:
                retrieval_top1 = float(ret.get("top1_retrieval", np.nan))
        bal = float(best["balanced_accuracy"]) if best is not None else np.nan
        if key == "e1_baseline":
            baseline_bal = bal
        rows.append(
            {
                "e1_key": key,
                "e1_short": cfg["short_label"],
                "training_mode": cfg.get("training_mode", ""),
                "contrastive_features": cfg.get("contrastive_features", ""),
                "best_model": best["model"] if best is not None else "",
                "balanced_accuracy": bal,
                "roc_auc": float(best.get("roc_auc", np.nan)) if best is not None else np.nan,
                "min_val_contrastive_loss": min_val_ctr,
                "xmodal_top1_retrieval": retrieval_top1,
                "selected_for_e2": key == E1_SELECTED_KEY,
            }
        )
    df = pd.DataFrame(rows)
    if not np.isnan(baseline_bal):
        df["delta_bal_acc_vs_baseline"] = df["balanced_accuracy"] - baseline_bal
    return df.sort_values("balanced_accuracy", ascending=False).reset_index(drop=True)


def best_classifier_row(metrics: pd.DataFrame) -> pd.Series | None:
    if metrics.empty or "balanced_accuracy" not in metrics.columns:
        return None
    sort_cols = [c for c in ("roc_auc", "f1") if c in metrics.columns]
    if sort_cols:
        return metrics.sort_values(["balanced_accuracy"] + sort_cols, ascending=False).iloc[0]
    return metrics.sort_values("balanced_accuracy", ascending=False).iloc[0]


def load_e2_classification(mode_key: str, comparison: dict[str, dict[str, str]]) -> pd.DataFrame:
    cfg = comparison[mode_key]
    df = load_classification_metrics_for(cfg["exp"], cfg["run_id"], mode_key)
    df["mode_short"] = cfg["short_label"]
    df["mode_label"] = cfg["label"]
    return df


def load_e2_reconstruction(mode_key: str, comparison: dict[str, dict[str, str]]) -> pd.DataFrame:
    cfg = comparison[mode_key]
    df = load_inference_reconstruction_for(cfg["exp"], cfg["run_id"], mode_key)
    df["mode_short"] = cfg["short_label"]
    df["mode_label"] = cfg["label"]
    return df


def build_e1_e2_summary_table(
    e1_keys: list[str],
    e2_comparison: dict[str, dict[str, str]],
) -> pd.DataFrame:
    rows: list[dict] = []
    for link in E1_E2_LINK_ROWS:
        e1_key = link["e1_key"]
        if e1_key not in e1_keys:
            continue
        e2_mode = link["e2_mode"]
        if e2_mode not in e2_comparison:
            continue
        e1_cfg = E1_EXPERIMENTS[e1_key]
        e1_exp = e1_cfg["exp"]
        analysis = link.get("e1_analysis_z18", e1_cfg["analysis_name"])
        e1_cls = load_phase1_classification(e1_exp, analysis)
        e1_best = best_classifier_row(e1_cls)

        e2_cls = load_e2_classification(e2_mode, e2_comparison)
        e2_best = best_classifier_row(e2_cls)
        e2_recon = load_e2_reconstruction(e2_mode, e2_comparison)
        test_recon = e2_recon[e2_recon["split"] == "test"]

        rows.append(
            {
                "e1_key": e1_key,
                "e1_short": link["e1_short"],
                "e1_feature_eval": link["e1_feature"],
                "e2_mode": e2_mode,
                "e2_short": link["e2_short"],
                "mri_condition": link["mri_condition"],
                "diffusion_space": link["diffusion_space"],
                "e1_best_model": e1_best["model"] if e1_best is not None else "",
                "e1_balanced_accuracy": float(e1_best["balanced_accuracy"]) if e1_best is not None else np.nan,
                "e1_roc_auc": float(e1_best.get("roc_auc", np.nan)) if e1_best is not None else np.nan,
                "e2_best_model": e2_best["model"] if e2_best is not None else "",
                "e2_balanced_accuracy": float(e2_best["balanced_accuracy"]) if e2_best is not None else np.nan,
                "e2_roc_auc": float(e2_best.get("roc_auc", np.nan)) if e2_best is not None else np.nan,
                "e2_test_psnr_mean": float(test_recon["PSNR"].mean()) if not test_recon.empty else np.nan,
                "e2_test_ssim_mean": float(test_recon["SSIM"].mean()) if not test_recon.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)
