"""Tests for PD clinical staging pipeline (HY1 vs HY2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_prompt_excludes_disease_labels_and_nhy():
    from code.clinical_staging import build_clinical_cohort

    cohort = build_clinical_cohort(require_radiology=False)
    prompt = cohort.iloc[0]["prompt"]
    assert "Subject aged" in prompt
    assert "Parkinson's patient" not in prompt
    assert "control subject" not in prompt
    assert "COHORT_DEFINITION" not in prompt
    assert "pd_nhy" not in prompt
    assert "numeric Hoehn" not in prompt


def test_cohort_pd_only_primary_split_counts():
    from code.clinical_staging import build_clinical_cohort, split_summary

    cohort = build_clinical_cohort(require_radiology=False)
    summary = split_summary(cohort)
    assert len(cohort) == 109
    assert summary["n_train"] == 89
    assert summary["n_test"] == 20
    assert summary["train"]["hy1"] == 42
    assert summary["train"]["hy2"] == 47
    assert summary["test"]["hy1"] == 9
    assert summary["test"]["hy2"] == 11
    assert set(cohort["stratum"]) <= {"pd_nhy1", "pd_nhy2"}


def test_export_clinical_pairs_columns(tmp_path):
    from code.clinical_staging import export_clinical_staging_artifacts

    out = tmp_path / "clinical_staging_pd"
    pairs = export_clinical_staging_artifacts(out_root=out, require_radiology=False)
    assert (out / "run_config.json").is_file()
    assert (out / "prompts.csv").is_file()
    assert (out / "clinical_pairs.csv").is_file()
    expected = {
        "ID",
        "split",
        "stratum",
        "stage_label",
        "prompt",
        "text_embedding_path",
        "representation_path",
    }
    assert expected <= set(pairs.columns)


def test_metric_row_staging_recall_columns():
    from code.nhy_staging_pd_classification import metric_row_staging_pd

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    proba = np.array([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.55, 0.45]])
    row = metric_row_staging_pd("toy", y_true, y_pred, proba)
    assert "recall_macro" in row
    assert "recall_hy1" in row
    assert "recall_hy2" in row


def _make_fixture_tree(tmp_path: Path) -> Path:
    from code.clinical_staging import export_clinical_staging_artifacts

    out = tmp_path / "clinical_staging_pd"
    pairs = export_clinical_staging_artifacts(out_root=out, require_radiology=False)

    reps_rad = tmp_path / "rad" / "reps"
    reps_rad.mkdir(parents=True)
    text_dir = out / "text_embeddings" / "reps"
    text_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    for row in pairs.itertuples(index=False):
        latent = rng.normal(size=(2, 2, 2)).astype(np.float32)
        np.savez(reps_rad / f"{row.ID}.npz", sampled_latent_final=latent)
        if not (text_dir / f"{row.ID}.npy").is_file():
            np.save(text_dir / f"{row.ID}.npy", rng.normal(size=768).astype(np.float32))

        pairs.loc[pairs["ID"] == row.ID, "representation_path"] = str(
            reps_rad / f"{row.ID}.npz"
        )
        pairs.loc[pairs["ID"] == row.ID, "text_embedding_path"] = str(
            text_dir / f"{row.ID}.npy"
        )

    pairs.to_csv(out / "clinical_pairs.csv", index=False)
    return out


def test_baselines_write_recall_metrics(tmp_path):
    from code.nhy_staging_pd_classification import run_staging_baselines

    out = _make_fixture_tree(tmp_path)
    metrics = run_staging_baselines(out_root=out)
    assert (out / "baselines" / "classification_metrics.csv").is_file()
    assert "recall_macro" in metrics.columns
    assert "recall_hy1" in metrics.columns


def test_stratified_batch_sampler_covers_both_classes():
    from code.clinical_radiology_clip import StratifiedBatchSampler

    labels = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    sampler = StratifiedBatchSampler(labels, batch_size=4, shuffle=False)
    batches = list(sampler)
    assert batches
    for batch in batches:
        batch_labels = labels[batch]
        if len(batch) >= 2:
            assert len(np.unique(batch_labels)) >= 2


def test_clip_forward_and_infonce_smoke():
    from code.clinical_radiology_clip import ClinicalRadiologyCLIP
    from losses.contrastive import CrossModalInfoNCE

    model = ClinicalRadiologyCLIP(image_dim=32, text_dim=16, proj_dim=8)
    img = torch.randn(4, 32, requires_grad=True)
    txt = torch.randn(4, 16)
    z_img, z_txt = model(img, txt)
    loss = CrossModalInfoNCE()(z_img, z_txt)
    assert z_img.shape == (4, 8)
    loss.backward()


def test_diagonal_retrieval_r1_is_alignment_not_staging():
    from code.clinical_radiology_clip import diagonal_retrieval_r1

    z = torch.eye(4)
    r1 = diagonal_retrieval_r1(z, z)
    assert r1 == 1.0


def test_early_stop_uses_val_supcon_not_epoch_one():
    from code.clinical_radiology_clip import (
        ClinicalRadiologyCLIP,
        PairedStagingDataset,
        train_clip_model_v2,
    )
    from torch.utils.data import DataLoader

    n = 12
    img = np.random.randn(n, 16).astype(np.float32)
    txt = np.random.randn(n, 8).astype(np.float32)
    labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    ds = PairedStagingDataset(img, txt, labels, [str(i) for i in range(n)])
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    model = ClinicalRadiologyCLIP(image_dim=16, text_dim=8, proj_dim=8)
    history = train_clip_model_v2(
        model,
        loader,
        loader,
        torch.device("cpu"),
        max_epochs=8,
        patience=20,
        min_epochs=3,
    )
    best_epoch = int(history.loc[history["val_supcon_loss"].idxmin(), "epoch"])
    assert best_epoch >= 3


def test_feature_scalers_fit_transform():
    from code.clinical_radiology_clip import FeatureScalers

    img = np.random.randn(20, 8).astype(np.float32) * 1000
    txt = np.random.randn(20, 4).astype(np.float32)
    scalers = FeatureScalers.fit(img, txt)
    img_s = scalers.transform_image(img)
    assert abs(img_s.mean()) < 0.5


def test_run_clip_v2_outputs(tmp_path):
    from code.clinical_radiology_clip import run_clinical_radiology_clip

    out = _make_fixture_tree(tmp_path)
    metrics = run_clinical_radiology_clip(
        out_root=out,
        batch_size=8,
        max_epochs=6,
        min_epochs=2,
        patience=20,
        device="cpu",
    )
    clip = out / "clip"
    assert (clip / "metrics_staging.csv").is_file()
    assert (clip / "metrics_alignment.csv").is_file()
    assert (clip / "checkpoints" / "feature_scalers.joblib").is_file()
    assert (clip / "fusion" / "classification_metrics.csv").is_file()
    assert "recall_hy2" in metrics.columns
    assert "balanced_accuracy" in metrics.columns
    fusion = pd.read_csv(clip / "fusion" / "classification_metrics.csv")
    assert "roc_auc_ovr_macro" in fusion.columns


def test_registry_clinical_staging_pd():
    registry = json.loads((ROOT / "experiments/registry.json").read_text(encoding="utf-8"))
    entry = registry["embedding_evals"]["clinical_staging_pd_nhy12"]
    assert entry["task"] == "pd_nhy12"
    assert entry["out_root"] == "result/clinical_staging_pd"
