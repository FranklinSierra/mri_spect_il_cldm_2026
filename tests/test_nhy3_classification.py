"""Tests for NHY3 embedding classification (CTL / H&Y1 / H&Y2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_stratum_map_and_join():
    from code.nhy3_classification import (
        STRATUM_TO_CLASS,
        join_embeddings_with_nhy3,
        load_nhy_label_map,
    )

    nhy = load_nhy_label_map(ROOT / "data_info/splits/nhy_v1/nhy_distribution.csv")
    assert set(nhy["stratum"].unique()) >= set(STRATUM_TO_CLASS)
    assert nhy["nhy3_label"].isin([0, 1, 2]).all()

    meta = pd.DataFrame(
        {
            "ID": ["3102", "3104", "3105"],
            "split": ["train", "train", "test"],
            "label": [1.0, 0.0, 1.0],
            "representation_path": ["a.npz", "b.npz", "c.npz"],
        }
    )
    merged = join_embeddings_with_nhy3(meta, nhy)
    assert merged.loc[merged["ID"] == "3104", "nhy3_label"].iloc[0] == 0
    assert merged.loc[merged["ID"] == "3102", "stratum"].iloc[0] == "pd_nhy1"
    assert merged.loc[merged["ID"] == "3105", "stratum"].iloc[0] == "pd_nhy2"


def test_registry_embedding_eval_nhy3():
    registry = json.loads((ROOT / "experiments/registry.json").read_text(encoding="utf-8"))
    entry = registry["embedding_evals"]["calib_roi3ch_uni_rois_reproc_nhy3"]
    assert entry["parent"] == "calib_roi3ch_uni_rois_reproc"
    assert entry["type"] == "embedding_eval"
    assert entry["task"] == "nhy3"
    assert entry["out_subdir"] == "classification_nhy3"
    assert entry["class_map"] == {"control": 0, "pd_nhy1": 1, "pd_nhy2": 2}
    emb = ROOT / entry["embedding_root"]
    # Soft check: path may exist in this workspace
    if emb.is_dir():
        assert (emb / "metadata.csv").is_file()


def test_metric_row_nhy3_shapes():
    from code.nhy3_classification import metric_row_nhy3

    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    proba = np.eye(3)[y_pred]
    row = metric_row_nhy3("toy", y_true, y_pred, proba)
    assert "f1_macro" in row
    assert "balanced_accuracy" in row
    assert "recall_control" in row


@pytest.mark.skipif(
    not (
        ROOT
        / "result/exp_nm_s2_frozen_calib_roi3ch_uni_rois_reproc/inference/calib_roi3ch_uni_rois_reproc/metadata.csv"
    ).is_file(),
    reason="reproc inference embeddings not present",
)
def test_run_nhy3_smoke(tmp_path):
    from code.nhy3_classification import DEFAULT_EMBEDDING_ROOT, run_nhy3_classification

    out = tmp_path / "classification_nhy3"
    metrics = run_nhy3_classification(
        embedding_root=DEFAULT_EMBEDDING_ROOT,
        out_dir=out,
        device="cpu",
    )
    assert (out / "classification_metrics.csv").is_file()
    assert (out / "run_config_nhy3.json").is_file()
    assert (out / "test_predictions.csv").is_file()
    assert "balanced_accuracy" in metrics.columns
    # Binary classification dir must remain untouched by this out_dir choice
    binary = DEFAULT_EMBEDDING_ROOT / "classification" / "classification_metrics.csv"
    assert binary.is_file()
