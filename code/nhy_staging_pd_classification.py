"""PD-only HY1 vs HY2 baselines on image and text embeddings (KNN k=5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from code.clinical_staging import (
    CLASS_NAMES,
    DEFAULT_CLINICAL_STAGING_ROOT,
    REPO_ROOT,
    _repo_relative,
)
from code.nhy3_classification import classifier_params


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path.resolve()
    candidate = REPO_ROOT / path
    if candidate.is_file():
        return candidate.resolve()
    return path.resolve()


def metric_row_staging_pd(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    **params: Any,
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    row: dict[str, Any] = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc_ovr_macro": np.nan,
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            row["roc_auc_ovr_macro"] = roc_auc_score(
                y_true, y_proba[:, 1] if y_proba.ndim == 2 and y_proba.shape[1] == 2 else y_proba
            )
        except ValueError:
            pass
    for i, name in enumerate(CLASS_NAMES):
        mask = y_true == i
        if mask.any():
            row[f"recall_{name}"] = recall_score(
                (y_true == i).astype(int), (y_pred == i).astype(int), zero_division=0
            )
            row[f"f1_{name}"] = f1_score(
                (y_true == i).astype(int), (y_pred == i).astype(int), zero_division=0
            )
    row.update(params)
    return row


def build_knn_pipeline(y_train: np.ndarray, k: int = 5) -> Pipeline:
    n_neighbors = min(k, len(y_train))
    if n_neighbors % 2 == 0:
        n_neighbors -= 1
    n_neighbors = max(1, n_neighbors)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")),
        ]
    )


def load_image_features(pairs: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = pairs[pairs["split"] == split]
    features, labels, ids = [], [], []
    for row in rows.itertuples(index=False):
        path = resolve_path(row.representation_path)
        payload = np.load(path)
        features.append(payload["sampled_latent_final"].reshape(-1))
        labels.append(int(row.stage_label))
        ids.append(str(row.ID))
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        ids,
    )


def load_text_features(pairs: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = pairs[pairs["split"] == split]
    features, labels, ids = [], [], []
    for row in rows.itertuples(index=False):
        path = resolve_path(row.text_embedding_path)
        features.append(np.load(path).reshape(-1))
        labels.append(int(row.stage_label))
        ids.append(str(row.ID))
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        ids,
    )


def run_staging_baselines(
    out_root: Path | str = DEFAULT_CLINICAL_STAGING_ROOT,
    pairs_csv: Path | None = None,
    knn_k: int = 5,
) -> pd.DataFrame:
    out_root = Path(out_root)
    out_dir = out_root / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = pairs_csv or (out_root / "clinical_pairs.csv")
    if not pairs_path.is_file():
        raise FileNotFoundError(
            f"Missing {pairs_path}. Run extract_clinical_text_embeddings.py first."
        )
    pairs = pd.read_csv(pairs_path, dtype={"ID": str})

    X_img_train, y_train, train_ids = load_image_features(pairs, "train")
    X_img_test, y_test, test_ids = load_image_features(pairs, "test")
    X_txt_train, y_txt_train, _ = load_text_features(pairs, "train")
    X_txt_test, y_txt_test, _ = load_text_features(pairs, "test")
    if not np.array_equal(y_train, y_txt_train) or not np.array_equal(y_test, y_txt_test):
        raise ValueError("Text/image label mismatch for staging baselines")

    rows = []
    prediction_rows = pd.DataFrame(
        {
            "ID": test_ids,
            "stage_label": y_test,
            "stratum": pairs.set_index("ID").loc[test_ids, "stratum"].tolist(),
            "class_name": [CLASS_NAMES[i] for i in y_test],
        }
    )

    baselines = {
        "KNN_image_k5": ("image", X_img_train, X_img_test),
        "KNN_text_k5": ("text", X_txt_train, X_txt_test),
    }
    for name, (_, X_tr, X_te) in baselines.items():
        model = build_knn_pipeline(y_train, k=knn_k)
        model.fit(X_tr, y_train)
        pred = model.predict(X_te)
        proba = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None
        rows.append(metric_row_staging_pd(name, y_test, pred, proba, **classifier_params(model)))
        prediction_rows[f"pred_{name}"] = pred
        if proba is not None:
            for i, cname in enumerate(CLASS_NAMES):
                prediction_rows[f"proba_{name}_{cname}"] = proba[:, i]

    metrics = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    metrics.to_csv(out_dir / "classification_metrics.csv", index=False)
    prediction_rows.to_csv(out_dir / "test_predictions.csv", index=False)

    for name in baselines:
        pred = prediction_rows[f"pred_{name}"].to_numpy()
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            out_dir / f"confusion_matrix_{name}.csv"
        )
        report = classification_report(
            y_test, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        )
        with open(out_dir / f"classification_report_{name}.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    run_config = {
        "analysis_id": "clinical_staging_pd_baselines",
        "task": "pd_nhy12_knn",
        "knn_k": knn_k,
        "class_names": CLASS_NAMES,
        "pairs_csv": _repo_relative(pairs_path.resolve()),
        "split_summary": {
            "train": int((pairs["split"] == "train").sum()),
            "test": int((pairs["split"] == "test").sum()),
        },
    }
    with open(out_dir / "run_config_baselines.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"[staging_baselines] wrote metrics → {out_dir / 'classification_metrics.csv'}")
    print(metrics[["model", "balanced_accuracy", "recall_macro", "f1_macro"]].to_string(index=False))
    return metrics
