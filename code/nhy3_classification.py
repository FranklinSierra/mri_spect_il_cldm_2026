"""NHY3 (CTL / H&Y1 / H&Y2) classification on existing DDIM embeddings.

Does not resample the LDM. Reuses inference metadata + npz latents and remaps
labels from data_info/splits/nhy_v1/nhy_distribution.csv. Writes under
classification_nhy3/ so binary classification/ stays intact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]

STRATUM_TO_CLASS = {
    "control": 0,
    "pd_nhy1": 1,
    "pd_nhy2": 2,
}
CLASS_NAMES = ["control", "pd_nhy1", "pd_nhy2"]
DEFAULT_NHY_CSV = REPO_ROOT / "data_info" / "splits" / "nhy_v1" / "nhy_distribution.csv"
DEFAULT_EMBEDDING_ROOT = (
    REPO_ROOT
    / "result"
    / "exp_nm_s2_frozen_calib_roi3ch_uni_rois_reproc"
    / "inference"
    / "calib_roi3ch_uni_rois_reproc"
)


def load_nhy_label_map(nhy_csv: Path | str = DEFAULT_NHY_CSV) -> pd.DataFrame:
    df = pd.read_csv(nhy_csv, dtype={"ID": str})
    missing = set(STRATUM_TO_CLASS) - set(df["stratum"].dropna().unique())
    if missing:
        raise ValueError(f"NHY CSV missing expected strata: {sorted(missing)}")
    df = df[["ID", "stratum", "NHY", "label"]].copy()
    df["nhy3_label"] = df["stratum"].map(STRATUM_TO_CLASS)
    if df["nhy3_label"].isna().any():
        bad = df.loc[df["nhy3_label"].isna(), "stratum"].unique().tolist()
        raise ValueError(f"Unknown stratum values in NHY CSV: {bad}")
    df["nhy3_label"] = df["nhy3_label"].astype(int)
    return df


def join_embeddings_with_nhy3(
    metadata: pd.DataFrame,
    nhy_df: pd.DataFrame,
) -> pd.DataFrame:
    """Keep inference train/test split; replace binary label with NHY3 classes."""
    meta = metadata.copy()
    meta["ID"] = meta["ID"].astype(str)
    merged = meta.merge(nhy_df[["ID", "stratum", "NHY", "nhy3_label"]], on="ID", how="left")
    if merged["nhy3_label"].isna().any():
        missing_ids = merged.loc[merged["nhy3_label"].isna(), "ID"].tolist()
        raise ValueError(f"IDs without NHY3 stratum: {missing_ids[:10]}")
    merged["nhy3_label"] = merged["nhy3_label"].astype(int)
    return merged


def load_feature_matrix_nhy3(
    merged: pd.DataFrame, split: str
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    rows = merged[merged["split"] == split].copy()
    features, labels, ids, strata = [], [], [], []
    for row in rows.itertuples(index=False):
        path = Path(row.representation_path)
        if not path.is_file():
            path = REPO_ROOT / row.representation_path
        payload = np.load(path)
        features.append(payload["sampled_latent_final"].reshape(-1))
        labels.append(int(row.nhy3_label))
        ids.append(str(row.ID))
        strata.append(str(row.stratum))
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        ids,
        strata,
    )


def build_classical_models_nhy3(y_train: np.ndarray) -> dict[str, Pipeline]:
    knn_auto = min(5, len(y_train))
    if knn_auto % 2 == 0:
        knn_auto -= 1
    knn_auto = max(1, knn_auto)
    knn_k2 = min(2, len(y_train))

    def pipe(classifier):
        return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])

    return {
        "KNN_auto": pipe(KNeighborsClassifier(n_neighbors=knn_auto, weights="distance")),
        "KNN_K2": pipe(KNeighborsClassifier(n_neighbors=knn_k2, weights="distance")),
        "LogisticRegression": pipe(
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            )
        ),
    }


class EmbeddingMLP3(nn.Module):
    def __init__(self, input_dim: int, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_dnn_nhy3(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    n_classes: int = 3,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    model = EmbeddingMLP3(X_train_scaled.shape[1], n_classes=n_classes).to(device)
    counts = np.bincount(y_train.astype(int), minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = (counts.sum() / (n_classes * counts)).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_scaled),
            torch.tensor(y_train.astype(np.int64)),
        ),
        batch_size=min(16, len(X_train)),
        shuffle=True,
    )
    X_test_t = torch.tensor(X_test_scaled, device=device)
    y_test_t = torch.tensor(y_test.astype(np.int64), device=device)

    best_state = None
    best_loss = float("inf")
    bad_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(300):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            test_loss = float(criterion(model(X_test_t), y_test_t).item())
        history.append(
            {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "test_loss": test_loss}
        )
        if test_loss < best_loss:
            best_loss = test_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= 30:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = probs.argmax(axis=1).astype(int)
    return pred, probs, pd.DataFrame(history)


def metric_row_nhy3(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    **params: Any,
) -> dict[str, Any]:
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
                y_true, y_proba, multi_class="ovr", average="macro"
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


def classifier_params(model: Pipeline) -> dict[str, Any]:
    clf = model.named_steps["classifier"]
    params: dict[str, Any] = {"classifier": clf.__class__.__name__}
    if hasattr(clf, "n_neighbors"):
        params["n_neighbors"] = clf.n_neighbors
    return params


def _proba_or_none(model: Pipeline, X: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None


def run_nhy3_classification(
    embedding_root: Path,
    out_dir: Path | None = None,
    nhy_csv: Path = DEFAULT_NHY_CSV,
    device: str | None = None,
    parent_experiment: str = "calib_roi3ch_uni_rois_reproc",
    analysis_id: str = "calib_roi3ch_uni_rois_reproc_nhy3",
) -> pd.DataFrame:
    embedding_root = Path(embedding_root)
    metadata_path = embedding_root / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    out_dir = Path(out_dir) if out_dir is not None else embedding_root / "classification_nhy3"
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    metadata = pd.read_csv(metadata_path, dtype={"ID": str})
    nhy_df = load_nhy_label_map(nhy_csv)
    merged = join_embeddings_with_nhy3(metadata, nhy_df)
    merged.to_csv(out_dir / "metadata_nhy3.csv", index=False)

    X_train, y_train, train_ids, train_strata = load_feature_matrix_nhy3(merged, "train")
    X_test, y_test, test_ids, test_strata = load_feature_matrix_nhy3(merged, "test")
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("NHY3 classification requires non-empty train and test splits.")

    split_summary = {
        "train": {CLASS_NAMES[i]: int((y_train == i).sum()) for i in range(3)},
        "test": {CLASS_NAMES[i]: int((y_test == i).sum()) for i in range(3)},
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    rows = []
    prediction_rows = pd.DataFrame(
        {
            "ID": test_ids,
            "nhy3_label": y_test,
            "stratum": test_strata,
            "class_name": [CLASS_NAMES[i] for i in y_test],
        }
    )

    for name, model in build_classical_models_nhy3(y_train).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = _proba_or_none(model, X_test)
        rows.append(metric_row_nhy3(name, y_test, pred, proba, **classifier_params(model)))
        prediction_rows[f"pred_{name}"] = pred
        if proba is not None:
            for i, cname in enumerate(CLASS_NAMES):
                prediction_rows[f"proba_{name}_{cname}"] = proba[:, i]

    dnn_pred, dnn_proba, history = train_dnn_nhy3(
        X_train, y_train, X_test, y_test, torch_device
    )
    rows.append(
        metric_row_nhy3("DNN_simple", y_test, dnn_pred, dnn_proba, classifier="EmbeddingMLP3")
    )
    prediction_rows["pred_DNN_simple"] = dnn_pred
    for i, cname in enumerate(CLASS_NAMES):
        prediction_rows[f"proba_DNN_simple_{cname}"] = dnn_proba[:, i]

    metrics = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    metrics.to_csv(out_dir / "classification_metrics.csv", index=False)
    prediction_rows.to_csv(out_dir / "test_predictions.csv", index=False)
    history.to_csv(out_dir / "dnn_training_history.csv", index=False)

    report = classification_report(
        y_test, dnn_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    with open(out_dir / "dnn_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    cm = confusion_matrix(y_test, dnn_pred, labels=[0, 1, 2])
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        out_dir / "dnn_confusion_matrix.csv"
    )

    # Best classical model confusion too
    best_name = metrics.iloc[0]["model"]
    if best_name != "DNN_simple":
        best_pred = prediction_rows[f"pred_{best_name}"].to_numpy()
        cm_best = confusion_matrix(y_test, best_pred, labels=[0, 1, 2])
        pd.DataFrame(cm_best, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            out_dir / f"confusion_matrix_{best_name}.csv"
        )

    run_config = {
        "analysis_id": analysis_id,
        "parent_experiment": parent_experiment,
        "type": "embedding_eval",
        "task": "nhy3",
        "class_map": STRATUM_TO_CLASS,
        "class_names": CLASS_NAMES,
        "embedding_root": str(embedding_root),
        "label_source": str(nhy_csv),
        "split_policy": "inference_metadata_split (primary reproc); NOT nhy_distribution.split",
        "split_summary": split_summary,
        "device": str(device),
        "outputs": str(out_dir),
    }
    with open(out_dir / "run_config_nhy3.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"[nhy3] wrote metrics → {out_dir / 'classification_metrics.csv'}")
    print(metrics[["model", "balanced_accuracy", "f1_macro", "accuracy"]].to_string(index=False))
    return metrics
