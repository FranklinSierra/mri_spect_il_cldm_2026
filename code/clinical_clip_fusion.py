"""Late fusion classifiers on projected CLIP embeddings [z_img; z_txt]."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from code.clinical_staging import CLASS_NAMES
from code.nhy_staging_pd_classification import metric_row_staging_pd


class FusionMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_fusion_classifiers(y_train: np.ndarray, knn_k: int = 5) -> dict[str, Pipeline]:
    n_neighbors = min(knn_k, len(y_train))
    if n_neighbors % 2 == 0:
        n_neighbors -= 1
    n_neighbors = max(1, n_neighbors)

    def pipe(clf):
        return Pipeline([("scaler", StandardScaler()), ("classifier", clf)])

    return {
        "Fusion_KNN_k5": pipe(KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")),
        "Fusion_LogisticRegression": pipe(
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
        ),
        "Fusion_SVM_linear": pipe(
            SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced", random_state=42)
        ),
        "Fusion_SVM_RBF": pipe(
            SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=42)
        ),
    }


def _fusion_classifier_params(model: Pipeline) -> dict[str, Any]:
    clf = model.named_steps["classifier"]
    params: dict[str, Any] = {"classifier": clf.__class__.__name__}
    if hasattr(clf, "n_neighbors"):
        params["n_neighbors"] = clf.n_neighbors
    if hasattr(clf, "kernel"):
        params["kernel"] = clf.kernel
    if hasattr(clf, "C"):
        params["C"] = clf.C
    return params


def train_fusion_dnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    model = FusionMLP(X_train_s.shape[1]).to(device)
    counts = np.bincount(y_train.astype(int), minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = (counts.sum() / (2 * counts)).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_s),
            torch.tensor(y_train.astype(np.int64)),
        ),
        batch_size=min(16, len(X_train_s)),
        shuffle=True,
    )
    X_test_t = torch.tensor(X_test_s, device=device)
    y_test_t = torch.tensor(y_test.astype(np.int64), device=device)

    best_state = None
    best_loss = float("inf")
    bad_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(200):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(X_test_t), y_test_t).item())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
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


def run_late_fusion_classifiers(
    z_img_train: np.ndarray,
    z_txt_train: np.ndarray,
    y_train: np.ndarray,
    z_img_test: np.ndarray,
    z_txt_test: np.ndarray,
    y_test: np.ndarray,
    test_ids: list[str],
    out_dir: Path,
    device: str | None = None,
    knn_k: int = 5,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.concatenate([z_img_train, z_txt_train], axis=1).astype(np.float32)
    X_test = np.concatenate([z_img_test, z_txt_test], axis=1).astype(np.float32)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    rows = []
    prediction_rows = pd.DataFrame(
        {
            "ID": test_ids,
            "stage_label": y_test,
            "class_name": [CLASS_NAMES[i] for i in y_test],
        }
    )

    for name, model in build_fusion_classifiers(y_train, knn_k=knn_k).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        rows.append(metric_row_staging_pd(name, y_test, pred, proba, **_fusion_classifier_params(model)))
        prediction_rows[f"pred_{name}"] = pred
        if proba is not None:
            for i, cname in enumerate(CLASS_NAMES):
                prediction_rows[f"proba_{name}_{cname}"] = proba[:, i]

    dnn_pred, dnn_proba, history = train_fusion_dnn(
        X_train, y_train, X_test, y_test, torch_device
    )
    rows.append(
        metric_row_staging_pd(
            "Fusion_DNN_simple",
            y_test,
            dnn_pred,
            dnn_proba,
            classifier="FusionMLP",
        )
    )
    prediction_rows["pred_Fusion_DNN_simple"] = dnn_pred
    for i, cname in enumerate(CLASS_NAMES):
        prediction_rows[f"proba_Fusion_DNN_simple_{cname}"] = dnn_proba[:, i]
    history.to_csv(out_dir / "dnn_training_history.csv", index=False)

    metrics = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    metrics.to_csv(out_dir / "classification_metrics.csv", index=False)
    prediction_rows.to_csv(out_dir / "test_predictions.csv", index=False)

    best_name = metrics.iloc[0]["model"]
    best_pred = prediction_rows[f"pred_{best_name}"].to_numpy()
    cm = confusion_matrix(y_test, best_pred, labels=[0, 1])
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        out_dir / f"confusion_matrix_{best_name}.csv"
    )
    report = classification_report(
        y_test, best_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    with open(out_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[clip_fusion] wrote metrics → {out_dir / 'classification_metrics.csv'}")
    return metrics
