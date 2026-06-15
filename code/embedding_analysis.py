"""Shared embedding classification and visualization (MRI phase-1 / SPECT latents)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

try:
    import umap
except ImportError:
    umap = None

import config


def load_embedding_matrix(rows: pd.DataFrame):
    features = []
    labels = []
    ids = []
    for row in rows.itertuples(index=False):
        embedding = nib.load(row.embedding_path).get_fdata().astype(np.float32).reshape(-1)
        features.append(embedding)
        labels.append(int(row.label))
        ids.append(str(row.ID))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), ids


def classifier_params(model):
    classifier = model.named_steps["classifier"]
    params = {"classifier": classifier.__class__.__name__}
    if hasattr(classifier, "n_neighbors"):
        params["n_neighbors"] = classifier.n_neighbors
    if hasattr(classifier, "kernel"):
        params["kernel"] = classifier.kernel
    if hasattr(classifier, "C"):
        params["C"] = classifier.C
    return params


def build_classical_models(y_train, exclude: set[str] | None = None):
    exclude = exclude or set()
    knn_auto = min(5, len(y_train))
    if knn_auto % 2 == 0:
        knn_auto -= 1
    knn_auto = max(1, knn_auto)
    knn_k2 = min(2, len(y_train))

    def pipe(classifier):
        return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])

    models = {
        "KNN_auto": pipe(KNeighborsClassifier(n_neighbors=knn_auto, weights="distance")),
        "KNN_K2": pipe(KNeighborsClassifier(n_neighbors=knn_k2, weights="distance")),
        "SVM_RBF": pipe(
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=42,
            )
        ),
        "SVM_linear": pipe(
            SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced", random_state=42)
        ),
        "LogisticRegression": pipe(
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
        ),
    }
    return {name: model for name, model in models.items() if name not in exclude}


def positive_scores(model, features):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return None


def metric_row(model_name, y_true, y_pred, y_score=None, **params):
    row = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score)
        if y_score is not None and len(np.unique(y_true)) == 2
        else np.nan,
    }
    row.update(params)
    return row


class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_dnn(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    model = EmbeddingMLP(X_train_scaled.shape[1]).to(config.device)
    class_counts = np.bincount(y_train.astype(int), minlength=2)
    pos_weight = torch.tensor(
        [class_counts[0] / max(class_counts[1], 1)], dtype=torch.float32, device=config.device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train.astype(np.float32), dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=config.device)
    y_test_tensor = torch.tensor(y_test.astype(np.float32), dtype=torch.float32, device=config.device)
    best_state = None
    best_loss = float("inf")
    bad_epochs = 0
    patience = 30
    history = []

    for epoch in range(300):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(config.device)
            yb = yb.to(config.device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test_tensor), y_test_tensor).item()
        history.append(
            {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "test_loss": test_loss}
        )
        if test_loss < best_loss:
            best_loss = test_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(X_test_tensor)).detach().cpu().numpy()
    return (scores >= 0.5).astype(int), scores, pd.DataFrame(history)


def label_name(label):
    return "control" if int(label) == 0 else "parkinson"


def projection_coordinates(X, method, n_components):
    X_scaled = StandardScaler().fit_transform(X)
    # Manifold methods on 40x48x40 latents: pre-reduce so viz matches MRI runtime.
    if method in {"tsne", "umap"} and X_scaled.shape[1] > 100:
        pre_n = min(50, X_scaled.shape[0] - 1, X_scaled.shape[1])
        X_scaled = PCA(n_components=pre_n, random_state=42).fit_transform(X_scaled)
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == "tsne":
        reducer = TSNE(
            n_components=n_components,
            random_state=42,
            perplexity=min(30, max(1, len(X_scaled) - 1)),
            init="pca",
            learning_rate="auto",
        )
    elif method == "umap":
        if umap is None:
            raise ImportError("umap-learn is required for UMAP visualizations.")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(15, max(2, len(X_scaled) - 1)),
            min_dist=0.1,
            metric="euclidean",
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown projection method: {method}")
    return reducer.fit_transform(X_scaled)


def save_projection_2d(df, method, out_dir, title_suffix: str):
    colors = {"control": "#3b6fd8", "parkinson": "#d83b5f"}
    plt.figure(figsize=(8, 6))
    for class_name in ["control", "parkinson"]:
        class_df = df[df["class"] == class_name]
        if class_df.empty:
            continue
        plt.scatter(
            class_df[f"{method}_1"],
            class_df[f"{method}_2"],
            color=colors[class_name],
            alpha=0.75,
            label=class_name,
        )
    plt.title(f"{method.upper()} 2D embeddings - {title_suffix}")
    plt.xlabel(f"{method}_1")
    plt.ylabel(f"{method}_2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{method}_2d.png", dpi=200)
    plt.close()


def save_visualizations(test_rows: pd.DataFrame, path_map: dict, title_suffix: str):
    if len(test_rows) < 2:
        return
    X, y, ids = load_embedding_matrix(test_rows)
    classes = [label_name(label) for label in y]
    for method in ["pca", "tsne"]:
        for n_components in [2, 3]:
            coords = projection_coordinates(X, method, n_components)
            columns = [f"{method}_{idx + 1}" for idx in range(n_components)]
            df = pd.DataFrame(coords, columns=columns)
            df["ID"] = ids
            df["label"] = y
            df["class"] = classes
            df["embedding_eval_split"] = "test"
            df.to_csv(path_map["visualizations"] / f"{method}_{n_components}d.csv", index=False)
            if n_components == 2:
                save_projection_2d(df, method, path_map["visualizations"], title_suffix)
            else:
                fig = px.scatter_3d(
                    df,
                    x=f"{method}_1",
                    y=f"{method}_2",
                    z=f"{method}_3",
                    color="class",
                    color_discrete_map={"control": "#3b6fd8", "parkinson": "#d83b5f"},
                    hover_data=["ID", "label", "class"],
                    title=f"{method.upper()} 3D embeddings - {title_suffix}",
                )
                fig.write_html(path_map["visualizations"] / f"{method}_3d.html")


def plot_confusion_and_roc(y_true, y_pred, y_score, path_map: dict):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (best model)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path_map["visualizations"] / "confusion_matrix.png", dpi=200)
    plt.close()

    if y_score is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path_map["visualizations"] / "roc_curve.png", dpi=200)
        plt.close()


def analyze_embeddings_with_eval_split(
    eval_split: pd.DataFrame,
    path_map: dict,
    *,
    modality: str,
    evaluation_protocol: str,
    viz_title_suffix: str = "internal test",
    skip_dnn: bool = False,
    exclude_classifiers: set[str] | None = None,
) -> dict:
    """Classify embeddings using a fixed train/test ID split (comparable across modalities)."""
    required = {"ID", "label", "embedding_eval_split", "embedding_path"}
    missing = required.difference(eval_split.columns)
    if missing:
        raise ValueError(f"eval_split missing columns: {sorted(missing)}")

    eval_split = eval_split.copy()
    eval_split["ID"] = eval_split["ID"].astype(str)
    eval_split.to_csv(path_map["analysis"] / "embedding_eval_split.csv", index=False)

    eval_train = eval_split[eval_split["embedding_eval_split"] == "train"]
    eval_test = eval_split[eval_split["embedding_eval_split"] == "test"]
    if eval_train.empty or eval_test.empty:
        raise ValueError("eval_split must contain non-empty train and test subsets.")

    X_train, y_train, _train_ids = load_embedding_matrix(eval_train)
    X_test, y_test, test_ids = load_embedding_matrix(eval_test)

    rows = []
    prediction_rows = pd.DataFrame({"ID": test_ids, "label": y_test})
    for model_name, model in build_classical_models(y_train, exclude=exclude_classifiers).items():
        print(f"[{modality}] training {model_name}...", flush=True)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = positive_scores(model, X_test)
        rows.append(metric_row(model_name, y_test, pred, score, **classifier_params(model)))
        prediction_rows[f"pred_{model_name}"] = pred
        if score is not None:
            prediction_rows[f"score_{model_name}"] = score

    if skip_dnn:
        print(f"[{modality}] skipping DNN_simple", flush=True)
    else:
        print(f"[{modality}] training DNN_simple...", flush=True)
        dnn_pred, dnn_score, history = train_dnn(X_train, y_train, X_test, y_test)
        rows.append(metric_row("DNN_simple", y_test, dnn_pred, dnn_score, classifier="EmbeddingMLP"))
        prediction_rows["pred_DNN_simple"] = dnn_pred
        prediction_rows["score_DNN_simple"] = dnn_score
        history.to_csv(path_map["classification"] / "dnn_training_history.csv", index=False)

    metrics = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    metrics.to_csv(path_map["classification"] / "classification_metrics.csv", index=False)
    metrics.to_json(path_map["classification"] / "metrics_summary.json", orient="records", indent=2)
    prediction_rows.to_csv(path_map["classification"] / "test_predictions.csv", index=False)

    best_model = metrics.iloc[0]["model"]
    best_pred = prediction_rows[f"pred_{best_model}"].to_numpy()
    best_score_column = f"score_{best_model}"
    best_score = (
        prediction_rows[best_score_column].to_numpy()
        if best_score_column in prediction_rows.columns
        else None
    )
    plot_confusion_and_roc(y_test, best_pred, best_score, path_map)
    print(f"[{modality}] saving projections...", flush=True)
    save_visualizations(eval_test, path_map, viz_title_suffix)

    report = classification_report(y_test, best_pred, digits=4)
    with open(path_map["classification"] / "classifier_report.txt", "w", encoding="utf-8") as file:
        file.write(f"Best model: {best_model}\n")
        file.write(f"Modality: {modality}\n")
        file.write(f"Protocol: {evaluation_protocol}\n\n")
        file.write(report)

    decision = {
        "modality": modality,
        "best_model": best_model,
        "best_balanced_accuracy": float(metrics.iloc[0]["balanced_accuracy"]),
        "best_roc_auc": None if pd.isna(metrics.iloc[0]["roc_auc"]) else float(metrics.iloc[0]["roc_auc"]),
        "evaluation_protocol": evaluation_protocol,
        "n_eval_train": int(len(eval_train)),
        "n_eval_test": int(len(eval_test)),
        "excluded_classifiers": sorted(exclude_classifiers or []),
        "skip_dnn": skip_dnn,
    }
    with open(path_map["classification"] / "quality_summary.json", "w", encoding="utf-8") as file:
        json.dump(decision, file, indent=2)
    return decision


def ensure_analysis_dirs(base: Path) -> dict:
    path_map = {
        "base": base,
        "analysis": base,
        "visualizations": base / "visualizations",
        "classification": base / "classification",
    }
    for key in ("analysis", "visualizations", "classification"):
        path_map[key].mkdir(parents=True, exist_ok=True)
    return path_map
