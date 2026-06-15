import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config


def _read_ids(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def _load_embeddings(split_ids, latent_root, labels_map, split_name):
    rows = []
    features = []
    labels = []
    for subject_id in split_ids:
        latent_path = Path(latent_root) / f"{subject_id}.nii.gz"
        if not latent_path.exists():
            continue
        embedding = nib.load(str(latent_path)).get_fdata().astype(np.float32).reshape(-1)
        features.append(embedding)
        label = int(labels_map[subject_id])
        labels.append(label)
        rows.append({"ID": subject_id, "split": split_name, "label": label, "latent_path": str(latent_path)})
    return rows, np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _plot_embedding_space(features, labels, splits, out_path, method):
    if len(features) < 2:
        return
    if method == "tsne":
        perplexity = min(30, max(2, len(features) - 1))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        proj = reducer.fit_transform(features)
    else:
        reducer = PCA(n_components=2, random_state=42)
        proj = reducer.fit_transform(features)
    plt.figure(figsize=(8, 6))
    for split in sorted(set(splits)):
        idx = np.where(splits == split)[0]
        plt.scatter(
            proj[idx, 0],
            proj[idx, 1],
            c=labels[idx],
            cmap="coolwarm",
            alpha=0.7,
            label=f"{split} (0/1)",
        )
    plt.title(f"{method.upper()} of MRI embeddings")
    plt.xlabel("component_1")
    plt.ylabel("component_2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _evaluate_classifier(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        decision = model.decision_function(X_test)
        y_score = (decision - decision.min()) / (decision.max() - decision.min() + 1e-8)

    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_score)) if len(np.unique(y_test)) == 2 else None,
    }
    return y_pred, y_score, metrics


def run_embedding_quality_gate():
    exp_name = str(config.exp).strip("/")
    base_result_dir = Path("result") / exp_name
    eval_dir = base_result_dir / "embedding_eval"
    fig_dir = base_result_dir / "figures"
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data_info = pd.read_csv("data_info/data_info.csv", dtype={"ID": str})
    labels_map = data_info.set_index("ID")["label"].to_dict()

    all_rows = []
    all_features = []
    all_labels = []
    all_splits = []
    split_payload = {}
    split_config = {"train": config.train, "validation": config.validation, "test": config.test}
    for split_name, split_file in split_config.items():
        split_ids = _read_ids(split_file)
        rows, X, y = _load_embeddings(split_ids, config.latent_MRI, labels_map, split_name)
        split_payload[split_name] = (X, y)
        all_rows.extend(rows)
        if len(X):
            all_features.append(X)
            all_labels.append(y)
            all_splits.extend([split_name] * len(y))

    if len(split_payload["train"][0]) == 0:
        raise ValueError("No MRI embeddings found for train split in data/latent_MRI.")
    if len(split_payload["test"][0]) == 0:
        raise ValueError("No MRI embeddings found for test split in data/latent_MRI.")

    metadata = pd.DataFrame(all_rows)
    metadata.to_csv(eval_dir / "embedding_metadata.csv", index=False)

    train_X, train_y = split_payload["train"]
    test_X, test_y = split_payload["test"]

    models = {
        "LogisticRegression": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42))]
        ),
        "SVM_RBF": Pipeline(
            [("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))]
        ),
    }

    summary = []
    prediction_df = pd.DataFrame({"ID": metadata[metadata["split"] == "test"]["ID"].tolist(), "label": test_y})
    best_name = None
    best_bal_acc = -1.0
    best_pred = None
    best_score = None
    for model_name, model in models.items():
        pred, score, metrics = _evaluate_classifier(model, train_X, train_y, test_X, test_y)
        summary.append({"model": model_name, **metrics})
        prediction_df[f"pred_{model_name}"] = pred
        prediction_df[f"score_{model_name}"] = score
        if metrics["balanced_accuracy"] > best_bal_acc:
            best_bal_acc = metrics["balanced_accuracy"]
            best_name = model_name
            best_pred = pred
            best_score = score

    summary_df = pd.DataFrame(summary).sort_values("balanced_accuracy", ascending=False)
    summary_df.to_json(eval_dir / "metrics_summary.json", orient="records", indent=2)
    prediction_df.to_csv(eval_dir / "per_subject_predictions.csv", index=False)

    report = classification_report(test_y, best_pred, digits=4)
    with open(eval_dir / "classifier_report.txt", "w", encoding="utf-8") as file:
        file.write(f"Best model: {best_name}\n\n")
        file.write(report)

    cm = confusion_matrix(test_y, best_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix ({best_name})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    fpr, tpr, _ = roc_curve(test_y, best_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{best_name} (AUC={roc_auc_score(test_y, best_score):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curve.png", dpi=200)
    plt.close()

    all_X = np.concatenate(all_features, axis=0) if all_features else np.empty((0, 2))
    all_y = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,))
    all_split_np = np.asarray(all_splits)
    _plot_embedding_space(all_X, all_y, all_split_np, fig_dir / "tsne_train_val_test.png", method="tsne")
    _plot_embedding_space(all_X, all_y, all_split_np, fig_dir / "pca_train_val_test.png", method="pca")

    quality_gate = {
        "best_model": best_name,
        "best_balanced_accuracy": float(best_bal_acc),
        "passes_threshold": bool(best_bal_acc >= 0.60),
        "threshold": 0.60,
    }
    with open(eval_dir / "quality_gate.json", "w", encoding="utf-8") as file:
        json.dump(quality_gate, file, indent=2)
    return quality_gate


if __name__ == "__main__":
    outcome = run_embedding_quality_gate()
    print(json.dumps(outcome, indent=2))
