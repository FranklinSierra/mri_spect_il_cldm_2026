import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    import umap
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    umap = None

import config
from dataset import OneDataset, resolve_nifti_path
from code.volume_shapes import get_latent_shape, get_volume_shape, get_working_volume_shape
from losses import spect_reconstruction_loss
from main import batch_metric_scores
from model import AAE, Discriminator
from utils import load_checkpoint, seed_torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a phase-1 MRI autoencoder, save MRI embeddings, and evaluate embedding quality."
    )
    parser.add_argument("--exp", default="exp_2")
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--lr", type=float, default=config.learning_rate)
    parser.add_argument("--validation-interval", type=int, default=config.validation_interval)
    parser.add_argument("--patience", type=int, default=config.early_stopping_patience)
    parser.add_argument("--max-subjects-per-split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    return parser.parse_args()


def output_root(exp):
    return Path("result") / exp.strip("/") / "phase1_mri"


def paths_for(exp):
    base = output_root(exp)
    return {
        "base": base,
        "checkpoint_dir": base / "checkpoints",
        "checkpoint": base / "checkpoints" / "MRI_AAE.pth.tar",
        "reconstruction": base / "reconstruction",
        "embeddings": base / "embeddings",
        "analysis": base / "embedding_analysis",
        "visualizations": base / "embedding_analysis" / "visualizations",
        "classification": base / "embedding_analysis" / "classification",
        "run_config": base / "run_config.json",
        "metadata": base / "embeddings_metadata.csv",
    }


def ensure_dirs(path_map):
    for key in ["checkpoint_dir", "reconstruction", "embeddings", "analysis", "visualizations", "classification"]:
        path_map[key].mkdir(parents=True, exist_ok=True)


def split_file(split):
    return {"train": config.train, "validation": config.validation, "test": config.test}[split]


def build_dataset(split, max_subjects=None):
    dataset = OneDataset(root_Abeta=config.whole_MRI, task=split_file(split), stage=split)
    if max_subjects is not None:
        dataset.images = dataset.images[:max_subjects]
        dataset.length_dataset = len(dataset.images)
        dataset.len = len(dataset.images)
    return dataset


def build_loader(split, batch_size, max_subjects=None, shuffle=False, drop_last=False):
    dataset = build_dataset(split, max_subjects=max_subjects)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.numworker,
        pin_memory=(config.device == "cuda"),
        drop_last=drop_last,
    )


def subject_id_from_name(name):
    if isinstance(name, (tuple, list)):
        name = name[0]
    name = Path(str(name)).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def reconstruction_training_loss(target, prediction):
    if config.USE_WEIGHTED_RECON_LOSS or config.USE_TAIL_RECON_LOSS:
        loss, _ = spect_reconstruction_loss(
            target,
            prediction,
            use_weighted=config.USE_WEIGHTED_RECON_LOSS,
            use_tail=config.USE_TAIL_RECON_LOSS,
            weighted_alpha=config.WEIGHTED_RECON_ALPHA,
            z_threshold=config.WEIGHTED_Z_THRESHOLD,
            tail_fraction=config.TAIL_RECON_FRACTION,
            tail_beta=config.TAIL_RECON_BETA,
        )
        return loss
    return torch.abs(target - prediction).mean()


def append_csv(path, row, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if not file_exists or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def should_validate(epoch, args):
    return (epoch + 1) % args.validation_interval == 0 or epoch == args.epochs - 1


def _volume_channel_shape() -> tuple[int, int, int, int]:
    """(C, D, H, W) tensor shape after dataset crop/pad."""
    working = get_working_volume_shape()
    return (1, *working)


def _shape_label(shape: tuple[int, ...]) -> str:
    return ",".join(str(int(x)) for x in shape)


def save_checkpoint(model, optimizer, path, epoch, score):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "selection_score": float(score),
            "modality": "MRI",
            "stage": "phase1_mri_autoencoder",
        },
        path,
    )


def evaluate_reconstruction(model, split, args, out_csv):
    loader = build_loader(split, batch_size=1, max_subjects=args.max_subjects_per_split, shuffle=False)
    model.eval()
    psnr_total = 0.0
    ssim_total = 0.0
    rows = []
    with torch.no_grad():
        for volume, names in tqdm(loader, desc=f"reconstruction:{split}"):
            volume = torch.as_tensor(np.expand_dims(volume, axis=1), dtype=torch.float32, device=config.device)
            reconstructed = model(volume)
            expected = _volume_channel_shape()
            if tuple(reconstructed.shape[1:]) != expected:
                raise ValueError(
                    f"Unexpected MRI reconstruction shape: {tuple(reconstructed.shape)} "
                    f"(expected {expected})"
                )
            batch_psnr, batch_ssim = batch_metric_scores(volume, reconstructed)
            subject_id = subject_id_from_name(names)
            rows.append({"ID": subject_id, "split": split, "PSNR": batch_psnr, "SSIM": batch_ssim})
            psnr_total += batch_psnr
            ssim_total += batch_ssim

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    length = max(len(rows), 1)
    return psnr_total / length, ssim_total / length


def train_mri_autoencoder(args, path_map):
    model = AAE().to(config.device)
    discriminator = Discriminator().to(config.device)
    opt_model = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    best_score = -float("inf")
    validations_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        discriminator.train()
        train_loader = build_loader(
            "train",
            batch_size=args.batch_size,
            max_subjects=args.max_subjects_per_split,
            shuffle=True,
            drop_last=True,
        )
        recon_loss_epoch = 0.0
        disc_loss_epoch = 0.0
        sample_count = 0

        for volume, _names in tqdm(train_loader, desc=f"train:mri_aae:epoch_{epoch + 1}"):
            volume = torch.as_tensor(np.expand_dims(volume, axis=1), dtype=torch.float32, device=config.device)
            reconstructed = model(volume)

            disc_real = discriminator(volume)
            disc_fake = discriminator(reconstructed)
            recon_loss = reconstruction_training_loss(volume, reconstructed)
            generator_loss = -torch.mean(disc_fake)
            loss = recon_loss * config.Lambda + generator_loss

            disc_loss_real = torch.mean(F.relu(1.0 - disc_real))
            disc_loss_fake = torch.mean(F.relu(1.0 + disc_fake))
            disc_loss = (disc_loss_real + disc_loss_fake) / 2

            opt_model.zero_grad()
            loss.backward(retain_graph=True)
            opt_disc.zero_grad()
            disc_loss.backward()
            opt_model.step()
            opt_disc.step()

            batch_size = volume.shape[0]
            sample_count += batch_size
            recon_loss_epoch += recon_loss.item() * batch_size
            disc_loss_epoch += disc_loss.item() * batch_size

        append_csv(
            path_map["reconstruction"] / "train.csv",
            {
                "Epoch": epoch + 1,
                "recon_loss": recon_loss_epoch / max(sample_count, 1),
                "disc_loss_epoch": disc_loss_epoch / max(sample_count, 1),
            },
            ["Epoch", "recon_loss", "disc_loss_epoch"],
        )

        if not should_validate(epoch, args):
            continue

        val_psnr, val_ssim = evaluate_reconstruction(
            model,
            "validation",
            args,
            path_map["reconstruction"] / "validation.csv",
        )
        score = val_psnr + val_ssim * 10
        append_csv(
            path_map["reconstruction"] / "validation_summary.csv",
            {"Epoch": epoch + 1, "PSNR": val_psnr, "SSIM": val_ssim, "selection_score": score},
            ["Epoch", "PSNR", "SSIM", "selection_score"],
        )

        if score > best_score:
            best_score = score
            validations_without_improvement = 0
            save_checkpoint(model, opt_model, path_map["checkpoint"], epoch + 1, best_score)
            test_psnr, test_ssim = evaluate_reconstruction(
                model,
                "test",
                args,
                path_map["reconstruction"] / "test.csv",
            )
            append_csv(
                path_map["reconstruction"] / "test_summary.csv",
                {"Epoch": epoch + 1, "PSNR": test_psnr, "SSIM": test_ssim, "selection_score": best_score},
                ["Epoch", "PSNR", "SSIM", "selection_score"],
            )
        else:
            validations_without_improvement += 1
            print(f"MRI AAE early stopping counter: {validations_without_improvement}/{args.patience}")
            if validations_without_improvement >= args.patience:
                print(f"Stopping MRI AAE early at epoch {epoch + 1}")
                break


def load_trained_aae(checkpoint_path, lr):
    model = AAE().to(config.device).eval()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    load_checkpoint(str(checkpoint_path), model, optimizer, lr)
    return model


def labels_map():
    data_info = pd.read_csv("data_info/data_info.csv", dtype={"ID": str})
    if data_info["ID"].duplicated().any():
        raise ValueError("Duplicate IDs found in data_info/data_info.csv")
    return data_info.set_index("ID")["label"].astype(int).to_dict()


def encode_embeddings(args, path_map):
    if not path_map["checkpoint"].exists():
        raise FileNotFoundError(f"Missing MRI AAE checkpoint: {path_map['checkpoint']}")
    model = load_trained_aae(path_map["checkpoint"], args.lr)
    label_lookup = labels_map()
    rows = []

    for split in ["train", "validation", "test"]:
        split_dir = path_map["embeddings"] / split
        split_dir.mkdir(parents=True, exist_ok=True)
        loader = build_loader(split, batch_size=1, max_subjects=args.max_subjects_per_split, shuffle=False)
        with torch.no_grad():
            for volume, names in tqdm(loader, desc=f"encode:mri_embeddings:{split}"):
                volume = torch.as_tensor(np.expand_dims(volume, axis=1), dtype=torch.float32, device=config.device)
                latent = model.encoder(volume)
                latent_shape = get_latent_shape()
                if tuple(latent.shape[1:]) != (1, *latent_shape):
                    raise ValueError(f"Unexpected MRI embedding shape: {tuple(latent.shape)}")
                subject_id = subject_id_from_name(names)
                latent_np = latent[0, 0].detach().cpu().numpy().astype(np.float32)
                reference = nib.load(resolve_nifti_path(config.whole_MRI, subject_id))
                out_path = split_dir / f"{subject_id}.nii.gz"
                nib.save(nib.Nifti1Image(latent_np, reference.affine), out_path)
                rows.append(
                    {
                        "ID": subject_id,
                        "split": split,
                        "label": int(label_lookup[subject_id]),
                        "embedding_path": str(out_path),
                        "embedding_shape": _shape_label((1, *get_latent_shape())),
                    }
                )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(path_map["metadata"], index=False)
    return metadata


def load_embedding_matrix(rows):
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


def build_classical_models(y_train):
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
        "SVM_RBF": pipe(SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=42)),
        "SVM_linear": pipe(SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced", random_state=42)),
        "LogisticRegression": pipe(LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)),
    }


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
        "roc_auc": roc_auc_score(y_true, y_score) if y_score is not None and len(np.unique(y_true)) == 2 else np.nan,
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
    pos_weight = torch.tensor([class_counts[0] / max(class_counts[1], 1)], dtype=torch.float32, device=config.device)
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
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "test_loss": test_loss})
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


def save_projection_2d(df, method, out_dir):
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
    plt.title(f"{method.upper()} 2D MRI-AE embeddings - internal test")
    plt.xlabel(f"{method}_1")
    plt.ylabel(f"{method}_2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{method}_2d.png", dpi=200)
    plt.close()


def save_visualizations(test_rows, path_map):
    if len(test_rows) < 2:
        return
    X, y, ids = load_embedding_matrix(test_rows)
    classes = [label_name(label) for label in y]
    for method in ["pca", "tsne", "umap"]:
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
                save_projection_2d(df, method, path_map["visualizations"])
            else:
                fig = px.scatter_3d(
                    df,
                    x=f"{method}_1",
                    y=f"{method}_2",
                    z=f"{method}_3",
                    color="class",
                    color_discrete_map={"control": "#3b6fd8", "parkinson": "#d83b5f"},
                    hover_data=["ID", "label", "class"],
                    title=f"{method.upper()} 3D MRI-AE embeddings - internal test",
                )
                fig.write_html(path_map["visualizations"] / f"{method}_3d.html")


def plot_confusion_and_roc(y_true, y_pred, y_score, path_map):
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


def analyze_embeddings(path_map):
    metadata = pd.read_csv(path_map["metadata"], dtype={"ID": str})
    train_rows = metadata[metadata["split"] == "train"].copy()
    if train_rows.empty:
        raise ValueError("No train MRI embeddings found for internal 80/20 evaluation.")
    class_counts = train_rows["label"].astype(int).value_counts()
    if len(train_rows) < 10 or len(class_counts) < 2 or class_counts.min() < 2:
        skipped = {
            "status": "skipped",
            "reason": "Not enough train embeddings per class for a stratified internal 80/20 evaluation.",
            "n_train_embeddings": int(len(train_rows)),
            "class_counts": {str(key): int(value) for key, value in class_counts.to_dict().items()},
            "evaluation_protocol": "train_original_internal_80_20",
        }
        train_rows.assign(embedding_eval_split="insufficient_data")[
            ["ID", "label", "embedding_eval_split", "embedding_path"]
        ].to_csv(path_map["analysis"] / "embedding_eval_split.csv", index=False)
        pd.DataFrame(columns=["model", "accuracy", "balanced_accuracy", "f1", "roc_auc"]).to_csv(
            path_map["classification"] / "classification_metrics.csv",
            index=False,
        )
        pd.DataFrame(columns=["ID", "label"]).to_csv(
            path_map["classification"] / "test_predictions.csv",
            index=False,
        )
        with open(path_map["classification"] / "quality_summary.json", "w", encoding="utf-8") as file:
            json.dump(skipped, file, indent=2)
        return skipped
    eval_train, eval_test = train_test_split(
        train_rows,
        test_size=0.20,
        stratify=train_rows["label"].astype(int),
        random_state=42,
    )
    eval_split = pd.concat(
        [
            eval_train.assign(embedding_eval_split="train"),
            eval_test.assign(embedding_eval_split="test"),
        ],
        ignore_index=True,
    )[["ID", "label", "embedding_eval_split", "embedding_path"]]
    eval_split.to_csv(path_map["analysis"] / "embedding_eval_split.csv", index=False)

    X_train, y_train, _train_ids = load_embedding_matrix(eval_train)
    X_test, y_test, test_ids = load_embedding_matrix(eval_test)

    rows = []
    prediction_rows = pd.DataFrame({"ID": test_ids, "label": y_test})
    for model_name, model in build_classical_models(y_train).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = positive_scores(model, X_test)
        rows.append(metric_row(model_name, y_test, pred, score, **classifier_params(model)))
        prediction_rows[f"pred_{model_name}"] = pred
        if score is not None:
            prediction_rows[f"score_{model_name}"] = score

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
    best_score = prediction_rows[best_score_column].to_numpy() if best_score_column in prediction_rows.columns else None
    plot_confusion_and_roc(y_test, best_pred, best_score, path_map)
    save_visualizations(eval_test, path_map)

    report = classification_report(y_test, best_pred, digits=4)
    with open(path_map["classification"] / "classifier_report.txt", "w", encoding="utf-8") as file:
        file.write(f"Best model: {best_model}\n\n")
        file.write(report)
    decision = {
        "best_model": best_model,
        "best_balanced_accuracy": float(metrics.iloc[0]["balanced_accuracy"]),
        "best_roc_auc": None if pd.isna(metrics.iloc[0]["roc_auc"]) else float(metrics.iloc[0]["roc_auc"]),
        "evaluation_protocol": "train_original_internal_80_20",
    }
    with open(path_map["classification"] / "quality_summary.json", "w", encoding="utf-8") as file:
        json.dump(decision, file, indent=2)
    return decision


def write_run_config(args, path_map):
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": args.exp,
        "stage": "phase1_mri_autoencoder",
        "architecture": "AAE",
        "input_modality": "MRI",
        "checkpoint": str(path_map["checkpoint"]),
        "output_root": str(path_map["base"]),
        "device": config.device,
        "paths": {
            "whole_MRI": config.whole_MRI,
            "train": config.train,
            "validation": config.validation,
            "test": config.test,
        },
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "validation_interval": args.validation_interval,
        "early_stopping_patience": args.patience,
        "max_subjects_per_split": args.max_subjects_per_split,
        "embedding_shape": _shape_label((1, *get_latent_shape())),
        "reconstruction_shape": _shape_label(_volume_channel_shape()),
        "volume_shape_on_disk": _shape_label(get_volume_shape()),
        "working_volume_shape": _shape_label(get_working_volume_shape()),
        "embedding_eval_protocol": "train_original_internal_80_20",
        "future_diffusion_usage": {
            "FREEZE_MRI_ENCODER": "1",
            "MRI_ENCODER_CHECKPOINT": str(path_map["checkpoint"]),
        },
    }
    with open(path_map["run_config"], "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main():
    args = parse_args()
    seed_torch(args.seed)
    path_map = paths_for(args.exp)
    ensure_dirs(path_map)
    write_run_config(args, path_map)
    print(f"Device: {config.device}")
    print(f"MRI phase-1 output: {path_map['base']}")

    if not args.skip_train:
        train_mri_autoencoder(args, path_map)
    if not args.skip_embeddings:
        encode_embeddings(args, path_map)
    if not args.skip_analysis:
        decision = analyze_embeddings(path_map)
        print(json.dumps(decision, indent=2))
    print(f"Done. Results saved in {path_map['base']}")


if __name__ == "__main__":
    main()
