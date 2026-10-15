"""Cross-modal CLIP-style training for PD staging (HY1 vs HY2) — v2."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Sampler

from code.clinical_clip_fusion import run_late_fusion_classifiers
from code.clinical_staging import (
    CLASS_NAMES,
    DEFAULT_CLINICAL_STAGING_ROOT,
    _repo_relative,
)
from code.nhy_staging_pd_classification import metric_row_staging_pd, resolve_path
from losses.contrastive import (
    ClassAwareCrossModalInfoNCE,
    ContrastiveProjector,
    SupervisedContrastiveLoss,
    l2_normalize,
)


@dataclass
class FeatureScalers:
    image: StandardScaler
    text: StandardScaler

    def transform_image(self, x: np.ndarray) -> np.ndarray:
        return self.image.transform(x).astype(np.float32)

    def transform_text(self, x: np.ndarray) -> np.ndarray:
        return self.text.transform(x).astype(np.float32)

    def save(self, path: Path) -> None:
        joblib.dump({"image": self.image, "text": self.text}, path)

    @classmethod
    def load(cls, path: Path) -> FeatureScalers:
        payload = joblib.load(path)
        return cls(image=payload["image"], text=payload["text"])

    @classmethod
    def fit(cls, img: np.ndarray, txt: np.ndarray) -> FeatureScalers:
        img_scaler = StandardScaler()
        txt_scaler = StandardScaler()
        img_scaler.fit(img)
        txt_scaler.fit(txt)
        return cls(image=img_scaler, text=txt_scaler)


class PairedStagingDataset(Dataset):
    def __init__(
        self,
        image_features: np.ndarray,
        text_features: np.ndarray,
        labels: np.ndarray,
        ids: list[str],
    ):
        self.image_features = torch.tensor(image_features, dtype=torch.float32)
        self.text_features = torch.tensor(text_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        return (
            self.image_features[idx],
            self.text_features[idx],
            self.labels[idx],
            self.ids[idx],
        )


class StratifiedBatchSampler(Sampler[list[int]]):
    """Yield batches with both HY1/HY2 when batch_size >= 2."""

    def __init__(self, labels: np.ndarray, batch_size: int, shuffle: bool = True):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.labels = labels.astype(int)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_indices = {
            cls: np.where(self.labels == cls)[0].tolist() for cls in np.unique(self.labels)
        }

    def __iter__(self) -> Iterator[list[int]]:
        pools = {k: v.copy() for k, v in self.class_indices.items()}
        if self.shuffle:
            for indices in pools.values():
                random.shuffle(indices)
        leftover: list[int] = []
        while any(pools.values()) or leftover:
            batch: list[int] = leftover[:]
            leftover = []
            if self.batch_size >= 2 and len(self.class_indices) >= 2:
                for cls in sorted(self.class_indices):
                    if pools[cls] and len(batch) < self.batch_size:
                        batch.append(pools[cls].pop())
            while len(batch) < self.batch_size:
                available = [c for c, idxs in pools.items() if idxs]
                if not available:
                    break
                cls = random.choice(available)
                batch.append(pools[cls].pop())
            if not batch:
                break
            yield batch

    def __len__(self) -> int:
        return max(1, int(np.ceil(len(self.labels) / self.batch_size)))


class ClinicalRadiologyCLIP(nn.Module):
    def __init__(self, image_dim: int, text_dim: int, proj_dim: int = 128):
        super().__init__()
        self.image_projector = ContrastiveProjector(image_dim, proj_dim)
        self.text_projector = ContrastiveProjector(text_dim, proj_dim)

    def forward(self, image_feat: torch.Tensor, text_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_img = l2_normalize(self.image_projector(image_feat))
        z_txt = l2_normalize(self.text_projector(text_feat))
        return z_img, z_txt


def load_paired_features(pairs: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows = pairs[pairs["split"] == split]
    img_feats, txt_feats, labels, ids = [], [], [], []
    for row in rows.itertuples(index=False):
        img_path = resolve_path(row.representation_path)
        txt_path = resolve_path(row.text_embedding_path)
        img_feats.append(np.load(img_path)["sampled_latent_final"].reshape(-1))
        txt_feats.append(np.load(txt_path).reshape(-1))
        labels.append(int(row.stage_label))
        ids.append(str(row.ID))
    return (
        np.asarray(img_feats, dtype=np.float32),
        np.asarray(txt_feats, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        ids,
    )


def diagonal_retrieval_r1(z_img: torch.Tensor, z_txt: torch.Tensor) -> float:
    """Alignment metric: fraction of rows where argmax_j sim(img_i, text_j) == i."""
    logits = torch.matmul(z_img, z_txt.T)
    pred = logits.argmax(dim=1)
    targets = torch.arange(z_img.shape[0], device=z_img.device)
    return float((pred == targets).float().mean().item())


def classify_by_text_prototypes(
    z_img: torch.Tensor,
    z_txt_train: torch.Tensor,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    proto_mat = build_class_prototypes(z_txt_train, y_train)
    logits = torch.matmul(z_img, proto_mat.T)
    probs = torch.softmax(logits, dim=1)
    return logits.argmax(dim=1).cpu().numpy().astype(int), probs.cpu().numpy()


def classify_by_image_prototypes(
    z_img: torch.Tensor,
    z_img_train: torch.Tensor,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    proto_mat = build_class_prototypes(z_img_train, y_train)
    logits = torch.matmul(z_img, proto_mat.T)
    probs = torch.softmax(logits, dim=1)
    return logits.argmax(dim=1).cpu().numpy().astype(int), probs.cpu().numpy()


def classify_by_avg_prototypes(
    z_img: torch.Tensor,
    z_img_train: torch.Tensor,
    z_txt_train: torch.Tensor,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    proto_txt = build_class_prototypes(z_txt_train, y_train)
    proto_img = build_class_prototypes(z_img_train, y_train)
    logits_txt = torch.matmul(z_img, proto_txt.T)
    logits_img = torch.matmul(z_img, proto_img.T)
    logits = 0.5 * (logits_txt + logits_img)
    probs = torch.softmax(logits, dim=1)
    return logits.argmax(dim=1).cpu().numpy().astype(int), probs.cpu().numpy()


def build_class_prototypes(z: torch.Tensor, y: np.ndarray) -> torch.Tensor:
    protos = []
    for cls in sorted(np.unique(y)):
        mask = torch.tensor(y == cls, device=z.device)
        protos.append(l2_normalize(z[mask].mean(dim=0, keepdim=True)))
    return torch.cat(protos, dim=0)


def knn_joint_predict(
    z_img_train: np.ndarray,
    z_txt_train: np.ndarray,
    y_train: np.ndarray,
    z_img_test: np.ndarray,
    z_txt_test: np.ndarray,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    X_train = np.concatenate([z_img_train, z_txt_train], axis=1)
    X_test = np.concatenate([z_img_test, z_txt_test], axis=1)
    n_neighbors = min(k, len(y_train))
    if n_neighbors % 2 == 0:
        n_neighbors -= 1
    n_neighbors = max(1, n_neighbors)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    return pred.astype(int), proba


def project_features(
    model: ClinicalRadiologyCLIP,
    img: np.ndarray,
    txt: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    z_imgs, z_txts = [], []
    with torch.no_grad():
        for start in range(0, len(img), batch_size):
            img_b = torch.tensor(img[start : start + batch_size], device=device)
            txt_b = torch.tensor(txt[start : start + batch_size], device=device)
            z_i, z_t = model(img_b, txt_b)
            z_imgs.append(z_i.cpu().numpy())
            z_txts.append(z_t.cpu().numpy())
    return np.concatenate(z_imgs, axis=0), np.concatenate(z_txts, axis=0)


def train_clip_model_v2(
    model: ClinicalRadiologyCLIP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lambda_infonce: float = 0.5,
    lambda_supcon: float = 1.0,
    temperature: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 20,
    min_epochs: int = 5,
) -> pd.DataFrame:
    infonce_crit = ClassAwareCrossModalInfoNCE(temperature=temperature)
    supcon_crit = SupervisedContrastiveLoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[dict[str, float]] = []
    best_state = None
    best_val_supcon = float("inf")
    bad_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        train_infonce, train_supcon, train_total = [], [], []
        train_r1 = []
        for img, txt, labels, _ids in train_loader:
            img, txt, labels = img.to(device), txt.to(device), labels.to(device)
            optimizer.zero_grad()
            z_img, z_txt = model(img, txt)
            loss_infonce = infonce_crit(z_img, z_txt, labels)
            loss_supcon = supcon_crit(z_img, labels)
            loss = lambda_infonce * loss_infonce + lambda_supcon * loss_supcon
            loss.backward()
            optimizer.step()
            train_infonce.append(float(loss_infonce.item()))
            train_supcon.append(float(loss_supcon.item()))
            train_total.append(float(loss.item()))
            train_r1.append(diagonal_retrieval_r1(z_img.detach(), z_txt.detach()))

        model.eval()
        val_infonce, val_supcon, val_total, val_r1 = [], [], [], []
        with torch.no_grad():
            for img, txt, labels, _ids in val_loader:
                img, txt, labels = img.to(device), txt.to(device), labels.to(device)
                z_img, z_txt = model(img, txt)
                li = infonce_crit(z_img, z_txt, labels)
                ls = supcon_crit(z_img, labels)
                val_infonce.append(float(li.item()))
                val_supcon.append(float(ls.item()))
                val_total.append(float((lambda_infonce * li + lambda_supcon * ls).item()))
                val_r1.append(diagonal_retrieval_r1(z_img, z_txt))

        epoch_val_supcon = float(np.mean(val_supcon)) if val_supcon else float("inf")
        history.append(
            {
                "epoch": epoch + 1,
                "train_infonce_loss": float(np.mean(train_infonce)) if train_infonce else np.nan,
                "train_supcon_loss": float(np.mean(train_supcon)) if train_supcon else np.nan,
                "train_total_loss": float(np.mean(train_total)) if train_total else np.nan,
                "train_retrieval_r1": float(np.mean(train_r1)) if train_r1 else np.nan,
                "val_infonce_loss": float(np.mean(val_infonce)) if val_infonce else np.nan,
                "val_supcon_loss": epoch_val_supcon,
                "val_total_loss": float(np.mean(val_total)) if val_total else np.nan,
                "val_retrieval_r1": float(np.mean(val_r1)) if val_r1 else np.nan,
            }
        )

        if epoch + 1 >= min_epochs and epoch_val_supcon < best_val_supcon:
            best_val_supcon = epoch_val_supcon
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        elif epoch + 1 >= min_epochs:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(history)


def evaluate_staging(
    z_img_test: np.ndarray,
    z_txt_test: np.ndarray,
    y_test: np.ndarray,
    z_img_train: np.ndarray,
    z_txt_train: np.ndarray,
    y_train: np.ndarray,
    test_ids: list[str],
    pairs: pd.DataFrame,
    proj_dim: int,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    z_img_te = torch.tensor(z_img_test, dtype=torch.float32)
    z_txt_te = torch.tensor(z_txt_test, dtype=torch.float32)
    z_img_tr = torch.tensor(z_img_train, dtype=torch.float32)
    z_txt_tr = torch.tensor(z_txt_train, dtype=torch.float32)

    staging_modes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pred, proba = classify_by_text_prototypes(z_img_te, z_txt_tr, y_train)
    staging_modes["Proto_text"] = (pred, proba)
    pred, proba = classify_by_image_prototypes(z_img_te, z_img_tr, y_train)
    staging_modes["Proto_image"] = (pred, proba)
    pred, proba = classify_by_avg_prototypes(z_img_te, z_img_tr, z_txt_tr, y_train)
    staging_modes["Proto_avg"] = (pred, proba)
    pred, proba = knn_joint_predict(z_img_train, z_txt_train, y_train, z_img_test, z_txt_test, k=5)
    staging_modes["KNN_joint"] = (pred, proba)

    rows = []
    prediction_rows = pd.DataFrame(
        {
            "ID": test_ids,
            "stage_label": y_test,
            "stratum": pairs.set_index("ID").loc[test_ids, "stratum"].tolist(),
            "class_name": [CLASS_NAMES[i] for i in y_test],
        }
    )
    for mode_name, (pred, proba) in staging_modes.items():
        rows.append(
            metric_row_staging_pd(
                mode_name,
                y_test,
                pred,
                proba,
                proj_dim=proj_dim,
                batch_size=batch_size,
            )
        )
        prediction_rows[f"pred_{mode_name}"] = pred
        prediction_rows[f"score_{mode_name}_hy2"] = proba[:, 1]

    metrics = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    return metrics, prediction_rows


def run_clinical_radiology_clip(
    out_root: Path | str = DEFAULT_CLINICAL_STAGING_ROOT,
    pairs_csv: Path | None = None,
    proj_dim: int = 128,
    batch_size: int = 16,
    val_fraction: float = 0.15,
    seed: int = 42,
    device: str | None = None,
    max_epochs: int = 200,
    patience: int = 20,
    min_epochs: int = 5,
    lambda_infonce: float = 0.5,
    lambda_supcon: float = 1.0,
    temperature: float = 0.1,
) -> pd.DataFrame:
    out_root = Path(out_root)
    clip_dir = out_root / "clip"
    ckpt_dir = clip_dir / "checkpoints"
    fusion_dir = clip_dir / "fusion"
    clip_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = pairs_csv or (out_root / "clinical_pairs.csv")
    pairs = pd.read_csv(pairs_path, dtype={"ID": str})

    img_train, txt_train, y_train, train_ids = load_paired_features(pairs, "train")
    img_test, txt_test, y_test, test_ids = load_paired_features(pairs, "test")

    idx = np.arange(len(y_train))
    tr_idx, val_idx = train_test_split(
        idx,
        test_size=val_fraction,
        random_state=seed,
        stratify=y_train,
    )

    scalers = FeatureScalers.fit(img_train[tr_idx], txt_train[tr_idx])
    scaler_path = ckpt_dir / "feature_scalers.joblib"
    scalers.save(scaler_path)

    img_train_s = scalers.transform_image(img_train)
    txt_train_s = scalers.transform_text(txt_train)
    img_test_s = scalers.transform_image(img_test)
    txt_test_s = scalers.transform_text(txt_test)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    model = ClinicalRadiologyCLIP(
        image_dim=img_train.shape[1],
        text_dim=txt_train.shape[1],
        proj_dim=proj_dim,
    ).to(torch_device)

    train_ds = PairedStagingDataset(
        img_train_s[tr_idx],
        txt_train_s[tr_idx],
        y_train[tr_idx],
        [train_ids[i] for i in tr_idx],
    )
    val_ds = PairedStagingDataset(
        img_train_s[val_idx],
        txt_train_s[val_idx],
        y_train[val_idx],
        [train_ids[i] for i in val_idx],
    )
    train_sampler = StratifiedBatchSampler(y_train[tr_idx], batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False)

    history = train_clip_model_v2(
        model,
        train_loader,
        val_loader,
        torch_device,
        lambda_infonce=lambda_infonce,
        lambda_supcon=lambda_supcon,
        temperature=temperature,
        max_epochs=max_epochs,
        patience=patience,
        min_epochs=min_epochs,
    )
    history.to_csv(clip_dir / "training_history.csv", index=False)
    torch.save(model.state_dict(), ckpt_dir / "projectors.pt")

    z_img_train, z_txt_train = project_features(model, img_train_s, txt_train_s, torch_device)
    z_img_test, z_txt_test = project_features(model, img_test_s, txt_test_s, torch_device)

    r1_test = diagonal_retrieval_r1(
        torch.tensor(z_img_test),
        torch.tensor(z_txt_test),
    )
    pd.DataFrame(
        [
            {
                "split": "test",
                "retrieval_r1": r1_test,
                "n_subjects": len(y_test),
            }
        ]
    ).to_csv(clip_dir / "metrics_alignment.csv", index=False)

    staging_metrics, prediction_rows = evaluate_staging(
        z_img_test,
        z_txt_test,
        y_test,
        z_img_train,
        z_txt_train,
        y_train,
        test_ids,
        pairs,
        proj_dim,
        batch_size,
    )
    staging_metrics.to_csv(clip_dir / "metrics_staging.csv", index=False)
    prediction_rows.to_csv(clip_dir / "test_predictions.csv", index=False)

    fusion_metrics = run_late_fusion_classifiers(
        z_img_train,
        z_txt_train,
        y_train,
        z_img_test,
        z_txt_test,
        y_test,
        test_ids,
        fusion_dir,
        device=str(device),
    )

    best_mode = staging_metrics.iloc[0]["model"]
    best_pred = prediction_rows[f"pred_{best_mode}"].to_numpy()
    cm = confusion_matrix(y_test, best_pred, labels=[0, 1])
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        clip_dir / "confusion_matrix.csv"
    )
    report = classification_report(
        y_test, best_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    with open(clip_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    run_config = {
        "analysis_id": "clinical_staging_pd_clip_v2",
        "version": "v2",
        "proj_dim": proj_dim,
        "batch_size": batch_size,
        "val_fraction": val_fraction,
        "seed": seed,
        "device": str(device),
        "lambda_infonce": lambda_infonce,
        "lambda_supcon": lambda_supcon,
        "temperature": temperature,
        "min_epochs": min_epochs,
        "patience": patience,
        "early_stop_metric": "val_supcon_loss",
        "checkpoint": _repo_relative((ckpt_dir / "projectors.pt").resolve()),
        "feature_scalers": _repo_relative(scaler_path.resolve()),
        "staging_modes": staging_metrics["model"].tolist(),
        "fusion_dir": _repo_relative(fusion_dir.resolve()),
    }
    with open(clip_dir / "run_config_clip.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"[clinical_clip v2] alignment R@1 test={r1_test:.3f}")
    print(f"[clinical_clip v2] staging → {clip_dir / 'metrics_staging.csv'}")
    print(staging_metrics[["model", "balanced_accuracy", "recall_macro", "f1_macro", "roc_auc_ovr_macro"]].to_string(index=False))
    print(f"[clinical_clip v2] fusion best BA={fusion_metrics.iloc[0]['balanced_accuracy']:.3f}")
    return staging_metrics
