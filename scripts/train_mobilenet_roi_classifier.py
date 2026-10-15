#!/usr/bin/env python3
"""Train MobileNetV2 axial-slice classifiers on ROI volumes (MRI or SPECT).

Subject-level predictions use majority vote over axial slices.
Early-stopping monitors subject-level validation ROC-AUC.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_META = ROOT / "data_info" / "data_info.csv"
DEFAULT_DATA_ROOT = ROOT / "data_rois"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modality", choices=["MRI", "SPECT"], required=True)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--meta-csv", type=Path, default=DEFAULT_META)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--min-nonzero-frac", type=float, default=0.01,
                   help="Skip axial slices with fewer non-zero voxels than this fraction.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def modality_dir(data_root: Path, modality: str) -> Path:
    if modality == "MRI":
        return data_root / "whole_MRI"
    return data_root / "whole_Abeta"


def apply_volume_norm(vol: np.ndarray, modality: str) -> np.ndarray:
    """MRI: z-score on voxels > 0. SPECT: leave as-is (region-normalized)."""
    x = vol.astype(np.float32, copy=True)
    if modality != "MRI":
        return x
    mask = x > 0
    if not np.any(mask):
        return x
    mu = float(x[mask].mean())
    sigma = float(x[mask].std())
    if sigma < 1e-8:
        sigma = 1.0
    out = np.zeros_like(x)
    out[mask] = (x[mask] - mu) / sigma
    return out


def load_volume(path: Path, modality: str) -> np.ndarray:
    vol = nib.load(str(path)).get_fdata(dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {vol.shape} for {path}")
    return apply_volume_norm(vol, modality)


def slice_to_rgb_uint8_scale(slc: np.ndarray) -> np.ndarray:
    """Map a float slice to roughly [0, 1] for ImageNet-pretrained input.

    SPECT values are left un-zscored but still need a display-range map for
    torchvision ImageNet normalization. Uses robust percentile scaling per slice.
    """
    x = slc.astype(np.float32)
    lo, hi = np.percentile(x, (1.0, 99.0))
    if hi <= lo:
        hi = float(x.max()) if float(x.max()) > float(x.min()) else float(x.min()) + 1.0
        lo = float(x.min())
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return x


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class AxialSliceDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        volumes: dict[str, np.ndarray],
        transform,
        min_nonzero_frac: float,
    ):
        self.transform = transform
        self.volumes = volumes
        self.samples: list[tuple[str, int, int]] = []  # sid, label, z
        for rec in records:
            sid = rec["ID"]
            label = int(rec["label"])
            vol = volumes[sid]
            for z in range(vol.shape[2]):
                slc = vol[:, :, z]
                if (slc != 0).mean() < min_nonzero_frac:
                    continue
                self.samples.append((sid, label, z))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sid, label, z = self.samples[idx]
        # Volume already modality-normalized (MRI z-score>0 / SPECT identity).
        # Percentile→[0,1] is only for ImageNet-pretrained RGB input, not a medical z-score.
        slc = slice_to_rgb_uint8_scale(self.volumes[sid][:, :, z])
        rgb = np.stack([slc, slc, slc], axis=-1)  # H,W,3 float [0,1]
        img = Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")
        x = self.transform(img)
        return x, label, sid


def build_mobilenet_v2(num_classes: int = 2) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def predict_slice_loader(model, loader, device):
    model.eval()
    rows = []
    for xb, yb, sids in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        y_np = yb.numpy()
        for i, sid in enumerate(sids):
            rows.append(
                {
                    "ID": sid,
                    "label": int(y_np[i]),
                    "pred_slice": int(preds[i]),
                    "prob_1": float(probs[i, 1]),
                }
            )
    return pd.DataFrame(rows)


def aggregate_subject(slice_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in slice_df.groupby("ID"):
        votes = g["pred_slice"].to_numpy()
        # majority vote; tie -> higher mean P(class=1)
        counts = np.bincount(votes, minlength=2)
        if counts[0] == counts[1]:
            pred = int(g["prob_1"].mean() >= 0.5)
        else:
            pred = int(np.argmax(counts))
        rows.append(
            {
                "ID": sid,
                "label": int(g["label"].iloc[0]),
                "pred": pred,
                "prob_1": float(g["prob_1"].mean()),
                "n_slices": int(len(g)),
                "vote_class0": int(counts[0]),
                "vote_class1": int(counts[1]),
            }
        )
    return pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)


def subject_metrics(subj_df: pd.DataFrame) -> dict:
    y = subj_df["label"].to_numpy()
    pred = subj_df["pred"].to_numpy()
    prob = subj_df["prob_1"].to_numpy()
    out = {
        "n_subjects": int(len(subj_df)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def evaluate(model, loader, device) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    slice_df = predict_slice_loader(model, loader, device)
    subj_df = aggregate_subject(slice_df)
    return subject_metrics(subj_df), slice_df, subj_df


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = args.out_dir or (args.data_root / "classifiers" / args.modality)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"modality={args.modality} device={device} out_dir={out_dir}")

    meta = pd.read_csv(args.meta_csv)
    meta["ID"] = meta["ID"].astype(str)
    meta["label"] = meta["label"].astype(int)
    if "split" not in meta.columns:
        raise ValueError(f"{args.meta_csv} must contain a split column")

    train_pool = meta[meta["split"] == "train"].copy()
    test_df = meta[meta["split"] == "test"].copy()
    if train_pool.empty or test_df.empty:
        raise RuntimeError("Empty train or test split in meta CSV")

    train_df, val_df = train_test_split(
        train_pool,
        test_size=args.val_fraction,
        random_state=args.seed,
        stratify=train_pool["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    split_table = pd.concat(
        [
            train_df.assign(clf_split="train"),
            val_df.assign(clf_split="val"),
            test_df.assign(clf_split="test"),
        ],
        ignore_index=True,
    )[["ID", "label", "split", "clf_split"]]
    split_table.to_csv(out_dir / "split_used.csv", index=False)

    vol_dir = modality_dir(args.data_root, args.modality)
    needed_ids = sorted(set(split_table["ID"]))
    volumes: dict[str, np.ndarray] = {}
    missing = []
    for sid in tqdm(needed_ids, desc=f"load {args.modality} volumes"):
        path = vol_dir / f"{sid}.nii.gz"
        if not path.exists():
            missing.append(sid)
            continue
        volumes[sid] = load_volume(path, args.modality)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} volumes under {vol_dir}: {missing[:10]}")

    transform = build_transform(args.image_size)

    def make_loader(frame: pd.DataFrame, shuffle: bool) -> DataLoader:
        records = frame[["ID", "label"]].to_dict("records")
        ds = AxialSliceDataset(records, volumes, transform, args.min_nonzero_frac)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df, shuffle=False)
    test_loader = make_loader(test_df, shuffle=False)
    print(
        f"slices train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"test={len(test_loader.dataset)} | subjects "
        f"train={len(train_df)} val={len(val_df)} test={len(test_df)}"
    )

    model = build_mobilenet_v2(num_classes=2).to(device)
    # CrossEntropyLoss expects raw logits (log-softmax is inside). Softmax is used
    # for probabilities / majority-vote aggregation at evaluation time.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_val_auc = -1.0
    best_epoch = -1
    patience_left = args.patience
    best_path = out_dir / "best_mobilenet_v2.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        t0 = time.time()
        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(yb)
            n += len(yb)
        train_loss = running / max(n, 1)

        val_metrics, _, _ = evaluate(model, val_loader, device)
        val_auc = val_metrics["roc_auc"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auc": val_auc,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} loss={train_loss:.4f} "
            f"val_auc={val_auc:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} ({row['seconds']:.1f}s)"
        )

        improved = np.isfinite(val_auc) and val_auc > best_val_auc + 1e-6
        if improved:
            best_val_auc = float(val_auc)
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_auc": best_val_auc,
                    "modality": args.modality,
                    "args": vars(args),
                },
                best_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop at epoch {epoch} (best epoch={best_epoch}, best_val_auc={best_val_auc:.4f})")
                break

    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics, test_slices, test_subjects = evaluate(model, test_loader, device)
    val_metrics, _, val_subjects = evaluate(model, val_loader, device)

    test_slices.to_csv(out_dir / "test_slice_predictions.csv", index=False)
    test_subjects.to_csv(out_dir / "test_subject_predictions.csv", index=False)
    val_subjects.to_csv(out_dir / "val_subject_predictions.csv", index=False)

    report = classification_report(
        test_subjects["label"],
        test_subjects["pred"],
        digits=4,
        zero_division=0,
    )
    (out_dir / "classification_report_test.txt").write_text(report)
    cm = confusion_matrix(test_subjects["label"], test_subjects["pred"])
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        out_dir / "confusion_matrix_test.csv"
    )

    metrics_row = {
        "modality": args.modality,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"val_{k}": v for k, v in val_metrics.items()},
    }
    pd.DataFrame([metrics_row]).to_csv(out_dir / "classification_metrics.csv", index=False)

    summary = {
        "modality": args.modality,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "n_train_subjects": int(len(train_df)),
        "n_val_subjects": int(len(val_df)),
        "n_test_subjects": int(len(test_df)),
        "n_train_slices": int(len(train_loader.dataset)),
        "n_val_slices": int(len(val_loader.dataset)),
        "n_test_slices": int(len(test_loader.dataset)),
        "axial_axis": 2,
        "volume_norm": "mri_zscore_gt0" if args.modality == "MRI" else "spect_identity",
        "aggregation": "majority_vote_slices",
        "model": "mobilenet_v2_imagenet",
        "loss": "CrossEntropyLoss(logits) + Softmax for probs",
        "checkpoint": str(best_path),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== TEST (subject-level, majority vote) ===")
    print(report)
    print(json.dumps(test_metrics, indent=2))
    print(f"Saved artifacts under {out_dir}")


if __name__ == "__main__":
    main()
