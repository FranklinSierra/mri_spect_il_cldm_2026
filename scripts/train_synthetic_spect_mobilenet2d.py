#!/usr/bin/env python3
"""Train MobileNetV2 axial-slice classifiers on synthetic SPECT NIfTIs.

Holds out a stratified fraction of the synthetic train subjects for validation.
Subject-level predictions use majority vote over axial slices in [z_min, z_max].
Early-stopping monitors subject-level validation ROC-AUC.
"""

from __future__ import annotations

import argparse
import json
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
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--meta-csv", type=Path, default=ROOT / "data_info" / "data_info.csv")
    p.add_argument("--z-min", type=int, default=68)
    p.add_argument("--z-max", type=int, default=87)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional SPECT 2D MobileNet checkpoint for full finetune.",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subject_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return path.stem


def list_volumes(root: Path) -> list[Path]:
    files = sorted(root.glob("*.nii.gz")) + sorted(root.glob("*.nii"))
    seen = set()
    unique = []
    for path in files:
        sid = subject_id_from_path(path)
        if sid in seen:
            continue
        seen.add(sid)
        unique.append(path)
    if not unique:
        raise FileNotFoundError(f"No NIfTI volumes under {root}")
    return unique


def slice_to_rgb_uint8_scale(slc: np.ndarray) -> np.ndarray:
    x = slc.astype(np.float32)
    lo, hi = np.percentile(x, (1.0, 99.0))
    if hi <= lo:
        hi = float(x.max()) if float(x.max()) > float(x.min()) else float(x.min()) + 1.0
        lo = float(x.min())
    return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)


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
        z_min: int,
        z_max: int,
    ):
        self.transform = transform
        self.volumes = volumes
        self.samples: list[tuple[str, int, int]] = []
        for rec in records:
            sid = rec["ID"]
            label = int(rec["label"])
            vol = volumes[sid]
            z_hi = min(z_max, vol.shape[2] - 1)
            z_lo = max(z_min, 0)
            for z in range(z_lo, z_hi + 1):
                self.samples.append((sid, label, z))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sid, label, z = self.samples[idx]
        slc = slice_to_rgb_uint8_scale(self.volumes[sid][:, :, z])
        rgb = np.stack([slc, slc, slc], axis=-1)
        img = Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")
        x = self.transform(img)
        return x, label, sid


def build_mobilenet_v2(num_classes: int = 2, *, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_model_weights(model: nn.Module, checkpoint: Path) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)


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


def records_from_dir(root: Path, labels: dict[str, int]) -> list[dict]:
    rows = []
    for path in list_volumes(root):
        sid = subject_id_from_path(path)
        if sid not in labels:
            raise KeyError(f"ID {sid} missing from metadata")
        rows.append({"ID": sid, "label": int(labels[sid]), "path": path})
    return rows


def main():
    args = parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise SystemExit(f"ERROR: --val-fraction must be in (0, 1), got {args.val_fraction}")
    if args.z_min > args.z_max:
        raise SystemExit(f"ERROR: --z-min ({args.z_min}) > --z-max ({args.z_max})")
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = pd.read_csv(args.meta_csv, dtype={"ID": str})
    meta["ID"] = meta["ID"].astype(str)
    labels = meta.set_index("ID")["label"].astype(int).to_dict()

    train_pool = records_from_dir(args.train_dir, labels)
    test_recs = records_from_dir(args.test_dir, labels)
    pool_ids = [r["ID"] for r in train_pool]
    pool_y = [r["label"] for r in train_pool]
    fit_ids, val_ids = train_test_split(
        pool_ids,
        test_size=args.val_fraction,
        random_state=args.seed,
        stratify=pool_y,
    )
    by_id = {r["ID"]: r for r in train_pool}
    fit_recs = [by_id[sid] for sid in fit_ids]
    val_recs = [by_id[sid] for sid in val_ids]

    split_table = pd.DataFrame(
        [{"ID": r["ID"], "label": r["label"], "clf_split": "train"} for r in fit_recs]
        + [{"ID": r["ID"], "label": r["label"], "clf_split": "val"} for r in val_recs]
        + [{"ID": r["ID"], "label": r["label"], "clf_split": "test"} for r in test_recs]
    )
    split_table.to_csv(out_dir / "split_used.csv", index=False)

    needed = fit_recs + val_recs + test_recs
    volumes: dict[str, np.ndarray] = {}
    for rec in tqdm(needed, desc="load synthetic volumes"):
        volumes[rec["ID"]] = nib.load(str(rec["path"])).get_fdata(dtype=np.float32)

    sample_shape = tuple(next(iter(volumes.values())).shape)
    transform = build_transform(args.image_size)

    def make_loader(recs: list[dict], shuffle: bool) -> DataLoader:
        return DataLoader(
            AxialSliceDataset(recs, volumes, transform, args.z_min, args.z_max),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    train_loader = make_loader(fit_recs, shuffle=True)
    val_loader = make_loader(val_recs, shuffle=False)
    test_loader = make_loader(test_recs, shuffle=False)

    init_ckpt = Path(args.init_checkpoint) if args.init_checkpoint else None
    if init_ckpt is not None and not init_ckpt.is_file():
        raise FileNotFoundError(f"Missing --init-checkpoint: {init_ckpt}")
    finetune_mode = "full" if init_ckpt is not None else "from_imagenet"
    model_name = (
        "mobilenet_v2_full_finetune_from_spect"
        if init_ckpt is not None
        else "mobilenet_v2_imagenet"
    )

    model = build_mobilenet_v2(num_classes=2, pretrained=init_ckpt is None).to(device)
    if init_ckpt is not None:
        load_model_weights(model, init_ckpt)
        print(f"Loaded init checkpoint: {init_ckpt}")

    print(
        f"device={device} train_pool={len(train_pool)} fit={len(fit_recs)} "
        f"val={len(val_recs)} test={len(test_recs)} "
        f"slices train/val/test="
        f"{len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)} "
        f"z=[{args.z_min},{args.z_max}] model={model_name}"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_val_auc = -1.0
    best_epoch = -1
    patience_left = args.patience
    best_path = out_dir / "best_mobilenet_v2.pt"

    config_payload = {
        "train_dir": str(args.train_dir),
        "test_dir": str(args.test_dir),
        "out_dir": str(out_dir),
        "meta_csv": str(args.meta_csv),
        "init_checkpoint": str(init_ckpt) if init_ckpt is not None else None,
        "finetune_mode": finetune_mode,
        "z_min": args.z_min,
        "z_max": args.z_max,
        "z_range_inclusive": True,
        "val_fraction": args.val_fraction,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
        "image_size": args.image_size,
        "volume_shape": list(sample_shape),
        "early_stop_metric": "val_roc_auc_subject_majority_vote",
        "aggregation": "majority_vote_slices",
        "model": model_name,
        "n_train_pool": len(train_pool),
        "n_fit": len(fit_recs),
        "n_val": len(val_recs),
        "n_test": len(test_recs),
        "fit_ids": sorted(fit_ids),
        "val_ids": sorted(val_ids),
        "test_ids": sorted(r["ID"] for r in test_recs),
    }
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        t0 = time.time()
        for xb, yb, _ in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
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
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_f1": val_metrics["f1"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
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
                    "args": {
                        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                    },
                },
                best_path,
            )
        else:
            patience_left -= 1
            print(f"early-stop counter: {args.patience - patience_left}/{args.patience}")
            if patience_left <= 0:
                print(
                    f"Early stop at epoch {epoch} "
                    f"(best epoch={best_epoch}, best_val_auc={best_val_auc:.4f})"
                )
                break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    val_metrics, val_slices, val_subjects = evaluate(model, val_loader, device)
    test_metrics, test_slices, test_subjects = evaluate(model, test_loader, device)

    val_slices.to_csv(out_dir / "val_slice_predictions.csv", index=False)
    val_subjects.to_csv(out_dir / "val_subject_predictions.csv", index=False)
    test_slices.to_csv(out_dir / "test_slice_predictions.csv", index=False)
    test_subjects.to_csv(out_dir / "test_subject_predictions.csv", index=False)

    report = classification_report(
        test_subjects["label"],
        test_subjects["pred"],
        digits=4,
        zero_division=0,
    )
    (out_dir / "classification_report_test.txt").write_text(report)
    cm = confusion_matrix(test_subjects["label"], test_subjects["pred"], labels=[0, 1])
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        out_dir / "confusion_matrix_test.csv"
    )

    metrics_valid = {"best_epoch": best_epoch, "best_val_auc": best_val_auc, **val_metrics}
    metrics_test = {"best_epoch": best_epoch, **test_metrics}
    (out_dir / "metrics_valid.json").write_text(json.dumps(metrics_valid, indent=2))
    (out_dir / "metrics_test.json").write_text(json.dumps(metrics_test, indent=2))

    metrics_row = {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"val_{k}": v for k, v in val_metrics.items()},
    }
    pd.DataFrame([metrics_row]).to_csv(out_dir / "classification_metrics.csv", index=False)

    summary = {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "n_train_subjects": len(fit_recs),
        "n_val_subjects": len(val_recs),
        "n_test_subjects": len(test_recs),
        "n_train_slices": int(len(train_loader.dataset)),
        "n_val_slices": int(len(val_loader.dataset)),
        "n_test_slices": int(len(test_loader.dataset)),
        "axial_axis": 2,
        "z_min": args.z_min,
        "z_max": args.z_max,
        "aggregation": "majority_vote_slices",
        "model": model_name,
        "init_checkpoint": str(init_ckpt) if init_ckpt is not None else None,
        "finetune_mode": finetune_mode,
        "loss": "CrossEntropyLoss(logits) + Softmax for probs",
        "checkpoint": str(best_path),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== VAL ===")
    print(json.dumps(metrics_valid, indent=2))
    print("\n=== TEST (subject-level, majority vote) ===")
    print(report)
    print(json.dumps(metrics_test, indent=2))
    print(f"Saved under {out_dir}")


if __name__ == "__main__":
    main()
