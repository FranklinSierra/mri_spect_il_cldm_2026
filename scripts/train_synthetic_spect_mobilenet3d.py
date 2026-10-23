#!/usr/bin/env python3
"""Train MobileNetV2-3D on synthetic SPECT NIfTIs.

Holds out a stratified fraction of the synthetic train subjects for validation.
Reports classification on the synthetic test subjects. One CE per volume.

Optional --z-min/--z-max crops the axial axis to a 2.5D subvolume.
Optional --init-checkpoint loads a SPECT-pretrained 3D/2.5D MobileNet for full finetune.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import nibabel as nib
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
mod = sys.modules.get("code")
if mod is not None and not hasattr(mod, "__path__"):
    del sys.modules["code"]

from code.mobilenet3d import build_mobilenet_v2_3d  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--meta-csv", type=Path, default=ROOT / "data_info" / "data_info.csv")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--z-min", type=int, default=None)
    p.add_argument("--z-max", type=int, default=None)
    p.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional SPECT 3D/2.5D MobileNet checkpoint for full finetune.",
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


def crop_axial(vol: np.ndarray, z_min: int | None, z_max: int | None) -> np.ndarray:
    if z_min is None and z_max is None:
        return vol
    if z_min is None or z_max is None:
        raise ValueError("Both --z-min and --z-max must be set together")
    if z_min > z_max:
        raise ValueError(f"--z-min ({z_min}) > --z-max ({z_max})")
    z_lo = max(int(z_min), 0)
    z_hi = min(int(z_max), vol.shape[2] - 1)
    return vol[:, :, z_lo : z_hi + 1]


class NiftiVolumeDataset(Dataset):
    def __init__(self, records: list[dict], z_min: int | None, z_max: int | None):
        self.records = records
        self.z_min = z_min
        self.z_max = z_max

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        vol = nib.load(rec["path"]).get_fdata().astype(np.float32)
        vol = crop_axial(vol, self.z_min, self.z_max)
        x = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
        return x, int(rec["label"]), rec["ID"]


def subject_metrics(y_true, y_pred, y_prob) -> dict:
    out = {
        "n_subjects": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[dict, pd.DataFrame]:
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
                    "pred": int(preds[i]),
                    "prob_1": float(probs[i, 1]),
                }
            )
    df = pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)
    metrics = subject_metrics(df["label"].to_numpy(), df["pred"].to_numpy(), df["prob_1"].to_numpy())
    return metrics, df


def records_from_dir(root: Path, labels: dict[str, int]) -> list[dict]:
    rows = []
    for path in list_volumes(root):
        sid = subject_id_from_path(path)
        if sid not in labels:
            raise KeyError(f"ID {sid} missing from metadata")
        rows.append({"ID": sid, "label": int(labels[sid]), "path": path})
    return rows


def load_model_weights(model: nn.Module, checkpoint: Path) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)


def main():
    args = parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise SystemExit(f"ERROR: --val-fraction must be in (0, 1), got {args.val_fraction}")
    if (args.z_min is None) ^ (args.z_max is None):
        raise SystemExit("ERROR: --z-min and --z-max must be provided together")
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

    print(
        f"device={device} train_pool={len(train_pool)} fit={len(fit_recs)} "
        f"val={len(val_recs)} test={len(test_recs)} batch_size={args.batch_size}"
    )
    sample_full = nib.load(fit_recs[0]["path"]).get_fdata().astype(np.float32)
    sample = crop_axial(sample_full, args.z_min, args.z_max)
    print(
        f"volume_shape={tuple(sample.shape)} "
        f"stored_range=[{float(sample.min()):.4f}, {float(sample.max()):.4f}]"
    )

    def make_loader(recs: list[dict], shuffle: bool) -> DataLoader:
        return DataLoader(
            NiftiVolumeDataset(recs, args.z_min, args.z_max),
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

    use_25d = args.z_min is not None
    if init_ckpt is not None:
        finetune_mode = "full"
        pretrained = False
        model_name = (
            "mobilenet_v2_3d_full_finetune_from_spect_2.5D"
            if use_25d
            else "mobilenet_v2_3d_full_finetune_from_spect"
        )
    else:
        finetune_mode = "from_imagenet"
        pretrained = not args.no_pretrained
        model_name = (
            "mobilenet_v2_3d_imagenet_inflated_2.5D"
            if use_25d
            else "mobilenet_v2_3d_imagenet_inflated"
        )

    model = build_mobilenet_v2_3d(
        num_classes=2,
        in_channels=1,
        pretrained=pretrained,
    ).to(device)
    if init_ckpt is not None:
        load_model_weights(model, init_ckpt)
        print(f"Loaded init checkpoint: {init_ckpt}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_val_ba = -1.0
    best_epoch = -1
    patience_left = args.patience
    best_path = out_dir / "best.pt"

    config_payload = {
        "train_dir": str(args.train_dir),
        "test_dir": str(args.test_dir),
        "out_dir": str(out_dir),
        "meta_csv": str(args.meta_csv),
        "init_checkpoint": str(init_ckpt) if init_ckpt is not None else None,
        "finetune_mode": finetune_mode,
        "val_fraction": args.val_fraction,
        "val_split": "stratified holdout from synthetic train",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
        "pretrained": pretrained,
        "volume_norm": "as stored in synthetic NIfTI (minmax restored to [0, 1])",
        "volume_shape": list(sample.shape),
        "early_stop_metric": "val_balanced_accuracy",
        "model": model_name,
        "n_train_pool": len(train_pool),
        "n_fit": len(fit_recs),
        "n_val": len(val_recs),
        "n_test": len(test_recs),
        "fit_ids": sorted(fit_ids),
        "val_ids": sorted(val_ids),
        "test_ids": sorted(r["ID"] for r in test_recs),
    }
    if use_25d:
        config_payload.update(
            {
                "z_min": args.z_min,
                "z_max": args.z_max,
                "z_range_inclusive": True,
                "mode": "2.5D_axial_subvolume",
            }
        )
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

        val_metrics, _ = evaluate(model, val_loader, device)
        val_ba = val_metrics["balanced_accuracy"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_balanced_accuracy": val_ba,
            "val_accuracy": val_metrics["accuracy"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_f1": val_metrics["f1"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
        print(
            f"epoch {epoch:03d}/{args.epochs} loss={train_loss:.4f} "
            f"val_BA={val_ba:.4f} val_auc={val_metrics['roc_auc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} ({row['seconds']:.1f}s)"
        )

        improved = np.isfinite(val_ba) and val_ba > best_val_ba + 1e-6
        if improved:
            best_val_ba = float(val_ba)
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_balanced_accuracy": best_val_ba,
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
                    f"(best epoch={best_epoch}, best_val_BA={best_val_ba:.4f})"
                )
                break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    val_metrics, val_df = evaluate(model, val_loader, device)
    test_metrics, test_df = evaluate(model, test_loader, device)
    val_df.to_csv(out_dir / "val_predictions.csv", index=False)
    test_df.to_csv(out_dir / "test_predictions.csv", index=False)

    report = classification_report(test_df["label"], test_df["pred"], digits=4, zero_division=0)
    (out_dir / "classification_report_test.txt").write_text(report)
    cm = confusion_matrix(test_df["label"], test_df["pred"], labels=[0, 1])
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        out_dir / "confusion_matrix_test.csv"
    )

    metrics_valid = {"best_epoch": best_epoch, "best_val_balanced_accuracy": best_val_ba, **val_metrics}
    metrics_test = {"best_epoch": best_epoch, **test_metrics}
    (out_dir / "metrics_valid.json").write_text(json.dumps(metrics_valid, indent=2))
    (out_dir / "metrics_test.json").write_text(json.dumps(metrics_test, indent=2))

    summary = {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_val_ba,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "n_train_subjects": len(fit_recs),
        "n_val_subjects": len(val_recs),
        "n_test_subjects": len(test_recs),
        "model": model_name,
        "init_checkpoint": str(init_ckpt) if init_ckpt is not None else None,
        "finetune_mode": finetune_mode,
        "early_stop_metric": "val_balanced_accuracy",
        "loss": "CrossEntropyLoss(logits) + Softmax for probs",
        "checkpoint": str(best_path),
    }
    if use_25d:
        summary.update(
            {
                "z_min": args.z_min,
                "z_max": args.z_max,
                "mode": "2.5D_axial_subvolume",
            }
        )
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== VAL ===")
    print(json.dumps(metrics_valid, indent=2))
    print("\n=== TEST ===")
    print(report)
    print(json.dumps(metrics_test, indent=2))
    print(f"Saved under {out_dir}")


if __name__ == "__main__":
    main()
