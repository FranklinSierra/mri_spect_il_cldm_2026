#!/usr/bin/env python3
"""Stage 1: pretrain the inflated ConvNeXt3D classifier on real SPECT (whole_Abeta).

Same role as scripts/train_spect_vit3d_pretrain.py (which this superseded after the
ViT3D backbone failed to learn the task at all -- see code/convnext3d.py docstring),
reusing the exact same fit/val/test subject IDs (read from the MobileNet run's
config.json) so all pretraining runs are on identical data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
mod = sys.modules.get("code")
if mod is not None and not hasattr(mod, "__path__"):
    del sys.modules["code"]

from code.convnext3d import build_convnext3d  # noqa: E402
from code.volume_augment3d import augment_batch  # noqa: E402
from code.vit3d_train_utils import (  # noqa: E402
    NiftiVolumeDataset,
    cosine_warmup_lr,
    evaluate,
    records_from_dir,
    set_seed,
)

DEFAULT_SPLIT_SOURCE = ROOT / "data_rois" / "classifiers" / "SPECT" / "3D" / "config.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spect-root", type=Path, default=ROOT / "data_rois" / "whole_Abeta")
    p.add_argument("--meta-csv", type=Path, default=ROOT / "data_info" / "data_info.csv")
    p.add_argument("--split-source", type=Path, default=DEFAULT_SPLIT_SOURCE)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-augment", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_cfg = json.loads(args.split_source.read_text())
    fit_ids, val_ids, test_ids = (
        set(split_cfg["fit_ids"]),
        set(split_cfg["val_ids"]),
        set(split_cfg["test_ids"]),
    )

    meta = pd.read_csv(args.meta_csv, dtype={"ID": str})
    meta["ID"] = meta["ID"].astype(str)
    labels = meta.set_index("ID")["label"].astype(int).to_dict()

    all_recs = records_from_dir(args.spect_root, labels)
    by_id = {r["ID"]: r for r in all_recs}
    missing = (fit_ids | val_ids | test_ids) - set(by_id)
    if missing:
        raise KeyError(f"IDs from split-source missing under {args.spect_root}: {sorted(missing)[:10]}")

    fit_recs = [by_id[sid] for sid in sorted(fit_ids)]
    val_recs = [by_id[sid] for sid in sorted(val_ids)]
    test_recs = [by_id[sid] for sid in sorted(test_ids)]

    split_table = pd.DataFrame(
        [{"ID": r["ID"], "label": r["label"], "clf_split": "train"} for r in fit_recs]
        + [{"ID": r["ID"], "label": r["label"], "clf_split": "val"} for r in val_recs]
        + [{"ID": r["ID"], "label": r["label"], "clf_split": "test"} for r in test_recs]
    )
    split_table.to_csv(out_dir / "split_used.csv", index=False)

    print(f"device={device} fit={len(fit_recs)} val={len(val_recs)} test={len(test_recs)} batch_size={args.batch_size}")

    def make_loader(recs, shuffle):
        return DataLoader(
            NiftiVolumeDataset(recs, target_shape=None),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    train_loader = make_loader(fit_recs, True)
    val_loader = make_loader(val_recs, False)
    test_loader = make_loader(test_recs, False)

    model = build_convnext3d(num_classes=2, in_channels=1, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config_payload = {
        "stage": "1_real_spect_pretrain",
        "spect_root": str(args.spect_root),
        "split_source": str(args.split_source),
        "out_dir": str(out_dir),
        "meta_csv": str(args.meta_csv),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "seed": args.seed,
        "augment": not args.no_augment,
        "volume_norm": "per_volume_zscore_gt0",
        "early_stop_metric": "val_balanced_accuracy",
        "model": "convnext_tiny_3d_inflated_imagenet",
        "n_fit": len(fit_recs),
        "n_val": len(val_recs),
        "n_test": len(test_recs),
        "fit_ids": sorted(fit_ids),
        "val_ids": sorted(val_ids),
        "test_ids": sorted(test_ids),
    }
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2))

    history = []
    best_val_ba = -1.0
    best_epoch = -1
    patience_left = args.patience
    best_path = out_dir / "best.pt"

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        t0 = time.time()
        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if not args.no_augment:
                xb = augment_batch(xb)
            lr_now = cosine_warmup_lr(global_step, total_steps, warmup_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr_now
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(yb)
            n += len(yb)
            global_step += 1
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
            "lr": lr_now,
            "seconds": time.time() - t0,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
        print(
            f"epoch {epoch:03d}/{args.epochs} loss={train_loss:.4f} "
            f"val_BA={val_ba:.4f} val_auc={val_metrics['roc_auc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} lr={lr_now:.2e} ({row['seconds']:.1f}s)"
        )

        improved = val_ba > best_val_ba + 1e-6
        if improved:
            best_val_ba, best_epoch, patience_left = float(val_ba), epoch, args.patience
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop at epoch {epoch} (best epoch={best_epoch}, best_val_BA={best_val_ba:.4f})")
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

    (out_dir / "metrics_valid.json").write_text(
        json.dumps({"best_epoch": best_epoch, "best_val_balanced_accuracy": best_val_ba, **val_metrics}, indent=2)
    )
    (out_dir / "metrics_test.json").write_text(json.dumps({"best_epoch": best_epoch, **test_metrics}, indent=2))

    summary = {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_val_ba,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "model": "convnext_tiny_3d_inflated_imagenet",
        "checkpoint": str(best_path),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== VAL ===")
    print(json.dumps(val_metrics, indent=2))
    print("\n=== TEST ===")
    print(report)
    print(json.dumps(test_metrics, indent=2))
    print(f"Saved under {out_dir}")


if __name__ == "__main__":
    main()
