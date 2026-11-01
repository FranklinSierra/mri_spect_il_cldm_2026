#!/usr/bin/env python3
"""Stage 2: full finetune the ConvNeXt3D classifier on synthetic SPECT NIfTIs.

Same role as scripts/train_synthetic_spect_mobilenet3d.py (full finetune from a
SPECT-pretrained checkpoint, one CE per volume, early-stop on val balanced accuracy),
but with the inflated ConvNeXt3D backbone, AdamW + cosine warmup, class-balanced
loss, and on-GPU 3D augmentation (code/volume_augment3d.py). Reuses the exact
fit/val/test subject IDs from the MobileNet synthetic run (--split-source) for an
apples-to-apples comparison.

Two earlier direct full-finetune recipes both failed on this backbone: a "heavy"
one (lr=5e-5, augment, class-weight, label-smoothing) produced near-random test
metrics, and a "simple" one copying the MobileNet recipe verbatim (lr=1e-4, no
augment, no class-weight) collapsed to predicting the majority class from epoch 1
(val_BA stuck at exactly 0.5 throughout). ConvNeXt's LayerNorm + layer_scale blocks
are known to be far more lr-sensitive in full finetune than BatchNorm-based nets
like MobileNetV2, and the stage-1 checkpoint here is a near-perfect (test BA=0.925)
fit on the trivially-separable real-SPECT task, making it easy to fall into a lazy
majority-class solution under an aggressive update. This script now supports the
same LP-FT (linear-probe-then-finetune) scheme used for ViT3D -- freeze the
backbone and train the head alone first to recalibrate it to the synthetic class
balance -- combined with a much lower full-finetune lr, to avoid that collapse.
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--meta-csv", type=Path, required=True)
    p.add_argument("--split-source", type=Path, required=True,
                    help="config.json of the MobileNet synthetic run to copy fit/val/test IDs from.")
    p.add_argument("--init-checkpoint", type=Path, required=True,
                    help="Stage-1 ConvNeXt3D checkpoint pretrained on real SPECT.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-class-weight", action="store_true")
    p.add_argument(
        "--linear-probe-epochs",
        type=int,
        default=10,
        help="Epochs with the backbone frozen (head-only) before unfreezing for full "
        "finetune. Guards against representation collapse to the majority class when "
        "the stage-1 checkpoint (near-perfect on real SPECT) sees out-of-distribution "
        "(synthetic) inputs under an aggressive update from epoch 1.",
    )
    p.add_argument("--probe-lr", type=float, default=1e-3)
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

    train_pool_recs = records_from_dir(args.train_dir, labels)
    test_recs_all = records_from_dir(args.test_dir, labels)
    by_id = {r["ID"]: r for r in train_pool_recs + test_recs_all}
    missing = (fit_ids | val_ids | test_ids) - set(by_id)
    if missing:
        raise KeyError(f"IDs from split-source missing under train/test dirs: {sorted(missing)[:10]}")

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

    model = build_convnext3d(num_classes=2, in_channels=1, pretrained=False).to(device)
    ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    print(f"Loaded init checkpoint: {args.init_checkpoint}")

    if args.no_class_weight:
        class_weight = None
    else:
        n_pos = sum(r["label"] for r in fit_recs)
        n_neg = len(fit_recs) - n_pos
        w0 = len(fit_recs) / (2.0 * max(n_neg, 1))
        w1 = len(fit_recs) / (2.0 * max(n_pos, 1))
        class_weight = torch.tensor([w0, w1], device=device)
        print(f"class_weight (0,1) = {w0:.3f}, {w1:.3f} (fit n_pos={n_pos} n_neg={n_neg})")
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config_payload = {
        "stage": "2_synthetic_full_finetune",
        "train_dir": str(args.train_dir),
        "test_dir": str(args.test_dir),
        "out_dir": str(out_dir),
        "meta_csv": str(args.meta_csv),
        "split_source": str(args.split_source),
        "init_checkpoint": str(args.init_checkpoint),
        "finetune_mode": "full",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "label_smoothing": args.label_smoothing,
        "patience": args.patience,
        "seed": args.seed,
        "augment": not args.no_augment,
        "class_weighted_loss": not args.no_class_weight,
        "linear_probe_epochs": args.linear_probe_epochs,
        "probe_lr": args.probe_lr,
        "volume_norm": "per_volume_zscore_gt0 (matches stage-1 real-SPECT pretraining input stats)",
        "early_stop_metric": "val_balanced_accuracy",
        "model": "convnext_tiny_3d_inflated_imagenet_full_finetune_from_spect",
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

    n_probe = min(args.linear_probe_epochs, args.epochs)
    n_full = args.epochs - n_probe
    total_steps = max(n_full, 1) * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    global_step = 0
    lr_now = args.probe_lr

    def run_epoch(optimizer, use_schedule: bool):
        nonlocal global_step, lr_now
        model.train()
        running, n = 0.0, 0
        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if not args.no_augment:
                xb = augment_batch(xb)
            if use_schedule:
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
            if use_schedule:
                global_step += 1
        return running / max(n, 1)

    def track_and_maybe_stop(epoch: int, phase: str, train_loss: float, early_stop: bool, t0: float):
        nonlocal best_val_ba, best_epoch, patience_left
        val_metrics, _ = evaluate(model, val_loader, device)
        val_ba = val_metrics["balanced_accuracy"]
        row = {
            "epoch": epoch,
            "phase": phase,
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
            f"[{phase}] epoch {epoch:03d}/{args.epochs} loss={train_loss:.4f} "
            f"val_BA={val_ba:.4f} val_auc={val_metrics['roc_auc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} lr={lr_now:.2e} ({row['seconds']:.1f}s)"
        )
        improved = val_ba > best_val_ba + 1e-6
        if improved:
            best_val_ba, best_epoch, patience_left = float(val_ba), epoch, args.patience
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, best_path)
            return False
        if early_stop:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop at epoch {epoch} (best epoch={best_epoch}, best_val_BA={best_val_ba:.4f})")
                return True
        return False

    # Phase A: freeze the backbone, train only the head. Recalibrates the head to
    # the synthetic class balance before any backbone update, guarding against the
    # majority-class collapse observed when full finetune started from epoch 1.
    if n_probe > 0:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("head")
        probe_optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=args.probe_lr, weight_decay=0.0
        )
        for epoch in range(1, n_probe + 1):
            t0 = time.time()
            train_loss = run_epoch(probe_optimizer, use_schedule=False)
            track_and_maybe_stop(epoch, "probe", train_loss, early_stop=False, t0=t0)

    # Phase B: unfreeze everything, full finetune with warmup + cosine decay at a
    # much lower lr than the direct-finetune attempts that collapsed.
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(n_probe + 1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(optimizer, use_schedule=True)
        if track_and_maybe_stop(epoch, "full", train_loss, early_stop=True, t0=t0):
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
        "n_train_subjects": len(fit_recs),
        "n_val_subjects": len(val_recs),
        "n_test_subjects": len(test_recs),
        "model": "convnext_tiny_3d_inflated_imagenet_full_finetune_from_spect",
        "init_checkpoint": str(args.init_checkpoint),
        "finetune_mode": "full",
        "early_stop_metric": "val_balanced_accuracy",
        "loss": "CrossEntropyLoss(logits, class_weight, label_smoothing) + Softmax for probs",
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
