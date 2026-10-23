#!/usr/bin/env python3
"""Frozen inference of a SPECT-trained MobileNetV2-3D on synthetic SPECT volumes.

No training: loads a checkpoint, freezes weights, evaluates on --test-dir only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
mod = sys.modules.get("code")
if mod is not None and not hasattr(mod, "__path__"):
    del sys.modules["code"]

from code.mobilenet3d import build_mobilenet_v2_3d  # noqa: E402
from scripts.train_synthetic_spect_mobilenet3d import (  # noqa: E402
    NiftiVolumeDataset,
    evaluate,
    load_model_weights,
    records_from_dir,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "data_rois/classifiers/SPECT/3D/best.pt",
    )
    p.add_argument("--meta-csv", type=Path, default=ROOT / "data_info" / "data_info.csv")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--z-min", type=int, default=None)
    p.add_argument("--z-max", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if (args.z_min is None) ^ (args.z_max is None):
        raise SystemExit("ERROR: --z-min and --z-max must be provided together")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = pd.read_csv(args.meta_csv, dtype={"ID": str})
    meta["ID"] = meta["ID"].astype(str)
    labels = meta.set_index("ID")["label"].astype(int).to_dict()
    test_recs = records_from_dir(args.test_dir, labels)

    loader = DataLoader(
        NiftiVolumeDataset(test_recs, args.z_min, args.z_max),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_mobilenet_v2_3d(num_classes=2, in_channels=1, pretrained=False).to(device)
    load_model_weights(model, args.checkpoint)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    metrics, pred_df = evaluate(model, loader, device)
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    report = classification_report(pred_df["label"], pred_df["pred"], digits=4, zero_division=0)
    (out_dir / "classification_report_test.txt").write_text(report)
    cm = confusion_matrix(pred_df["label"], pred_df["pred"], labels=[0, 1])
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        out_dir / "confusion_matrix_test.csv"
    )
    (out_dir / "metrics_test.json").write_text(json.dumps(metrics, indent=2))

    config = {
        "mode": "frozen_inference",
        "test_dir": str(args.test_dir),
        "out_dir": str(out_dir),
        "checkpoint": str(args.checkpoint),
        "meta_csv": str(args.meta_csv),
        "batch_size": args.batch_size,
        "z_min": args.z_min,
        "z_max": args.z_max,
        "model": "mobilenet_v2_3d_frozen_from_spect",
        "finetune_mode": "frozen",
        "n_test": len(test_recs),
        "test_ids": sorted(r["ID"] for r in test_recs),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    summary = {
        "device": str(device),
        "mode": "frozen_inference",
        "checkpoint": str(args.checkpoint),
        "test_metrics": metrics,
        "n_test_subjects": len(test_recs),
        "model": "mobilenet_v2_3d_frozen_from_spect",
        "finetune_mode": "frozen",
        "trainable_params": 0,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print(report)
    print(json.dumps(metrics, indent=2))
    print(f"Saved under {out_dir}")


if __name__ == "__main__":
    main()
