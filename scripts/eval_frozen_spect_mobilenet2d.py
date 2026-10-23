#!/usr/bin/env python3
"""Frozen inference of a SPECT-trained MobileNetV2-2D on synthetic SPECT volumes.

No training: loads a checkpoint, freezes weights, evaluates on --test-dir only
with axial slices in [z_min, z_max] and subject-level majority vote.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
TRAIN_2D = ROOT / "scripts" / "train_synthetic_spect_mobilenet2d.py"


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train_synthetic_spect_mobilenet2d", TRAIN_2D)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


t2d = _load_train_module()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "data_rois/classifiers/SPECT/2D/best_mobilenet_v2.pt",
    )
    p.add_argument("--meta-csv", type=Path, default=ROOT / "data_info" / "data_info.csv")
    p.add_argument("--z-min", type=int, default=68)
    p.add_argument("--z-max", type=int, default=87)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    return p.parse_args()


def main():
    args = parse_args()
    if args.z_min > args.z_max:
        raise SystemExit(f"ERROR: --z-min ({args.z_min}) > --z-max ({args.z_max})")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = pd.read_csv(args.meta_csv, dtype={"ID": str})
    meta["ID"] = meta["ID"].astype(str)
    labels = meta.set_index("ID")["label"].astype(int).to_dict()
    test_recs = t2d.records_from_dir(args.test_dir, labels)

    volumes: dict[str, object] = {}
    for rec in tqdm(test_recs, desc="load synthetic volumes"):
        volumes[rec["ID"]] = nib.load(str(rec["path"])).get_fdata(dtype=np.float32)

    transform = t2d.build_transform(args.image_size)
    loader = DataLoader(
        t2d.AxialSliceDataset(test_recs, volumes, transform, args.z_min, args.z_max),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = t2d.build_mobilenet_v2(num_classes=2, pretrained=False).to(device)
    t2d.load_model_weights(model, args.checkpoint)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    metrics, slice_df, subj_df = t2d.evaluate(model, loader, device)
    slice_df.to_csv(out_dir / "test_slice_predictions.csv", index=False)
    subj_df.to_csv(out_dir / "test_subject_predictions.csv", index=False)

    report = classification_report(subj_df["label"], subj_df["pred"], digits=4, zero_division=0)
    (out_dir / "classification_report_test.txt").write_text(report)
    cm = confusion_matrix(subj_df["label"], subj_df["pred"], labels=[0, 1])
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
        "z_min": args.z_min,
        "z_max": args.z_max,
        "z_range_inclusive": True,
        "aggregation": "majority_vote_slices",
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "model": "mobilenet_v2_frozen_from_spect",
        "finetune_mode": "frozen",
        "n_test": len(test_recs),
        "n_test_slices": int(len(loader.dataset)),
        "test_ids": sorted(r["ID"] for r in test_recs),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    summary = {
        "device": str(device),
        "mode": "frozen_inference",
        "checkpoint": str(args.checkpoint),
        "test_metrics": metrics,
        "n_test_subjects": len(test_recs),
        "n_test_slices": int(len(loader.dataset)),
        "axial_axis": 2,
        "z_min": args.z_min,
        "z_max": args.z_max,
        "aggregation": "majority_vote_slices",
        "model": "mobilenet_v2_frozen_from_spect",
        "finetune_mode": "frozen",
        "trainable_params": 0,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print(report)
    print(json.dumps(metrics, indent=2))
    print(f"Saved under {out_dir}")


if __name__ == "__main__":
    main()
