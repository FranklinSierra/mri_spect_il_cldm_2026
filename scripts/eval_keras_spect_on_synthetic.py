#!/usr/bin/env python3
"""Evaluate a Keras MobileNet trained on real SPECT against synthetic SPECT NIfTIs.

Subject-level decision is the majority vote of axial slices, matching the
classifier that produced model.keras. ROC-AUC uses the mean P(class=1) across
slices. Reported metrics match scripts/train_synthetic_spect_mobilenet3d.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
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

FST_ROOT = Path("/workspace/IL-FST")
if not FST_ROOT.exists():
    FST_ROOT = Path("/workspace/IL-CLDM/IL-FST")
sys.path.insert(0, str(FST_ROOT))

from fst.slice_data import map_volume_to_unit_interval, slice_indices, slice_to_rgb  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        type=Path,
        default=FST_ROOT / "classifiers/classifiers_roi_4mm/spect/model.keras",
    )
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--meta-csv", type=Path, default=Path("/workspace/IL-CLDM/data_info/data_info.csv"))
    p.add_argument("--z-min", type=int, default=78)
    p.add_argument("--z-max", type=int, default=112)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


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


def tiles_for_volume(path: Path, z_min: int, z_max: int) -> np.ndarray:
    raw = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    unit = map_volume_to_unit_interval(raw, "spect")
    zs = slice_indices(z_min, z_max, unit.shape, axis=2)
    if not zs:
        raise ValueError(f"{path.name} shape {unit.shape} has no slices in [{z_min}, {z_max}]")
    return np.stack([slice_to_rgb(unit[:, :, z]) for z in zs], axis=0)


def main():
    args = parse_args()
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    meta = pd.read_csv(args.meta_csv, dtype={"ID": str})
    labels = {str(r.ID): int(r.label) for r in meta.itertuples(index=False)}

    paths = list_volumes(args.test_dir)
    print(f"Loading {args.model}", flush=True)
    model = tf.keras.models.load_model(str(args.model))
    print(f"subjects={len(paths)} gpus={tf.config.list_physical_devices('GPU')}", flush=True)

    rows = []
    for path in paths:
        sid = subject_id_from_path(path)
        if sid not in labels:
            raise KeyError(f"ID {sid} missing from {args.meta_csv}")
        tiles = tiles_for_volume(path, args.z_min, args.z_max)
        probs = model.predict(tiles, batch_size=args.batch_size, verbose=0)
        pred_slice = np.argmax(probs, axis=1)
        votes = pd.Series(pred_slice).value_counts()
        rows.append(
            {
                "ID": sid,
                "label": int(labels[sid]),
                "pred": int(votes.idxmax()),
                "prob_1": float(probs[:, 1].mean()),
                "n_slices": int(len(pred_slice)),
            }
        )
        print(
            f"{sid} label={labels[sid]} pred={rows[-1]['pred']} "
            f"prob_1={rows[-1]['prob_1']:.4f} slices={rows[-1]['n_slices']}",
            flush=True,
        )

    df = pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)
    metrics = subject_metrics(df["label"].to_numpy(), df["pred"].to_numpy(), df["prob_1"].to_numpy())
    report = classification_report(df["label"], df["pred"], digits=4, zero_division=0)
    cm = confusion_matrix(df["label"], df["pred"], labels=[0, 1])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_out = df[["ID", "label", "pred", "prob_1"]]
    pred_out.to_csv(out_dir / "test_predictions.csv", index=False)
    (out_dir / "classification_report_test.txt").write_text(report)
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        out_dir / "confusion_matrix_test.csv"
    )
    (out_dir / "metrics_test.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "model": str(args.model),
                "test_dir": str(args.test_dir),
                "preprocess": "spect_p1_p99_to_unit",
                "slice_axis": 2,
                "slice_range": [args.z_min, args.z_max],
                "tile_size": [256, 256],
                "aggregation": "majority_vote_slices",
                "prob_1": "mean softmax class 1 across slices",
                "n_test": len(df),
            },
            indent=2,
        )
    )

    print("\n=== TEST ===", flush=True)
    print(report, flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
