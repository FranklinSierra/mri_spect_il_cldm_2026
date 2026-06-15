#!/usr/bin/env python3
"""Side-by-side table: SPECT E1 vs MRI E2 embedding classification (matched 118/30 split)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_metrics(path: Path, modality: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["modality"] = modality
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spect-metrics",
        type=Path,
        default=ROOT / "result/exp_nm_s1_spect/embedding_analysis/classification/classification_metrics.csv",
    )
    parser.add_argument(
        "--mri-metrics",
        type=Path,
        default=ROOT
        / "result/exp_nm_s1_mri/phase1_mri/embedding_analysis/classification/classification_metrics.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs/tables/spect_mri_embedding_comparison.csv",
    )
    args = parser.parse_args()

    spect = load_metrics(args.spect_metrics, "SPECT_E1")
    mri = load_metrics(args.mri_metrics, "MRI_E2")
    merged = pd.concat([spect, mri], ignore_index=True)
    cols = [
        "modality",
        "model",
        "balanced_accuracy",
        "roc_auc",
        "accuracy",
        "f1",
    ]
    merged = merged[cols].sort_values(["modality", "balanced_accuracy"], ascending=[True, False])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(merged.to_string(index=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
