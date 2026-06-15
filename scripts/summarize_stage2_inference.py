#!/usr/bin/env python3
"""Aggregate stage-2 inference reconstruction and classification metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPERIMENTS = [
    ("exp_nm_s2_baseline", "baseline"),
    ("exp_nm_s2_mri_frozen", "frozen"),
    ("exp_nm_s2_mri_finetune", "finetune"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="nm_stage2_v1")
    parser.add_argument("--output-dir", default="result/stage2_inference_summary")
    return parser.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    recon_rows = []
    class_rows = []
    manifest = {"run_id": args.run_id, "experiments": {}}

    for exp_name, label in EXPERIMENTS:
        inf_dir = Path("result") / exp_name / "inference" / args.run_id
        if not inf_dir.is_dir():
            print(f"Warning: missing {inf_dir}")
            continue

        manifest["experiments"][label] = {"exp": exp_name, "inference_dir": str(inf_dir)}

        recon_path = inf_dir / "reconstruction_metrics.csv"
        if recon_path.exists():
            recon = pd.read_csv(recon_path, dtype={"ID": str})
            test_recon = recon[recon["split"] == "test"]
            if len(test_recon):
                recon_rows.append({
                    "experiment": exp_name,
                    "mode": label,
                    "n_test": len(test_recon),
                    "mean_PSNR": test_recon["PSNR"].mean(),
                    "mean_SSIM": test_recon["SSIM"].mean(),
                    "std_PSNR": test_recon["PSNR"].std(),
                    "std_SSIM": test_recon["SSIM"].std(),
                })

        class_path = inf_dir / "classification" / "classification_metrics.csv"
        if class_path.exists():
            metrics = pd.read_csv(class_path)
            for _, row in metrics.iterrows():
                class_rows.append({
                    "experiment": exp_name,
                    "mode": label,
                    "model": row["model"],
                    "accuracy": row["accuracy"],
                    "balanced_accuracy": row["balanced_accuracy"],
                    "f1": row["f1"],
                    "roc_auc": row.get("roc_auc", float("nan")),
                })

    if recon_rows:
        pd.DataFrame(recon_rows).to_csv(out_root / "reconstruction_test_summary.csv", index=False)
    if class_rows:
        pd.DataFrame(class_rows).to_csv(out_root / "classification_test_summary.csv", index=False)

    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote summary to {out_root}")


if __name__ == "__main__":
    main()
