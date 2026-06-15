#!/usr/bin/env python3
"""Regenerate cohort and metrics tables for docs/REPORTE_PROYECTO.md."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "tables"


def cohort_table(df: pd.DataFrame) -> pd.DataFrame:
    labels = df["label"].map({0: "control", 1: "parkinson"})
    counts = pd.crosstab(df["split"], labels)
    counts.loc["total"] = counts.sum()
    return counts


def metrics_by_group(metrics_path: Path, info: pd.DataFrame) -> pd.DataFrame:
    m = pd.read_csv(metrics_path)
    merged = m.copy()
    if "label" not in merged.columns:
        merged = merged.merge(info[["ID", "label"]], on="ID", how="left")
    if merged["ID"].dtype != info["ID"].dtype:
        merged["ID"] = merged["ID"].astype(str)
        info = info.copy()
        info["ID"] = info["ID"].astype(str)
    merged["population"] = merged["label"].map(
        {0: "control", 1: "parkinson", 0.0: "control", 1.0: "parkinson"}
    )
    return (
        merged.groupby(["split", "population"], dropna=False)
        .agg(PSNR_mean=("PSNR", "mean"), SSIM_mean=("SSIM", "mean"), n=("ID", "count"))
        .round(4)
        .reset_index()
    )


def markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=True)
    except ImportError:
        return df.to_string()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--inference-metrics",
        type=Path,
        default=ROOT / "result/exp_1/inference/20260508_160032/reconstruction_metrics.csv",
    )
    parser.add_argument(
        "--phase1-test",
        type=Path,
        default=ROOT / "result/exp_2/phase1_mri/reconstruction/test.csv",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    info = pd.read_csv(ROOT / "data_info/data_info.csv")
    cohort = cohort_table(info)
    cohort.to_csv(args.out_dir / "cohort_by_split_label.csv")

    sections = ["# Tablas generadas automáticamente\n", "## Cohorte\n", markdown_table(cohort), ""]

    if args.inference_metrics.exists():
        inf = metrics_by_group(args.inference_metrics, info)
        inf.to_csv(args.out_dir / "inference_synthesis_metrics.csv", index=False)
        sections.extend(
            ["## Síntesis MRI→SPECT (inferencia)\n", markdown_table(inf.set_index(["split", "population"])), ""]
        )

    if args.phase1_test.exists():
        p1 = pd.read_csv(args.phase1_test)
        info_id = info.copy()
        info_id["ID"] = info_id["ID"].astype(str)
        p1["ID"] = p1["ID"].astype(str)
        merged = p1.merge(info_id[["ID", "label"]], on="ID")
        merged["population"] = merged["label"].map({0: "control", 1: "parkinson"})
        summary = (
            merged.groupby("population")
            .agg(PSNR_mean=("PSNR", "mean"), SSIM_mean=("SSIM", "mean"), n=("ID", "count"))
            .round(4)
        )
        summary.to_csv(args.out_dir / "phase1_mri_test_recon.csv")
        sections.extend(["## Phase-1 MRI AAE (test)\n", markdown_table(summary), ""])

    out_md = args.out_dir / "generated_tables.md"
    out_md.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"CSV dir: {args.out_dir}")


if __name__ == "__main__":
    main()
