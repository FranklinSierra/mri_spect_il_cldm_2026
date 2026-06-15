#!/usr/bin/env python3
"""Preflight checks before running stage-1/stage-2 experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def present_ids(data_root: Path) -> set[str]:
    ids = set()
    for p in (data_root / "whole_MRI").glob("*.nii*"):
        name = p.name
        if name.endswith(".nii.gz"):
            ids.add(name[:-7])
        elif name.endswith(".nii"):
            ids.add(name[:-4])
    return ids


def check_split(split_name: str, ids: list[str], present: set[str]) -> list[str]:
    missing = sorted(set(ids) - present)
    print(f"{split_name}: {len(ids)} IDs | missing in data: {len(missing)}")
    if missing:
        print(f"  missing {split_name}: {missing}")
    return missing


def main() -> int:
    data_root = ROOT / "data"
    info_root = ROOT / "data_info"

    present = present_ids(data_root)
    print(f"Data IDs in data/whole_MRI: {len(present)}")

    train_ids = read_ids(info_root / "train.txt")
    val_ids = read_ids(info_root / "validation.txt")
    test_ids = read_ids(info_root / "test.txt")

    missing = []
    missing += check_split("train", train_ids, present)
    missing += check_split("validation", val_ids, present)
    missing += check_split("test", test_ids, present)

    data_info = pd.read_csv(info_root / "data_info.csv", dtype={"ID": str})
    data_info_ids = set(data_info["ID"])
    diff_csv = sorted(data_info_ids - present)
    print(f"data_info.csv rows: {len(data_info)} | IDs missing in data: {len(diff_csv)}")
    if diff_csv:
        print(f"  missing data_info IDs: {diff_csv}")

    latent_abeta_count = len(list((data_root / "latent_Abeta").glob("*.nii*")))
    latent_mri_count = len(list((data_root / "latent_MRI").glob("*.nii*")))
    print(f"latent_Abeta files: {latent_abeta_count}")
    print(f"latent_MRI files: {latent_mri_count}")
    if latent_abeta_count == 0:
        print("WARNING: latent_Abeta is empty (expected before running stage 1 SPECT + encoding).")
    if latent_mri_count == 0:
        print("INFO: latent_MRI is empty (normal unless MRI embeddings have been encoded).")

    return 1 if (missing or diff_csv) else 0


if __name__ == "__main__":
    raise SystemExit(main())
