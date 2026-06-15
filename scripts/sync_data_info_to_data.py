#!/usr/bin/env python3
"""Sync data_info files to subject IDs currently present in data/whole_MRI."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _nifti_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return path.stem


def _read_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--info-dir", type=Path, default=ROOT / "data_info")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    info_dir = args.info_dir.resolve()

    mri_dir = data_root / "whole_MRI"
    if not mri_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {mri_dir}")

    present_ids = sorted(_nifti_stem(p) for p in mri_dir.glob("*.nii*"))
    present_set = set(present_ids)
    if not present_ids:
        raise RuntimeError(f"No NIfTI files found in {mri_dir}")

    # data_info.csv
    data_info_path = info_dir / "data_info.csv"
    data_info = pd.read_csv(data_info_path, dtype={"ID": str})
    before_rows = len(data_info)
    data_info = data_info[data_info["ID"].isin(present_set)].copy()
    keep_cols = [c for c in ("ID", "label", "split") if c in data_info.columns]
    data_info = data_info[keep_cols]
    data_info = data_info.drop_duplicates(subset=["ID"], keep="first")
    data_info = data_info.sort_values(by=["split", "ID"]).reset_index(drop=True)

    # split txt files
    split_removed: dict[str, list[str]] = {}
    for split_name in ("train", "validation", "test"):
        split_path = info_dir / f"{split_name}.txt"
        ids = _read_ids(split_path)
        kept = [sid for sid in ids if sid in present_set]
        split_removed[split_name] = sorted(set(ids) - set(kept))
        if not args.dry_run:
            _write_ids(split_path, kept)

    # embedding split file (optional)
    embedding_split_path = info_dir / "embedding_eval_split.csv"
    embedding_before = 0
    embedding_after = 0
    if embedding_split_path.exists():
        emb = pd.read_csv(embedding_split_path, dtype={"ID": str})
        embedding_before = len(emb)
        emb = emb[emb["ID"].isin(present_set)].copy().reset_index(drop=True)
        embedding_after = len(emb)
        if not args.dry_run:
            emb.to_csv(embedding_split_path, index=False)

    if not args.dry_run:
        data_info.to_csv(data_info_path, index=False)

    removed_csv = sorted(set(pd.read_csv(data_info_path, dtype={"ID": str})["ID"]) - present_set) if args.dry_run else []
    # reporting
    print(f"Data IDs in {mri_dir}: {len(present_ids)}")
    print(f"data_info.csv rows: {before_rows} -> {len(data_info)}")
    if args.dry_run:
        dropped = sorted(set(pd.read_csv(data_info_path, dtype={"ID": str})["ID"]) - present_set)
        print(f"Would drop from data_info.csv: {dropped}")
    else:
        dropped_count = before_rows - len(data_info)
        print(f"Dropped {dropped_count} rows from data_info.csv")
    for split_name in ("train", "validation", "test"):
        print(f"{split_name}.txt removed IDs: {split_removed[split_name]}")
    if embedding_split_path.exists():
        print(f"embedding_eval_split.csv rows: {embedding_before} -> {embedding_after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
