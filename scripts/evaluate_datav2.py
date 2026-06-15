#!/usr/bin/env python3
"""Verify dataV2/ matches data/ cohort (IDs, regular files, integrity; no crop)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from datav2_common import (
    MODALITIES,
    ROOT,
    file_md5,
    list_subject_ids,
    nifti_metadata,
    resolve_source,
    volume_path,
)


def read_split_ids(info_dir: Path, split_name: str) -> list[str]:
    path = info_dir / f"{split_name}.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def split_summary(info_dir: Path, subject_ids: set[str]) -> dict:
    if not (info_dir / "data_info.csv").exists():
        return {}
    info = pd.read_csv(info_dir / "data_info.csv", dtype={"ID": str})
    info = info[info["ID"].isin(subject_ids)]
    summary = {}
    for split in ("train", "test", "validation"):
        split_ids = set(read_split_ids(info_dir, split))
        present = split_ids & subject_ids
        summary[split] = {"expected": len(split_ids), "present_in_datav2": len(present)}
    if "label" in info.columns:
        summary["label_counts"] = info["label"].value_counts().astype(int).to_dict()
    return summary


def evaluate_subject(
    reference_data: Path,
    data_root: Path,
    subject_id: str,
    *,
    check_hash: bool,
) -> dict:
    row = {"id": subject_id, "status": "ok", "modalities": {}, "errors": []}
    for modality in MODALITIES:
        entry = {"modality": modality}
        try:
            ref_link = volume_path(reference_data, modality, subject_id)
            source = resolve_source(ref_link)
            dest = volume_path(data_root, modality, subject_id)

            if dest.is_symlink():
                raise ValueError(f"Expected regular file, got symlink: {dest}")
            if not dest.is_file():
                raise FileNotFoundError(f"Missing copy: {dest}")

            entry["dest_path"] = str(dest)
            entry["source_path"] = str(source)
            entry["dest_size"] = dest.stat().st_size
            entry["source_size"] = source.stat().st_size

            if entry["dest_size"] != entry["source_size"]:
                raise ValueError(
                    f"Size mismatch dest={entry['dest_size']} source={entry['source_size']}"
                )

            if check_hash:
                entry["md5_match"] = file_md5(dest) == file_md5(source)
                if not entry["md5_match"]:
                    raise ValueError("MD5 mismatch vs resolved source")

            entry.update(nifti_metadata(dest, compute_stats=False))
        except Exception as exc:
            entry["error"] = str(exc)
            row["errors"].append(f"{modality}: {exc}")
            row["status"] = "error"
        row["modalities"][modality] = entry
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--reference-data", type=Path, default=ROOT / "data")
    parser.add_argument("--info-dir", type=Path, default=ROOT / "data_info")
    parser.add_argument(
        "--check-hash",
        action="store_true",
        help="Compare MD5 with resolved source (slower)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "reports" / "datav2_evaluation.json",
    )
    args = parser.parse_args()

    reference_data = args.reference_data.resolve()
    data_root = args.data_root.resolve()
    info_dir = args.info_dir.resolve()

    ref_ids = set(list_subject_ids(reference_data))
    try:
        v2_ids = set(list_subject_ids(data_root))
    except Exception as exc:
        print(f"Cannot list dataV2 IDs: {exc}")
        return 1

    missing = sorted(ref_ids - v2_ids)
    extra = sorted(v2_ids - ref_ids)

    subjects = [
        evaluate_subject(reference_data, data_root, sid, check_hash=args.check_hash)
        for sid in sorted(ref_ids)
    ]
    failed = [s for s in subjects if s["status"] != "ok"]

    report = {
        "reference_data": str(reference_data),
        "data_root": str(data_root),
        "n_reference_ids": len(ref_ids),
        "n_datav2_ids": len(v2_ids),
        "missing_in_datav2": missing,
        "extra_in_datav2": extra,
        "n_ok": len(subjects) - len(failed),
        "n_failed": len(failed),
        "failed_ids": [s["id"] for s in failed],
        "split_summary": split_summary(info_dir, v2_ids),
        "subjects": subjects,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Reference IDs: {len(ref_ids)} | dataV2 IDs: {len(v2_ids)}")
    print(f"Missing: {len(missing)} | Extra: {len(extra)} | Failed checks: {len(failed)}")
    print(f"Report: {args.out}")

    if missing or extra or failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
