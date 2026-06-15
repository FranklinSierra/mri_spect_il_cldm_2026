#!/usr/bin/env python3
"""Audit resolved NIfTI sources for IDs listed under data/ (no crop checks)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from datav2_common import (
    MODALITIES,
    ROOT,
    list_subject_ids,
    nifti_metadata,
    resolve_source,
    volume_path,
)


def audit_subject(reference_data: Path, subject_id: str, compute_stats: bool) -> dict:
    row = {"id": subject_id, "modalities": {}, "status": "ok", "errors": []}
    for modality in MODALITIES:
        entry = {"modality": modality}
        try:
            link_path = volume_path(reference_data, modality, subject_id)
            source = resolve_source(link_path)
            entry["link_path"] = str(link_path)
            entry["source_path"] = str(source)
            if not source.is_file():
                raise FileNotFoundError(f"Resolved source is not a file: {source}")
            entry.update(nifti_metadata(source, compute_stats=compute_stats))
        except Exception as exc:
            entry["error"] = str(exc)
            row["errors"].append(f"{modality}: {exc}")
            row["status"] = "error"
        row["modalities"][modality] = entry
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-data",
        type=Path,
        default=ROOT / "data",
        help="Directory with whole_MRI/ and whole_Abeta/ (default: data/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "reports" / "fsierra_source_audit.json",
    )
    parser.add_argument(
        "--no-voxel-stats",
        action="store_true",
        help="Skip min/max/mean (header and shape only)",
    )
    args = parser.parse_args()

    reference_data = args.reference_data.resolve()
    subject_ids = list_subject_ids(reference_data)

    subjects = [
        audit_subject(reference_data, sid, compute_stats=not args.no_voxel_stats)
        for sid in subject_ids
    ]
    ok = [s for s in subjects if s["status"] == "ok"]
    failed = [s for s in subjects if s["status"] != "ok"]

    report = {
        "reference_data": str(reference_data),
        "n_subjects": len(subject_ids),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "failed_ids": [s["id"] for s in failed],
        "subjects": subjects,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Reference: {reference_data}")
    print(f"Subjects: {len(subject_ids)} | OK: {len(ok)} | Failed: {len(failed)}")
    print(f"Report: {args.out}")
    if failed:
        for row in failed[:10]:
            print(f"  FAIL {row['id']}: {'; '.join(row['errors'])}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
