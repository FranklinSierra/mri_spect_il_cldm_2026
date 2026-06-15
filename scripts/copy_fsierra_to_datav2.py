#!/usr/bin/env python3
"""Copy resolved NIfTI volumes from data/ symlinks into dataV2/ (byte-identical)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from datav2_common import MODALITIES, ROOT, list_subject_ids, resolve_source, volume_path


def ensure_layout(dest_root: Path) -> None:
    for sub in MODALITIES + ("latent_Abeta", "latent_MRI"):
        (dest_root / sub).mkdir(parents=True, exist_ok=True)


def copy_subject(
    reference_data: Path,
    dest_root: Path,
    subject_id: str,
    *,
    dry_run: bool,
    force: bool,
) -> list[tuple[Path, Path]]:
    copied = []
    for modality in MODALITIES:
        link_path = volume_path(reference_data, modality, subject_id)
        source = resolve_source(link_path)
        if not source.is_file():
            raise FileNotFoundError(f"Source not found for {subject_id} {modality}: {source}")
        dest = dest_root / modality / link_path.name
        if dest.exists() and not force:
            if dest.is_symlink():
                raise RuntimeError(f"Refusing to overwrite symlink: {dest}")
            if dest.stat().st_size == source.stat().st_size:
                continue
            raise FileExistsError(f"Destination exists (use --force): {dest}")
        if dry_run:
            print(f"[dry-run] {source} -> {dest}")
        else:
            shutil.copy2(source, dest)
        copied.append((source, dest))
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-data",
        type=Path,
        default=ROOT / "data",
        help="Symlink tree with whole_MRI/ and whole_Abeta/",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=ROOT / "data",
        help="Output root (default: data/)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    reference_data = args.reference_data.resolve()
    dest_root = args.dest.resolve()
    subject_ids = list_subject_ids(reference_data)

    if not args.dry_run:
        ensure_layout(dest_root)

    errors: list[tuple[str, str]] = []
    n_copied = 0
    for subject_id in subject_ids:
        try:
            pairs = copy_subject(
                reference_data,
                dest_root,
                subject_id,
                dry_run=args.dry_run,
                force=args.force,
            )
            n_copied += len(pairs)
        except Exception as exc:
            errors.append((subject_id, str(exc)))

    print(f"Reference: {reference_data}")
    print(f"Destination: {dest_root}")
    print(f"Subjects: {len(subject_ids)} | File copies: {n_copied} | Errors: {len(errors)}")
    if errors:
        for sid, msg in errors[:10]:
            print(f"  ERROR {sid}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
