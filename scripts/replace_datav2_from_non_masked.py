#!/usr/bin/env python3
"""Replace dataV2 volumes with temp_data/non_masked (IL-CLDM folder layout)."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "temp_data" / "non_masked"
DEST_ROOT = ROOT / "data"
MODALITY_MAP = {
    "whole_MRI": ("mri",),
    "whole_Abeta": ("spect",),
}


def nifti_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return path.stem


def extract_id_from_non_masked(path: Path) -> str:
    m = re.match(r"(\d+)_", path.name)
    if not m:
        raise ValueError(f"Cannot parse ID from {path}")
    return m.group(1)


def build_source_index(source_root: Path) -> dict[tuple[str, str], Path]:
    """Map (modality_folder, subject_id) -> source path. modality_folder is mri or spect."""
    index: dict[tuple[str, str], Path] = {}
    for path in sorted(source_root.rglob("*.nii.gz")):
        parts = path.parts
        if "mri" in parts:
            folder = "mri"
        elif "spect" in parts:
            folder = "spect"
        else:
            continue
        sid = extract_id_from_non_masked(path)
        key = (folder, sid)
        if key in index:
            raise RuntimeError(f"Duplicate source for {key}: {index[key]} vs {path}")
        index[key] = path
    return index


def list_dest_ids(dest_root: Path) -> list[str]:
    mri_dir = dest_root / "whole_MRI"
    if not mri_dir.is_dir():
        raise FileNotFoundError(f"Missing {mri_dir}")
    return sorted(nifti_stem(p) for p in mri_dir.glob("*.nii*"))


def clear_directory(dir_path: Path) -> int:
    removed = 0
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        return 0
    for item in dir_path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
            removed += 1
        elif item.is_dir():
            shutil.rmtree(item)
            removed += 1
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--dest", type=Path, default=DEST_ROOT)
    parser.add_argument(
        "--ids-from-dest",
        action="store_true",
        default=True,
        help="Use subject IDs currently listed under dest/whole_MRI (default)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_root = args.source.resolve()
    dest_root = args.dest.resolve()
    if not source_root.is_dir():
        print(f"Source not found: {source_root}", file=sys.stderr)
        return 1

    index = build_source_index(source_root)
    subject_ids = list_dest_ids(dest_root)
    print(f"Subjects to copy: {len(subject_ids)}")
    print(f"Source: {source_root}")
    print(f"Dest: {dest_root}")

    errors: list[str] = []
    copies: list[tuple[Path, Path]] = []

    for sid in subject_ids:
        for dest_mod, src_folders in MODALITY_MAP.items():
            src_path = None
            for folder in src_folders:
                candidate = index.get((folder, sid))
                if candidate is not None:
                    src_path = candidate
                    break
            if src_path is None:
                errors.append(f"{sid}: missing {src_folders} in non_masked")
                continue
            dest_path = dest_root / dest_mod / f"{sid}.nii.gz"
            copies.append((src_path, dest_path))

    if errors:
        print("Missing sources:", file=sys.stderr)
        for e in errors[:20]:
            print(f"  {e}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)
        return 1

    if args.dry_run:
        for src, dst in copies[:5]:
            print(f"[dry-run] {src} -> {dst}")
        print(f"... total {len(copies)} files")
        return 0

    for sub in ("whole_MRI", "whole_Abeta", "latent_Abeta", "latent_MRI"):
        n = clear_directory(dest_root / sub)
        print(f"Cleared {sub}/ ({n} items removed)")

    for src, dst in copies:
        shutil.copy2(src, dst)

    n_mri = len(list((dest_root / "whole_MRI").glob("*.nii.gz")))
    n_spect = len(list((dest_root / "whole_Abeta").glob("*.nii.gz")))
    print(f"Done. whole_MRI: {n_mri} | whole_Abeta: {n_spect}")
    return 0 if n_mri == n_spect == len(subject_ids) else 1


if __name__ == "__main__":
    sys.exit(main())
