"""Shared helpers for dataV2 inspect / copy / evaluate scripts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ("whole_MRI", "whole_Abeta")


def nifti_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return path.stem


def list_subject_ids(reference_data: Path) -> list[str]:
    mri_dir = reference_data / "whole_MRI"
    if not mri_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {mri_dir}")
    ids = sorted(nifti_stem(p) for p in mri_dir.glob("*.nii*"))
    if not ids:
        raise RuntimeError(f"No NIfTI files found in {mri_dir}")
    return ids


def volume_path(reference_data: Path, modality: str, subject_id: str) -> Path:
    for suffix in (".nii.gz", ".nii"):
        path = reference_data / modality / f"{subject_id}{suffix}"
        if path.exists() or path.is_symlink():
            return path
    raise FileNotFoundError(
        f"No NIfTI for ID '{subject_id}' in {reference_data / modality}"
    )


def resolve_source(path: Path) -> Path:
    return path.resolve(strict=False)


def nifti_metadata(path: Path, compute_stats: bool = True) -> dict:
    img = nib.load(str(path))
    header = img.header
    out = {
        "shape": list(img.shape),
        "zooms": [float(z) for z in header.get_zooms()[:3]],
        "dtype": str(img.get_data_dtype()),
        "file_size_bytes": path.stat().st_size,
    }
    if compute_stats:
        data = np.asanyarray(img.dataobj, dtype=np.float32)
        out["min"] = float(np.nanmin(data))
        out["max"] = float(np.nanmax(data))
        out["mean"] = float(np.nanmean(data))
        out["has_nan"] = bool(np.isnan(data).any())
    return out


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
