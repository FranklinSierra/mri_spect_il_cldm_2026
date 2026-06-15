"""Visualize paired MRI / SPECT volumes with IL-CLDM crop and z-score friendly scaling."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

CROP_SLICES = (slice(10, 170), slice(18, 210), slice(10, 170))
CROP_SHAPE = (160, 192, 160)


def load_cropped(root: Path, subject_id: str, *, apply_pipeline_zscore: bool = True) -> np.ndarray:
    for suffix in (".nii.gz", ".nii"):
        path = root / f"{subject_id}{suffix}"
        if path.exists():
            if apply_pipeline_zscore:
                from dataset import nifti_to_numpy

                vol = nifti_to_numpy(str(path), zscore=True)
            else:
                vol = nib.load(str(path)).get_fdata(dtype=np.float32)
            return vol[CROP_SLICES]
    raise FileNotFoundError(f"No NIfTI for {subject_id} in {root}")


def brain_mask(vol: np.ndarray, abs_thresh: float = 0.05) -> np.ndarray:
    """Mask for z-score volumes: brain voxels deviate from background ~0."""
    return np.abs(vol) > abs_thresh


def display_range(
    vol: np.ndarray,
    *,
    symmetric: bool = True,
    percentiles: tuple[float, float] = (2.0, 98.0),
    mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Contrast limits for imshow.

    symmetric=True uses +/- p98(|v|) — appropriate for z-score data.
    Using only voxels > 0 fails on z-score MRI (much tissue is negative).
    """
    if mask is None:
        mask = brain_mask(vol)
    vals = vol[mask] if mask.any() else vol.ravel()
    if vals.size == 0:
        return float(vol.min()), float(vol.max())
    if symmetric:
        limit = float(np.percentile(np.abs(vals), percentiles[1]))
        limit = max(limit, 1e-6)
        return -limit, limit
    return float(np.percentile(vals, percentiles[0])), float(np.percentile(vals, percentiles[1]))


def show_slices(
    vol: np.ndarray,
    title: str,
    *,
    cmap: str = "gray",
    symmetric: bool = True,
    spect_hot: bool = False,
) -> plt.Figure:
    z, y, x = vol.shape[2] // 2, vol.shape[1] // 2, vol.shape[0] // 2
    slices = [
        ("axial", vol[:, :, z].T),
        ("coronal", vol[:, y, :].T),
        ("sagittal", vol[x, :, :].T),
    ]
    if spect_hot:
        vmin, vmax = display_range(vol, symmetric=False, percentiles=(5.0, 99.5))
        cmap = "hot"
    else:
        vmin, vmax = display_range(vol, symmetric=symmetric)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (plane, sl) in zip(axes, slices):
        im = ax.imshow(sl, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title} — {plane}")
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.suptitle(f"{title}  (display: [{vmin:.2f}, {vmax:.2f}])")
    plt.tight_layout()
    return fig


def show_axial_pair(
    mri: np.ndarray,
    spect: np.ndarray,
    subject_id: str,
    z: int | None = None,
) -> plt.Figure:
    if z is None:
        z = mri.shape[2] // 2
    mri_lim = display_range(mri, symmetric=True)
    spect_lim = display_range(spect, symmetric=False, percentiles=(5.0, 99.5))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(mri[:, :, z].T, cmap="gray", origin="lower", vmin=mri_lim[0], vmax=mri_lim[1])
    axes[0].set_title(f"MRI — axial z={z}")
    axes[0].axis("off")
    axes[1].imshow(spect[:, :, z].T, cmap="hot", origin="lower", vmin=spect_lim[0], vmax=spect_lim[1])
    axes[1].set_title(f"SPECT — axial z={z}")
    axes[1].axis("off")
    fig.suptitle(f"Sujeto {subject_id} — crop IL-CLDM (z-score display)")
    plt.tight_layout()
    return fig


def plot_subject(
    data_root: Path,
    subject_id: str,
    *,
    show: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    mri = load_cropped(data_root / "whole_MRI", subject_id)
    spect = load_cropped(data_root / "whole_Abeta", subject_id)
    show_slices(mri, f"MRI {subject_id}", symmetric=True)
    if show:
        plt.show()
    show_slices(spect, f"SPECT {subject_id}", spect_hot=True)
    if show:
        plt.show()
    show_axial_pair(mri, spect, subject_id)
    if show:
        plt.show()
    return mri, spect
