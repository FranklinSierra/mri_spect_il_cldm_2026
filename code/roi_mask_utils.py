"""Load basal ganglia ROI mask for IL-CLDM crop space (no z-score)."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from code.volume_shapes import (
    LEGACY_LATENT_SHAPE,
    LEGACY_VOLUME_SHAPE,
    get_latent_shape,
    get_volume_shape,
    get_working_volume_shape,
    pad_volume_to_working_shape,
    volume_to_latent_shape,
)
from dataset import crop

# Legacy aliases (full-brain IL-CLDM crop).
CROP_SHAPE = LEGACY_VOLUME_SHAPE
LATENT_SHAPE = LEGACY_LATENT_SHAPE


def load_roi_mask_volume(path: str | Path) -> np.ndarray:
    """Load binary ROI volume in native array layout (same as MRI/SPECT get_fdata)."""
    vol = nib.load(str(path)).get_fdata(dtype=np.float32)
    mask = (vol > 0.5).astype(np.float32)
    if mask.sum() == 0:
        raise ValueError(f"ROI mask has no active voxels: {path}")
    return mask


def crop_roi_mask(vol: np.ndarray) -> np.ndarray:
    return crop(vol)


def load_roi_mask_for_pipeline(path: str | Path) -> np.ndarray:
    """Load mask aligned with current volume profile (legacy crop or ROI patch as-is)."""
    import config

    mask = load_roi_mask_volume(path)
    if not config.SKIP_CROP:
        mask = crop_roi_mask(mask)
    expected = get_volume_shape()
    if tuple(mask.shape) != expected:
        raise ValueError(f"Expected ROI mask shape {expected}, got {mask.shape} ({path})")
    if getattr(config, "ROI_VOLUME_SHAPE", None):
        mask = pad_volume_to_working_shape(mask, raw_shape=expected, working_shape=get_working_volume_shape())
    if mask.sum() == 0:
        raise ValueError(f"ROI mask empty: {path}")
    return mask


def load_roi_mask_cropped(path: str | Path) -> np.ndarray:
    """Legacy name: mask in pipeline volume space."""
    return load_roi_mask_for_pipeline(path)


def load_roi_mask_tensor(
    device: torch.device | str,
    path: str | Path | None = None,
) -> torch.Tensor:
    """Return (1, 1, D, H, W) float mask on device."""
    import config

    mask_path = Path(path or config.ROI_MASK_PATH)
    if not mask_path.is_file():
        raise FileNotFoundError(f"ROI mask not found: {mask_path}")
    mask = load_roi_mask_for_pipeline(mask_path)
    vol_shape = get_working_volume_shape()
    return torch.as_tensor(mask, device=device, dtype=torch.float32).view(1, 1, *vol_shape)


def downsample_roi_to_latent(mask_volume: np.ndarray) -> np.ndarray:
    """Downsample volume-space mask to latent grid (two stride-2 max-pools)."""
    expected_latent = volume_to_latent_shape(mask_volume.shape)
    tensor = torch.as_tensor(mask_volume, dtype=torch.float32).view(1, 1, *mask_volume.shape)
    tensor = F.max_pool3d(tensor, kernel_size=2, stride=2)
    tensor = F.max_pool3d(tensor, kernel_size=2, stride=2)
    out = tensor.squeeze(0).squeeze(0).cpu().numpy()
    if tuple(out.shape) != expected_latent:
        raise ValueError(f"Expected latent ROI shape {expected_latent}, got {out.shape}")
    return out


def load_roi_latent_buffer(path: str | Path | None = None) -> torch.Tensor:
    """Return (1, 1, Dz, Dy, Dx) ROI mask for UNet input channel."""
    import config

    mask_path = Path(path or config.ROI_MASK_PATH)
    mask = load_roi_mask_for_pipeline(mask_path)
    latent = downsample_roi_to_latent(mask)
    latent_shape = get_latent_shape()
    return torch.as_tensor(latent, dtype=torch.float32).view(1, 1, *latent_shape)


def try_load_roi_mask_tensor(device: torch.device | str) -> torch.Tensor | None:
    """Load ROI tensor or return None if disabled / missing."""
    import config

    if not config.USE_ROI_LOSS:
        return None
    mask_path = Path(config.ROI_MASK_PATH)
    if not mask_path.is_file():
        print(f"Warning: USE_ROI_LOSS=1 but mask missing at {mask_path}; ROI disabled.")
        return None
    try:
        return load_roi_mask_tensor(device, mask_path)
    except ValueError as exc:
        print(f"Warning: ROI mask invalid ({exc}); ROI disabled.")
        return None


def try_load_diffusion_roi_latent(device: torch.device | str) -> torch.Tensor | None:
    """Latent-grid ROI mask for DIFFUSION_ROI_WEIGHT (primary x0 L1). None if off/missing."""
    import config

    weight = float(getattr(config, "DIFFUSION_ROI_WEIGHT", 0.0) or 0.0)
    if weight <= 0:
        return None
    mask_path = Path(config.ROI_MASK_PATH)
    if not mask_path.is_file():
        print(
            f"Warning: DIFFUSION_ROI_WEIGHT={weight} but mask missing at {mask_path}; "
            "falling back to plain L1."
        )
        return None
    try:
        return load_roi_latent_buffer(mask_path).to(device)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Warning: diffusion ROI latent invalid ({exc}); falling back to plain L1.")
        return None


def try_load_roi_mask_numpy() -> np.ndarray | None:
    """Load ROI mask for metrics, or None if missing."""
    import config

    mask_path = Path(config.ROI_MASK_PATH)
    if not mask_path.is_file():
        return None
    try:
        return load_roi_mask_for_pipeline(mask_path)
    except ValueError as exc:
        print(f"Warning: ROI mask invalid ({exc}); ROI metrics disabled.")
        return None
