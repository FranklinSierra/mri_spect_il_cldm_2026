"""Load basal ganglia ROI mask for IL-CLDM crop space (no z-score)."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from dataset import crop

CROP_SHAPE = (160, 192, 160)
LATENT_SHAPE = (40, 48, 40)


def load_roi_mask_volume(path: str | Path) -> np.ndarray:
    """Load binary ROI volume in native array layout (same as MRI/SPECT get_fdata)."""
    vol = nib.load(str(path)).get_fdata(dtype=np.float32)
    mask = (vol > 0.5).astype(np.float32)
    if mask.sum() == 0:
        raise ValueError(f"ROI mask has no active voxels: {path}")
    return mask


def crop_roi_mask(vol: np.ndarray) -> np.ndarray:
    return crop(vol)


def load_roi_mask_cropped(path: str | Path) -> np.ndarray:
    mask = crop_roi_mask(load_roi_mask_volume(path))
    if mask.shape != CROP_SHAPE:
        raise ValueError(f"Expected cropped ROI shape {CROP_SHAPE}, got {mask.shape}")
    if mask.sum() == 0:
        raise ValueError(f"ROI mask empty after crop: {path}")
    return mask


def load_roi_mask_tensor(
    device: torch.device | str,
    path: str | Path | None = None,
) -> torch.Tensor:
    """Return (1, 1, D, H, W) float mask on device."""
    import config

    mask_path = Path(path or config.ROI_MASK_PATH)
    if not mask_path.is_file():
        raise FileNotFoundError(f"ROI mask not found: {mask_path}")
    mask = load_roi_mask_cropped(mask_path)
    return torch.as_tensor(mask, device=device, dtype=torch.float32).view(1, 1, *CROP_SHAPE)


def downsample_roi_to_latent(mask_cropped: np.ndarray) -> np.ndarray:
    """Downsample cropped ROI (160,192,160) to latent grid (40,48,40) via max-pool."""
    tensor = torch.as_tensor(mask_cropped, dtype=torch.float32).view(1, 1, *mask_cropped.shape)
    tensor = F.max_pool3d(tensor, kernel_size=2, stride=2)
    tensor = F.max_pool3d(tensor, kernel_size=2, stride=2)
    out = tensor.squeeze(0).squeeze(0).cpu().numpy()
    if tuple(out.shape) != LATENT_SHAPE:
        raise ValueError(f"Expected latent ROI shape {LATENT_SHAPE}, got {out.shape}")
    return out


def load_roi_latent_buffer(path: str | Path | None = None) -> torch.Tensor:
    """Return (1, 1, 40, 48, 40) ROI mask for UNet input channel."""
    import config

    mask_path = Path(path or config.ROI_MASK_PATH)
    mask = load_roi_mask_cropped(mask_path)
    latent = downsample_roi_to_latent(mask)
    return torch.as_tensor(latent, dtype=torch.float32).view(1, 1, *LATENT_SHAPE)


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


def try_load_roi_mask_numpy() -> np.ndarray | None:
    """Load cropped ROI mask for metrics, or None if missing."""
    import config

    mask_path = Path(config.ROI_MASK_PATH)
    if not mask_path.is_file():
        return None
    try:
        return load_roi_mask_cropped(mask_path)
    except ValueError as exc:
        print(f"Warning: ROI mask invalid ({exc}); ROI metrics disabled.")
        return None
