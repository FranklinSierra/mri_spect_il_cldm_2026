"""Volume / latent grid shapes for IL-CLDM (legacy full-brain vs ROI patches)."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

LEGACY_VOLUME_SHAPE = (160, 192, 160)
LEGACY_LATENT_SHAPE = (40, 48, 40)

_SHAPE_CACHE: dict[str, tuple[int, int, int]] = {}


def parse_shape_triplet(value: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in value.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated integers, got {value!r}")
    return tuple(parts)  # type: ignore[return-value]


def volume_to_latent_shape(volume_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Match encoder output / two stride-2 max-pools on the ROI mask."""
    tensor = torch.zeros(1, 1, *volume_shape)
    tensor = F.max_pool3d(tensor, kernel_size=2, stride=2)
    tensor = F.max_pool3d(tensor, kernel_size=2, stride=2)
    return tuple(int(x) for x in tensor.shape[2:])


def _aae_preserves_shape(volume_shape: tuple[int, int, int]) -> bool:
    from model import AAE

    model = AAE()
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 1, *volume_shape)
        y = model(x)
    return tuple(y.shape[2:]) == volume_shape


def resolve_working_volume_shape(raw_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Pad each axis so AAE encoder+decoder preserves spatial size."""
    if _aae_preserves_shape(raw_shape):
        return raw_shape

    padded = list(raw_shape)
    for _ in range(32):
        if _aae_preserves_shape(tuple(padded)):
            return tuple(padded)
        with torch.no_grad():
            from model import AAE

            model = AAE()
            model.eval()
            y = model(torch.zeros(1, 1, *padded))
        out_shape = tuple(int(x) for x in y.shape[2:])
        for axis in range(3):
            if out_shape[axis] < padded[axis]:
                padded[axis] += 1
            elif out_shape[axis] > padded[axis]:
                padded[axis] += out_shape[axis] - padded[axis]
    raise ValueError(f"Could not find AAE-safe padding for volume shape {raw_shape}")


def pad_volume_to_working_shape(
    volume: np.ndarray,
    raw_shape: tuple[int, int, int] | None = None,
    working_shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Zero-pad trailing edges so the array matches the working pipeline shape."""
    raw_shape = raw_shape or tuple(int(x) for x in volume.shape[:3])
    working_shape = working_shape or get_working_volume_shape()
    if raw_shape == working_shape:
        return volume
    out = np.zeros(working_shape, dtype=volume.dtype)
    out[: raw_shape[0], : raw_shape[1], : raw_shape[2]] = volume
    return out


def crop_volume_to_raw_shape(
    volume: np.ndarray,
    raw_shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Drop trailing zero-padding from the AAE/UNet working grid."""
    raw_shape = raw_shape or get_volume_shape()
    if tuple(int(x) for x in volume.shape[:3]) == raw_shape:
        return volume
    return volume[: raw_shape[0], : raw_shape[1], : raw_shape[2]]


def get_volume_shape() -> tuple[int, int, int]:
    import config

    key = "volume"
    if key in _SHAPE_CACHE:
        return _SHAPE_CACHE[key]

    if getattr(config, "ROI_VOLUME_SHAPE", None):
        shape = tuple(int(x) for x in config.ROI_VOLUME_SHAPE)
    elif config.SKIP_CROP:
        ref = Path(config.path)
        if not ref.is_file():
            mri_root = Path(config.whole_MRI)
            candidates = sorted(mri_root.glob("*.nii.gz"))
            if not candidates:
                raise FileNotFoundError(f"No NIfTI in {mri_root} to infer ROI volume shape")
            ref = candidates[0]
        shape = tuple(int(x) for x in nib.load(str(ref)).shape)
    else:
        shape = LEGACY_VOLUME_SHAPE

    _SHAPE_CACHE[key] = shape
    return shape


def get_working_volume_shape() -> tuple[int, int, int]:
    """Pipeline tensor shape after optional ROI padding for AAE/UNet."""
    import config

    key = "working_volume"
    if key in _SHAPE_CACHE:
        return _SHAPE_CACHE[key]

    if getattr(config, "ROI_VOLUME_SHAPE", None):
        shape = resolve_working_volume_shape(get_volume_shape())
    else:
        shape = get_volume_shape()

    _SHAPE_CACHE[key] = shape
    return shape


def get_latent_shape() -> tuple[int, int, int]:
    import config

    key = "latent"
    if key in _SHAPE_CACHE:
        return _SHAPE_CACHE[key]

    if getattr(config, "ROI_VOLUME_SHAPE", None):
        shape = volume_to_latent_shape(get_working_volume_shape())
    else:
        shape = LEGACY_LATENT_SHAPE

    _SHAPE_CACHE[key] = shape
    return shape


def clear_shape_cache() -> None:
    _SHAPE_CACHE.clear()
