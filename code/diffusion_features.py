"""Gated E2 diffusion feature spaces: z18k | h64 | h64_gap1."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn

import config

FeatureSpace = Literal["z18k", "h64", "h64_gap1"]


def resolve_diffusion_feature_space(value: str | None = None) -> FeatureSpace:
    raw = (value if value is not None else getattr(config, "DIFFUSION_FEATURE_SPACE", "z18k"))
    space = str(raw).strip().lower()
    if space in ("", "z", "z18", "latent", "bottleneck"):
        return "z18k"
    if space in ("z18k", "h64", "h64_gap1"):
        return space  # type: ignore[return-value]
    raise ValueError(f"Unknown DIFFUSION_FEATURE_SPACE={raw!r}; expected z18k|h64|h64_gap1")


def diffusion_channels(space: FeatureSpace | None = None) -> int:
    space = space or resolve_diffusion_feature_space()
    return 64 if space == "h64" else 1


def unet_in_channels(space: FeatureSpace | None = None, use_roi: bool | None = None) -> int:
    space = space or resolve_diffusion_feature_space()
    use_roi = config.UNET_ROI_CHANNEL if use_roi is None else use_roi
    c = diffusion_channels(space)
    return 2 * c + int(bool(use_roi))


def channel_mean_to_one(volume: torch.Tensor) -> torch.Tensor:
    """Mean over channel dim → (B,1,D,H,W) or (1,D,H,W) preserving spatial grid."""
    if volume.ndim == 5:
        return volume.mean(dim=1, keepdim=True)
    if volume.ndim == 4:
        return volume.mean(dim=0, keepdim=True)
    raise ValueError(f"Expected 4D/5D tensor for channel GAP, got shape {tuple(volume.shape)}")


def ensure_batch_channels(volume: torch.Tensor | np.ndarray, channels: int) -> torch.Tensor:
    """Coerce numpy/torch latent to (B,C,D,H,W)."""
    if isinstance(volume, np.ndarray):
        volume = torch.as_tensor(volume, dtype=torch.float32)
    else:
        volume = volume.float()

    if volume.ndim == 3:
        # (D,H,W) → (1,1,D,H,W)
        volume = volume.unsqueeze(0).unsqueeze(0)
    elif volume.ndim == 4:
        # (C,D,H,W) or (B,D,H,W) — treat as (C,D,H,W) when C matches
        if volume.shape[0] == channels:
            volume = volume.unsqueeze(0)
        else:
            # (B,D,H,W) with C=1 implied
            volume = volume.unsqueeze(1)
    elif volume.ndim == 5:
        pass
    else:
        raise ValueError(f"Unexpected latent ndim={volume.ndim} shape={tuple(volume.shape)}")

    if volume.shape[1] != channels:
        raise ValueError(
            f"Latent channels {volume.shape[1]} != expected {channels} (shape={tuple(volume.shape)})"
        )
    return volume


def apply_feature_space_to_map(volume: torch.Tensor, space: FeatureSpace) -> torch.Tensor:
    """Map encoder outputs / cached h64 into the diffusion feature tensor."""
    if space == "h64":
        if volume.shape[1] != 64:
            raise ValueError(f"h64 expects 64 channels, got {tuple(volume.shape)}")
        return volume
    if space == "h64_gap1":
        if volume.shape[1] == 64:
            return channel_mean_to_one(volume)
        if volume.shape[1] == 1:
            return volume
        raise ValueError(f"h64_gap1 expects 64 or 1 channels, got {tuple(volume.shape)}")
    # z18k
    if volume.shape[1] != 1:
        raise ValueError(f"z18k expects 1 channel, got {tuple(volume.shape)}")
    return volume


def encode_mri_condition(mri_encoder: nn.Module, mri: torch.Tensor, space: FeatureSpace) -> torch.Tensor:
    """Frozen/finetune MRI encoder → condition map matching diffusion channels."""
    if space == "z18k":
        return mri_encoder(mri)
    latent, h64 = mri_encoder(mri, return_hidden=True)
    if space == "h64":
        return h64
    return channel_mean_to_one(h64)


def encode_mri_dual(
    mri_encoder: nn.Module,
    mri: torch.Tensor,
    space: FeatureSpace | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single encoder pass → (concat condition map, h64 map for spatial FiLM)."""
    space = space or resolve_diffusion_feature_space()
    latent, h64 = mri_encoder(mri, return_hidden=True)
    if space == "z18k":
        mri_cond = apply_feature_space_to_map(latent, space)
    elif space == "h64":
        mri_cond = h64
    else:
        mri_cond = channel_mean_to_one(h64)
    return mri_cond, h64


class SpectBottleneckProjector(nn.Module):
    """Bonus: map h64 → z18k using the frozen last Conv3d of a SPECT encoder."""

    def __init__(self, spect_encoder: nn.Module):
        super().__init__()
        last = spect_encoder.model[-1]
        if not isinstance(last, nn.Conv3d):
            raise TypeError(f"Expected Conv3d bottleneck, got {type(last)}")
        self.proj = nn.Conv3d(
            last.in_channels,
            last.out_channels,
            kernel_size=last.kernel_size,
            stride=last.stride,
            padding=last.padding,
            bias=last.bias is not None,
        )
        self.proj.load_state_dict(last.state_dict())
        self.proj.requires_grad_(False)

    def forward(self, h64: torch.Tensor) -> torch.Tensor:
        return self.proj(h64)


def batch_latent_to_tensor(latent_batch, channels: int, device) -> torch.Tensor:
    """Collated dataset latent → (B,C,D,H,W) on device."""
    t = torch.as_tensor(latent_batch, device=device, dtype=torch.float32)
    if t.ndim == 4:
        # (B,D,H,W) classic NIfTI collation
        t = t.unsqueeze(1)
    if t.ndim != 5:
        raise ValueError(f"Expected latent batch 4D/5D, got {tuple(t.shape)}")
    if t.shape[1] != channels:
        raise ValueError(
            f"Latent batch channels {t.shape[1]} != DIFFUSION_CHANNELS={channels} "
            f"(shape={tuple(t.shape)})"
        )
    return t


def decode_h64_via_bottleneck(aae: nn.Module, h64: torch.Tensor) -> torch.Tensor:
    """Optional SPECT image decode: h64 → Conv64→1 → decoder."""
    projector = SpectBottleneckProjector(aae.encoder)
    projector.to(h64.device)
    z18k = projector(h64)
    return aae.decoder(z18k)
