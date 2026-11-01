"""Lightweight on-GPU 3D augmentation for small-N volumetric classification.

The baseline MobileNet-3D scripts (train_synthetic_spect_mobilenet3d.py,
train_mobilenet_roi_classifier.py) train with no augmentation at all, which is a
likely source of overfitting on ~124 training volumes. These transforms are applied
batch-wise on the GPU (no CPU/scipy cost) and are intentionally conservative: small
rigid-ish jitter plus mild intensity noise, nothing that could plausibly flip the
clinical signal (e.g. no left-right flip, since striatal uptake asymmetry can itself
be diagnostic).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _rotation_matrix(angles_rad: torch.Tensor) -> torch.Tensor:
    """angles_rad: (B, 3) -> (B, 3, 3) rotation matrices (Rz @ Ry @ Rx)."""
    b = angles_rad.shape[0]
    device = angles_rad.device
    ax, ay, az = angles_rad[:, 0], angles_rad[:, 1], angles_rad[:, 2]
    one = torch.ones(b, device=device)
    zero = torch.zeros(b, device=device)

    cx, sx = torch.cos(ax), torch.sin(ax)
    rx = torch.stack(
        [one, zero, zero, zero, cx, -sx, zero, sx, cx], dim=1
    ).reshape(b, 3, 3)

    cy, sy = torch.cos(ay), torch.sin(ay)
    ry = torch.stack(
        [cy, zero, sy, zero, one, zero, -sy, zero, cy], dim=1
    ).reshape(b, 3, 3)

    cz, sz = torch.cos(az), torch.sin(az)
    rz = torch.stack(
        [cz, -sz, zero, sz, cz, zero, zero, zero, one], dim=1
    ).reshape(b, 3, 3)

    return rz @ ry @ rx


def random_affine_3d(
    x: torch.Tensor,
    max_degrees: float = 8.0,
    max_translate_frac: float = 0.04,
    max_scale_delta: float = 0.05,
    p: float = 0.8,
) -> torch.Tensor:
    """Random small rotation + isotropic scale + translation, independently per sample.

    x: (B, C, D, H, W). Returns a tensor of the same shape.
    """
    b = x.shape[0]
    device = x.device
    apply = (torch.rand(b, 1, device=device) < p).float()  # (B,1), 0 -> identity
    deg2rad = 3.141592653589793 / 180.0
    angles = (torch.rand(b, 3, device=device) * 2 - 1) * max_degrees * deg2rad * apply
    rot = _rotation_matrix(angles)  # (B,3,3)
    scale = 1.0 + (torch.rand(b, 1, 1, device=device) * 2 - 1) * max_scale_delta * apply.unsqueeze(-1)
    translate = (torch.rand(b, 3, device=device) * 2 - 1) * max_translate_frac * apply

    theta = rot * scale  # (B,3,3)
    theta = torch.cat([theta, translate.unsqueeze(-1)], dim=-1)  # (B,3,4)

    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(
        x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


def random_intensity(
    x: torch.Tensor,
    scale_delta: float = 0.1,
    noise_std_frac: float = 0.02,
    p: float = 0.8,
) -> torch.Tensor:
    """Per-sample multiplicative scale jitter + additive Gaussian noise.

    Noise std is relative to each sample's own std, so it adapts to both the
    SPECT-identity range (real data, ~[0, 2]) and the min-max synthetic range ([0, 1]).
    """
    b = x.shape[0]
    device = x.device
    apply = (torch.rand(b, 1, 1, 1, 1, device=device) < p).float()
    scale = 1.0 + (torch.rand(b, 1, 1, 1, 1, device=device) * 2 - 1) * scale_delta * apply
    x = x * scale
    sample_std = x.flatten(1).std(dim=1).view(b, 1, 1, 1, 1).clamp_min(1e-6)
    noise = torch.randn_like(x) * sample_std * noise_std_frac * apply
    return x + noise


def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Apply both augmentations; each sample independently decides whether/how."""
    x = random_affine_3d(x)
    x = random_intensity(x)
    return x
