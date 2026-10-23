"""Differentiable SSIM for stage-2 image-space diffusion loss (PASTA-style)."""

from __future__ import annotations

from math import exp

import torch
import torch.nn.functional as F


def _gaussian_1d(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32)
    gauss = torch.exp(-((coords - window_size // 2) ** 2) / (2.0 * sigma**2))
    return gauss / gauss.sum()


def create_window_3d(window_size: int, channel: int, sigma: float = 1.5) -> torch.Tensor:
    """Separable 3D Gaussian window shaped ``(C, 1, W, W, W)`` for grouped conv3d."""
    g1 = _gaussian_1d(window_size, sigma).unsqueeze(1)
    g2 = g1.mm(g1.t())
    g3 = g1.mm(g2.reshape(1, -1)).reshape(window_size, window_size, window_size)
    window = g3.float().unsqueeze(0).unsqueeze(0)
    return window.expand(channel, 1, window_size, window_size, window_size).contiguous()


def ssim3d(
    img1: torch.Tensor,
    img2: torch.Tensor,
    *,
    window_size: int = 11,
    data_range: float = 2.0,
    size_average: bool = True,
) -> torch.Tensor:
    """Mean SSIM over a 5D batch ``(N, C, D, H, W)``.

    ``data_range`` should match the intensity span (minmax decoder tanh → 2.0).
    Constants follow Wang et al. scaled by ``data_range`` (PASTA uses range=1).
    """
    if img1.shape != img2.shape:
        raise ValueError(f"SSIM shape mismatch: {tuple(img1.shape)} vs {tuple(img2.shape)}")
    if img1.ndim != 5:
        raise ValueError(f"ssim3d expects 5D (N,C,D,H,W), got {tuple(img1.shape)}")

    channel = img1.shape[1]
    window = create_window_3d(window_size, channel).to(device=img1.device, dtype=img1.dtype)
    padding = window_size // 2

    mu1 = F.conv3d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv3d(img2, window, padding=padding, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(dim=(1, 2, 3, 4))


def ssim_loss_3d(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 11,
    data_range: float = 2.0,
) -> torch.Tensor:
    """``1 - SSIM`` for use as a training loss term (PASTA adds this unweighted)."""
    return 1.0 - ssim3d(
        prediction,
        target,
        window_size=window_size,
        data_range=data_range,
        size_average=True,
    )
