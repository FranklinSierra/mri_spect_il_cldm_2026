"""Intensity-aware reconstruction losses for z-score SPECT/MRI volumes."""

from __future__ import annotations

import torch


def uptake_weights(
    target: torch.Tensor,
    alpha: float = 3.0,
    z_threshold: float = 1.0,
) -> torch.Tensor:
    """Higher weight where target z-score exceeds z_threshold."""
    return 1.0 + alpha * torch.relu(target - z_threshold)


def combined_spect_weights(
    target: torch.Tensor,
    *,
    weighted_alpha: float = 3.0,
    z_threshold: float = 1.0,
    roi_mask: torch.Tensor | None = None,
    roi_alpha: float = 0.0,
) -> torch.Tensor:
    """Base weight 1 + peak-aware uptake + optional ROI boost."""
    weights = uptake_weights(target, alpha=weighted_alpha, z_threshold=z_threshold)
    if roi_mask is not None and roi_alpha > 0:
        weights = weights + roi_alpha * roi_mask
    return weights


def roi_weighted_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    roi_mask: torch.Tensor | None,
    roi_weight: float,
) -> torch.Tensor:
    """Mean L1 with outside weight 1 and ROI weight ``roi_weight``.

    ``roi_mask`` is broadcast to ``prediction`` (e.g. latent 1×1×Dz×Dy×Dx).
    When ``roi_weight <= 0`` or ``roi_mask`` is None, falls back to plain L1.
    """
    err = torch.abs(prediction - target)
    if roi_mask is None or float(roi_weight) <= 0:
        return err.mean()
    w = 1.0 + (float(roi_weight) - 1.0) * roi_mask
    return (w * err).mean()


def weighted_l1_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    alpha: float = 3.0,
    z_threshold: float = 1.0,
    roi_mask: torch.Tensor | None = None,
    roi_alpha: float = 0.0,
) -> torch.Tensor:
    weights = combined_spect_weights(
        target,
        weighted_alpha=alpha,
        z_threshold=z_threshold,
        roi_mask=roi_mask,
        roi_alpha=roi_alpha,
    )
    return (weights * torch.abs(target - prediction)).mean()


def tail_l1_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    top_fraction: float = 0.05,
) -> torch.Tensor:
    """L1 on voxels in the top fraction of target intensities (per volume)."""
    batch = target.shape[0]
    flat = target.flatten(1)
    k = max(1, int(top_fraction * flat.shape[1]))
    threshold = torch.topk(flat, k, dim=1).values.min(dim=1, keepdim=True).values
    threshold = threshold.view(batch, *([1] * (target.ndim - 1)))
    mask = target >= threshold
    if not mask.any():
        return torch.zeros((), device=target.device, dtype=target.dtype)
    return torch.abs(target[mask] - prediction[mask]).mean()


def spect_reconstruction_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    use_weighted: bool = True,
    use_tail: bool = True,
    weighted_alpha: float = 3.0,
    z_threshold: float = 1.0,
    tail_fraction: float = 0.05,
    tail_beta: float = 1.0,
    roi_mask: torch.Tensor | None = None,
    roi_alpha: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combined image-space loss for SPECT decode (peak-aware + optional ROI)."""
    parts: dict[str, torch.Tensor] = {}
    if use_weighted:
        parts["weighted_l1"] = weighted_l1_loss(
            target,
            prediction,
            alpha=weighted_alpha,
            z_threshold=z_threshold,
            roi_mask=roi_mask,
            roi_alpha=roi_alpha,
        )
        total = parts["weighted_l1"]
    else:
        if roi_mask is not None and roi_alpha > 0:
            weights = 1.0 + roi_alpha * roi_mask
            parts["l1"] = (weights * torch.abs(target - prediction)).mean()
        else:
            parts["l1"] = torch.abs(target - prediction).mean()
        total = parts["l1"]

    if use_tail:
        parts["tail_l1"] = tail_l1_loss(target, prediction, top_fraction=tail_fraction)
        total = total + tail_beta * parts["tail_l1"]

    return total, parts
