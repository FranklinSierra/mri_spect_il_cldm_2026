"""Supervised contrastive and cross-modal InfoNCE losses for E1 MRI training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def flatten_feature_volume(volume: torch.Tensor) -> torch.Tensor:
    """Flatten spatial (and channel) dims: (B, ...) -> (B, D)."""
    if volume.ndim < 2:
        raise ValueError(f"Expected batched tensor, got shape {tuple(volume.shape)}")
    return volume.reshape(volume.shape[0], -1)


def pool_latent_volume(latent: torch.Tensor) -> torch.Tensor:
    """Global average pool over spatial dims: (B, C, D, H, W) -> (B, C)."""
    if latent.ndim != 5:
        raise ValueError(f"Expected 5D latent (B,C,D,H,W), got shape {tuple(latent.shape)}")
    return latent.flatten(2).mean(dim=2)


class ContrastiveProjector(nn.Module):
    """MLP projector for contrastive embeddings."""

    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al., 2020)."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"features must be (N, D), got {tuple(features.shape)}")
        labels = labels.reshape(-1).long()
        n = features.shape[0]
        if n < 2:
            return features.new_zeros(())

        features = l2_normalize(features.float())
        sim = torch.matmul(features, features.T) / self.temperature
        logits_mask = torch.ones((n, n), device=features.device, dtype=torch.bool)
        logits_mask.fill_diagonal_(False)

        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = labels_eq & logits_mask
        neg_mask = ~labels_eq & logits_mask

        losses = []
        for i in range(n):
            pos = pos_mask[i]
            if not pos.any():
                continue
            denom_logits = sim[i][logits_mask[i]]
            log_prob = sim[i][pos] - torch.logsumexp(denom_logits, dim=0)
            losses.append(-log_prob.mean())

        if not losses:
            return features.new_zeros(())
        return torch.stack(losses).mean()


class CrossModalInfoNCE(nn.Module):
    """InfoNCE aligning MRI embeddings to SPECT embeddings (diagonal = same subject)."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_mri: torch.Tensor, z_spect: torch.Tensor) -> torch.Tensor:
        if z_mri.shape != z_spect.shape or z_mri.ndim != 2:
            raise ValueError(
                f"z_mri and z_spect must match shape (N, D); got {tuple(z_mri.shape)} vs {tuple(z_spect.shape)}"
            )
        n = z_mri.shape[0]
        if n < 1:
            return z_mri.new_zeros(())

        z_mri = l2_normalize(z_mri.float())
        z_spect = l2_normalize(z_spect.float())
        logits = torch.matmul(z_mri, z_spect.T) / self.temperature
        targets = torch.arange(n, device=z_mri.device, dtype=torch.long)
        return F.cross_entropy(logits, targets)


class ClassAwareCrossModalInfoNCE(nn.Module):
    """Cross-modal InfoNCE with soft-negative masking: same-class subjects excluded from denominator."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_img: torch.Tensor,
        z_txt: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if z_img.shape != z_txt.shape or z_img.ndim != 2:
            raise ValueError(
                f"z_img and z_txt must match shape (N, D); got {tuple(z_img.shape)} vs {tuple(z_txt.shape)}"
            )
        n = z_img.shape[0]
        if n < 1:
            return z_img.new_zeros(())

        z_img = l2_normalize(z_img.float())
        z_txt = l2_normalize(z_txt.float())
        logits = torch.matmul(z_img, z_txt.T) / self.temperature
        labels = labels.reshape(-1).long()

        losses: list[torch.Tensor] = []
        for i in range(n):
            pos = logits[i, i]
            not_self = torch.arange(n, device=z_img.device) != i
            cross_class = labels != labels[i]
            neg_mask = not_self & cross_class
            if not neg_mask.any():
                neg_mask = not_self
            if not neg_mask.any():
                continue
            denom = torch.cat([pos.unsqueeze(0), logits[i][neg_mask]])
            losses.append(-pos + torch.logsumexp(denom, dim=0))

        if not losses:
            return z_img.new_zeros(())
        return torch.stack(losses).mean()
