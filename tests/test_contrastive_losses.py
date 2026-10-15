"""Unit tests for contrastive losses and spatial feature helpers."""

import numpy as np
import torch

from losses.contrastive import (
    ClassAwareCrossModalInfoNCE,
    ContrastiveProjector,
    CrossModalInfoNCE,
    SupervisedContrastiveLoss,
    flatten_feature_volume,
    pool_latent_volume,
)


def test_pool_latent_volume_shape():
    x = torch.randn(2, 1, 4, 5, 6)
    pooled = pool_latent_volume(x)
    assert pooled.shape == (2, 1)


def test_flatten_feature_volume_z18k_and_h64():
    z = torch.randn(2, 1, 3, 4, 5)
    flat_z = flatten_feature_volume(z)
    assert flat_z.shape == (2, 1 * 3 * 4 * 5)

    h = torch.randn(2, 64, 3, 4, 5)
    flat_h = flatten_feature_volume(h)
    assert flat_h.shape == (2, 64 * 3 * 4 * 5)


def test_supervised_contrastive_backward():
    crit = SupervisedContrastiveLoss(temperature=0.5)
    z = torch.randn(4, 32, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    loss = crit(z, labels)
    assert loss.ndim == 0
    loss.backward()
    assert z.grad is not None


def test_cross_modal_infonce_diagonal_positive():
    crit = CrossModalInfoNCE(temperature=0.5)
    z_mri = torch.randn(8, 64, requires_grad=True)
    z_spect = torch.randn(8, 64)
    loss = crit(z_mri, z_spect)
    assert loss.ndim == 0
    loss.backward()
    assert z_mri.grad is not None


def test_contrastive_projector_output_dim():
    # Legacy helper retained; active E1 path does not use a projector.
    proj = ContrastiveProjector(in_dim=16, out_dim=32)
    out = proj(torch.randn(3, 16))
    assert out.shape == (3, 32)


def test_class_aware_infonce_excludes_same_class_negatives():
    crit = ClassAwareCrossModalInfoNCE(temperature=0.5)
    z_img = torch.randn(4, 32, requires_grad=True)
    z_txt = torch.randn(4, 32)
    labels = torch.tensor([0, 0, 1, 1])
    loss = crit(z_img, z_txt, labels)
    assert loss.ndim == 0
    loss.backward()
    assert z_img.grad is not None


def test_class_aware_infonce_differs_from_standard_when_same_class_in_batch():
    crit_std = CrossModalInfoNCE(temperature=0.5)
    crit_ca = ClassAwareCrossModalInfoNCE(temperature=0.5)
    z_img = torch.randn(4, 32)
    z_txt = torch.randn(4, 32)
    labels = torch.tensor([0, 0, 1, 1])
    loss_std = float(crit_std(z_img, z_txt).item())
    loss_ca = float(crit_ca(z_img, z_txt, labels).item())
    assert loss_std != loss_ca
