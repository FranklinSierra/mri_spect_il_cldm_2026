"""LatentClsHead forward and training-step smoke test."""

import torch
import torch.nn as nn

from model import LATENT_FLAT_DIM, LatentClsHead


def test_latent_cls_head_forward_shape():
    head = LatentClsHead()
    x0 = torch.randn(2, 1, 40, 48, 40)
    logits = head(x0)
    assert logits.shape == (2,)


def test_cls_loss_backward():
    head = LatentClsHead()
    x0 = torch.randn(2, 1, 40, 48, 40, requires_grad=True)
    labels = torch.tensor([0.0, 1.0])
    logits = head(x0)
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()
    assert x0.grad is not None
    assert any(p.grad is not None for p in head.parameters())


def test_flat_dim_constant():
    assert LATENT_FLAT_DIM == 40 * 48 * 40
