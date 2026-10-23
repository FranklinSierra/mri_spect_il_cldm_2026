"""Unit tests for differentiable 3D SSIM used by R4 image-space loss."""

import torch

from losses.ssim import ssim3d, ssim_loss_3d


def test_ssim3d_identical_near_one():
    x = torch.randn(2, 1, 24, 28, 32)
    score = ssim3d(x, x, window_size=7, data_range=2.0)
    assert score.item() > 0.99


def test_ssim_loss_identical_near_zero_mismatch_higher():
    x = torch.randn(1, 1, 24, 28, 32)
    loss_same = ssim_loss_3d(x, x, window_size=7, data_range=2.0)
    y = x + 1.5
    loss_diff = ssim_loss_3d(y, x, window_size=7, data_range=2.0)
    assert loss_same.item() < 0.02
    assert loss_diff.item() > loss_same.item()


def test_ssim_loss_has_finite_gradient():
    target = torch.randn(1, 1, 24, 28, 32)
    pred = torch.randn(1, 1, 24, 28, 32, requires_grad=True)
    loss = ssim_loss_3d(pred, target, window_size=7, data_range=2.0)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0
