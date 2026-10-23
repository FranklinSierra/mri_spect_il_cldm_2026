import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from losses import spect_reconstruction_loss, combined_spect_weights
from losses.spect import roi_weighted_l1


def test_roi_weighted_l1_boosts_roi_error():
    pred = torch.zeros(1, 1, 2, 2, 2)
    target = torch.zeros_like(pred)
    target[0, 0, 0, 0, 0] = 1.0
    roi = torch.zeros_like(pred)
    roi[0, 0, 0, 0, 0] = 1.0
    plain = roi_weighted_l1(pred, target, None, 0.0)
    weighted = roi_weighted_l1(pred, target, roi, 10.0)
    assert abs(plain.item() - (1.0 / 8.0)) < 1e-6
    assert weighted.item() > plain.item()
    assert abs(weighted.item() - (10.0 / 8.0)) < 1e-6


def test_roi_weighted_l1_off_matches_plain():
    pred = torch.randn(2, 1, 4, 4, 4)
    target = torch.randn_like(pred)
    roi = torch.ones_like(pred)
    assert abs(roi_weighted_l1(pred, target, roi, 0.0).item() - torch.abs(pred - target).mean().item()) < 1e-6


def test_combined_spect_weights_roi_boost():
    target = torch.ones(1, 1, 2, 2, 2)
    roi = torch.zeros_like(target)
    roi[0, 0, 0, 0, 0] = 1.0
    w = combined_spect_weights(target, roi_mask=roi, roi_alpha=3.0)
    assert w[0, 0, 0, 0, 0].item() > w[0, 0, 1, 1, 1].item()


def test_roi_weighted_spect_reconstruction_loss():
    target = torch.zeros(1, 1, 4, 4, 4)
    pred = torch.zeros_like(target)
    target[0, 0, 2, 2, 2] = 2.0
    roi = torch.zeros_like(target)
    roi[0, 0, 2, 2, 2] = 1.0

    loss_base, _ = spect_reconstruction_loss(
        target, pred, use_weighted=False, use_tail=False
    )
    loss_roi, _ = spect_reconstruction_loss(
        target,
        pred,
        use_weighted=False,
        use_tail=False,
        roi_mask=roi,
        roi_alpha=3.0,
    )
    assert loss_roi.item() > loss_base.item()
