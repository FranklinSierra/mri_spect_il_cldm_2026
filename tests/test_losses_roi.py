import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from losses import spect_reconstruction_loss, combined_spect_weights


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
