import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from code.roi_mask_utils import LATENT_SHAPE
from model import build_unet


def test_build_unet_default_two_channels():
    net = build_unet(in_channel=2, use_roi_channel=False)
    assert net.downs[0].weight.shape[1] == 2
    assert not hasattr(net, "roi_latent")


def test_build_unet_three_channels_with_roi_buffer(monkeypatch, tmp_path):
    mask_path = tmp_path / "mask.nii.gz"
    import nibabel as nib
    import numpy as np

    vol = np.zeros((180, 228, 180), dtype=np.float32)
    vol[80, 120, 80] = 1.0
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(mask_path))

    monkeypatch.setattr(config, "ROI_MASK_PATH", str(mask_path))
    net = build_unet(in_channel=3, use_roi_channel=True)
    assert net.downs[0].weight.shape[1] == 3
    assert hasattr(net, "roi_latent")
    assert tuple(net.roi_latent.shape[-3:]) == LATENT_SHAPE


def test_concat_inputs_expands_roi_batch():
    net = build_unet(in_channel=3, use_roi_channel=True)
    net.roi_latent = torch.ones(1, 1, *LATENT_SHAPE)
    x = torch.zeros(4, 1, *LATENT_SHAPE)
    cond = torch.zeros(4, 1, *LATENT_SHAPE)
    out = net._concat_inputs(x, cond)
    assert out.shape == (4, 3, *LATENT_SHAPE)
    assert out[:, 2].eq(1).all()
