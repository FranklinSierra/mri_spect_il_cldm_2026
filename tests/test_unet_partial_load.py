import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import UNet
from utils import _adapt_unet_input_conv


def _first_conv_state(in_channels):
    net = UNet(in_channel=in_channels)
    return net.state_dict()["downs.0.weight"]


def test_adapt_unet_input_conv_expands_two_to_three():
    ckpt_w = _first_conv_state(2)
    model_w = _first_conv_state(3)
    state_dict = {"downs.0.weight": ckpt_w.clone()}
    model_state = {"downs.0.weight": model_w}

    adapted = _adapt_unet_input_conv(state_dict, model_state)
    new_w = adapted["downs.0.weight"]

    assert new_w.shape == model_w.shape
    assert torch.allclose(new_w[:, :2], ckpt_w)
    assert torch.all(new_w[:, 2:].eq(0))


def test_adapt_unet_input_conv_unchanged_when_shapes_match():
    w = _first_conv_state(2)
    state_dict = {"downs.0.weight": w.clone()}
    model_state = {"downs.0.weight": w.clone()}
    adapted = _adapt_unet_input_conv(state_dict, model_state)
    assert torch.allclose(adapted["downs.0.weight"], w)
