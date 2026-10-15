"""Smoke tests for inflated MobileNetV2-3D."""

import torch

from code.mobilenet3d import build_mobilenet_v2_3d, inflate_conv2d


def test_inflate_conv2d_shapes():
    conv2d = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
    conv3d = inflate_conv2d(conv2d)
    assert conv3d.weight.shape == (32, 3, 3, 3, 3)
    assert conv3d.stride == (2, 2, 2)


def test_mobilenet_v2_3d_forward_logits():
    model = build_mobilenet_v2_3d(num_classes=2, in_channels=1, pretrained=False)
    x = torch.randn(2, 1, 32, 40, 40)
    y = model(x)
    assert y.shape == (2, 2)
