"""MobileNetV2 inflated to 3D for single-channel volumetric classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


def _as_triple(value, *, depth_from_spatial: bool = True) -> tuple[int, int, int]:
    """Expand a 2D kernel/stride/padding scalar or pair to 3D (D,H,W)."""
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 2:
        raise ValueError(f"Expected int or length-2 sequence, got {value!r}")
    h, w = int(value[0]), int(value[1])
    d = h if depth_from_spatial else 1
    return (d, h, w)


def inflate_conv2d(conv2d: nn.Conv2d, *, depth_kernel_from_spatial: bool = True) -> nn.Conv3d:
    """Create Conv3d matching conv2d; optionally copy inflated ImageNet weights later."""
    kernel_size = _as_triple(conv2d.kernel_size, depth_from_spatial=depth_kernel_from_spatial)
    stride = _as_triple(conv2d.stride, depth_from_spatial=True)
    padding = _as_triple(conv2d.padding, depth_from_spatial=depth_kernel_from_spatial)
    dilation = _as_triple(conv2d.dilation, depth_from_spatial=True)
    return nn.Conv3d(
        in_channels=conv2d.in_channels,
        out_channels=conv2d.out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=conv2d.groups,
        bias=conv2d.bias is not None,
        padding_mode=conv2d.padding_mode,
    )


def inflate_batchnorm2d(bn2d: nn.BatchNorm2d) -> nn.BatchNorm3d:
    bn3d = nn.BatchNorm3d(
        num_features=bn2d.num_features,
        eps=bn2d.eps,
        momentum=bn2d.momentum,
        affine=bn2d.affine,
        track_running_stats=bn2d.track_running_stats,
    )
    bn3d.weight.data.copy_(bn2d.weight.data)
    bn3d.bias.data.copy_(bn2d.bias.data)
    bn3d.running_mean.copy_(bn2d.running_mean)
    bn3d.running_var.copy_(bn2d.running_var)
    bn3d.num_batches_tracked.copy_(bn2d.num_batches_tracked)
    return bn3d


def _inflate_conv_weights(src: nn.Conv2d, dst: nn.Conv3d) -> None:
    """Repeat 2D kernel along depth and divide by depth size (classic inflation)."""
    w2d = src.weight.data  # (out, in/groups, kh, kw)
    kt = dst.weight.shape[2]
    w3d = w2d.unsqueeze(2).repeat(1, 1, kt, 1, 1) / float(kt)
    dst.weight.data.copy_(w3d)
    if src.bias is not None and dst.bias is not None:
        dst.bias.data.copy_(src.bias.data)


def _convert_module(module: nn.Module) -> nn.Module:
    """Recursively replace Conv2d/BatchNorm2d with 3D counterparts and inflate weights."""
    if isinstance(module, nn.Conv2d):
        conv3d = inflate_conv2d(module)
        _inflate_conv_weights(module, conv3d)
        return conv3d
    if isinstance(module, nn.BatchNorm2d):
        return inflate_batchnorm2d(module)
    if isinstance(module, nn.AdaptiveAvgPool2d):
        return nn.AdaptiveAvgPool3d(1)

    if isinstance(module, nn.Sequential):
        return nn.Sequential(*[_convert_module(child) for child in module.children()])

    for name, child in list(module.named_children()):
        setattr(module, name, _convert_module(child))
    return module


class MobileNetV2_3D(nn.Module):
    """Inflated MobileNetV2: (B,1,D,H,W) → logits (B, num_classes)."""

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)
        features_3d = _convert_module(backbone.features)

        # First conv: RGB ImageNet → single SPECT channel (mean over RGB in-weights).
        first = features_3d[0][0]
        if not isinstance(first, nn.Conv3d):
            raise TypeError(f"Expected first feature Conv3d, got {type(first)}")
        if in_channels != first.in_channels:
            new_first = nn.Conv3d(
                in_channels=in_channels,
                out_channels=first.out_channels,
                kernel_size=first.kernel_size,
                stride=first.stride,
                padding=first.padding,
                dilation=first.dilation,
                groups=1,
                bias=first.bias is not None,
                padding_mode=first.padding_mode,
            )
            with torch.no_grad():
                # Average RGB input filters → 1 channel
                new_first.weight.copy_(first.weight.mean(dim=1, keepdim=True))
                if first.bias is not None and new_first.bias is not None:
                    new_first.bias.copy_(first.bias)
            features_3d[0][0] = new_first

        self.features = features_3d
        last_channels = backbone.last_channel
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channels, num_classes),
        )
        # Init classifier from scratch (not ImageNet 1000-way)
        nn.init.normal_(self.classifier[1].weight, std=0.01)
        nn.init.zeros_(self.classifier[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected (B,C,D,H,W), got shape {tuple(x.shape)}")
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool3d(x, 1)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_mobilenet_v2_3d(num_classes: int = 2, in_channels: int = 1, pretrained: bool = True) -> MobileNetV2_3D:
    return MobileNetV2_3D(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
    )
