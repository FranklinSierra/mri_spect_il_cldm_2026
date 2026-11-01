"""ConvNeXt-Tiny inflated to 3D for single-channel volumetric SPECT classification.

Same "inflate a 2D ImageNet backbone" convention as code/mobilenet3d.py (reuses its
Conv2d->Conv3d inflation helpers directly), extended to ConvNeXt's extra building
blocks: the channels-last LayerNorm2d, the `Permute` modules that switch between
NCHW and NHWC around the pointwise MLP, and the per-channel `layer_scale` parameter
inside each CNBlock.

Why ConvNeXt instead of the originally-planned ViT3D: an inflated ViT3D (patch
embedding + ImageNet-pretrained attention blocks) was empirically unable to even
memorize the ~124-volume synthetic training set (train accuracy stuck at the
majority-class rate, confirmed via gradient-norm logging that this wasn't a
vanishing-gradient issue), while an inflated CNN with the same data pipeline
overfits normally within a few epochs. ConvNeXt keeps the convolutional locality
bias that makes that possible, while still being a materially more modern backbone
than MobileNetV2 (depthwise 7x7 kernels, LayerNorm, GELU, layer-scale residuals).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights
from torchvision.models.convnext import CNBlock, LayerNorm2d, Permute

from code.mobilenet3d import _inflate_conv_weights, inflate_conv2d


class LayerNorm3d(nn.LayerNorm):
    """Channels-last LayerNorm for (B, C, D, H, W) tensors (3D analogue of torchvision's LayerNorm2d)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 4, 1, 2, 3)


class Permute3d(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.permute(x, self.dims)


def _permute_2d_to_3d(dims: list[int]) -> list[int]:
    """[0,2,3,1] (NCHW->NHWC) -> [0,2,3,4,1]; [0,3,1,2] (NHWC->NCHW) -> [0,4,1,2,3]."""
    if dims == [0, 2, 3, 1]:
        return [0, 2, 3, 4, 1]
    if dims == [0, 3, 1, 2]:
        return [0, 4, 1, 2, 3]
    raise ValueError(f"Unexpected ConvNeXt Permute dims {dims}")


def _convert_module(module: nn.Module) -> nn.Module:
    """Recursively replace 2D ConvNeXt submodules with their 3D counterparts."""
    if isinstance(module, nn.Conv2d):
        conv3d = inflate_conv2d(module)
        _inflate_conv_weights(module, conv3d)
        return conv3d
    if isinstance(module, LayerNorm2d):
        ln3d = LayerNorm3d(module.normalized_shape, eps=module.eps, elementwise_affine=module.elementwise_affine)
        if module.elementwise_affine:
            with torch.no_grad():
                ln3d.weight.copy_(module.weight)
                ln3d.bias.copy_(module.bias)
        return ln3d
    if isinstance(module, nn.AdaptiveAvgPool2d):
        return nn.AdaptiveAvgPool3d(module.output_size)
    if isinstance(module, Permute):
        return Permute3d(_permute_2d_to_3d(module.dims))

    if isinstance(module, CNBlock):
        # layer_scale broadcasts over (B,C,D,H,W); add the extra singleton depth dim.
        with torch.no_grad():
            module.layer_scale = nn.Parameter(module.layer_scale.data.unsqueeze(-1))

    for name, child in list(module.named_children()):
        setattr(module, name, _convert_module(child))
    return module


class ConvNeXt3D(nn.Module):
    """Inflated ConvNeXt-Tiny: (B,1,D,H,W) -> logits (B, num_classes).

    Unlike the ViT3D patch grid, plain convolutions tolerate any spatial/depth size
    (no exact-divisibility requirement), so volumes don't need cropping/padding to a
    canonical shape -- the final AdaptiveAvgPool3d(1) absorbs whatever remainder.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.convnext_tiny(weights=weights)
        features_3d = _convert_module(backbone.features)

        # First conv: RGB ImageNet -> single SPECT channel (mean over RGB in-weights).
        stem_conv = features_3d[0][0]
        if not isinstance(stem_conv, nn.Conv3d):
            raise TypeError(f"Expected first feature Conv3d, got {type(stem_conv)}")
        if in_channels != stem_conv.in_channels:
            new_stem = nn.Conv3d(
                in_channels=in_channels,
                out_channels=stem_conv.out_channels,
                kernel_size=stem_conv.kernel_size,
                stride=stem_conv.stride,
                padding=stem_conv.padding,
                dilation=stem_conv.dilation,
                groups=1,
                bias=stem_conv.bias is not None,
                padding_mode=stem_conv.padding_mode,
            )
            with torch.no_grad():
                new_stem.weight.copy_(stem_conv.weight.mean(dim=1, keepdim=True))
                if stem_conv.bias is not None and new_stem.bias is not None:
                    new_stem.bias.copy_(stem_conv.bias)
            features_3d[0][0] = new_stem

        self.features = features_3d
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        last_channels = backbone.classifier[-1].in_features
        self.norm = _convert_module(backbone.classifier[0])  # LayerNorm2d -> LayerNorm3d
        self.flatten = nn.Flatten(1)
        self.head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(last_channels, num_classes))
        nn.init.trunc_normal_(self.head[1].weight, std=0.02)
        nn.init.zeros_(self.head[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected (B,C,D,H,W), got shape {tuple(x.shape)}")
        x = self.features(x)
        x = self.avgpool(x)
        x = self.norm(x)
        x = self.flatten(x)
        return self.head(x)


def build_convnext3d(num_classes: int = 2, in_channels: int = 1, pretrained: bool = True) -> ConvNeXt3D:
    return ConvNeXt3D(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)
