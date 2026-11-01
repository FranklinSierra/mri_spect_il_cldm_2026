"""Vision Transformer (ViT-Small/16) inflated to 3D for volumetric SPECT classification.

Follows the same "inflate a 2D ImageNet backbone" convention as code/mobilenet3d.py,
but for a transformer: the patch-embedding conv and the positional embeddings are the
only pieces that depend on 2D vs 3D topology, so only those are adapted. The attention
and MLP blocks operate on a flat token sequence and are copied unchanged from the
pretrained 2D ViT (ViViT-style "central frame" inflation: every depth slice starts from
the same pretrained spatial positional embedding, and the model learns to tell slices
apart from there).
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def _inflate_patch_embed(conv2d: nn.Conv2d, depth_patch_size: int, in_channels: int) -> nn.Conv3d:
    """Conv2d(3, embed_dim, 16, 16) -> Conv3d(in_channels, embed_dim, depth_patch_size, 16, 16).

    The spatial kernel stays 16x16 (that's what's being inflated from the pretrained
    2D patch embed); only the depth kernel size is a free parameter, so depth
    resolution can be tuned independently of the pretrained spatial patch size.
    """
    out_channels = conv2d.out_channels
    kh, kw = conv2d.kernel_size
    conv3d = nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(depth_patch_size, kh, kw),
        stride=(depth_patch_size, kh, kw),
        bias=conv2d.bias is not None,
    )
    with torch.no_grad():
        # RGB -> single channel: average input filters (same trick as mobilenet3d.py).
        w2d = conv2d.weight.data.mean(dim=1, keepdim=True)  # (out, 1, kh, kw)
        kt = depth_patch_size
        w3d = w2d.unsqueeze(2).repeat(1, 1, kt, 1, 1) / float(kt)  # (out, 1, kt, kh, kw)
        conv3d.weight.data.copy_(w3d)
        if conv2d.bias is not None and conv3d.bias is not None:
            conv3d.bias.data.copy_(conv2d.bias.data)
    return conv3d


def _inflate_pos_embed(pos_embed_2d: torch.Tensor, grid_3d: tuple[int, int, int]) -> torch.Tensor:
    """(1, 1+14*14, dim) -> (1, 1+dt*gh*gw, dim).

    Spatial part is bicubically resized to the new (gh, gw) grid, then tiled identically
    across the dt depth positions (every depth slice starts identical; the model learns
    to differentiate them during fine-tuning).
    """
    dt, gh, gw = grid_3d
    dim = pos_embed_2d.shape[-1]
    n_patches_2d = pos_embed_2d.shape[1] - 1
    side = int(round(n_patches_2d**0.5))
    if side * side != n_patches_2d:
        raise ValueError(f"Expected a square grid of patch positions, got {n_patches_2d}")

    cls_pos = pos_embed_2d[:, :1, :]  # (1, 1, dim)
    patch_pos = pos_embed_2d[:, 1:, :].reshape(1, side, side, dim).permute(0, 3, 1, 2)  # (1,dim,side,side)
    patch_pos = F.interpolate(patch_pos, size=(gh, gw), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, gh * gw, dim)  # (1, gh*gw, dim)

    patch_pos_3d = patch_pos.unsqueeze(1).repeat(1, dt, 1, 1).reshape(1, dt * gh * gw, dim)
    return torch.cat([cls_pos, patch_pos_3d], dim=1)


class VisionTransformer3D(nn.Module):
    """Inflated ViT-Small/16: (B, 1, D, H, W) -> logits (B, num_classes).

    Expects D, H, W all divisible by `patch_size` (default 16); callers should
    crop/pad volumes to such a shape before feeding this model (e.g. 96x112x112).
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        pretrained: bool = True,
        depth_patch_size: int = 16,
        volume_shape: tuple[int, int, int] = (96, 112, 112),
        drop_path_rate: float = 0.1,
        timm_name: str = "vit_small_patch16_224.augreg_in21k_ft_in1k",
    ):
        super().__init__()
        d, h, w = volume_shape
        spatial_patch = 16  # fixed: matches the pretrained 2D patch embed kernel
        if d % depth_patch_size or h % spatial_patch or w % spatial_patch:
            raise ValueError(
                f"volume_shape {volume_shape} must be divisible by "
                f"(depth_patch_size={depth_patch_size}, spatial_patch={spatial_patch})"
            )
        self.depth_patch_size = depth_patch_size
        self.grid = (d // depth_patch_size, h // spatial_patch, w // spatial_patch)

        vit2d = timm.create_model(timm_name, pretrained=pretrained, drop_path_rate=drop_path_rate)
        self.embed_dim = vit2d.embed_dim

        self.patch_embed = _inflate_patch_embed(vit2d.patch_embed.proj, depth_patch_size, in_channels)
        self.cls_token = nn.Parameter(vit2d.cls_token.data.clone())
        self.pos_embed = nn.Parameter(_inflate_pos_embed(vit2d.pos_embed.data, self.grid))
        self.pos_drop = vit2d.pos_drop
        self.blocks = vit2d.blocks
        self.norm = vit2d.norm

        self.head = nn.Linear(self.embed_dim, num_classes)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected (B,C,D,H,W), got shape {tuple(x.shape)}")
        b = x.shape[0]
        x = self.patch_embed(x)  # (B, dim, dt, gh, gw)
        x = x.flatten(2).transpose(1, 2)  # (B, dt*gh*gw, dim)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]  # cls token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def build_vit3d(
    num_classes: int = 2,
    in_channels: int = 1,
    pretrained: bool = True,
    volume_shape: tuple[int, int, int] = (96, 112, 112),
    drop_path_rate: float = 0.1,
    depth_patch_size: int = 16,
) -> VisionTransformer3D:
    return VisionTransformer3D(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        volume_shape=volume_shape,
        drop_path_rate=drop_path_rate,
        depth_patch_size=depth_patch_size,
    )
