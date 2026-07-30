"""Tests for spatial contrastive feature path (encoder h64 + balanced sampler)."""

import torch

from code.run_phase1_mri_experiment import (
    BalancedBatchSampler,
    contrastive_feature_tensor,
)
from code.volume_shapes import get_latent_shape, get_working_volume_shape
from model import Encoder


def test_encoder_return_hidden_shapes():
    enc = Encoder().eval()
    working = get_working_volume_shape()
    latent_shape = get_latent_shape()
    x = torch.zeros(1, 1, *working)
    with torch.no_grad():
        latent_only = enc(x)
        latent, h64 = enc(x, return_hidden=True)
    assert tuple(latent_only.shape) == (1, 1, *latent_shape)
    assert tuple(latent.shape) == (1, 1, *latent_shape)
    assert tuple(h64.shape) == (1, 64, *latent_shape)
    assert torch.allclose(latent_only, latent)


def test_contrastive_feature_tensor_modes():
    latent = torch.randn(2, 1, 3, 4, 5)
    h64 = torch.randn(2, 64, 3, 4, 5)
    z18k = contrastive_feature_tensor(latent, h64, "z18k")
    zh = contrastive_feature_tensor(latent, h64, "h64")
    assert z18k.shape == (2, 1 * 3 * 4 * 5)
    assert zh.shape == (2, 64 * 3 * 4 * 5)


def test_balanced_batch_sampler_four_plus_four():
    labels = [0] * 10 + [1] * 12
    sampler = BalancedBatchSampler(labels, batch_size=8, drop_last=True, seed=0)
    assert len(sampler) == 2  # min(10,12)//4 = 2
    for batch in sampler:
        assert len(batch) == 8
        batch_labels = [labels[i] for i in batch]
        assert batch_labels.count(0) == 4
        assert batch_labels.count(1) == 4
