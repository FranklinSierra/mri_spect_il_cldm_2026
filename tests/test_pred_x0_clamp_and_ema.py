"""Tests for pred_x0 clamp, label embedding size, EMA warmup, attn_res, SVM gate."""

from __future__ import annotations

import os

import pytest
import torch


@pytest.fixture(autouse=True)
def _cpu_device(monkeypatch):
    monkeypatch.setenv("ILCLDM_DEVICE", "cpu")
    monkeypatch.setenv("ILCLDM_PRED_X0_CLAMP", "4.0")
    monkeypatch.setenv("ILCLDM_LABEL_DROPOUT_PROB", "0.0")
    monkeypatch.setenv("ILCLDM_USE_FILM", "0")
    monkeypatch.setenv("ILCLDM_ATTN_RES", "4")
    monkeypatch.setenv("ILCLDM_INCLUDE_SVM", "0")
    import importlib
    import config

    importlib.reload(config)
    yield
    importlib.reload(config)


def test_predict_x0_is_clamped(monkeypatch):
    import importlib
    import config
    from main import Diffusion

    monkeypatch.setenv("ILCLDM_PRED_X0_CLAMP", "4.0")
    importlib.reload(config)
    diff = Diffusion()
    # Force extreme noise prediction at high t → unclamped pred_x0 would explode.
    t = torch.tensor([999], dtype=torch.long)
    x_t = torch.randn(1, 1, 4, 4, 4)
    bad_eps = torch.ones_like(x_t) * 10.0
    pred = diff.predict_x0(x_t, t, bad_eps)
    assert float(pred.abs().max()) <= 4.0 + 1e-5


def test_label_embedding_grows_with_dropout(monkeypatch):
    import importlib
    import config
    from model import UNet

    monkeypatch.setenv("ILCLDM_LABEL_DROPOUT_PROB", "0.0")
    importlib.reload(config)
    net0 = UNet(in_channel=2, out_channel=1, image_size=8, attn_res=(2,), use_film=False)
    assert net0.label_emb.num_embeddings == 2
    assert net0.null_label_id is None

    monkeypatch.setenv("ILCLDM_LABEL_DROPOUT_PROB", "0.15")
    importlib.reload(config)
    net1 = UNet(in_channel=2, out_channel=1, image_size=8, attn_res=(2,), use_film=False)
    assert net1.label_emb.num_embeddings == 3
    assert net1.null_label_id == 2


def test_ema_warmup_schedule():
    from model import EMA

    ema = EMA(beta=0.995, beta_start=0.9, warmup_steps=10, step_start_ema=2)
    # During copy phase
    ema.step = 0
    assert abs(ema.current_beta() - 0.9) < 1e-9
    # Mid warmup (step=2 is first averaging step → steps_after=0)
    ema.step = 2
    assert abs(ema.current_beta() - 0.9) < 1e-9
    ema.step = 7  # steps_after=5 → halfway
    beta = ema.current_beta()
    assert 0.94 < beta < 0.96
    ema.step = 100
    assert abs(ema.current_beta() - 0.995) < 1e-9


def test_attn_res_five_enables_blocks(monkeypatch):
    import importlib
    import config
    from model import UNet

    monkeypatch.setenv("ILCLDM_ATTN_RES", "5")
    importlib.reload(config)
    # image_size=40 → 40,20,10,5 — with attn_res=(5,) downs/ups at res=5 get attn + mid
    net = UNet(in_channel=2, out_channel=1, image_size=40, attn_res=(5,), use_film=False)
    assert net.count_attn_blocks() > 1


def test_svm_excluded_by_default(monkeypatch):
    import importlib
    import config
    from inference import build_classical_models

    monkeypatch.setenv("ILCLDM_INCLUDE_SVM", "0")
    importlib.reload(config)
    models = build_classical_models([0, 1, 0, 1, 0, 1])
    assert "SVM_RBF" not in models
    assert "LogisticRegression" in models
