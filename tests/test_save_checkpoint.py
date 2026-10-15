"""save_checkpoint must allow optimizer=None (finetuned MRI encoder dump)."""

from pathlib import Path

import torch
from torch import nn

from utils import save_checkpoint


def test_save_checkpoint_without_optimizer(tmp_path: Path):
    model = nn.Linear(2, 2)
    path = tmp_path / "MRI_Encoder_finetuned.pth.tar"
    save_checkpoint(model, None, filename=str(path))
    ckpt = torch.load(path, map_location="cpu")
    assert "state_dict" in ckpt
    assert ckpt["optimizer"] is None
    model.load_state_dict(ckpt["state_dict"])
