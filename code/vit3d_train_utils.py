"""Shared helpers for the inflated-3D-backbone SPECT classifier training scripts
(ViT3D and ConvNeXt3D alike -- despite the filename, nothing here is ViT-specific).

Mirrors the conventions of scripts/train_synthetic_spect_mobilenet3d.py (same metric
set, same output file names) so results are directly comparable to the MobileNet
baseline, but factors out the parts shared between the stage-1 (real SPECT) and
stage-2 (synthetic SPECT) scripts.
"""

from __future__ import annotations

import random
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset

TARGET_SHAPE = (96, 112, 112)  # (D, H, W); divisible by ViT patch size 16


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subject_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return path.stem


def list_volumes(root: Path) -> list[Path]:
    files = sorted(root.glob("*.nii.gz")) + sorted(root.glob("*.nii"))
    seen = set()
    unique = []
    for path in files:
        sid = subject_id_from_path(path)
        if sid in seen:
            continue
        seen.add(sid)
        unique.append(path)
    if not unique:
        raise FileNotFoundError(f"No NIfTI volumes under {root}")
    return unique


def records_from_dir(root: Path, labels: dict[str, int]) -> list[dict]:
    rows = []
    for path in list_volumes(root):
        sid = subject_id_from_path(path)
        if sid not in labels:
            raise KeyError(f"ID {sid} missing from metadata")
        rows.append({"ID": sid, "label": int(labels[sid]), "path": path})
    return rows


def center_crop_or_pad(vol: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    """Center crop (or zero-pad) each axis independently to `target`."""
    out = vol
    for axis, t in enumerate(target):
        n = out.shape[axis]
        if n == t:
            continue
        if n > t:
            lo = (n - t) // 2
            out = np.take(out, range(lo, lo + t), axis=axis)
        else:
            pad_total = t - n
            pad_lo = pad_total // 2
            pad_hi = pad_total - pad_lo
            pad_width = [(0, 0)] * out.ndim
            pad_width[axis] = (pad_lo, pad_hi)
            out = np.pad(out, pad_width, mode="constant", constant_values=0.0)
    return out


def zscore_volume(vol: np.ndarray) -> np.ndarray:
    """Per-volume z-score over voxels > 0.

    Real SPECT (whole_Abeta) and the synthetic SPECT NIfTIs are stored in very
    different raw ranges (real: identity scale ~[0.09, 2.1], mean~1.1; synthetic:
    min-max restored to ~[0, 1], mean~0.4). Feeding those two ranges verbatim to a
    ViT pretrained with ImageNet-normalized (~N(0,1)) inputs causes the attention
    blocks to collapse (near input-invariant CLS token) on whichever domain is
    farther from what the backbone last saw. Z-scoring each volume independently
    removes that global scale/offset gap so both domains land in a comparable
    range at the model's input.
    """
    x = vol.astype(np.float32, copy=True)
    mask = x > 1e-6
    if not np.any(mask):
        return x
    mu = float(x[mask].mean())
    sigma = float(x[mask].std())
    if sigma < 1e-8:
        sigma = 1.0
    return (x - mu) / sigma


class NiftiVolumeDataset(Dataset):
    """Loads a NIfTI volume per subject, z-scored and (optionally) center-cropped/padded.

    `target_shape=None` skips the crop/pad step and keeps each volume's native shape
    (fine for plain-conv backbones like ConvNeXt3D, which tolerate any D/H/W via
    AdaptiveAvgPool3d at the head; the ViT3D patch grid needs the fixed TARGET_SHAPE).
    """

    def __init__(self, records: list[dict], target_shape: tuple[int, int, int] | None = TARGET_SHAPE):
        self.records = records
        self.target_shape = target_shape

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        vol = nib.load(rec["path"]).get_fdata().astype(np.float32)
        vol = zscore_volume(vol)
        if self.target_shape is not None:
            vol = center_crop_or_pad(vol, self.target_shape)
        x = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
        return x, int(rec["label"]), rec["ID"]


def subject_metrics(y_true, y_pred, y_prob) -> dict:
    out = {
        "n_subjects": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[dict, pd.DataFrame]:
    model.eval()
    rows = []
    for xb, yb, sids in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        y_np = yb.numpy()
        for i, sid in enumerate(sids):
            rows.append(
                {
                    "ID": sid,
                    "label": int(y_np[i]),
                    "pred": int(preds[i]),
                    "prob_1": float(probs[i, 1]),
                }
            )
    df = pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)
    metrics = subject_metrics(df["label"].to_numpy(), df["pred"].to_numpy(), df["prob_1"].to_numpy())
    return metrics, df


def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    import math

    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
