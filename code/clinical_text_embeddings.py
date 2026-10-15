"""ClinicalBERT text embedding extraction for PD staging prompts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from code.clinical_staging import (
    DEFAULT_CLINICAL_STAGING_ROOT,
    REPO_ROOT,
    _repo_relative,
    export_clinical_staging_artifacts,
)

DEFAULT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
DEFAULT_GPU = "7"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    candidate = REPO_ROOT / path
    if candidate.is_file():
        return candidate
    return path


def l2_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def load_clinicalbert(
    model_name: str = DEFAULT_MODEL_NAME,
    device: torch.device | None = None,
) -> tuple[AutoTokenizer, AutoModel]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def encode_prompts_cls(
    prompts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 512,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded)
        cls = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        for row in cls:
            embeddings.append(l2_normalize(row))
    return np.stack(embeddings, axis=0).astype(np.float32)


def extract_text_embeddings(
    out_root: Path | str = DEFAULT_CLINICAL_STAGING_ROOT,
    pairs_csv: Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 16,
    device: str | None = None,
    refresh_pairs: bool = True,
) -> pd.DataFrame:
    out_root = Path(out_root)
    if refresh_pairs or not (out_root / "clinical_pairs.csv").is_file():
        export_clinical_staging_artifacts(out_root=out_root, require_radiology=False)

    pairs_path = pairs_csv or (out_root / "clinical_pairs.csv")
    pairs = pd.read_csv(pairs_path, dtype={"ID": str})

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    tokenizer, model = load_clinicalbert(model_name, torch_device)
    reps_dir = out_root / "text_embeddings" / "reps"
    reps_dir.mkdir(parents=True, exist_ok=True)

    prompts = pairs["prompt"].astype(str).tolist()
    embeddings = encode_prompts_cls(prompts, tokenizer, model, torch_device, batch_size=batch_size)

    metadata_rows = []
    for row, emb in zip(pairs.itertuples(index=False), embeddings, strict=True):
        rep_path = reps_dir / f"{row.ID}.npy"
        np.save(rep_path, emb)
        metadata_rows.append(
            {
                "ID": row.ID,
                "split": row.split,
                "stratum": row.stratum,
                "stage_label": int(row.stage_label),
                "embedding_path": _repo_relative(rep_path.resolve()),
            }
        )

    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(out_root / "text_embeddings" / "metadata.csv", index=False)

    run_config = {
        "model_name": model_name,
        "pooling": "cls",
        "normalization": "l2",
        "embedding_dim": int(embeddings.shape[1]),
        "batch_size": batch_size,
        "device": str(device),
        "n_subjects": int(len(metadata)),
        "out_dir": _repo_relative((out_root / "text_embeddings").resolve()),
    }
    with open(out_root / "text_embeddings" / "run_config_text.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"[clinical_text] wrote {len(metadata)} embeddings → {out_root / 'text_embeddings'}")
    return metadata
