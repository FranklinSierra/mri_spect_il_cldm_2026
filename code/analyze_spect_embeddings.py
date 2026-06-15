#!/usr/bin/env python3
"""
Classify SPECT autoencoder latents (E1) with the same train/test IDs as MRI phase-1 (E2).

Uses latent NIfTI in data/latent_Abeta/ and the embedding_eval_split.csv produced by
run_phase1_mri_experiment (80/20 stratified on cohort train, random_state=42).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

import config
from code.embedding_analysis import analyze_embeddings_with_eval_split, ensure_analysis_dirs
from utils import seed_torch

DEFAULT_MRI_EVAL_SPLIT = (
    ROOT / "result/exp_nm_s1_mri/phase1_mri/embedding_analysis/embedding_eval_split.csv"
)
DEFAULT_SPECT_EXP = "exp_nm_s1_spect"
DEFAULT_LATENT_ROOT = ROOT / "data/latent_Abeta"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default=DEFAULT_SPECT_EXP, help="E1 experiment name (output folder).")
    parser.add_argument(
        "--latent-root",
        type=Path,
        default=DEFAULT_LATENT_ROOT,
        help="Directory with SPECT latent NIfTI from E1 encoding.",
    )
    parser.add_argument(
        "--eval-split",
        type=Path,
        default=DEFAULT_MRI_EVAL_SPLIT,
        help="Fixed train/test split CSV (same as MRI E2 embedding analysis).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exclude-classifier",
        action="append",
        default=None,
        help="Skip classifiers (repeatable). Default on 76k-dim latents: SVM_RBF SVM_linear.",
    )
    parser.add_argument("--skip-dnn", action="store_true", help="Skip DNN_simple (faster).")
    return parser.parse_args()


def latent_path(latent_root: Path, subject_id: str) -> Path:
    for suffix in (".nii.gz", ".nii"):
        path = latent_root / f"{subject_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No SPECT latent for ID '{subject_id}' in {latent_root}")


def build_spect_eval_split(mri_split: pd.DataFrame, latent_root: Path) -> pd.DataFrame:
    rows = []
    missing = []
    for row in mri_split.itertuples(index=False):
        subject_id = str(row.ID)
        try:
            path = latent_path(latent_root, subject_id)
        except FileNotFoundError:
            missing.append(subject_id)
            continue
        rows.append(
            {
                "ID": subject_id,
                "label": int(row.label),
                "embedding_eval_split": str(row.embedding_eval_split),
                "embedding_path": str(path.resolve()),
            }
        )
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} subjects from MRI eval split lack SPECT latents in {latent_root}: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    return pd.DataFrame(rows)


def build_train_cohort_metadata(latent_root: Path) -> pd.DataFrame:
    """Metadata for all encoded train latents (reference list)."""
    labels = pd.read_csv(ROOT / "data_info/data_info.csv", dtype={"ID": str}).set_index("ID")["label"]
    train_ids = [
        line.strip()
        for line in (ROOT / "data_info/train.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    for subject_id in train_ids:
        try:
            path = latent_path(latent_root, subject_id)
        except FileNotFoundError:
            continue
        rows.append(
            {
                "ID": subject_id,
                "split": "train",
                "label": int(labels[subject_id]),
                "embedding_path": str(path.resolve()),
                "embedding_shape": "1,40,48,40",
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    seed_torch(args.seed)

    if not args.eval_split.is_file():
        raise FileNotFoundError(
            f"MRI eval split not found: {args.eval_split}\n"
            "Run E2 (s1_mri) through embedding analysis first, or pass --eval-split."
        )
    if not args.latent_root.is_dir():
        raise FileNotFoundError(f"Latent root not found: {args.latent_root}")

    mri_split = pd.read_csv(args.eval_split, dtype={"ID": str})
    spect_eval = build_spect_eval_split(mri_split, args.latent_root)

    out_base = ROOT / "result" / args.exp.strip("/") / "embedding_analysis"
    path_map = ensure_analysis_dirs(out_base)

    metadata = build_train_cohort_metadata(args.latent_root)
    metadata.to_csv(out_base / "spect_latents_metadata.csv", index=False)

    protocol = (
        "matched_mri_phase1_eval_split_80_20 "
        f"(source={args.eval_split.relative_to(ROOT) if args.eval_split.is_relative_to(ROOT) else args.eval_split})"
    )
    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": args.exp,
        "modality": "SPECT_AAE_latent",
        "latent_root": str(args.latent_root),
        "aae_checkpoint": str(ROOT / "result" / args.exp.strip("/") / "AAE.pth.tar"),
        "mri_eval_split_source": str(args.eval_split),
        "evaluation_protocol": protocol,
        "n_latents_train_cohort": int(len(metadata)),
        "n_eval_train": int((spect_eval["embedding_eval_split"] == "train").sum()),
        "n_eval_test": int((spect_eval["embedding_eval_split"] == "test").sum()),
        "excluded_classifiers": args.exclude_classifier,
        "skip_dnn": args.skip_dnn,
    }
    with open(out_base / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2)

    exclude_list = args.exclude_classifier
    if exclude_list is None:
        exclude_list = ["SVM_RBF", "SVM_linear"]
    exclude = {c for c in exclude_list if c}
    decision = analyze_embeddings_with_eval_split(
        spect_eval,
        path_map,
        modality="SPECT",
        evaluation_protocol=protocol,
        viz_title_suffix="SPECT-AE internal test (MRI-matched split)",
        skip_dnn=args.skip_dnn,
        exclude_classifiers=exclude,
    )
    print(json.dumps(decision, indent=2))
    print(f"Done. Results: {out_base}")


if __name__ == "__main__":
    main()
