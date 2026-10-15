#!/usr/bin/env python3
"""Train/evaluate clinical radiology CLIP v2 for PD HY1 vs HY2 staging.

Usage:
    python3 scripts/train_clinical_radiology_clip.py
    python3 scripts/train_clinical_radiology_clip.py --gpu 7 --lambda-infonce 0.5 --lambda-supcon 1.0

Default: GPU 7, seed 42, hybrid loss with early stop on val SupCon loss.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from code.clinical_radiology_clip import run_clinical_radiology_clip  # noqa: E402
from code.clinical_staging import DEFAULT_CLINICAL_STAGING_ROOT  # noqa: E402
from code.clinical_text_embeddings import DEFAULT_GPU  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_CLINICAL_STAGING_ROOT,
    )
    parser.add_argument("--pairs-csv", type=Path, default=None)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--lambda-infonce", type=float, default=0.5)
    parser.add_argument("--lambda-supcon", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--gpu", default=DEFAULT_GPU, help="CUDA_VISIBLE_DEVICES index (default: 7)")
    parser.add_argument("--device", default=None, help="cuda|cpu override")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    os.chdir(REPO_ROOT)
    run_clinical_radiology_clip(
        out_root=args.out_root,
        pairs_csv=args.pairs_csv,
        proj_dim=args.proj_dim,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_epochs=args.min_epochs,
        lambda_infonce=args.lambda_infonce,
        lambda_supcon=args.lambda_supcon,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
