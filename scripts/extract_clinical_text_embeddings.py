#!/usr/bin/env python3
"""Extract ClinicalBERT embeddings for PD staging neutral prompts.

Usage:
    python3 scripts/extract_clinical_text_embeddings.py
    python3 scripts/extract_clinical_text_embeddings.py --gpu 7

Default out-root: result/clinical_staging_pd
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from code.clinical_staging import DEFAULT_CLINICAL_STAGING_ROOT  # noqa: E402
from code.clinical_text_embeddings import DEFAULT_GPU, extract_text_embeddings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_CLINICAL_STAGING_ROOT,
        help="Clinical staging output root",
    )
    parser.add_argument("--model-name", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu", default=DEFAULT_GPU, help="CUDA_VISIBLE_DEVICES index (default: 7)")
    parser.add_argument("--device", default=None, help="cuda|cpu override")
    parser.add_argument(
        "--skip-pairs-export",
        action="store_true",
        help="Do not regenerate clinical_pairs.csv before encoding",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    os.chdir(REPO_ROOT)
    extract_text_embeddings(
        out_root=args.out_root,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        refresh_pairs=not args.skip_pairs_export,
    )


if __name__ == "__main__":
    main()
