#!/usr/bin/env python3
"""Classify existing DDIM embeddings into CTL / H&Y1 / H&Y2 (NHY3).

Usage:
    python3 scripts/classify_embeddings_nhy3.py
    python3 scripts/classify_embeddings_nhy3.py --embedding-root <path> --gpu 0

Default parent: calib_roi3ch_uni_rois_reproc (does not overwrite binary classification/).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from code.nhy3_classification import (  # noqa: E402
    DEFAULT_EMBEDDING_ROOT,
    DEFAULT_NHY_CSV,
    run_nhy3_classification,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=DEFAULT_EMBEDDING_ROOT,
        help="Inference dir with metadata.csv + reps/ (default: reproc parent)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <embedding-root>/classification_nhy3",
    )
    parser.add_argument(
        "--nhy-csv",
        type=Path,
        default=DEFAULT_NHY_CSV,
        help="NHY stratum table (control / pd_nhy1 / pd_nhy2)",
    )
    parser.add_argument(
        "--parent-experiment",
        default="calib_roi3ch_uni_rois_reproc",
    )
    parser.add_argument(
        "--analysis-id",
        default="calib_roi3ch_uni_rois_reproc_nhy3",
    )
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES index")
    parser.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    os.chdir(REPO_ROOT)
    run_nhy3_classification(
        embedding_root=args.embedding_root,
        out_dir=args.out_dir,
        nhy_csv=args.nhy_csv,
        device=args.device,
        parent_experiment=args.parent_experiment,
        analysis_id=args.analysis_id,
    )


if __name__ == "__main__":
    main()
