#!/usr/bin/env python3
"""Run PD HY1/HY2 KNN baselines on image and text embeddings.

Usage:
    python3 scripts/classify_nhy_staging_pd.py
    python3 scripts/classify_nhy_staging_pd.py --out-root result/clinical_staging_pd
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from code.clinical_staging import DEFAULT_CLINICAL_STAGING_ROOT  # noqa: E402
from code.nhy_staging_pd_classification import run_staging_baselines  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_CLINICAL_STAGING_ROOT,
    )
    parser.add_argument("--pairs-csv", type=Path, default=None)
    parser.add_argument("--knn-k", type=int, default=5)
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    run_staging_baselines(
        out_root=args.out_root,
        pairs_csv=args.pairs_csv,
        knn_k=args.knn_k,
    )


if __name__ == "__main__":
    main()
