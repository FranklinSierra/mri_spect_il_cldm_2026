#!/usr/bin/env python3
"""Example: visualize MRI + SPECT with correct z-score contrast (see viz_pairs.py)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viz_pairs import plot_subject

if __name__ == "__main__":
    data_root = ROOT / "data"
    subject_id = "3102"
    plot_subject(data_root, subject_id, show=True)
