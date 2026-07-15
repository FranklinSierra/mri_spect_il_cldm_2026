"""Domain bucket helpers for stage2_experiments_report."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from code.stage2_report_utils import domain_buckets_for_modes, load_registry, subset_by_modes


def test_domain_buckets_for_three_mode_wave():
    reg = load_registry()["experiments"]
    modes = ["h05_roi3ch", "calib_roi3ch_uni", "calib_roi3ch_uni_rois"]
    buckets = domain_buckets_for_modes(modes, reg)
    assert len(buckets) == 2
    assert buckets[0][0] == "fullbrain"
    assert set(buckets[0][1]) == {"h05_roi3ch", "calib_roi3ch_uni"}
    assert buckets[1][0] == "roi_patch"
    assert buckets[1][1] == ["calib_roi3ch_uni_rois"]


def test_subset_by_modes():
    import pandas as pd

    df = pd.DataFrame({"mode": ["a", "b", "a"], "x": [1, 2, 3]})
    out = subset_by_modes(df, ["a"])
    assert len(out) == 2
    assert out["mode"].tolist() == ["a", "a"]
