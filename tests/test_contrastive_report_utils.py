"""Smoke tests for contrastive_report_utils."""

from code.contrastive_report_utils import (
    E1_ORDER,
    E1_SELECTED_KEY,
    E2_ORDER,
    available_e1_experiments,
    build_e1_e2_summary_table,
    build_e1_leaderboard,
    e2_comparison_from_registry,
    validate_e1_artifacts,
    validate_e2_artifacts,
)


def test_e1_e2_artifacts_available():
    e1 = available_e1_experiments()
    assert "e1_baseline" in e1
    assert "e1_xmodal_h64" in e1
    assert len(e1) == len(E1_ORDER)
    e2 = e2_comparison_from_registry()
    for mode in E2_ORDER:
        assert mode in e2
    assert not validate_e1_artifacts(list(e1.keys()))
    assert not validate_e2_artifacts(e2)


def test_transversal_summary_shape():
    e1 = available_e1_experiments()
    e2 = e2_comparison_from_registry()
    keys = [k for k in E1_ORDER if k in e1]
    df = build_e1_e2_summary_table(keys, e2)
    assert len(df) == 4
    assert "e2_balanced_accuracy" in df.columns


def test_e1_leaderboard_ranks_xmodal_h64_first():
    e1 = available_e1_experiments()
    lb = build_e1_leaderboard(e1)
    assert lb.iloc[0]["e1_key"] == E1_SELECTED_KEY
    assert bool(lb.loc[lb["e1_key"] == E1_SELECTED_KEY, "selected_for_e2"].iloc[0])
