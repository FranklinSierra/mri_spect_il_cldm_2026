"""Tests for NHY stratified split generator."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.ppmi_master import load_ppmi_subjects, stratum_for_subject
from scripts.generate_nhy_split import cohort_ids_from_data, main as generate_main


@pytest.fixture(scope="module")
def ppmi_subjects():
    xlsx = ROOT / "PPMI_MASTER_v2.xlsx"
    if not xlsx.is_file():
        pytest.skip("PPMI_MASTER_v2.xlsx not available")
    return load_ppmi_subjects(xlsx)


def test_cohort_matches_ppmi(ppmi_subjects):
    cohort = set(cohort_ids_from_data())
    assert len(cohort) == 184
    assert cohort.issubset(set(ppmi_subjects.keys()))


def test_stratum_labels(ppmi_subjects):
    cohort = cohort_ids_from_data()
    strata = {stratum_for_subject(ppmi_subjects[i]) for i in cohort}
    assert strata == {"control", "pd_nhy1", "pd_nhy2"}


def test_generate_split_artifacts(tmp_path, monkeypatch):
    xlsx = ROOT / "PPMI_MASTER_v2.xlsx"
    if not xlsx.is_file():
        pytest.skip("PPMI_MASTER_v2.xlsx not available")
    out = tmp_path / "nhy_v1"
    monkeypatch.setattr(
        "scripts.generate_nhy_split.parse_args",
        lambda: type(
            "Args",
            (),
            {"xlsx": xlsx, "out": out, "seed": 42, "test_fraction": 36 / 184},
        )(),
    )
    assert generate_main() == 0
    train = {line.strip() for line in (out / "train.txt").read_text().splitlines() if line.strip()}
    test = {line.strip() for line in (out / "test.txt").read_text().splitlines() if line.strip()}
    assert len(train) == 148
    assert len(test) == 36
    assert not train & test
    manifest = json.loads((out / "split_manifest.json").read_text())
    assert manifest["n_train"] == 148
    assert manifest["n_test"] == 36
