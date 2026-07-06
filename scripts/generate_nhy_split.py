#!/usr/bin/env python3
"""Generate NHY-stratified train/test split under data_info/splits/nhy_v1/."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.ppmi_master import load_ppmi_subjects, stratum_for_subject
from dataset import resolve_nifti_path

DEFAULT_XLSX = ROOT / "PPMI_MASTER_v2.xlsx"
DEFAULT_OUT = ROOT / "data_info" / "splits" / "nhy_v1"
DEFAULT_SEED = 42


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(sorted(ids, key=int)) + "\n")


def cohort_ids_from_data() -> list[str]:
    mri_root = ROOT / "data" / "whole_MRI"
    ids = []
    for suffix in (".nii.gz", ".nii"):
        for path in mri_root.glob(f"*{suffix}"):
            name = path.name
            if name.endswith(".nii.gz"):
                ids.append(name[:-7])
            elif name.endswith(".nii"):
                ids.append(name[:-4])
    return sorted(set(ids), key=int)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=36 / 184,
        help="Fraction of cohort assigned to test (default 36/184).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = load_ppmi_subjects(args.xlsx)
    cohort_ids = cohort_ids_from_data()
    missing_xlsx = [i for i in cohort_ids if i not in subjects]
    if missing_xlsx:
        raise SystemExit(f"IDs missing from PPMI xlsx: {missing_xlsx[:10]} ...")

    strata = [stratum_for_subject(subjects[i]) for i in cohort_ids]
    if "pd_nhy_missing" in strata:
        raise SystemExit("Some PD subjects lack NHY in PPMI_MASTER_v2.xlsx")

    train_ids, test_ids = train_test_split(
        cohort_ids,
        test_size=args.test_fraction,
        stratify=strata,
        random_state=args.seed,
    )
    train_set = set(train_ids)
    test_set = set(test_ids)
    assert not train_set & test_set

    write_ids(out_dir / "train.txt", train_ids)
    write_ids(out_dir / "test.txt", test_ids)
    write_ids(out_dir / "validation.txt", test_ids)

    info = pd.read_csv(ROOT / "data_info" / "data_info.csv", dtype={"ID": str})
    info_nhy = info[info["ID"].isin(cohort_ids)].copy()
    info_nhy["split"] = info_nhy["ID"].apply(lambda x: "train" if x in train_set else "test")
    info_nhy.to_csv(out_dir / "data_info.csv", index=False)

    rows = []
    for sid in cohort_ids:
        subj = subjects[sid]
        cohort = str(subj.get("COHORT", "")).split(".")[0]
        nhy = str(subj.get("NHY", "")).split(".")[0] if cohort == "1" else ""
        label_row = info_nhy.loc[info_nhy["ID"] == sid, "label"]
        label = int(label_row.iloc[0]) if len(label_row) else -1
        rows.append(
            {
                "ID": sid,
                "label": label,
                "COHORT": cohort,
                "NHY": nhy,
                "stratum": stratum_for_subject(subj),
                "split": "train" if sid in train_set else "test",
            }
        )
    dist_df = pd.DataFrame(rows)
    dist_df.to_csv(out_dir / "nhy_distribution.csv", index=False)

    primary_train = set(read_ids(ROOT / "data_info" / "train.txt"))
    primary_test = set(read_ids(ROOT / "data_info" / "test.txt"))
    moved_to_train = sorted(test_set & primary_test, key=int)
    moved_to_test = sorted(train_set & primary_train, key=int)

    def nhy_counts(ids: set[str]) -> dict[str, int]:
        sub = dist_df[dist_df["ID"].isin(ids)]
        pd_sub = sub[sub["COHORT"] == "1"]
        return {
            "n": len(sub),
            "control": int((sub["COHORT"] == "2").sum()),
            "pd_nhy1": int((pd_sub["NHY"] == "1").sum()),
            "pd_nhy2": int((pd_sub["NHY"] == "2").sum()),
        }

    manifest = {
        "split_profile": "nhy_v1",
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "n_total": len(cohort_ids),
        "n_train": len(train_set),
        "n_test": len(test_set),
        "stratification": ["control", "pd_nhy1", "pd_nhy2"],
        "primary_split": {
            "moved_test_to_train": moved_to_train,
            "moved_train_to_test": moved_to_test,
            "n_moved_test_to_train": len(moved_to_train),
            "n_moved_train_to_test": len(moved_to_test),
        },
        "distribution": {
            "nhy_v1_train": nhy_counts(train_set),
            "nhy_v1_test": nhy_counts(test_set),
            "primary_train": nhy_counts(primary_train),
            "primary_test": nhy_counts(primary_test),
        },
        "paths": {
            "train": str(out_dir / "train.txt"),
            "test": str(out_dir / "test.txt"),
            "validation": str(out_dir / "validation.txt"),
        },
    }
    with open(out_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    for sid in cohort_ids:
        resolve_nifti_path(str(ROOT / "data" / "whole_MRI"), sid)

    print(f"Wrote split to {out_dir}")
    print(json.dumps(manifest["distribution"], indent=2))
    print(f"Moved test->train: {len(moved_to_train)} | train->test: {len(moved_to_test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
