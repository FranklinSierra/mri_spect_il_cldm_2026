"""PD-only H&Y staging (HY1 vs HY2): cohort, neutral prompts, and artifact export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CLINICAL_STAGING_ROOT = REPO_ROOT / "result" / "clinical_staging_pd"
DEFAULT_CLINICAL_CSV = REPO_ROOT / "MDS-UPDRS_Part_III_28Oct2025_estandarizado_pd_mbrs.csv"
DEFAULT_NHY_CSV = REPO_ROOT / "data_info" / "splits" / "nhy_v1" / "nhy_distribution.csv"
DEFAULT_RADIOLOGY_ROOT = (
    REPO_ROOT
    / "result"
    / "exp_nm_s2_frozen_calib_roi3ch_uni_rois_reproc"
    / "inference"
    / "calib_roi3ch_uni_rois_reproc"
)
DEFAULT_TRAIN_SPLIT = REPO_ROOT / "data_info" / "train.txt"
DEFAULT_TEST_SPLIT = REPO_ROOT / "data_info" / "test.txt"
DEFAULT_PARENT_EXPERIMENT = "calib_roi3ch_uni_rois_reproc"

STAGING_STRATA = frozenset({"pd_nhy1", "pd_nhy2"})
STAGE_TO_LABEL = {"pd_nhy1": 0, "pd_nhy2": 1}
CLASS_NAMES = ["hy1", "hy2"]

EXCLUDED_PROMPT_COLUMNS = frozenset(
    {
        "NHY",
        "H & Y Stage",
        "H&Y Stage",
        "H & Y",
        "HY Stage",
        "COHORT_DEFINITION",
        "PATNO",
        "EVENT_ID",
        "grupo",
        "AGE_AT_VISIT",
        "SEX",
        "Genero",
        "Edad",
        "ID",
        "split",
        "stratum",
        "stage_label",
        "representation_path",
        "text_embedding_path",
        "prompt",
    }
)

DESCRIPCIONES_EN: dict[str, str] = {
    "Edad": "age",
    "Genero": "gender",
    "SEX": "sex",
    "AGE_AT_VISIT": "age at visit",
    "NP3BRADY": "bradykinesia",
    "NP3FACXP": "reduced facial expression",
    "NP3FRZGT": "freezing of gait",
    "NP3FTAPL": "left foot tapping",
    "NP3FTAPR": "right foot tapping",
    "NP3GAIT": "gait impairment",
    "NP3HMOVL": "left hand movement",
    "NP3HMOVR": "right hand movement",
    "NP3KTRML": "left hand kinesia",
    "NP3KTRMR": "right hand kinesia",
    "NP3LGAGL": "left leg agility",
    "NP3LGAGR": "right leg agility",
    "NP3POSTR": "posture",
    "NP3PRSPL": "left hand postural tremor",
    "NP3PRSPR": "right hand postural tremor",
    "NP3PSTBL": "postural instability",
    "NP3PTRML": "left hand resting tremor",
    "NP3PTRMR": "right hand resting tremor",
    "NP3RIGLL": "left leg rigidity",
    "NP3RIGLU": "left arm rigidity",
    "NP3RIGN": "neck rigidity",
    "NP3RIGRL": "right leg rigidity",
    "NP3RIGRU": "right arm rigidity",
    "NP3RISNG": "difficulty rising from chair",
    "NP3RTALJ": "left leg action tremor",
    "NP3RTALL": "left leg action tremor",
    "NP3RTALU": "left arm action tremor",
    "NP3RTARL": "right leg action tremor",
    "NP3RTARU": "right arm action tremor",
    "NP3RTCON": "tremor constancy",
    "NP3SPCH": "speech impairment",
    "NP3TOT": "MDS-UPDRS III total score",
    "NP3TTAPL": "left hand tapping",
    "NP3TTAPR": "right hand tapping",
    "DBSYN": "deep brain stimulation symptoms",
    "DYSKPRES": "dyskinesia presence",
    "DYSKIRAT": "dyskinesia rating",
    "MBRS": "dyskinesia scale",
    "H & Y Stage": "Hoehn & Yahr stage",
    "NHY": "numeric Hoehn & Yahr stage",
    "DBSOFFTM": "time DBS switched off",
    "DBSONTM": "time DBS switched on",
    "HRDBSOFF": "hours from DBS off to NUPDRS3 exam",
    "HRDBSON": "hours from DBS on to NUPDRS3 exam",
    "HRPOSTMED": "hours from last PD medication to NUPDRS3 exam",
    "OFFEXAM": "OFF exam performed",
    "OFFNORSN": "reason OFF exam not performed",
    "ONEXAM": "ON exam performed",
    "ONNORSN": "reason ON exam not performed",
    "ONOFFORDER": "order of OFF/ON Part III exams",
    "PDMEDYN": "Parkinson medication",
    "PDSTATE": "current functional state",
    "PDTRTMNT": "treatment received",
    "Pat_id": "patient ID",
    "PATNO": "patient ID",
    "grupo": "patient group",
    "HIFUYN": "HIFU performed",
    "DBSOFFYN": "DBS off during exam",
    "DBSONYN": "DBS on during exam",
}


def load_split_ids(train_path: Path | str, test_path: Path | str) -> dict[str, set[str]]:
    train_path = Path(train_path)
    test_path = Path(test_path)
    train_ids = {line.strip() for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    test_ids = {line.strip() for line in test_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"Primary split overlap between train and test: {sorted(overlap)[:5]}")
    return {"train": train_ids, "test": test_ids}


def load_staging_label_map(nhy_csv: Path | str = DEFAULT_NHY_CSV) -> pd.DataFrame:
    df = pd.read_csv(nhy_csv, dtype={"ID": str})
    df = df[df["stratum"].isin(STAGING_STRATA)].copy()
    df["stage_label"] = df["stratum"].map(STAGE_TO_LABEL)
    if df["stage_label"].isna().any():
        bad = df.loc[df["stage_label"].isna(), "stratum"].unique().tolist()
        raise ValueError(f"Unknown staging stratum: {bad}")
    df["stage_label"] = df["stage_label"].astype(int)
    return df[["ID", "stratum", "NHY", "stage_label"]].copy()


def assign_primary_split(ids: pd.Series, split_map: dict[str, set[str]]) -> pd.Series:
    def _split_for_id(subject_id: str) -> str:
        if subject_id in split_map["train"]:
            return "train"
        if subject_id in split_map["test"]:
            return "test"
        raise ValueError(f"ID {subject_id} not in primary train/test split files")

    return ids.astype(str).map(_split_for_id)


def _format_sex(value: Any) -> str:
    if pd.isna(value):
        return "not reported"
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return str(value)
    return "male" if numeric == 1 else "female"


def _format_value(col: str, value: Any) -> str:
    if pd.isna(value):
        return "not reported"
    if col in {"Genero", "SEX"}:
        return _format_sex(value)
    return str(value)


def build_neutral_prompt_sinhy(row: pd.Series, columns: list[str]) -> str:
    age = row.get("AGE_AT_VISIT", row.get("Edad", pd.NA))
    sex_col = "SEX" if "SEX" in row.index else "Genero"
    sex = _format_sex(row.get(sex_col, pd.NA))
    age_text = "not reported" if pd.isna(age) else str(age)
    parts = [f"Subject aged {age_text}. Sex: {sex}."]

    for col in columns:
        if col in EXCLUDED_PROMPT_COLUMNS:
            continue
        if col.startswith("prompts"):
            continue
        desc = DESCRIPCIONES_EN.get(col, col.replace("_", " ").lower())
        parts.append(f"{desc}: {_format_value(col, row[col])}")

    return " ".join(parts)


def build_prompts_for_cohort(clinical_df: pd.DataFrame) -> pd.Series:
    prompt_cols = [c for c in clinical_df.columns if c not in EXCLUDED_PROMPT_COLUMNS]
    return clinical_df.apply(lambda row: build_neutral_prompt_sinhy(row, prompt_cols), axis=1)


def _repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_clinical_csv(clinical_csv: Path | str = DEFAULT_CLINICAL_CSV) -> pd.DataFrame:
    df = pd.read_csv(clinical_csv, dtype={"PATNO": str})
    df["ID"] = df["PATNO"].astype(str)
    return df


def load_radiology_metadata(radiology_root: Path | str = DEFAULT_RADIOLOGY_ROOT) -> pd.DataFrame | None:
    metadata_path = Path(radiology_root) / "metadata.csv"
    if not metadata_path.is_file():
        return None
    meta = pd.read_csv(metadata_path, dtype={"ID": str})
    return meta


def build_clinical_cohort(
    clinical_csv: Path | str = DEFAULT_CLINICAL_CSV,
    nhy_csv: Path | str = DEFAULT_NHY_CSV,
    train_split: Path | str = DEFAULT_TRAIN_SPLIT,
    test_split: Path | str = DEFAULT_TEST_SPLIT,
    radiology_root: Path | str = DEFAULT_RADIOLOGY_ROOT,
    require_radiology: bool = False,
) -> pd.DataFrame:
    split_map = load_split_ids(train_split, test_split)
    labels = load_staging_label_map(nhy_csv)
    clinical = load_clinical_csv(clinical_csv)

    cohort = labels.merge(clinical, on="ID", how="left", validate="one_to_one")
    missing_clinical = cohort[cohort["PATNO"].isna()]["ID"].tolist()
    if missing_clinical:
        raise ValueError(f"PD staging IDs missing in clinical CSV: {missing_clinical[:10]}")

    cohort["split"] = assign_primary_split(cohort["ID"], split_map)
    clinical_cols = [
        c
        for c in clinical.columns
        if c not in EXCLUDED_PROMPT_COLUMNS and not c.startswith("prompts")
    ]
    prompt_by_id = {
        str(row["ID"]): build_neutral_prompt_sinhy(row, clinical_cols)
        for _, row in clinical.iterrows()
    }
    cohort["prompt"] = cohort["ID"].map(prompt_by_id)

    meta = load_radiology_metadata(radiology_root)
    if meta is None:
        if require_radiology:
            raise FileNotFoundError(f"Missing radiology metadata under {radiology_root}")
        cohort["representation_path"] = pd.NA
        return cohort

    meta = meta.copy()
    meta["ID"] = meta["ID"].astype(str)
    merged = cohort.merge(
        meta[["ID", "representation_path"]],
        on="ID",
        how="left",
        validate="one_to_one",
    )
    missing_rep = merged[merged["representation_path"].isna()]["ID"].tolist()
    if missing_rep:
        raise ValueError(f"IDs missing radiology representation_path: {missing_rep[:10]}")
    for idx, row in merged.iterrows():
        rep_path = Path(row["representation_path"])
        if not rep_path.is_file():
            rep_path = REPO_ROOT / row["representation_path"]
        if not rep_path.is_file():
            raise FileNotFoundError(f"Missing representation file for ID {row['ID']}: {rep_path}")
        merged.at[idx, "representation_path"] = _repo_relative(rep_path.resolve())

    return merged


def split_summary(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"train": {}, "test": {}, "n_train": 0, "n_test": 0}
    for split_name in ("train", "test"):
        part = df[df["split"] == split_name]
        summary[split_name] = {
            CLASS_NAMES[0]: int((part["stage_label"] == 0).sum()),
            CLASS_NAMES[1]: int((part["stage_label"] == 1).sum()),
        }
        summary[f"n_{split_name}"] = int(len(part))
    return summary


def export_clinical_staging_artifacts(
    out_root: Path | str = DEFAULT_CLINICAL_STAGING_ROOT,
    clinical_csv: Path | str = DEFAULT_CLINICAL_CSV,
    nhy_csv: Path | str = DEFAULT_NHY_CSV,
    train_split: Path | str = DEFAULT_TRAIN_SPLIT,
    test_split: Path | str = DEFAULT_TEST_SPLIT,
    radiology_root: Path | str = DEFAULT_RADIOLOGY_ROOT,
    parent_experiment: str = DEFAULT_PARENT_EXPERIMENT,
    require_radiology: bool = False,
) -> pd.DataFrame:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cohort = build_clinical_cohort(
        clinical_csv=clinical_csv,
        nhy_csv=nhy_csv,
        train_split=train_split,
        test_split=test_split,
        radiology_root=radiology_root,
        require_radiology=require_radiology,
    )

    text_emb_dir = out_root / "text_embeddings" / "reps"
    text_emb_dir.mkdir(parents=True, exist_ok=True)

    pairs = cohort[
        [
            "ID",
            "split",
            "stratum",
            "stage_label",
            "prompt",
            "representation_path",
        ]
    ].copy()
    pairs["text_embedding_path"] = pairs["ID"].map(
        lambda sid: _repo_relative(text_emb_dir / f"{sid}.npy")
    )
    pairs.to_csv(out_root / "clinical_pairs.csv", index=False)

    cohort[["ID", "prompt"]].to_csv(out_root / "prompts.csv", index=False)

    run_config = {
        "analysis_id": "clinical_staging_pd_nhy12",
        "parent_experiment": parent_experiment,
        "parent_radiology_root": _repo_relative(Path(radiology_root)),
        "clinical_csv": _repo_relative(Path(clinical_csv)),
        "label_source": _repo_relative(Path(nhy_csv)),
        "split_source": {
            "train": _repo_relative(Path(train_split)),
            "test": _repo_relative(Path(test_split)),
        },
        "split_policy": "primary data_info train/test; PD-only pd_nhy1 vs pd_nhy2",
        "prompt_variant": "neutral_sinHY_en",
        "class_map": STAGE_TO_LABEL,
        "class_names": CLASS_NAMES,
        "out_root": _repo_relative(out_root.resolve()),
        "split_summary": split_summary(cohort),
        "n_subjects": int(len(cohort)),
    }
    with open(out_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    return pairs
