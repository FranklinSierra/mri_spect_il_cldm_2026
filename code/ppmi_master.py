"""Read PPMI_MASTER_v2.xlsx without openpyxl (stdlib XML parser)."""

from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
HEADER_ROW = 2
DATA_START_ROW = 3


def _col_idx(col: str) -> int:
    n = 0
    for char in col:
        n = n * 26 + ord(char) - 64
    return n - 1


def _col_ref(cell_ref: str) -> str:
    return re.match(r"([A-Z]+)", cell_ref).group(1)


def _load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for si in root.findall(f"{NS}si"):
        texts = [(t.text or "") for t in si.iter(f"{NS}t")]
        strings.append("".join(texts))
    return strings


def _load_rows(zf: zipfile.ZipFile) -> dict[int, dict[int, str]]:
    strings = _load_shared_strings(zf)
    sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    rows_dict: dict[int, dict[int, str]] = {}
    for row in sheet.findall(f".//{NS}row"):
        r_idx = int(row.get("r"))
        for cell in row.findall(f"{NS}c"):
            ci = _col_idx(_col_ref(cell.get("r")))
            value_el = cell.find(f"{NS}v")
            if value_el is None:
                val = ""
            elif cell.get("t") == "s":
                val = strings[int(value_el.text)]
            else:
                val = value_el.text or ""
            rows_dict.setdefault(r_idx, {})[ci] = val
    return rows_dict


def load_ppmi_subjects(xlsx_path: str | Path) -> dict[str, dict[str, str]]:
    """Return PATNO -> {column_name: value} for first row per subject."""
    path = Path(xlsx_path)
    with zipfile.ZipFile(path) as zf:
        rows_dict = _load_rows(zf)

    header = rows_dict[HEADER_ROW]
    subjects: dict[str, dict[str, str]] = {}
    for r_idx, vals in rows_dict.items():
        if r_idx < DATA_START_ROW:
            continue
        patno_i = next(i for i, name in header.items() if name == "PATNO")
        if patno_i not in vals or not vals[patno_i]:
            continue
        pid = str(int(float(vals[patno_i])))
        if pid in subjects:
            continue
        subjects[pid] = {header[i]: vals.get(i, "") for i in header}

    return subjects


def stratum_for_subject(subject: dict[str, str]) -> str:
    cohort = str(subject.get("COHORT", "")).split(".")[0]
    if cohort == "2":
        return "control"
    nhy = str(subject.get("NHY", "")).strip()
    if not nhy or nhy.lower() == "nan":
        return "pd_nhy_missing"
    return f"pd_nhy{nhy.split('.')[0]}"
