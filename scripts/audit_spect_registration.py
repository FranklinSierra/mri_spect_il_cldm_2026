#!/usr/bin/env python3
"""Per-subject SPECT re-registration for miscoregistered whole_Abeta volumes.

Strategy 1B (atlas-anchored): the flagged SPECT volumes share a systematic
spatial displacement (~40 mm anterior + ~18 mm inferior) with respect to the
MNI basal-ganglia atlas mask. To recover the alignment we:

  1. Anchor to the FIXED atlas mask (a clean MNI landmark), NOT to the noisy
     per-subject brain centre of mass (which over-shifted volumes previously).
  2. Coarse init: translate the SPECT so that the centre of mass of its uptake
     (voxels > p90) matches  COM(atlas mask) + REF_UPTAKE_OFFSET, where the
     reference offset is the median relationship measured over the 166 well
     registered ("OK") subjects. This places a displaced subject exactly where
     the good cohort sits in MNI space.
  3. Fine registration: rigid mutual-information refinement of SPECT -> subject
     MRI, initialised with the coarse translation, with a safeguard that reverts
     to the translation-only result if the refinement drifts.

Why not align uptake COM directly onto the mask COM?  At p90 the SPECT "uptake"
is essentially whole-brain (~700k voxels), so its COM sits ~30 vox SUPERIOR to
the (inferior-central) basal-ganglia mask even in perfectly registered subjects.
Aligning it straight onto the mask would push every volume ~30 vox down. The
REF_UPTAKE_OFFSET encodes that natural relationship so the correction matches
the good cohort.

Non-destructive: corrected volumes + transforms + overlays are written to a
separate folder; data/whole_Abeta is never modified.

Requires: antspyx, scipy, nibabel, matplotlib.

Example
-------
    python scripts/audit_spect_registration.py                 # flagged IDs
    python scripts/audit_spect_registration.py --ids 3107 3134
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import ants
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# IDs flagged as registration failures by the whole_Abeta QC audit
# (docs/reports/whole_Abeta_registration_summary.csv, fail_type != OK).
DEFAULT_FLAGGED = [
    3107, 3134, 3174, 3185, 3307, 3309, 3360, 3367, 3591, 3592,
    3593, 3826, 3829, 3830, 3834, 3838,
]

# Median COM(SPECT uptake > p90) - COM(atlas mask), measured over the 166 "OK"
# subjects (full-MNI voxel space). std ~ [1.5, 3.6, 6.8], i.e. a tight, stable
# landmark relationship. Recompute with --recompute-reference if the cohort
# or mask changes.
REF_UPTAKE_OFFSET = np.array([-1.64, 7.17, 30.09], dtype=float)

RIM = 3  # voxels zeroed at each face to remove boundary reconstruction artifacts
UPTAKE_PCT = 90.0  # percentile defining "uptake" voxels for COM anchoring


# --------------------------------------------------------------------------- #
# Preprocessing helpers
# --------------------------------------------------------------------------- #
def rim_zero(vol: np.ndarray, rim: int = RIM) -> np.ndarray:
    """Zero a boundary shell; guards against any boundary reconstruction spikes."""
    out = vol.astype(np.float32, copy=True)
    out[:rim] = 0
    out[-rim:] = 0
    out[:, :rim] = 0
    out[:, -rim:] = 0
    out[:, :, :rim] = 0
    out[:, :, -rim:] = 0
    return out


def clip_prep(img: "ants.ANTsImage", pct: float, rim: int = RIM) -> "ants.ANTsImage":
    """Rim-zero + upper-percentile clip so MI is not dominated by outliers."""
    data = rim_zero(img.numpy(), rim)
    pos = data[data > 0]
    hi = float(np.percentile(pos, pct)) if pos.size else float(data.max())
    data = np.clip(data, 0.0, hi)
    return ants.from_numpy(
        data, origin=img.origin, spacing=img.spacing, direction=img.direction
    )


def brain_mask_np(vol: np.ndarray, pct: float = 40.0, rim: int = RIM) -> np.ndarray:
    """Filled brain/head mask from a rim-zeroed volume via a positive-percentile cut."""
    data = rim_zero(vol, rim)
    pos = data[data > 0]
    if pos.size == 0:
        return np.zeros(vol.shape, dtype=bool)
    return ndi.binary_fill_holes(data > float(np.percentile(pos, pct)))


def brain_mask_image(mri: "ants.ANTsImage", dilate: int = 6) -> "ants.ANTsImage":
    """Dilated MRI brain mask used to focus the MI metric on the head."""
    mask = ndi.binary_dilation(brain_mask_np(mri.numpy(), 40.0, rim=0), iterations=dilate)
    return ants.from_numpy(
        mask.astype(np.float32),
        origin=mri.origin,
        spacing=mri.spacing,
        direction=mri.direction,
    )


def uptake_com(vol: np.ndarray, pct: float = UPTAKE_PCT, rim: int = RIM) -> np.ndarray:
    """Centre of mass of the SPECT uptake voxels (> percentile of positives)."""
    data = rim_zero(vol, rim)
    pos = data[data > 0]
    if pos.size == 0:
        return np.array(vol.shape, dtype=float) / 2.0
    uptake = data > float(np.percentile(pos, pct))
    idx = np.array(np.where(uptake))
    return idx.mean(axis=1) if idx.size else np.array(vol.shape, dtype=float) / 2.0


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def uptake_ref_deviation(
    spect_np: np.ndarray, mask_com: np.ndarray
) -> float:
    """Distance (vox) of uptake COM from its expected good-cohort position.

    0 == exactly where well-registered subjects sit; the flagged volumes start
    at ~40-50 vox before correction.
    """
    target = mask_com + REF_UPTAKE_OFFSET
    return float(np.linalg.norm(uptake_com(spect_np) - target))


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    b = b > 0
    denom = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / denom) if denom else 0.0


# --------------------------------------------------------------------------- #
# Initialisation + registration
# --------------------------------------------------------------------------- #
def atlas_com_init(
    mri: "ants.ANTsImage",
    spect: "ants.ANTsImage",
    mask_com: np.ndarray,
    tmp_dir: Path,
) -> str:
    """Translation putting SPECT uptake COM at COM(mask) + REF_UPTAKE_OFFSET.

    Anchored to the fixed atlas landmark and calibrated to the good cohort, so a
    displaced subject lands exactly where well-registered subjects sit in MNI.
    """
    src_idx = uptake_com(spect.numpy())
    dst_idx = mask_com + REF_UPTAKE_OFFSET
    src_pt = np.array(
        ants.transform_index_to_physical_point(spect, [int(round(x)) for x in src_idx])
    )
    dst_pt = np.array(
        ants.transform_index_to_physical_point(mri, [int(round(x)) for x in dst_idx])
    )
    tx = ants.new_ants_transform(transform_type="AffineTransform", dimension=3)
    # apply_transforms maps fixed->moving; to move content from src to dst the
    # translation is (src_pt - dst_pt).
    params = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, *(src_pt - dst_pt)], dtype=float)
    ants.set_ants_transform_parameters(tx, params)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / f"atlasinit_{abs(hash(tuple(src_pt - dst_pt)))}.mat"
    ants.write_transform(tx, str(out))
    return str(out)


def register_subject(
    sid: str, data_root: Path, mask_com: np.ndarray, tmp_dir: Path
) -> tuple["ants.ANTsImage", "ants.ANTsImage", "ants.ANTsImage", list[str], str]:
    mri = ants.image_read(str(data_root / "whole_MRI" / f"{sid}.nii.gz"))
    spect = ants.image_read(str(data_root / "whole_Abeta" / f"{sid}.nii.gz"))

    mri_prep = clip_prep(mri, 99.5, rim=0)
    spect_prep = clip_prep(spect, 99.0, rim=RIM)
    bmask = brain_mask_image(mri)
    init = atlas_com_init(mri, spect, mask_com, tmp_dir)

    # Output source: original intensities, boundary shell zeroed.
    spect_out = ants.from_numpy(
        rim_zero(spect.numpy(), RIM),
        origin=spect.origin,
        spacing=spect.spacing,
        direction=spect.direction,
    )

    # Coarse (translation-only) result from the atlas anchor.
    warped_init = ants.apply_transforms(
        fixed=mri, moving=spect_out, transformlist=[init], interpolator="linear"
    )
    dev_init = uptake_ref_deviation(warped_init.numpy(), mask_com)

    # Fine rigid MI refinement against the subject MRI.
    reg = ants.registration(
        fixed=mri_prep,
        moving=spect_prep,
        type_of_transform="Rigid",
        initial_transform=init,
        aff_metric="mattes",
        mask=bmask,
    )
    warped_rig = ants.apply_transforms(
        fixed=mri, moving=spect_out, transformlist=reg["fwdtransforms"], interpolator="linear"
    )
    dev_rig = uptake_ref_deviation(warped_rig.numpy(), mask_com)

    # Safeguard: keep the fine refinement only if it stays close to the anchor.
    if dev_rig <= dev_init + 4.0:
        return mri, spect, warped_rig, reg["fwdtransforms"], "rigid"
    return mri, spect, warped_init, [init], "coarse_only"


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def _slice_at(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return vol[idx, :, :].T
    if axis == 1:
        return vol[:, idx, :].T
    return vol[:, :, idx].T


def overlay_figure(
    sid: str,
    mri: np.ndarray,
    spect_orig: np.ndarray,
    spect_reg: np.ndarray,
    mask: np.ndarray,
    mask_com: np.ndarray,
    dev_before: float,
    dev_after: float,
    out_path: Path,
) -> None:
    # Slice through the atlas mask centre so the striatal region is always shown.
    cx, cy, cz = [int(round(v)) for v in mask_com]
    planes = [(2, cz, "axial"), (1, cy, "coronal"), (0, cx, "sagittal")]
    reg_pos = spect_reg[spect_reg > 0]
    vmax = float(np.percentile(reg_pos, 99.5)) if reg_pos.size else 1.0

    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    col_titles = ["MRI", "SPECT original", "SPECT corregido", "Corregido + máscara"]
    for r, (axis, idx, pname) in enumerate(planes):
        mri_sl = _slice_at(mri, axis, idx)
        so_sl = _slice_at(spect_orig, axis, idx)
        sr_sl = _slice_at(spect_reg, axis, idx)
        mk_sl = _slice_at(mask.astype(float), axis, idx)

        axes[r, 0].imshow(mri_sl, cmap="gray", origin="lower")
        axes[r, 1].imshow(
            np.ma.masked_where(so_sl <= 0, so_sl), cmap="hot", origin="lower", vmax=vmax
        )
        axes[r, 2].imshow(
            np.ma.masked_where(sr_sl <= 0, sr_sl), cmap="hot", origin="lower", vmax=vmax
        )
        axes[r, 3].imshow(mri_sl, cmap="gray", origin="lower")
        axes[r, 3].imshow(
            np.ma.masked_where(sr_sl < vmax * 0.35, sr_sl),
            cmap="hot",
            alpha=0.6,
            origin="lower",
            vmax=vmax,
        )
        if mk_sl.max() > 0:
            axes[r, 3].contour(
                mk_sl, levels=[0.5], colors="lime", linewidths=0.9, origin="lower"
            )
        for c in range(4):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=11)
        axes[r, 0].set_ylabel(pname, fontsize=11)

    fig.suptitle(
        f"Sujeto {sid} — reg. Estrategia 1B (uptake→atlas + rígido) | "
        f"desviación uptake vs referencia: {dev_before:.1f} → {dev_after:.1f} vox",
        fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_flagged_ids(summary_csv: Path) -> list[str]:
    if not summary_csv.is_file():
        return [str(i) for i in DEFAULT_FLAGGED]
    ids: list[str] = []
    with summary_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("fail_type", "OK") != "OK":
                sid = int(float(row["ID"]))
                ids.append(str(sid))
    return ids or [str(i) for i in DEFAULT_FLAGGED]


def compute_reference_offset(data_root: Path, summary_csv: Path, mask_com: np.ndarray) -> np.ndarray:
    import pandas as pd

    summ = pd.read_csv(summary_csv)
    ok_ids = summ.loc[summ["fail_type"] == "OK", "ID"].astype(int).tolist()
    offs = []
    for sid in ok_ids:
        p = data_root / "whole_Abeta" / f"{sid}.nii.gz"
        if not p.is_file():
            continue
        vol = nib.load(str(p)).get_fdata(dtype=np.float32)
        offs.append(uptake_com(vol) - mask_com)
    offs = np.array(offs)
    med = np.median(offs, axis=0)
    print(f"Recomputed REF_UPTAKE_OFFSET over {len(offs)} OK subjects: {np.round(med, 3)}")
    return med


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "whole_Abeta_registered",
        help="Where corrected volumes are written (never overwrites whole_Abeta).",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "reports" / "registration_fixes",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=REPO_ROOT / "data" / "mask_basal_ganglia.nii.gz",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=REPO_ROOT / "docs" / "reports" / "whole_Abeta_registration_summary.csv",
    )
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument(
        "--recompute-reference",
        action="store_true",
        help="Recompute REF_UPTAKE_OFFSET from OK subjects (slow, ~2-3 min).",
    )
    args = parser.parse_args()

    global REF_UPTAKE_OFFSET

    mask = nib.load(str(args.mask)).get_fdata(dtype=np.float32) > 0.5
    mask_com = np.array(np.where(mask)).mean(axis=1)

    if args.recompute_reference:
        REF_UPTAKE_OFFSET = compute_reference_offset(
            args.data_root, args.summary_csv, mask_com
        )

    ids = args.ids if args.ids else load_flagged_ids(args.summary_csv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    tx_dir = args.out_dir / "transforms"
    tx_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.out_dir / "_tmp"

    rows = []
    print(f"Atlas mask COM (MNI vox): {np.round(mask_com, 1)}")
    print(f"REF_UPTAKE_OFFSET: {np.round(REF_UPTAKE_OFFSET, 2)}")
    print(f"Processing {len(ids)} subjects -> {args.out_dir} (non-destructive)\n")

    for sid in ids:
        try:
            mri, spect, warped, fwd, mode = register_subject(
                sid, args.data_root, mask_com, tmp_dir
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {sid}: ERROR {exc}")
            rows.append({"ID": sid, "status": "error", "error": str(exc)})
            continue

        mri_np = mri.numpy()
        so_np = spect.numpy()
        sr_np = warped.numpy()
        dev_before = uptake_ref_deviation(so_np, mask_com)
        dev_after = uptake_ref_deviation(sr_np, mask_com)
        # Signal coverage: well-registered SPECT fills ~97% of voxels; flagged
        # volumes are much sparser (upstream mask likely clipped the signal).
        cov_before = float((so_np > 0).mean() * 100.0)
        cov_after = float((sr_np > 0).mean() * 100.0)

        out_nii = args.out_dir / f"{sid}.nii.gz"
        ants.image_write(warped, str(out_nii))
        if fwd:
            try:
                ants.write_transform(
                    ants.read_transform(fwd[0]), str(tx_dir / f"{sid}.mat")
                )
            except Exception:  # noqa: BLE001
                pass

        fig_path = args.fig_dir / f"{sid}_registration.png"
        overlay_figure(
            sid, mri_np, so_np, sr_np, mask, mask_com, dev_before, dev_after, fig_path
        )

        aligned = dev_after <= 12.0
        low_signal = cov_after < 60.0
        status = "OK" if aligned and not low_signal else (
            "SEÑAL_BAJA" if aligned else "REVISAR"
        )
        print(
            f"  {sid}: uptake dev vs ref {dev_before:5.1f} -> {dev_after:4.1f} vox "
            f"({mode:11s}) | cobertura {cov_after:4.1f}%  [{status}]"
        )
        rows.append(
            {
                "ID": sid,
                "status": "ok",
                "mode": mode,
                "uptake_dev_before": round(dev_before, 2),
                "uptake_dev_after": round(dev_after, 2),
                "signal_coverage_before_pct": round(cov_before, 2),
                "signal_coverage_after_pct": round(cov_after, 2),
                "flag": status,
                "corrected_volume": str(out_nii),
                "overlay": str(fig_path),
            }
        )

    csv_path = args.fig_dir / "registration_metrics.csv"
    if rows:
        keys = sorted({k for r in rows for k in r})
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nCorrected volumes: {args.out_dir}")
    print(f"Overlays + metrics: {args.fig_dir}")
    print("\nIMPORTANTE: whole_Abeta NO fue modificado. Valida los overlays antes")
    print("de sustituir cualquier volumen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
