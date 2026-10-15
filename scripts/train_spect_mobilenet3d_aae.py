#!/usr/bin/env python3
"""Train MobileNetV2-3D on SPECT volumes reconstructed by a frozen E1 AAE (ROI 4 mm).

One CE logit pair per volume. Early-stop on validation balanced accuracy.
Does not train E2 / FST. Defaults target whole_Abeta_4mm + spect_rois_4mm AAE.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SPECT_ROOT = "data_rois/whole_Abeta_4mm"
DEFAULT_AAE_CKPT = "result/exp_nm_s1_spect_rois_4mm/AAE.pth.tar"
DEFAULT_OUT = "result/exp_nm_cls_spect_aae_mnet3d"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spect-root", default=DEFAULT_SPECT_ROOT)
    p.add_argument("--aae-checkpoint", default=DEFAULT_AAE_CKPT)
    p.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT))
    p.add_argument("--meta-csv", type=Path, default=ROOT / "data_info" / "data_info.csv")
    p.add_argument("--train-list", type=Path, default=ROOT / "data_info" / "train.txt")
    p.add_argument("--val-list", type=Path, default=ROOT / "data_info" / "validation.txt")
    p.add_argument("--test-list", type=Path, default=ROOT / "data_info" / "test.txt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force re-encode AAE reconstructions into recon_cache/",
    )
    return p.parse_args()


def refuse_wrong_paths(spect_root: str, aae_ckpt: str) -> None:
    sr = spect_root.replace("\\", "/")
    if "_4mm" not in Path(sr).name and not sr.rstrip("/").endswith("whole_Abeta_4mm"):
        raise SystemExit(f"ERROR: refusing non-4mm SPECT root: {spect_root}")
    if "whole_Abeta_4mm" not in sr:
        raise SystemExit(f"ERROR: expected whole_Abeta_4mm in path, got {spect_root}")
    ck = aae_ckpt.replace("\\", "/")
    if "spect_rois_4mm" not in ck:
        raise SystemExit(f"ERROR: refusing AAE ckpt (expected spect_rois_4mm): {aae_ckpt}")
    if not Path(aae_ckpt).is_file():
        raise SystemExit(f"ERROR: missing AAE checkpoint: {aae_ckpt}")
    if not Path(spect_root).is_dir():
        raise SystemExit(f"ERROR: missing SPECT dir: {spect_root}")


def read_id_list(path: Path) -> list[str]:
    ids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # train.txt may be bare IDs or filenames
        stem = Path(line).name
        if stem.endswith(".nii.gz"):
            stem = stem[: -len(".nii.gz")]
        elif stem.endswith(".nii"):
            stem = stem[: -len(".nii")]
        ids.append(stem)
    return ids


def resolve_nifti(root: Path, subject_id: str) -> Path:
    for name in (f"{subject_id}.nii.gz", f"{subject_id}.nii", subject_id):
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No NIfTI for {subject_id} under {root}")


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ReconVolumeDataset(Dataset):
    def __init__(self, records: list[dict], cache_dir: Path):
        self.records = records
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        sid = rec["ID"]
        path = self.cache_dir / f"{sid}.npy"
        vol = np.load(path).astype(np.float32)
        x = torch.from_numpy(vol).unsqueeze(0)  # (1,D,H,W)
        y = int(rec["label"])
        return x, y, sid


@torch.no_grad()
def build_recon_cache(
    *,
    subject_ids: list[str],
    spect_root: Path,
    aae_ckpt: Path,
    cache_dir: Path,
    device: torch.device,
    rebuild: bool,
) -> None:
    import config
    from dataset import crop, nifti_to_numpy
    from model import AAE
    from utils import load_checkpoint

    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [sid for sid in subject_ids if rebuild or not (cache_dir / f"{sid}.npy").is_file()]
    if not missing:
        print(f"recon_cache complete ({len(subject_ids)} subjects) at {cache_dir}")
        return

    print(f"Building recon cache for {len(missing)}/{len(subject_ids)} subjects → {cache_dir}")
    aae = AAE().to(device)
    opt = torch.optim.Adam(aae.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    load_checkpoint(str(aae_ckpt), aae, opt, config.learning_rate)
    aae.eval()
    for p in aae.parameters():
        p.requires_grad_(False)

    for sid in tqdm(missing, desc="aae_recon_cache"):
        nifti_path = resolve_nifti(spect_root, sid)
        vol = nifti_to_numpy(nifti_path, modality="spect")
        vol = crop(vol)
        tensor = torch.as_tensor(vol, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        recon = aae(tensor).squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        np.save(cache_dir / f"{sid}.npy", recon)

    # Drop reference affine sidecar optional: store shape note
    meta = {
        "n_cached": len(list(cache_dir.glob("*.npy"))),
        "aae_checkpoint": str(aae_ckpt),
        "spect_root": str(spect_root),
        "working_note": "volumes are crop/pad working shape after AAE recon",
    }
    (cache_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2))


def subject_metrics(y_true, y_pred, y_prob) -> dict:
    out = {
        "n_subjects": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[dict, pd.DataFrame]:
    model.eval()
    rows = []
    for xb, yb, sids in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        y_np = yb.numpy()
        for i, sid in enumerate(sids):
            rows.append(
                {
                    "ID": sid,
                    "label": int(y_np[i]),
                    "pred": int(preds[i]),
                    "prob_1": float(probs[i, 1]),
                }
            )
    df = pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)
    metrics = subject_metrics(df["label"].to_numpy(), df["pred"].to_numpy(), df["prob_1"].to_numpy())
    return metrics, df


def main():
    args = parse_args()
    refuse_wrong_paths(args.spect_root, args.aae_checkpoint)
    set_seed(args.seed)

    # Ensure ROI + legacy norm before config/dataset import side effects matter for crop.
    os.environ.setdefault("ILCLDM_SKIP_CROP", "1")
    os.environ["ILCLDM_ROI_VOLUME_SHAPE"] = os.environ.get("ILCLDM_ROI_VOLUME_SHAPE", "96,112,114")
    os.environ["ILCLDM_IMAGE_VOLUME_NORM"] = "legacy"
    os.environ["ILCLDM_WHOLE_ABETA"] = args.spect_root

    # Repo root must precede stdlib so `import code` resolves to ./code/ package.
    if str(ROOT) in sys.path:
        sys.path.remove(str(ROOT))
    sys.path.insert(0, str(ROOT))
    # Drop stdlib `code` if something already imported it (not a package).
    mod = sys.modules.get("code")
    if mod is not None and not hasattr(mod, "__path__"):
        del sys.modules["code"]

    import config  # noqa: F401 — reads env set above
    from code.mobilenet3d import build_mobilenet_v2_3d
    from code.volume_shapes import get_working_volume_shape
    from utils import seed_torch

    seed_torch(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "recon_cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} spect={args.spect_root} aae={args.aae_checkpoint} "
        f"out={out_dir} working_shape={get_working_volume_shape()}"
    )

    meta = pd.read_csv(args.meta_csv, dtype={"ID": str})
    meta["ID"] = meta["ID"].astype(str)
    labels = meta.set_index("ID")["label"].astype(int).to_dict()

    def records_for(list_path: Path) -> list[dict]:
        rows = []
        for sid in read_id_list(list_path):
            if sid not in labels:
                raise KeyError(f"ID {sid} missing from {args.meta_csv}")
            rows.append({"ID": sid, "label": int(labels[sid])})
        return rows

    train_recs = records_for(args.train_list)
    val_recs = records_for(args.val_list)
    test_recs = records_for(args.test_list)
    all_ids = sorted({r["ID"] for r in train_recs + val_recs + test_recs})

    build_recon_cache(
        subject_ids=all_ids,
        spect_root=Path(args.spect_root),
        aae_ckpt=Path(args.aae_checkpoint),
        cache_dir=cache_dir,
        device=device,
        rebuild=args.rebuild_cache,
    )

    def make_loader(recs: list[dict], shuffle: bool) -> DataLoader:
        ds = ReconVolumeDataset(recs, cache_dir)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    train_loader = make_loader(train_recs, shuffle=True)
    val_loader = make_loader(val_recs, shuffle=False)
    test_loader = make_loader(test_recs, shuffle=False)
    print(
        f"subjects train={len(train_recs)} val={len(val_recs)} test={len(test_recs)} "
        f"batch_size={args.batch_size}"
    )

    model = build_mobilenet_v2_3d(
        num_classes=2,
        in_channels=1,
        pretrained=not args.no_pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_val_ba = -1.0
    best_epoch = -1
    patience_left = args.patience
    best_path = out_dir / "best.pt"

    config_payload = {
        "spect_root": args.spect_root,
        "aae_checkpoint": args.aae_checkpoint,
        "out_dir": str(out_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
        "pretrained": not args.no_pretrained,
        "image_volume_norm": "legacy",
        "working_volume_shape": list(get_working_volume_shape()),
        "early_stop_metric": "val_balanced_accuracy",
        "model": "mobilenet_v2_3d_inflated",
        "n_train": len(train_recs),
        "n_val": len(val_recs),
        "n_test": len(test_recs),
    }
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        t0 = time.time()
        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(yb)
            n += len(yb)
        train_loss = running / max(n, 1)

        val_metrics, _ = evaluate(model, val_loader, device)
        val_ba = val_metrics["balanced_accuracy"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_balanced_accuracy": val_ba,
            "val_accuracy": val_metrics["accuracy"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_f1": val_metrics["f1"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} loss={train_loss:.4f} "
            f"val_BA={val_ba:.4f} val_auc={val_metrics['roc_auc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} ({row['seconds']:.1f}s)"
        )

        improved = np.isfinite(val_ba) and val_ba > best_val_ba + 1e-6
        if improved:
            best_val_ba = float(val_ba)
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_balanced_accuracy": best_val_ba,
                    "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                },
                best_path,
            )
        else:
            patience_left -= 1
            print(f"early-stop counter: {args.patience - patience_left}/{args.patience}")
            if patience_left <= 0:
                print(
                    f"Early stop at epoch {epoch} "
                    f"(best epoch={best_epoch}, best_val_BA={best_val_ba:.4f})"
                )
                break

    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    val_metrics, val_df = evaluate(model, val_loader, device)
    test_metrics, test_df = evaluate(model, test_loader, device)
    val_df.to_csv(out_dir / "val_predictions.csv", index=False)
    test_df.to_csv(out_dir / "test_predictions.csv", index=False)

    report = classification_report(
        test_df["label"], test_df["pred"], digits=4, zero_division=0
    )
    (out_dir / "classification_report_test.txt").write_text(report)
    cm = confusion_matrix(test_df["label"], test_df["pred"])
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        out_dir / "confusion_matrix_test.csv"
    )

    metrics_valid = {"best_epoch": best_epoch, "best_val_balanced_accuracy": best_val_ba, **val_metrics}
    metrics_test = {"best_epoch": best_epoch, **test_metrics}
    (out_dir / "metrics_valid.json").write_text(json.dumps(metrics_valid, indent=2))
    (out_dir / "metrics_test.json").write_text(json.dumps(metrics_test, indent=2))

    print("\n=== VAL ===")
    print(json.dumps(metrics_valid, indent=2))
    print("\n=== TEST ===")
    print(report)
    print(json.dumps(metrics_test, indent=2))
    print(f"Saved under {out_dir}")

    if test_metrics["balanced_accuracy"] <= 0.5 + 1e-6:
        print(
            "GATE: test BA is not clearly above chance (0.5). "
            "Do not open FST with 0.8 CE weight on this classifier."
        )
    else:
        print(
            f"GATE: test BA={test_metrics['balanced_accuracy']:.4f} > 0.5 "
            "(still compare vs MRI recon ~0.64 before FST)."
        )


if __name__ == "__main__":
    main()
