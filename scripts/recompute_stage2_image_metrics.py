#!/usr/bin/env python3
"""Re-evaluate stage-2 image PSNR/SSIM for a finished run and rewrite CSVs.

Use when training wrote image metrics before minmax restore was applied inside
``batch_metric_scores`` (GT in [0,1] vs decoder tanh [-1,1]). Latent MSE/MAE
columns are left unchanged; only PSNR/SSIM (and ROI) are refreshed for the
saved ``Unet.pth.tar`` checkpoint.

Example:
    python3 scripts/recompute_stage2_image_metrics.py \\
        calib_roi3ch_uni_rois_reproc_x0_minmax_kl --gpu 0
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "experiments" / "registry.json"


def _load_registry():
    import json

    with open(REGISTRY_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_env(registry, key):
    experiments = registry["experiments"]
    if key not in experiments:
        raise SystemExit(f"Unknown key {key!r}")
    entry = experiments[key]
    env = {}
    base_key = entry.get("base")
    if base_key:
        env.update(registry[base_key])
    env.update(entry.get("overrides", {}))
    env["ILCLDM_EXP"] = entry["exp"].rstrip("/") + "/"
    if entry.get("parent"):
        env["ILCLDM_PARENT_EXPERIMENT"] = entry["parent"]
    env["ILCLDM_REGISTRY_KEY"] = key
    return entry, env


def _archive(path: Path) -> Path | None:
    if not path.is_file():
        return None
    dest = path.with_name(path.stem + "_buggy_prerestore" + path.suffix)
    if dest.exists():
        dest = path.with_name(path.stem + "_buggy_prerestore_bak" + path.suffix)
    shutil.copy2(path, dest)
    return dest


def _best_latent_epoch(latent_csv: Path) -> int | None:
    if not latent_csv.is_file():
        return None
    with open(latent_csv, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    best = min(rows, key=lambda r: float(r["latent_MSE"]))
    return int(best["Epoch"])


def _write_image_csv(path: Path, epoch: int, metrics: dict, with_roi: bool) -> None:
    header = ["Epoch", "PSNR", "SSIM"]
    row = [epoch, metrics["psnr"], metrics["ssim"]]
    if with_roi:
        header.extend(["PSNR_roi", "SSIM_roi"])
        row.extend([metrics.get("psnr_roi", 0.0), metrics.get("ssim_roi", 0.0)])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(row)


def _patch_latent_csv(path: Path, epoch: int, metrics: dict) -> None:
    if not path.is_file():
        return
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else [
            "Epoch",
            "latent_MSE",
            "latent_MAE",
            "PSNR",
            "SSIM",
        ]
    for row in rows:
        if int(row["Epoch"]) == epoch:
            row["PSNR"] = str(metrics["psnr"])
            row["SSIM"] = str(metrics["ssim"])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key", help="Registry key of the finished stage-2 run")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES index")
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch label to write (default: best latent_MSE in validation_latent.csv)",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    registry = _load_registry()
    entry, env = _resolve_env(registry, args.key)
    for name, value in env.items():
        os.environ[name] = str(value)

    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))

    import config
    from code.roi_mask_utils import try_load_roi_mask_numpy
    from inference import (
        apply_manifest_to_config,
        load_mri_encoder,
        resolve_unet_build_config,
    )
    from main import Diffusion, evaluate_ldm_split
    from model import AAE, build_unet
    from utils import load_checkpoint

    apply_manifest_to_config(entry["exp"])
    result_dir = REPO_ROOT / "result" / entry["exp"].rstrip("/")
    ckpt = result_dir / "Unet.pth.tar"
    if not ckpt.is_file():
        raise SystemExit(f"Missing checkpoint: {ckpt}")

    device = config.device
    aae = AAE().to(device)
    load_checkpoint(config.CHECKPOINT_AAE, aae, None, config.learning_rate)
    aae.eval()

    mri_encoder = None
    mode = getattr(config, "MRI_ENCODER_MODE", "frozen")
    if mode not in ("none", "raw", ""):
        mri_encoder, _ = load_mri_encoder(
            mode, config.MRI_ENCODER_CHECKPOINT, device
        )

    use_roi, in_ch = resolve_unet_build_config(entry["exp"], ckpt)
    unet = build_unet(use_roi_channel=use_roi, in_channel=in_ch).to(device)
    load_checkpoint(str(ckpt), unet, None, config.learning_rate)
    unet.eval()

    diffusion = Diffusion()
    roi_mask_np = try_load_roi_mask_numpy()
    epoch = args.epoch or _best_latent_epoch(result_dir / "validation_latent.csv") or 0

    print(f"[recompute] exp={entry['exp']} ckpt={ckpt} epoch_label={epoch}")
    print(f"[recompute] IMAGE_VOLUME_NORM={config.IMAGE_VOLUME_NORM}")

    for split_name, task, stage, image_csv, latent_csv in (
        (
            "validation",
            config.validation,
            "validation",
            result_dir / "validation.csv",
            result_dir / "validation_latent.csv",
        ),
        (
            "test",
            config.test,
            "test",
            result_dir / "test.csv",
            result_dir / "test_latent.csv",
        ),
    ):
        archived = _archive(image_csv)
        if archived:
            print(f"[recompute] archived {image_csv.name} -> {archived.name}")
        metrics = evaluate_ldm_split(
            diffusion,
            unet,
            aae,
            mri_encoder,
            task,
            stage,
            roi_mask_np=roi_mask_np,
            track_latent=True,
        )
        _write_image_csv(image_csv, epoch, metrics, with_roi=roi_mask_np is not None)
        _patch_latent_csv(latent_csv, epoch, metrics)
        print(
            f"[recompute] {split_name}: PSNR={metrics['psnr']:.4f} "
            f"SSIM={metrics['ssim']:.4f} "
            f"PSNR_roi={metrics.get('psnr_roi', float('nan')):.4f} "
            f"latent_MSE={metrics.get('latent_mse')}"
        )

    note = result_dir / "IMAGE_METRICS_FIX.txt"
    note.write_text(
        "Image PSNR/SSIM in validation.csv / test.csv were recomputed with "
        "prepare_prediction_for_metrics (minmax tanh [-1,1] -> [0,1]).\n"
        "Per-epoch history from the buggy train loop is in *_buggy_prerestore.csv.\n"
        f"Registry key: {args.key}\n"
        f"Checkpoint: {ckpt.name}\n"
        f"Epoch label (best latent at recompute time): {epoch}\n",
        encoding="utf-8",
    )
    print(f"[recompute] wrote {note}")


if __name__ == "__main__":
    main()
