import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    import umap
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    umap = None

import config
from dataset import OneDataset, TwoDataset, resolve_nifti_path
from code.volume_shapes import get_latent_shape, get_volume_shape, get_working_volume_shape
from losses import (
    CrossModalInfoNCE,
    SupervisedContrastiveLoss,
    flatten_feature_volume,
    spect_reconstruction_loss,
)
from main import batch_metric_scores
from model import AAE, Discriminator
from utils import load_checkpoint, seed_torch


CONTRASTIVE_FEATURE_CHOICES = ("z18k", "h64")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a phase-1 MRI autoencoder, save MRI embeddings, and evaluate embedding quality."
    )
    parser.add_argument("--exp", default="exp_2")
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--lr", type=float, default=config.learning_rate)
    parser.add_argument("--validation-interval", type=int, default=config.validation_interval)
    parser.add_argument("--patience", type=int, default=config.early_stopping_patience)
    parser.add_argument("--max-subjects-per-split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument(
        "--embeddings-name",
        default="embeddings",
        help="Subdir under phase1_mri for encoded features (e.g. embeddings_z18k)",
    )
    parser.add_argument(
        "--analysis-name",
        default="embedding_analysis",
        help="Subdir under phase1_mri for classification/viz (e.g. embedding_analysis_z18k)",
    )
    parser.add_argument(
        "--eval-features",
        choices=list(CONTRASTIVE_FEATURE_CHOICES),
        default=None,
        help="Override feature space for encode/analysis only (e.g. z18k on an h64-trained run)",
    )
    parser.add_argument(
        "--training-mode",
        choices=["recon", "supcon", "xmodal"],
        default=config.MRI_TRAINING_MODE,
        help="recon=baseline AAE; supcon=+Supervised Contrastive; xmodal=+cross-modal InfoNCE",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=config.MRI_INIT_CHECKPOINT or None,
        help="Warm-start MRI AAE from this checkpoint (e.g. exp_nm_s1_mri_rois)",
    )
    parser.add_argument(
        "--spect-checkpoint",
        default=config.SPECT_AAE_CHECKPOINT or None,
        help="Frozen SPECT AAE for xmodal mode (e.g. exp_nm_s1_spect_rois/AAE.pth.tar)",
    )
    parser.add_argument("--supcon-lambda", type=float, default=config.SUPCON_LAMBDA)
    parser.add_argument("--xmodal-lambda", type=float, default=config.XMODAL_LAMBDA)
    parser.add_argument("--contrastive-temp", type=float, default=config.CONTRASTIVE_TEMP)
    parser.add_argument(
        "--contrastive-features",
        choices=list(CONTRASTIVE_FEATURE_CHOICES),
        default=getattr(config, "CONTRASTIVE_FEATURES", "z18k"),
        help="z18k=flatten bottleneck latent; h64=flatten pre-bottleneck 64-ch map",
    )
    parser.add_argument(
        "--baseline-exp",
        default="exp_nm_s1_mri_rois",
        help="Baseline experiment for embedding comparison table",
    )
    return parser.parse_args()


def output_root(exp):
    return Path("result") / exp.strip("/") / "phase1_mri"


def _safe_subdir_name(name: str, *, kind: str) -> str:
    cleaned = (name or "").strip().strip("/")
    if not cleaned or cleaned in (".", "..") or "/" in cleaned or "\\" in cleaned:
        raise ValueError(f"Invalid --{kind} name: {name!r}")
    return cleaned


def paths_for(exp, embeddings_name: str = "embeddings", analysis_name: str = "embedding_analysis"):
    base = output_root(exp)
    embeddings_name = _safe_subdir_name(embeddings_name, kind="embeddings-name")
    analysis_name = _safe_subdir_name(analysis_name, kind="analysis-name")
    return {
        "base": base,
        "checkpoint_dir": base / "checkpoints",
        "checkpoint": base / "checkpoints" / "MRI_AAE.pth.tar",
        "reconstruction": base / "reconstruction",
        "embeddings": base / embeddings_name,
        "analysis": base / analysis_name,
        "visualizations": base / analysis_name / "visualizations",
        "classification": base / analysis_name / "classification",
        "run_config": base / "run_config.json",
        "metadata": base / f"{embeddings_name}_metadata.csv",
        "embeddings_name": embeddings_name,
        "analysis_name": analysis_name,
    }


def resolve_eval_feature_mode(args) -> str:
    """Feature space for encode/analysis (may differ from the training contrastive space)."""
    if getattr(args, "eval_features", None):
        return args.eval_features
    if args.training_mode in ("supcon", "xmodal") or args.skip_train:
        return args.contrastive_features
    return "z18k"


def ensure_dirs(path_map):
    for key in ["checkpoint_dir", "reconstruction", "embeddings", "analysis", "visualizations", "classification"]:
        path_map[key].mkdir(parents=True, exist_ok=True)


def split_file(split):
    return {"train": config.train, "validation": config.validation, "test": config.test}[split]


def build_dataset(split, max_subjects=None):
    dataset = OneDataset(root_Abeta=config.whole_MRI, task=split_file(split), stage=split)
    if max_subjects is not None:
        dataset.images = dataset.images[:max_subjects]
        dataset.length_dataset = len(dataset.images)
        dataset.len = len(dataset.images)
    return dataset


def build_loader(split, batch_size, max_subjects=None, shuffle=False, drop_last=False):
    dataset = build_dataset(split, max_subjects=max_subjects)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.numworker,
        pin_memory=(config.device == "cuda"),
        drop_last=drop_last,
    )


class LabeledMRIDataset(torch.utils.data.Dataset):
    """MRI volumes with control/parkinson labels for SupCon training."""

    def __init__(self, split: str, max_subjects=None):
        self.base = build_dataset(split, max_subjects=max_subjects)
        self.label_lookup = labels_map()
        self.labels = []
        for image_id in self.base.images:
            subject_id = str(image_id)
            if subject_id not in self.label_lookup:
                raise ValueError(f"Missing label for ID '{subject_id}'")
            self.labels.append(int(self.label_lookup[subject_id]))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        volume, name = self.base[index]
        label = np.array([self.labels[index]], dtype=np.float32)
        return volume, label, name


class BalancedBatchSampler(torch.utils.data.Sampler):
    """Yield batches with equal counts per binary class (e.g. 4+4 for B=8)."""

    def __init__(self, labels, batch_size: int, drop_last: bool = True, seed: int = 0):
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if batch_size < 2 or batch_size % 2 != 0:
            raise ValueError(f"BalancedBatchSampler requires even batch_size>=2, got {batch_size}")
        classes = sorted(np.unique(labels).tolist())
        if len(classes) != 2:
            raise ValueError(f"BalancedBatchSampler expects exactly 2 classes, got {classes}")
        self.batch_size = int(batch_size)
        self.per_class = self.batch_size // 2
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self.class_to_indices = {c: np.where(labels == c)[0].tolist() for c in classes}
        for c, idxs in self.class_to_indices.items():
            if len(idxs) < self.per_class:
                raise ValueError(f"Class {c} has {len(idxs)} samples < per_class={self.per_class}")

    def __len__(self):
        min_len = min(len(v) for v in self.class_to_indices.values())
        n = min_len // self.per_class
        if not self.drop_last and (min_len % self.per_class):
            n += 1
        return n

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        shuffled = {c: rng.permutation(idxs).tolist() for c, idxs in self.class_to_indices.items()}
        classes = list(shuffled.keys())
        n_batches = len(self)
        for b in range(n_batches):
            batch = []
            for c in classes:
                start = b * self.per_class
                end = start + self.per_class
                chunk = shuffled[c][start:end]
                if len(chunk) < self.per_class:
                    # wrap with replacement for incomplete tail when drop_last=False
                    need = self.per_class - len(chunk)
                    chunk = chunk + rng.choice(shuffled[c], size=need, replace=True).tolist()
                batch.extend(chunk)
            rng.shuffle(batch)
            yield batch


def build_labeled_mri_loader(
    split,
    batch_size,
    max_subjects=None,
    shuffle=False,
    drop_last=False,
    balanced: bool = False,
    seed: int = 0,
):
    dataset = LabeledMRIDataset(split, max_subjects=max_subjects)
    if balanced:
        sampler = BalancedBatchSampler(dataset.labels, batch_size=batch_size, drop_last=drop_last, seed=seed)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=config.numworker,
            pin_memory=(config.device == "cuda"),
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.numworker,
        pin_memory=(config.device == "cuda"),
        drop_last=drop_last,
    )


def contrastive_feature_tensor(latent: torch.Tensor, h64: torch.Tensor | None, feature_mode: str) -> torch.Tensor:
    """Flatten selected encoder map for contrastive loss (no projector / no GAP)."""
    mode = (feature_mode or "z18k").strip().lower()
    if mode == "h64":
        if h64 is None:
            raise ValueError("h64 features requested but hidden map is None")
        return flatten_feature_volume(h64)
    if mode == "z18k":
        return flatten_feature_volume(latent)
    raise ValueError(f"Unknown contrastive feature mode: {feature_mode!r}")


def encode_volume_maps(encoder, volume: torch.Tensor):
    """Return (latent, h64) from encoder."""
    return encoder(volume, return_hidden=True)


def build_paired_loader(split, batch_size, max_subjects=None, shuffle=False, drop_last=False):
    dataset = TwoDataset(
        root_MRI=config.whole_MRI,
        root_Abeta=config.whole_Abeta,
        task=split_file(split),
        stage=split,
    )
    if max_subjects is not None:
        dataset.images = dataset.images[:max_subjects]
        dataset.length_dataset = len(dataset.images)
        dataset.len = len(dataset.images)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.numworker,
        pin_memory=(config.device == "cuda"),
        drop_last=drop_last,
    )


def load_model_weights(model, checkpoint_path: str | Path):
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=config.device, weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state, strict=True)
    print(f"=> Loaded weights from {path}")


def subject_id_from_name(name):
    if isinstance(name, (tuple, list)):
        name = name[0]
    name = Path(str(name)).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def reconstruction_training_loss(target, prediction):
    if config.USE_WEIGHTED_RECON_LOSS or config.USE_TAIL_RECON_LOSS:
        loss, _ = spect_reconstruction_loss(
            target,
            prediction,
            use_weighted=config.USE_WEIGHTED_RECON_LOSS,
            use_tail=config.USE_TAIL_RECON_LOSS,
            weighted_alpha=config.WEIGHTED_RECON_ALPHA,
            z_threshold=config.WEIGHTED_Z_THRESHOLD,
            tail_fraction=config.TAIL_RECON_FRACTION,
            tail_beta=config.TAIL_RECON_BETA,
        )
        return loss
    return torch.abs(target - prediction).mean()


def append_csv(path, row, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if not file_exists or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def should_validate(epoch, args):
    return (epoch + 1) % args.validation_interval == 0 or epoch == args.epochs - 1


def _volume_channel_shape() -> tuple[int, int, int, int]:
    """(C, D, H, W) tensor shape after dataset crop/pad."""
    working = get_working_volume_shape()
    return (1, *working)


def _shape_label(shape: tuple[int, ...]) -> str:
    return ",".join(str(int(x)) for x in shape)


def save_checkpoint(model, optimizer, path, epoch, score):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "selection_score": float(score),
            "modality": "MRI",
            "stage": "phase1_mri_autoencoder",
        },
        path,
    )


def evaluate_reconstruction(model, split, args, out_csv):
    loader = build_loader(split, batch_size=1, max_subjects=args.max_subjects_per_split, shuffle=False)
    model.eval()
    psnr_total = 0.0
    ssim_total = 0.0
    rows = []
    with torch.no_grad():
        for volume, names in tqdm(loader, desc=f"reconstruction:{split}"):
            volume = torch.as_tensor(np.expand_dims(volume, axis=1), dtype=torch.float32, device=config.device)
            reconstructed = model(volume)
            expected = _volume_channel_shape()
            if tuple(reconstructed.shape[1:]) != expected:
                raise ValueError(
                    f"Unexpected MRI reconstruction shape: {tuple(reconstructed.shape)} "
                    f"(expected {expected})"
                )
            batch_psnr, batch_ssim = batch_metric_scores(volume, reconstructed)
            subject_id = subject_id_from_name(names)
            rows.append({"ID": subject_id, "split": split, "PSNR": batch_psnr, "SSIM": batch_ssim})
            psnr_total += batch_psnr
            ssim_total += batch_ssim

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    length = max(len(rows), 1)
    return psnr_total / length, ssim_total / length


def evaluate_contrastive_validation(
    mode,
    model,
    args,
    *,
    feature_mode: str,
    supcon_crit=None,
    xmodal_crit=None,
    spect_model=None,
):
    """Mean contrastive loss on validation (no grad). Uses drop_last batches."""
    if mode == "supcon":
        if supcon_crit is None:
            raise ValueError("supcon validation requires criterion")
        loader = build_labeled_mri_loader(
            "validation",
            batch_size=args.batch_size,
            max_subjects=args.max_subjects_per_split,
            shuffle=False,
            drop_last=True,
            balanced=True,
            seed=args.seed + 10_000,
        )
    elif mode == "xmodal":
        if xmodal_crit is None or spect_model is None:
            raise ValueError("xmodal validation requires criterion and spect model")
        loader = build_paired_loader(
            "validation",
            batch_size=args.batch_size,
            max_subjects=args.max_subjects_per_split,
            shuffle=False,
            drop_last=True,
        )
    else:
        raise ValueError(f"No contrastive validation for mode={mode!r}")

    if len(loader) == 0:
        raise RuntimeError(
            f"Validation loader empty for contrastive eval (batch_size={args.batch_size}, drop_last=True). "
            "Reduce --batch-size or add validation subjects."
        )

    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"contrastive_val:{mode}:{feature_mode}"):
            if mode == "supcon":
                volume, labels, _names = batch
                volume = torch.as_tensor(np.expand_dims(volume, axis=1), dtype=torch.float32, device=config.device)
                labels_t = torch.as_tensor(labels, device=config.device).reshape(-1).long()
                latent, h64 = encode_volume_maps(model.encoder, volume)
                z = contrastive_feature_tensor(latent, h64, feature_mode)
                batch_loss = float(supcon_crit(z, labels_t).item())
            else:
                MRI, SPECT, _names, _labels = batch
                volume = torch.as_tensor(np.expand_dims(MRI, axis=1), dtype=torch.float32, device=config.device)
                spect_volume = torch.as_tensor(
                    np.expand_dims(SPECT, axis=1), dtype=torch.float32, device=config.device
                )
                latent, h64 = encode_volume_maps(model.encoder, volume)
                z_mri = contrastive_feature_tensor(latent, h64, feature_mode)
                spect_latent, spect_h64 = encode_volume_maps(spect_model.encoder, spect_volume)
                z_spect = contrastive_feature_tensor(spect_latent, spect_h64, feature_mode)
                batch_loss = float(xmodal_crit(z_mri, z_spect).item())
            n = volume.shape[0]
            total += batch_loss * n
            count += n
    return total / max(count, 1)


def train_mri_autoencoder(args, path_map):
    mode = args.training_mode
    feature_mode = args.contrastive_features
    model = AAE().to(config.device)
    if args.init_checkpoint:
        load_model_weights(model, args.init_checkpoint)

    discriminator = Discriminator().to(config.device)
    opt_model = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    supcon_crit = None
    xmodal_crit = None
    spect_model = None

    if mode == "supcon":
        supcon_crit = SupervisedContrastiveLoss(temperature=args.contrastive_temp)
    elif mode == "xmodal":
        if not args.spect_checkpoint:
            raise ValueError("xmodal mode requires --spect-checkpoint")
        spect_model = AAE().to(config.device)
        load_model_weights(spect_model, args.spect_checkpoint)
        spect_model.eval()
        spect_model.requires_grad_(False)
        xmodal_crit = CrossModalInfoNCE(temperature=args.contrastive_temp)

    best_score = -float("inf")
    validations_without_improvement = 0
    train_header = ["Epoch", "recon_loss", "disc_loss_epoch"]
    if mode == "supcon":
        train_header.append("supcon_loss")
    if mode == "xmodal":
        train_header.append("xmodal_loss")

    for epoch in range(args.epochs):
        model.train()
        discriminator.train()

        if mode == "supcon":
            train_loader = build_labeled_mri_loader(
                "train",
                batch_size=args.batch_size,
                max_subjects=args.max_subjects_per_split,
                drop_last=True,
                balanced=True,
                seed=args.seed + epoch,
            )
        elif mode == "xmodal":
            train_loader = build_paired_loader(
                "train",
                batch_size=args.batch_size,
                max_subjects=args.max_subjects_per_split,
                shuffle=True,
                drop_last=True,
            )
        else:
            train_loader = build_loader(
                "train",
                batch_size=args.batch_size,
                max_subjects=args.max_subjects_per_split,
                shuffle=True,
                drop_last=True,
            )

        recon_loss_epoch = 0.0
        disc_loss_epoch = 0.0
        supcon_loss_epoch = 0.0
        xmodal_loss_epoch = 0.0
        sample_count = 0
        supcon_steps = 0
        xmodal_steps = 0

        for batch in tqdm(train_loader, desc=f"train:mri_aae:{mode}:{feature_mode}:epoch_{epoch + 1}"):
            if mode == "supcon":
                volume, labels, _names = batch
                volume = torch.as_tensor(np.expand_dims(volume, axis=1), dtype=torch.float32, device=config.device)
                labels_t = torch.as_tensor(labels, device=config.device).reshape(-1).long()
            elif mode == "xmodal":
                MRI, SPECT, _names, _labels = batch
                volume = torch.as_tensor(np.expand_dims(MRI, axis=1), dtype=torch.float32, device=config.device)
                spect_volume = torch.as_tensor(np.expand_dims(SPECT, axis=1), dtype=torch.float32, device=config.device)
            else:
                volume, _names = batch
                volume = torch.as_tensor(np.expand_dims(volume, axis=1), dtype=torch.float32, device=config.device)

            # Single encoder forward: reuse latent for reconstruction; optional h64 for contrastive.
            if mode in ("supcon", "xmodal"):
                latent, h64 = encode_volume_maps(model.encoder, volume)
            else:
                latent = model.encoder(volume)
                h64 = None
            reconstructed = model.decoder(latent)

            disc_fake = discriminator(reconstructed)
            recon_loss = reconstruction_training_loss(volume, reconstructed)
            generator_loss = -torch.mean(disc_fake)
            loss = recon_loss * config.Lambda + generator_loss

            if mode == "supcon" and supcon_crit is not None:
                z = contrastive_feature_tensor(latent, h64, feature_mode)
                supcon_loss = supcon_crit(z, labels_t)
                loss = loss + args.supcon_lambda * supcon_loss
                supcon_loss_epoch += supcon_loss.detach().item() * volume.shape[0]
                supcon_steps += volume.shape[0]

            if mode == "xmodal" and xmodal_crit is not None and spect_model is not None:
                z_mri = contrastive_feature_tensor(latent, h64, feature_mode)
                with torch.no_grad():
                    spect_latent, spect_h64 = encode_volume_maps(spect_model.encoder, spect_volume)
                    z_spect = contrastive_feature_tensor(spect_latent, spect_h64, feature_mode)
                xmodal_loss = xmodal_crit(z_mri, z_spect)
                loss = loss + args.xmodal_lambda * xmodal_loss
                xmodal_loss_epoch += xmodal_loss.detach().item() * volume.shape[0]
                xmodal_steps += volume.shape[0]

            opt_model.zero_grad()
            loss.backward()
            opt_disc.zero_grad()
            opt_model.step()

            disc_real = discriminator(volume)
            disc_fake_det = discriminator(reconstructed.detach())
            disc_loss_real = torch.mean(F.relu(1.0 - disc_real))
            disc_loss_fake = torch.mean(F.relu(1.0 + disc_fake_det))
            disc_loss = (disc_loss_real + disc_loss_fake) / 2
            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()

            batch_size = volume.shape[0]
            sample_count += batch_size
            recon_loss_epoch += recon_loss.item() * batch_size
            disc_loss_epoch += disc_loss.item() * batch_size

        row = {
            "Epoch": epoch + 1,
            "recon_loss": recon_loss_epoch / max(sample_count, 1),
            "disc_loss_epoch": disc_loss_epoch / max(sample_count, 1),
        }
        if mode == "supcon":
            row["supcon_loss"] = supcon_loss_epoch / max(supcon_steps, 1)
        if mode == "xmodal":
            row["xmodal_loss"] = xmodal_loss_epoch / max(xmodal_steps, 1)
        append_csv(path_map["reconstruction"] / "train.csv", row, train_header)

        if not should_validate(epoch, args):
            continue

        val_psnr, val_ssim = evaluate_reconstruction(
            model,
            "validation",
            args,
            path_map["reconstruction"] / "validation.csv",
        )
        recon_score = val_psnr + val_ssim * 10
        val_ctr_loss = None
        if mode in ("supcon", "xmodal"):
            val_ctr_loss = evaluate_contrastive_validation(
                mode,
                model,
                args,
                feature_mode=feature_mode,
                supcon_crit=supcon_crit,
                xmodal_crit=xmodal_crit,
                spect_model=spect_model,
            )
            score = -float(val_ctr_loss)
            selection_metric = "neg_val_contrastive_loss"
        else:
            score = recon_score
            selection_metric = "psnr_ssim"

        val_row = {
            "Epoch": epoch + 1,
            "PSNR": val_psnr,
            "SSIM": val_ssim,
            "recon_selection_score": recon_score,
            "selection_score": score,
            "selection_metric": selection_metric,
        }
        if val_ctr_loss is not None:
            val_row["val_contrastive_loss"] = val_ctr_loss
        val_header = [
            "Epoch",
            "PSNR",
            "SSIM",
            "recon_selection_score",
            "val_contrastive_loss",
            "selection_score",
            "selection_metric",
        ]
        append_csv(path_map["reconstruction"] / "validation_summary.csv", val_row, val_header)

        if score > best_score:
            best_score = score
            validations_without_improvement = 0
            print(
                f"New best checkpoint at epoch {epoch + 1} "
                f"({selection_metric}={score:.6f}"
                + (f", val_contrastive_loss={val_ctr_loss:.6f}" if val_ctr_loss is not None else "")
                + f", PSNR={val_psnr:.4f})"
            )
            save_checkpoint(model, opt_model, path_map["checkpoint"], epoch + 1, best_score)
            test_psnr, test_ssim = evaluate_reconstruction(
                model,
                "test",
                args,
                path_map["reconstruction"] / "test.csv",
            )
            append_csv(
                path_map["reconstruction"] / "test_summary.csv",
                {
                    "Epoch": epoch + 1,
                    "PSNR": test_psnr,
                    "SSIM": test_ssim,
                    "selection_score": best_score,
                    "selection_metric": selection_metric,
                    "val_contrastive_loss": val_ctr_loss if val_ctr_loss is not None else "",
                    "contrastive_features": feature_mode if mode in ("supcon", "xmodal") else "",
                },
                [
                    "Epoch",
                    "PSNR",
                    "SSIM",
                    "selection_score",
                    "selection_metric",
                    "val_contrastive_loss",
                    "contrastive_features",
                ],
            )
        else:
            validations_without_improvement += 1
            print(f"MRI AAE early stopping counter: {validations_without_improvement}/{args.patience}")
            if validations_without_improvement >= args.patience:
                print(f"Stopping MRI AAE early at epoch {epoch + 1}")
                break


def load_trained_aae(checkpoint_path, lr):
    model = AAE().to(config.device).eval()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    load_checkpoint(str(checkpoint_path), model, optimizer, lr)
    return model


def labels_map():
    data_info = pd.read_csv("data_info/data_info.csv", dtype={"ID": str})
    if data_info["ID"].duplicated().any():
        raise ValueError("Duplicate IDs found in data_info/data_info.csv")
    return data_info.set_index("ID")["label"].astype(int).to_dict()


def encode_embeddings(args, path_map):
    if not path_map["checkpoint"].exists():
        raise FileNotFoundError(f"Missing MRI AAE checkpoint: {path_map['checkpoint']}")
    model = load_trained_aae(path_map["checkpoint"], args.lr)
    label_lookup = labels_map()
    feature_mode = resolve_eval_feature_mode(args)
    rows = []

    for split in ["train", "validation", "test"]:
        split_dir = path_map["embeddings"] / split
        split_dir.mkdir(parents=True, exist_ok=True)
        loader = build_loader(split, batch_size=1, max_subjects=args.max_subjects_per_split, shuffle=False)
        with torch.no_grad():
            for volume, names in tqdm(loader, desc=f"encode:mri_embeddings:{feature_mode}:{split}"):
                volume = torch.as_tensor(np.expand_dims(volume, axis=1), dtype=torch.float32, device=config.device)
                latent, h64 = encode_volume_maps(model.encoder, volume)
                latent_shape = get_latent_shape()
                if tuple(latent.shape[1:]) != (1, *latent_shape):
                    raise ValueError(f"Unexpected MRI embedding shape: {tuple(latent.shape)}")
                subject_id = subject_id_from_name(names)
                reference = nib.load(resolve_nifti_path(config.whole_MRI, subject_id))
                if feature_mode == "h64":
                    feat_np = h64[0].detach().cpu().numpy().astype(np.float32)
                    out_path = split_dir / f"{subject_id}_h64.npy"
                    np.save(out_path, feat_np)
                    shape_label = _shape_label(tuple(feat_np.shape))
                else:
                    feat_np = latent[0, 0].detach().cpu().numpy().astype(np.float32)
                    out_path = split_dir / f"{subject_id}.nii.gz"
                    nib.save(nib.Nifti1Image(feat_np, reference.affine), out_path)
                    shape_label = _shape_label((1, *get_latent_shape()))
                rows.append(
                    {
                        "ID": subject_id,
                        "split": split,
                        "label": int(label_lookup[subject_id]),
                        "embedding_path": str(out_path),
                        "embedding_shape": shape_label,
                        "contrastive_features": feature_mode,
                    }
                )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(path_map["metadata"], index=False)
    return metadata


def load_embedding_matrix(rows):
    features = []
    labels = []
    ids = []
    for row in rows.itertuples(index=False):
        path = Path(row.embedding_path)
        if path.suffix == ".npy":
            embedding = np.load(path).astype(np.float32).reshape(-1)
        else:
            embedding = nib.load(str(path)).get_fdata().astype(np.float32).reshape(-1)
        features.append(embedding)
        labels.append(int(row.label))
        ids.append(str(row.ID))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), ids


def classifier_params(model):
    classifier = model.named_steps["classifier"]
    params = {"classifier": classifier.__class__.__name__}
    if hasattr(classifier, "n_neighbors"):
        params["n_neighbors"] = classifier.n_neighbors
    if hasattr(classifier, "kernel"):
        params["kernel"] = classifier.kernel
    if hasattr(classifier, "C"):
        params["C"] = classifier.C
    return params


def build_classical_models(y_train):
    knn_auto = min(5, len(y_train))
    if knn_auto % 2 == 0:
        knn_auto -= 1
    knn_auto = max(1, knn_auto)
    knn_k2 = min(2, len(y_train))

    def pipe(classifier):
        return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])

    return {
        "KNN_auto": pipe(KNeighborsClassifier(n_neighbors=knn_auto, weights="distance")),
        "KNN_K2": pipe(KNeighborsClassifier(n_neighbors=knn_k2, weights="distance")),
        "SVM_RBF": pipe(SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=42)),
        "SVM_linear": pipe(SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced", random_state=42)),
        "LogisticRegression": pipe(LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)),
    }


def positive_scores(model, features):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return None


def metric_row(model_name, y_true, y_pred, y_score=None, **params):
    row = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score) if y_score is not None and len(np.unique(y_true)) == 2 else np.nan,
    }
    row.update(params)
    return row


class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_dnn(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    model = EmbeddingMLP(X_train_scaled.shape[1]).to(config.device)
    class_counts = np.bincount(y_train.astype(int), minlength=2)
    pos_weight = torch.tensor([class_counts[0] / max(class_counts[1], 1)], dtype=torch.float32, device=config.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train.astype(np.float32), dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=config.device)
    y_test_tensor = torch.tensor(y_test.astype(np.float32), dtype=torch.float32, device=config.device)
    best_state = None
    best_loss = float("inf")
    bad_epochs = 0
    patience = 30
    history = []

    for epoch in range(300):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(config.device)
            yb = yb.to(config.device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test_tensor), y_test_tensor).item()
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "test_loss": test_loss})
        if test_loss < best_loss:
            best_loss = test_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(X_test_tensor)).detach().cpu().numpy()
    return (scores >= 0.5).astype(int), scores, pd.DataFrame(history)


def label_name(label):
    return "control" if int(label) == 0 else "parkinson"


def projection_coordinates(X, method, n_components):
    X_scaled = StandardScaler().fit_transform(X)
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == "tsne":
        reducer = TSNE(
            n_components=n_components,
            random_state=42,
            perplexity=min(30, max(1, len(X_scaled) - 1)),
            init="pca",
            learning_rate="auto",
        )
    elif method == "umap":
        if umap is None:
            raise ImportError("umap-learn is required for UMAP visualizations.")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(15, max(2, len(X_scaled) - 1)),
            min_dist=0.1,
            metric="euclidean",
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown projection method: {method}")
    return reducer.fit_transform(X_scaled)


def save_projection_2d(df, method, out_dir):
    colors = {"control": "#3b6fd8", "parkinson": "#d83b5f"}
    plt.figure(figsize=(8, 6))
    for class_name in ["control", "parkinson"]:
        class_df = df[df["class"] == class_name]
        if class_df.empty:
            continue
        plt.scatter(
            class_df[f"{method}_1"],
            class_df[f"{method}_2"],
            color=colors[class_name],
            alpha=0.75,
            label=class_name,
        )
    plt.title(f"{method.upper()} 2D MRI-AE embeddings - internal test")
    plt.xlabel(f"{method}_1")
    plt.ylabel(f"{method}_2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{method}_2d.png", dpi=200)
    plt.close()


def save_visualizations(test_rows, path_map):
    if len(test_rows) < 2:
        return
    X, y, ids = load_embedding_matrix(test_rows)
    classes = [label_name(label) for label in y]
    for method in ["pca", "tsne", "umap"]:
        for n_components in [2, 3]:
            coords = projection_coordinates(X, method, n_components)
            columns = [f"{method}_{idx + 1}" for idx in range(n_components)]
            df = pd.DataFrame(coords, columns=columns)
            df["ID"] = ids
            df["label"] = y
            df["class"] = classes
            df["embedding_eval_split"] = "test"
            df.to_csv(path_map["visualizations"] / f"{method}_{n_components}d.csv", index=False)
            if n_components == 2:
                save_projection_2d(df, method, path_map["visualizations"])
            else:
                fig = px.scatter_3d(
                    df,
                    x=f"{method}_1",
                    y=f"{method}_2",
                    z=f"{method}_3",
                    color="class",
                    color_discrete_map={"control": "#3b6fd8", "parkinson": "#d83b5f"},
                    hover_data=["ID", "label", "class"],
                    title=f"{method.upper()} 3D MRI-AE embeddings - internal test",
                )
                fig.write_html(path_map["visualizations"] / f"{method}_3d.html")


def plot_confusion_and_roc(y_true, y_pred, y_score, path_map):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (best model)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path_map["visualizations"] / "confusion_matrix.png", dpi=200)
    plt.close()

    if y_score is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path_map["visualizations"] / "roc_curve.png", dpi=200)
        plt.close()


def analyze_embeddings(path_map):
    metadata = pd.read_csv(path_map["metadata"], dtype={"ID": str})
    train_rows = metadata[metadata["split"] == "train"].copy()
    if train_rows.empty:
        raise ValueError("No train MRI embeddings found for internal 80/20 evaluation.")
    class_counts = train_rows["label"].astype(int).value_counts()
    if len(train_rows) < 10 or len(class_counts) < 2 or class_counts.min() < 2:
        skipped = {
            "status": "skipped",
            "reason": "Not enough train embeddings per class for a stratified internal 80/20 evaluation.",
            "n_train_embeddings": int(len(train_rows)),
            "class_counts": {str(key): int(value) for key, value in class_counts.to_dict().items()},
            "evaluation_protocol": "train_original_internal_80_20",
        }
        train_rows.assign(embedding_eval_split="insufficient_data")[
            ["ID", "label", "embedding_eval_split", "embedding_path"]
        ].to_csv(path_map["analysis"] / "embedding_eval_split.csv", index=False)
        pd.DataFrame(columns=["model", "accuracy", "balanced_accuracy", "f1", "roc_auc"]).to_csv(
            path_map["classification"] / "classification_metrics.csv",
            index=False,
        )
        pd.DataFrame(columns=["ID", "label"]).to_csv(
            path_map["classification"] / "test_predictions.csv",
            index=False,
        )
        with open(path_map["classification"] / "quality_summary.json", "w", encoding="utf-8") as file:
            json.dump(skipped, file, indent=2)
        return skipped
    eval_train, eval_test = train_test_split(
        train_rows,
        test_size=0.20,
        stratify=train_rows["label"].astype(int),
        random_state=42,
    )
    eval_split = pd.concat(
        [
            eval_train.assign(embedding_eval_split="train"),
            eval_test.assign(embedding_eval_split="test"),
        ],
        ignore_index=True,
    )[["ID", "label", "embedding_eval_split", "embedding_path"]]
    eval_split.to_csv(path_map["analysis"] / "embedding_eval_split.csv", index=False)

    X_train, y_train, _train_ids = load_embedding_matrix(eval_train)
    X_test, y_test, test_ids = load_embedding_matrix(eval_test)

    rows = []
    prediction_rows = pd.DataFrame({"ID": test_ids, "label": y_test})
    for model_name, model in build_classical_models(y_train).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = positive_scores(model, X_test)
        rows.append(metric_row(model_name, y_test, pred, score, **classifier_params(model)))
        prediction_rows[f"pred_{model_name}"] = pred
        if score is not None:
            prediction_rows[f"score_{model_name}"] = score

    dnn_pred, dnn_score, history = train_dnn(X_train, y_train, X_test, y_test)
    rows.append(metric_row("DNN_simple", y_test, dnn_pred, dnn_score, classifier="EmbeddingMLP"))
    prediction_rows["pred_DNN_simple"] = dnn_pred
    prediction_rows["score_DNN_simple"] = dnn_score
    history.to_csv(path_map["classification"] / "dnn_training_history.csv", index=False)

    metrics = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    metrics.to_csv(path_map["classification"] / "classification_metrics.csv", index=False)
    metrics.to_json(path_map["classification"] / "metrics_summary.json", orient="records", indent=2)
    prediction_rows.to_csv(path_map["classification"] / "test_predictions.csv", index=False)

    best_model = metrics.iloc[0]["model"]
    best_pred = prediction_rows[f"pred_{best_model}"].to_numpy()
    best_score_column = f"score_{best_model}"
    best_score = prediction_rows[best_score_column].to_numpy() if best_score_column in prediction_rows.columns else None
    plot_confusion_and_roc(y_test, best_pred, best_score, path_map)
    save_visualizations(eval_test, path_map)

    report = classification_report(y_test, best_pred, digits=4)
    with open(path_map["classification"] / "classifier_report.txt", "w", encoding="utf-8") as file:
        file.write(f"Best model: {best_model}\n\n")
        file.write(report)
    decision = {
        "best_model": best_model,
        "best_balanced_accuracy": float(metrics.iloc[0]["balanced_accuracy"]),
        "best_roc_auc": None if pd.isna(metrics.iloc[0]["roc_auc"]) else float(metrics.iloc[0]["roc_auc"]),
        "evaluation_protocol": "train_original_internal_80_20",
    }
    with open(path_map["classification"] / "quality_summary.json", "w", encoding="utf-8") as file:
        json.dump(decision, file, indent=2)
    return decision


def evaluate_crossmodal_retrieval(args, path_map, spect_checkpoint: str | Path):
    """Top-1 MRI->SPECT retrieval on validation (paired subjects)."""
    if not path_map["checkpoint"].exists():
        raise FileNotFoundError(f"Missing MRI checkpoint: {path_map['checkpoint']}")
    feature_mode = args.contrastive_features
    mri_model = load_trained_aae(path_map["checkpoint"], args.lr)
    spect_model = AAE().to(config.device)
    load_model_weights(spect_model, spect_checkpoint)
    spect_model.eval()

    loader = build_paired_loader(
        "validation",
        batch_size=1,
        max_subjects=args.max_subjects_per_split,
        shuffle=False,
    )
    mri_feats = []
    spect_feats = []
    ids = []
    with torch.no_grad():
        for MRI, SPECT, names, _labels in tqdm(loader, desc=f"crossmodal:validation:{feature_mode}"):
            MRI_t = torch.as_tensor(np.expand_dims(MRI, axis=1), dtype=torch.float32, device=config.device)
            SPECT_t = torch.as_tensor(np.expand_dims(SPECT, axis=1), dtype=torch.float32, device=config.device)
            mri_latent, mri_h64 = encode_volume_maps(mri_model.encoder, MRI_t)
            spect_latent, spect_h64 = encode_volume_maps(spect_model.encoder, SPECT_t)
            z_mri = contrastive_feature_tensor(mri_latent, mri_h64, feature_mode)
            z_spect = contrastive_feature_tensor(spect_latent, spect_h64, feature_mode)
            mri_feats.append(z_mri.detach().cpu().numpy().reshape(-1))
            spect_feats.append(z_spect.detach().cpu().numpy().reshape(-1))
            ids.append(subject_id_from_name(names))

    if len(mri_feats) < 2:
        result = {"status": "skipped", "reason": "not enough validation pairs", "n": len(mri_feats)}
    else:
        mri_mat = np.asarray(mri_feats, dtype=np.float32)
        spect_mat = np.asarray(spect_feats, dtype=np.float32)
        mri_mat = mri_mat / (np.linalg.norm(mri_mat, axis=1, keepdims=True) + 1e-8)
        spect_mat = spect_mat / (np.linalg.norm(spect_mat, axis=1, keepdims=True) + 1e-8)
        sim = mri_mat @ spect_mat.T
        preds = sim.argmax(axis=1)
        targets = np.arange(len(ids))
        top1 = float((preds == targets).mean())
        result = {
            "status": "ok",
            "split": "validation",
            "n_subjects": len(ids),
            "top1_retrieval": top1,
            "random_chance": 1.0 / len(ids),
            "contrastive_features": feature_mode,
            "subject_ids": ids,
        }

    out_path = path_map["classification"] / "crossmodal_retrieval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump({k: v for k, v in result.items() if k != "subject_ids"}, file, indent=2)
    return result


def compare_embedding_baselines(args, path_map, classification_decision):
    """Write comparison table vs baseline mri_rois (bal.acc + optional retrieval)."""
    baseline_root = output_root(args.baseline_exp)
    baseline_metrics = baseline_root / "embedding_analysis" / "classification" / "classification_metrics.csv"
    current_metrics = path_map["classification"] / "classification_metrics.csv"

    rows = []
    if baseline_metrics.exists() and current_metrics.exists():
        base_df = pd.read_csv(baseline_metrics)
        cur_df = pd.read_csv(current_metrics)
        base_best = base_df.sort_values("balanced_accuracy", ascending=False).iloc[0]
        cur_best = cur_df.sort_values("balanced_accuracy", ascending=False).iloc[0]
        rows.append(
            {
                "experiment": args.baseline_exp,
                "training_mode": "recon",
                "best_model": base_best["model"],
                "balanced_accuracy": float(base_best["balanced_accuracy"]),
                "roc_auc": float(base_best["roc_auc"]) if pd.notna(base_best.get("roc_auc")) else np.nan,
            }
        )
        rows.append(
            {
                "experiment": args.exp,
                "training_mode": args.training_mode,
                "contrastive_features": args.contrastive_features if args.training_mode in ("supcon", "xmodal") else "",
                "best_model": cur_best["model"],
                "balanced_accuracy": float(cur_best["balanced_accuracy"]),
                "roc_auc": float(cur_best["roc_auc"]) if pd.notna(cur_best.get("roc_auc")) else np.nan,
            }
        )

    retrieval_path = path_map["classification"] / "crossmodal_retrieval.json"
    if retrieval_path.exists():
        with open(retrieval_path, encoding="utf-8") as file:
            retrieval = json.load(file)
        if retrieval.get("status") == "ok":
            rows.append(
                {
                    "experiment": args.exp,
                    "metric": "top1_retrieval_validation",
                    "value": retrieval["top1_retrieval"],
                    "random_chance": retrieval["random_chance"],
                }
            )

    if not rows:
        return None

    out_dir = Path("docs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    feat = args.contrastive_features if args.training_mode in ("supcon", "xmodal") else "recon"
    out_name = f"mri_e1_{args.training_mode}_{feat}_vs_{args.baseline_exp}.csv"
    out_path = out_dir / out_name
    pd.DataFrame(rows).to_csv(out_path, index=False)

    summary = {
        "comparison_csv": str(out_path),
        "current_experiment": args.exp,
        "baseline_experiment": args.baseline_exp,
        "training_mode": args.training_mode,
        "contrastive_features": args.contrastive_features if args.training_mode in ("supcon", "xmodal") else None,
        "classification": classification_decision,
    }
    with open(path_map["classification"] / "baseline_comparison.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def write_run_config(args, path_map):
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": args.exp,
        "stage": "phase1_mri_autoencoder",
        "architecture": "AAE",
        "input_modality": "MRI",
        "checkpoint": str(path_map["checkpoint"]),
        "output_root": str(path_map["base"]),
        "device": config.device,
        "paths": {
            "whole_MRI": config.whole_MRI,
            "train": config.train,
            "validation": config.validation,
            "test": config.test,
        },
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "validation_interval": args.validation_interval,
        "early_stopping_patience": args.patience,
        "max_subjects_per_split": args.max_subjects_per_split,
        "embedding_shape": (
            _shape_label((64, *get_latent_shape()))
            if args.training_mode in ("supcon", "xmodal") and args.contrastive_features == "h64"
            else _shape_label((1, *get_latent_shape()))
        ),
        "reconstruction_shape": _shape_label(_volume_channel_shape()),
        "volume_shape_on_disk": _shape_label(get_volume_shape()),
        "working_volume_shape": _shape_label(get_working_volume_shape()),
        "embedding_eval_protocol": "train_original_internal_80_20",
        "training_mode": args.training_mode,
        "checkpoint_selection": (
            "neg_val_contrastive_loss"
            if args.training_mode in ("supcon", "xmodal")
            else "psnr_ssim"
        ),
        "supcon_lambda": args.supcon_lambda if args.training_mode == "supcon" else None,
        "xmodal_lambda": args.xmodal_lambda if args.training_mode == "xmodal" else None,
        "contrastive_temp": args.contrastive_temp,
        "contrastive_features": args.contrastive_features if args.training_mode in ("supcon", "xmodal") else None,
        "projector": False,
        "init_checkpoint": args.init_checkpoint,
        "spect_checkpoint": args.spect_checkpoint if args.training_mode == "xmodal" else None,
        "baseline_exp": args.baseline_exp,
        "future_diffusion_usage": {
            "FREEZE_MRI_ENCODER": "1",
            "MRI_ENCODER_CHECKPOINT": str(path_map["checkpoint"]),
        },
    }
    with open(path_map["run_config"], "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main():
    args = parse_args()
    seed_torch(args.seed)
    path_map = paths_for(
        args.exp,
        embeddings_name=args.embeddings_name,
        analysis_name=args.analysis_name,
    )
    ensure_dirs(path_map)
    # Preserve original training run_config when only re-encoding / analyzing.
    if not args.skip_train:
        write_run_config(args, path_map)
    else:
        eval_cfg = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "experiment": args.exp,
            "stage": "phase1_mri_eval_only",
            "checkpoint": str(path_map["checkpoint"]),
            "eval_features": resolve_eval_feature_mode(args),
            "embeddings_dir": str(path_map["embeddings"]),
            "analysis_dir": str(path_map["analysis"]),
            "metadata": str(path_map["metadata"]),
            "skip_train": True,
        }
        with open(path_map["analysis"] / "eval_config.json", "w", encoding="utf-8") as file:
            json.dump(eval_cfg, file, indent=2)
    print(f"Device: {config.device}")
    print(f"MRI phase-1 output: {path_map['base']}")
    print(
        f"Eval features={resolve_eval_feature_mode(args)} "
        f"embeddings={path_map['embeddings_name']} analysis={path_map['analysis_name']}"
    )

    if not args.skip_train:
        train_mri_autoencoder(args, path_map)
    if not args.skip_embeddings:
        encode_embeddings(args, path_map)
    if not args.skip_analysis:
        decision = analyze_embeddings(path_map)
        print(json.dumps(decision, indent=2))
        if args.training_mode == "xmodal" and args.spect_checkpoint:
            retrieval = evaluate_crossmodal_retrieval(args, path_map, args.spect_checkpoint)
            print(json.dumps(retrieval, indent=2))
        comparison = compare_embedding_baselines(args, path_map, decision)
        if comparison is not None:
            print(json.dumps(comparison, indent=2))
    print(f"Done. Results saved in {path_map['base']}")


if __name__ == "__main__":
    main()
