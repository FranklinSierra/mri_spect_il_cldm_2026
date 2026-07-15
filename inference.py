import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import umap

import config
from code.roi_mask_utils import try_load_roi_mask_numpy
from code.volume_shapes import (
    clear_shape_cache,
    crop_volume_to_raw_shape,
    get_latent_shape,
    get_volume_shape,
    get_working_volume_shape,
    parse_shape_triplet,
)
from dataset import TwoDataset, resolve_nifti_path
from main import Diffusion, batch_metric_scores, batch_metric_scores_roi
from model import AAE, MRIEncoder, UNet, LatentClsHead, build_unet
from utils import load_checkpoint, load_cls_head_checkpoint, seed_torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MRI->SPECT reconstructions and latent representations."
    )
    parser.add_argument("--exp", default=config.exp.rstrip("/"))
    parser.add_argument("--run-id", default="auto")
    parser.add_argument("--sampling-steps", type=int, default=config.sampling_steps)
    parser.add_argument("--ddim-eta", type=float, default=config.ddim_eta)
    parser.add_argument("--save-synthetic-spect", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--splits", nargs="+", default=None, choices=["train", "validation", "test"])
    parser.add_argument("--max-subjects-per-split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-conditioning-audit", action="store_true")
    parser.add_argument("--audit-name", default="mri_conditioning_audit_v1")
    parser.add_argument("--include-raw-mri-baseline", action="store_true")
    parser.add_argument("--embedding-eval-split", default="data_info/embedding_eval_split.csv")
    parser.add_argument(
        "--mri-encoder-mode",
        choices=["raw", "frozen", "finetune"],
        default=None,
        help="Stage-2 MRI conditioning mode (default: infer from --exp name).",
    )
    parser.add_argument(
        "--mri-encoder-checkpoint",
        default=None,
        help="MRI encoder weights (frozen/finetune). Default: env or exp-specific path.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip subjects that already have a representation .npz for this split.",
    )
    return parser.parse_args()


def strip_nifti_suffix(name):
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def subject_id_from_name(name):
    if isinstance(name, (tuple, list)):
        name = name[0]
    return strip_nifti_suffix(Path(str(name)).name)


def task_file_for_split(split):
    return {
        "train": config.train,
        "validation": config.validation,
        "test": config.test,
    }[split]


def checkpoint_paths(exp):
    exp = exp.rstrip("/")
    exp_dir = _RESULT_ROOT / exp
    aae_override = os.getenv("ILCLDM_AAE_CHECKPOINT")
    if aae_override:
        checkpoint_aae = Path(aae_override)
    elif getattr(config, "CHECKPOINT_AAE", None):
        checkpoint_aae = Path(config.CHECKPOINT_AAE)
    else:
        checkpoint_aae = exp_dir / "AAE.pth.tar"
    return checkpoint_aae, exp_dir / "Unet.pth.tar"


def infer_mri_encoder_mode(exp, explicit_mode):
    if explicit_mode is not None:
        return explicit_mode
    exp_lower = exp.rstrip("/").lower()
    if "frozen" in exp_lower:
        return "frozen"
    if "finetune" in exp_lower:
        return "finetune"
    if "baseline" in exp_lower:
        return "raw"
    return config.MRI_ENCODER_MODE if config.MRI_ENCODER_MODE in {"raw", "frozen", "finetune"} else "raw"


def default_phase1_mri_encoder_checkpoint():
    return Path("result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar")


def resolve_mri_encoder_checkpoint(exp, mode, explicit_checkpoint):
    if mode == "raw":
        return None
    if explicit_checkpoint:
        return Path(explicit_checkpoint)
    env_path = os.getenv("ILCLDM_MRI_ENCODER_CHECKPOINT")
    if env_path:
        return Path(env_path)
    exp_dir = Path("result") / exp.rstrip("/")
    if mode == "finetune":
        finetuned = exp_dir / "MRI_Encoder_finetuned.pth.tar"
        if finetuned.exists():
            return finetuned
    return default_phase1_mri_encoder_checkpoint()


def load_mri_encoder_state_dict(checkpoint_path, device):
    encoder_state = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in encoder_state:
        encoder_state = encoder_state["state_dict"]
    if any(key.startswith("encoder.") for key in encoder_state.keys()):
        encoder_state = {
            key.replace("encoder.", "", 1): value
            for key, value in encoder_state.items()
            if key.startswith("encoder.")
        }
    return encoder_state


def load_mri_encoder(mode, checkpoint_path, device):
    if mode == "raw":
        print("=> MRI encoder mode raw: no checkpoint loaded")
        return None, None
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Missing MRI encoder checkpoint for mode={mode}: {checkpoint_path}")
    resolved = Path(checkpoint_path).resolve()
    print(f"=> Loading checkpoint (MRI encoder): {resolved}")
    mri_encoder = MRIEncoder().to(device)
    mri_encoder.load_state_dict(
        load_mri_encoder_state_dict(checkpoint_path, device),
        strict=True,
    )
    mri_encoder.eval()
    mri_encoder.requires_grad_(False)
    return mri_encoder, Path(checkpoint_path)


def run_id_value(run_id):
    if run_id != "auto":
        return run_id
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def inference_output_dir(exp, run_id):
    return Path("result") / exp.rstrip("/") / "inference" / run_id


def audit_output_dir(exp, run_id):
    return Path("result") / exp.rstrip("/") / "embeddings" / "conditioning_mri" / run_id


_REPO_ROOT = Path(__file__).resolve().parent
_RESULT_ROOT = _REPO_ROOT / "result"


def unet_build_config_from_checkpoint(checkpoint_path: str | Path) -> tuple[bool, int]:
    """Infer ROI channel and input width from a saved UNet checkpoint."""
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    in_channels = int(state_dict["downs.0.weight"].shape[1])
    use_roi_channel = "roi_latent" in state_dict
    return use_roi_channel, in_channels


def unet_build_config_from_manifest(exp):
    """Infer UNet ROI channel from run_manifest.json (falls back to config/env)."""
    manifest_path = _RESULT_ROOT / exp.rstrip("/") / "run_manifest.json"
    use_roi_channel = config.UNET_ROI_CHANNEL
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        cfg = manifest.get("config", {})
        if "UNET_ROI_CHANNEL" in cfg:
            use_roi_channel = bool(cfg["UNET_ROI_CHANNEL"])
        elif "UNET_IN_CHANNELS" in cfg:
            in_channels = int(cfg["UNET_IN_CHANNELS"])
            return in_channels > 2, in_channels
    in_channels = 2 + int(use_roi_channel)
    return use_roi_channel, in_channels


def resolve_unet_build_config(
    exp: str,
    checkpoint_path: str | Path | None = None,
) -> tuple[bool, int]:
    """Build config for UNet loading; checkpoint shapes win over manifest/env."""
    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if ckpt_path.is_file():
            return unet_build_config_from_checkpoint(ckpt_path)
    return unet_build_config_from_manifest(exp)


def _manifest_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip() in {"1", "true", "True"}


def apply_manifest_to_config(exp: str) -> dict | None:
    """Apply split paths and checkpoints from run_manifest.json to live config."""
    manifest_path = _RESULT_ROOT / exp.rstrip("/") / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    env = manifest.get("env_overrides", {})
    cfg = manifest.get("config", {})
    env_to_attr = {
        "ILCLDM_TRAIN_FILE": "train",
        "ILCLDM_TEST_FILE": "test",
        "ILCLDM_VALIDATION_FILE": "validation",
        "ILCLDM_LATENT_ABETA": "latent_Abeta",
        "ILCLDM_AAE_CHECKPOINT": "CHECKPOINT_AAE",
        "ILCLDM_MRI_ENCODER_CHECKPOINT": "MRI_ENCODER_CHECKPOINT",
        "ILCLDM_WHOLE_MRI": "whole_MRI",
        "ILCLDM_WHOLE_ABETA": "whole_Abeta",
        "ILCLDM_REFERENCE_NIFTI": "path",
        "ILCLDM_LATENT_STATS_PATH": "LATENT_STATS_PATH",
        "ILCLDM_ROI_MASK_PATH": "ROI_MASK_PATH",
    }
    for env_key, attr in env_to_attr.items():
        value = env.get(env_key) or cfg.get(attr)
        if value:
            setattr(config, attr, value)

    skip_crop = env.get("ILCLDM_SKIP_CROP")
    if skip_crop is not None:
        config.SKIP_CROP = _manifest_bool(skip_crop)

    roi_shape = env.get("ILCLDM_ROI_VOLUME_SHAPE")
    if roi_shape:
        config.ROI_VOLUME_SHAPE = parse_shape_triplet(str(roi_shape))

    unet_roi = env.get("ILCLDM_UNET_ROI_CHANNEL")
    if unet_roi is None and "UNET_ROI_CHANNEL" in cfg:
        unet_roi = cfg["UNET_ROI_CHANNEL"]
    if unet_roi is not None:
        config.UNET_ROI_CHANNEL = _manifest_bool(unet_roi)
        config.UNET_IN_CHANNELS = 2 + int(config.UNET_ROI_CHANNEL)

    calib = env.get("ILCLDM_LATENT_CALIBRATION")
    if calib is None and "LATENT_CALIBRATION" in cfg:
        calib = cfg["LATENT_CALIBRATION"]
    if calib is not None:
        config.LATENT_CALIBRATION = _manifest_bool(calib)

    if skip_crop is not None or roi_shape:
        clear_shape_cache()
    return manifest


def load_cls_head_for_exp(exp: str, device) -> LatentClsHead | None:
    path = _RESULT_ROOT / exp.rstrip("/") / "ClsHead.pth.tar"
    if not path.is_file():
        return None
    head = LatentClsHead().to(device)
    load_cls_head_checkpoint(head, path)
    head.eval()
    return head


def load_models(exp, device):
    checkpoint_aae, checkpoint_unet = checkpoint_paths(exp)
    if not checkpoint_aae.exists():
        raise FileNotFoundError(f"Missing AAE checkpoint: {checkpoint_aae}")
    if not checkpoint_unet.exists():
        raise FileNotFoundError(f"Missing UNet checkpoint: {checkpoint_unet}")

    aae = AAE().to(device).eval()
    load_checkpoint(
        str(checkpoint_aae), aae, None, config.learning_rate, model_name="SPECT AAE"
    )

    use_roi_channel, in_channels = resolve_unet_build_config(exp, checkpoint_unet)
    unet = build_unet(in_channel=in_channels, use_roi_channel=use_roi_channel).to(device).eval()
    load_checkpoint(
        str(checkpoint_unet), unet, None, config.learning_rate, model_name="UNet"
    )

    return aae, unet, checkpoint_aae, checkpoint_unet


def _embedding_to_nifti(reference_path, embedding):
    reference = nib.load(reference_path)
    return nib.Nifti1Image(embedding.astype(np.float32), reference.affine, reference.header)


def extract_conditioning_embeddings(args, unet, device, out_dir):
    rep_root = out_dir / "representations"
    nifti_root = out_dir / "nifti"
    rep_root.mkdir(parents=True, exist_ok=True)
    nifti_root.mkdir(parents=True, exist_ok=True)
    metadata_rows = []

    for split in args.splits:
        rep_dir = rep_root / split
        nifti_dir = nifti_root / split
        rep_dir.mkdir(parents=True, exist_ok=True)
        nifti_dir.mkdir(parents=True, exist_ok=True)
        loader = build_loader(split, args.batch_size, args.max_subjects_per_split)
        print(f"Extracting conditioning embeddings for {split} ({len(loader.dataset)} subjects)...")
        for MRI, _Abeta, names, labels in tqdm(loader, desc=f"audit:{split}"):
            MRI = torch.as_tensor(np.expand_dims(MRI, axis=1), dtype=torch.float32, device=device)
            labels_np = labels.detach().cpu().numpy().reshape(-1).astype(np.float32)
            with torch.no_grad():
                cond = unet.cond(MRI)
            for i in range(cond.shape[0]):
                subject_id = subject_id_from_name(names[i] if isinstance(names, (tuple, list)) else names)
                cond_np = cond[i].detach().cpu().numpy().astype(np.float32)
                latent_shape = get_latent_shape()
                if cond_np.shape != (1, *latent_shape):
                    raise ValueError(f"Unexpected conditioning embedding shape for {subject_id}: {cond_np.shape}")

                npz_path = rep_dir / f"{subject_id}.npz"
                np.savez_compressed(
                    npz_path,
                    ID=np.array(subject_id),
                    split=np.array(split),
                    label=np.array(labels_np[i], dtype=np.float32),
                    conditioning_mri_embedding=cond_np,
                )

                reference_path = resolve_nifti_path(config.whole_MRI, subject_id)
                cond_img = _embedding_to_nifti(reference_path, cond_np[0])
                nifti_path = nifti_dir / f"{subject_id}.nii.gz"
                nib.save(cond_img, nifti_path)
                metadata_rows.append(
                    {
                        "ID": subject_id,
                        "split": split,
                        "label": float(labels_np[i]),
                        "conditioning_embedding_path": str(npz_path),
                        "conditioning_nifti_path": str(nifti_path),
                        "embedding_shape": "1,40,48,40",
                    }
                )

    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(out_dir / "metadata.csv", index=False)
    return metadata


def _load_conditioning_feature_matrix(metadata, split):
    split_rows = metadata[metadata["split"] == split].copy()
    features = []
    labels = []
    ids = []
    for row in split_rows.itertuples(index=False):
        payload = np.load(row.conditioning_embedding_path)
        embedding = payload["conditioning_mri_embedding"].astype(np.float32).reshape(-1)
        features.append(embedding)
        labels.append(int(row.label))
        ids.append(row.ID)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), ids


def latent_id(path):
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def ensure_embedding_eval_split(split_path):
    split_path = Path(split_path)
    if split_path.exists():
        split_df = pd.read_csv(split_path, dtype={"ID": str})
        required = {"ID", "label", "embedding_eval_split"}
        missing = required.difference(split_df.columns)
        if missing:
            raise ValueError(f"{split_path} is missing required columns: {sorted(missing)}")
        return split_df

    latent_dir = Path(config.latent_Abeta)
    latent_paths = sorted(list(latent_dir.glob("*.nii")) + list(latent_dir.glob("*.nii.gz")))
    if not latent_paths:
        raise FileNotFoundError(f"No phase-1 SPECT latents found in {latent_dir}")

    info = pd.read_csv("data_info/data_info.csv", dtype={"ID": str})
    info["label"] = info["label"].astype(int)
    latent_df = pd.DataFrame(
        {
            "ID": [latent_id(path) for path in latent_paths],
            "latent_path": [str(path) for path in latent_paths],
        }
    )
    merged = latent_df.merge(info[["ID", "label"]], on="ID", how="inner").reset_index(drop=True)
    if merged.empty:
        raise ValueError("Could not match phase-1 SPECT latents with data_info.csv labels.")

    train_meta, test_meta = train_test_split(
        merged,
        test_size=0.20,
        stratify=merged["label"].to_numpy(),
        random_state=42,
    )
    train_meta = train_meta.copy()
    test_meta = test_meta.copy()
    train_meta["embedding_eval_split"] = "train"
    test_meta["embedding_eval_split"] = "test"
    split_df = (
        pd.concat([train_meta, test_meta], ignore_index=True)
        [["ID", "label", "embedding_eval_split"]]
        .sort_values(["embedding_eval_split", "ID"])
        .reset_index(drop=True)
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(split_path, index=False)
    return split_df


def _load_conditioning_feature_matrix_by_eval_split(metadata, split_df, eval_split):
    metadata = metadata.copy()
    metadata["ID"] = metadata["ID"].astype(str)
    split_df = split_df.copy()
    split_df["ID"] = split_df["ID"].astype(str)
    wanted = split_df[split_df["embedding_eval_split"] == eval_split][["ID", "label"]]
    rows = wanted.merge(metadata, on="ID", how="left", suffixes=("_eval", ""))
    missing = rows[rows["conditioning_embedding_path"].isna()]
    if len(missing):
        raise ValueError(
            f"Missing conditioning embeddings for {len(missing)} IDs in embedding_eval_split={eval_split}: "
            f"{missing['ID'].head(10).tolist()}"
        )

    features = []
    labels = []
    ids = []
    for row in rows.itertuples(index=False):
        payload = np.load(row.conditioning_embedding_path)
        features.append(payload["conditioning_mri_embedding"].astype(np.float32).reshape(-1))
        labels.append(int(row.label_eval))
        ids.append(row.ID)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), ids


def _evaluate_models(X_train, y_train, X_test, y_test, prediction_ids, device=None, class_dir=None):
    rows = []
    prediction_rows = pd.DataFrame({"ID": prediction_ids, "label": y_test})
    models = build_classical_models(y_train)
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = positive_scores(model, X_test)
        rows.append(metric_row(name, y_test, pred, score, **classifier_params(model)))
        prediction_rows[f"pred_{name}"] = pred
        if score is not None:
            prediction_rows[f"score_{name}"] = score
    if device is not None:
        dnn_pred, dnn_score, history = train_dnn(X_train, y_train, X_test, y_test, device)
        rows.append(metric_row("DNN_simple", y_test, dnn_pred, dnn_score, classifier="EmbeddingMLP"))
        prediction_rows["pred_DNN_simple"] = dnn_pred
        prediction_rows["score_DNN_simple"] = dnn_score
        if class_dir is not None:
            history.to_csv(class_dir / "dnn_training_history.csv", index=False)
    return pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False), prediction_rows


def _plot_confusion_and_roc(y_true, y_pred, y_score, figures_dir):
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
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    if y_score is not None and len(np.unique(y_true)) == 2:
        from sklearn.metrics import roc_curve

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
        plt.savefig(figures_dir / "roc_curve.png", dpi=200)
        plt.close()


def _plot_embedding_projection(X, y, split_names, output_path, mode="tsne"):
    if len(X) < 2:
        return
    if mode == "tsne":
        perplexity = min(30, max(1, len(X) - 1))
        projector = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        proj = projector.fit_transform(X)
    else:
        projector = PCA(n_components=2, random_state=42)
        proj = projector.fit_transform(X)
    plt.figure(figsize=(8, 6))
    unique_splits = sorted(set(split_names.tolist()))
    for split in unique_splits:
        idx = np.where(split_names == split)[0]
        plt.scatter(
            proj[idx, 0],
            proj[idx, 1],
            c=y[idx],
            cmap="coolwarm",
            alpha=0.65,
            label=split,
        )
    plt.xlabel("component_1")
    plt.ylabel("component_2")
    plt.title(f"{mode.upper()} projection of conditioning embeddings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _projection_coordinates(X, method, n_components):
    if len(X) < 2:
        return np.empty((0, n_components), dtype=np.float32)
    X_scaled = StandardScaler().fit_transform(X)
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == "tsne":
        perplexity = min(30, max(1, len(X_scaled) - 1))
        reducer = TSNE(
            n_components=n_components,
            random_state=42,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
        )
    elif method == "umap":
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


def _label_name(label):
    return "control" if int(label) == 0 else "parkinson"


def _projection_dataframe(coords, ids, labels, splits, method, eval_splits=None):
    columns = [f"{method}_{idx + 1}" for idx in range(coords.shape[1])]
    df = pd.DataFrame(coords, columns=columns)
    df["ID"] = ids
    df["label"] = labels
    df["class"] = [_label_name(label) for label in labels]
    df["split"] = splits
    if eval_splits is not None:
        df["embedding_eval_split"] = eval_splits
    return df


def _save_projection_2d(df, method, out_dir):
    x_col = f"{method}_1"
    y_col = f"{method}_2"
    colors = {"control": "#3b6fd8", "parkinson": "#d83b5f"}
    plt.figure(figsize=(8, 6))
    for class_name in ["control", "parkinson"]:
        class_df = df[df["class"] == class_name]
        if class_df.empty:
            continue
        plt.scatter(
            class_df[x_col],
            class_df[y_col],
            color=colors[class_name],
            alpha=0.7,
            label=class_name,
        )
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{method.upper()} 2D conditioning MRI embeddings - test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{method}_2d.png", dpi=200)
    plt.close()


def _save_projection_3d(df, method, out_dir):
    fig = px.scatter_3d(
        df,
        x=f"{method}_1",
        y=f"{method}_2",
        z=f"{method}_3",
        color="class",
        color_discrete_map={"control": "#3b6fd8", "parkinson": "#d83b5f"},
        hover_data=["ID", "label", "class", "split"],
        title=f"{method.upper()} 3D conditioning MRI embeddings - test",
    )
    fig.write_html(out_dir / f"{method}_3d.html")


def save_conditioning_visualizations(metadata, out_dir, split_df=None, eval_split="test"):
    viz_dir = out_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    plot_metadata = metadata.copy()
    plot_metadata["ID"] = plot_metadata["ID"].astype(str)
    if split_df is not None:
        split_df = split_df.copy()
        split_df["ID"] = split_df["ID"].astype(str)
        split_df = split_df[split_df["embedding_eval_split"] == eval_split][
            ["ID", "label", "embedding_eval_split"]
        ]
        plot_metadata = split_df.merge(
            plot_metadata,
            on="ID",
            how="inner",
            suffixes=("_eval", ""),
        )
        label_column = "label_eval"
    elif "split" in plot_metadata.columns and eval_split in set(plot_metadata["split"].astype(str)):
        plot_metadata = plot_metadata[plot_metadata["split"].astype(str) == eval_split]
        label_column = "label"
    else:
        label_column = "label"

    features = []
    labels = []
    splits = []
    eval_splits = []
    ids = []
    for row in plot_metadata.itertuples(index=False):
        payload = np.load(row.conditioning_embedding_path)
        features.append(payload["conditioning_mri_embedding"].astype(np.float32).reshape(-1))
        labels.append(int(getattr(row, label_column)))
        splits.append(row.split)
        eval_splits.append(getattr(row, "embedding_eval_split", eval_split))
        ids.append(row.ID)
    if len(features) < 2:
        return

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    split_array = np.asarray(splits)
    eval_split_array = np.asarray(eval_splits)
    for method in ["pca", "tsne", "umap"]:
        coords_2d = _projection_coordinates(X, method, 2)
        df_2d = _projection_dataframe(coords_2d, ids, y, split_array, method, eval_split_array)
        df_2d.to_csv(viz_dir / f"{method}_2d.csv", index=False)
        _save_projection_2d(df_2d, method, viz_dir)

        coords_3d = _projection_coordinates(X, method, 3)
        df_3d = _projection_dataframe(coords_3d, ids, y, split_array, method, eval_split_array)
        df_3d.to_csv(viz_dir / f"{method}_3d.csv", index=False)
        _save_projection_3d(df_3d, method, viz_dir)


def write_conditioning_manifest(out_dir, args, checkpoint_unet):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": args.exp,
        "embedding_type": "conditioning_mri",
        "source": "UNet.cond(MRI)",
        "checkpoint_unet": str(checkpoint_unet),
        "device": config.device,
        "paths": {
            "whole_MRI": config.whole_MRI,
            "train": config.train,
            "validation": config.validation,
            "test": config.test,
        },
        "splits": args.splits,
        "batch_size": args.batch_size,
        "max_subjects_per_split": args.max_subjects_per_split,
        "include_raw_mri_baseline": args.include_raw_mri_baseline,
        "embedding_eval_split": args.embedding_eval_split,
        "classification_protocol": "phase1_train_internal_80_20",
        "expected_embedding_shape": "1,40,48,40",
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


def _raw_mri_baseline(args, device):
    train_loader = build_loader("train", args.batch_size, args.max_subjects_per_split)
    test_loader = build_loader("test", args.batch_size, args.max_subjects_per_split)
    X_train, y_train, X_test, y_test = [], [], [], []
    for MRI, _Abeta, _names, labels in train_loader:
        X_train.append(np.expand_dims(MRI, axis=1).reshape(len(labels), -1))
        y_train.append(labels.detach().cpu().numpy().reshape(-1))
    for MRI, _Abeta, _names, labels in test_loader:
        X_test.append(np.expand_dims(MRI, axis=1).reshape(len(labels), -1))
        y_test.append(labels.detach().cpu().numpy().reshape(-1))
    X_train = np.concatenate(X_train, axis=0).astype(np.float32)
    y_train = np.concatenate(y_train, axis=0).astype(np.int64)
    X_test = np.concatenate(X_test, axis=0).astype(np.float32)
    y_test = np.concatenate(y_test, axis=0).astype(np.int64)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = model.predict_proba(X_test)[:, 1]
    return metric_row("MRI_raw_LogisticRegression", y_test, pred, score, classifier="LogisticRegression")


def _load_latest_latent_baseline(exp):
    base = Path("result") / exp.rstrip("/") / "inference"
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name != "mri_conditioning_audit_v1"]
    if not candidates:
        return None
    latest = sorted(candidates, key=lambda x: x.name, reverse=True)[0]
    metrics_path = latest / "classification" / "classification_metrics.csv"
    if not metrics_path.exists():
        return None
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return None
    best = metrics.sort_values("balanced_accuracy", ascending=False).iloc[0].to_dict()
    best["source_run"] = str(latest)
    return best


def run_conditioning_audit(args, unet, device):
    out_dir = audit_output_dir(args.exp, run_id_value(args.run_id))
    class_dir = out_dir / "classification"
    fig_dir = out_dir / "visualizations"
    class_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    _, checkpoint_unet = checkpoint_paths(args.exp)

    if "train" not in args.splits:
        raise ValueError("Conditioning audit requires the train split.")

    metadata = extract_conditioning_embeddings(args, unet, device, out_dir)
    write_conditioning_manifest(out_dir, args, checkpoint_unet)
    split_df = ensure_embedding_eval_split(args.embedding_eval_split)
    split_df.to_csv(class_dir / "embedding_eval_split.csv", index=False)
    X_train, y_train, _train_ids = _load_conditioning_feature_matrix_by_eval_split(
        metadata,
        split_df,
        "train",
    )
    X_test, y_test, test_ids = _load_conditioning_feature_matrix_by_eval_split(
        metadata,
        split_df,
        "test",
    )
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Conditioning audit requires non-empty train and test embeddings.")

    cls_metrics, predictions = _evaluate_models(
        X_train,
        y_train,
        X_test,
        y_test,
        test_ids,
        device=device,
        class_dir=class_dir,
    )
    cls_metrics.to_csv(class_dir / "classification_metrics.csv", index=False)
    cls_metrics.to_json(class_dir / "metrics_summary.json", orient="records", indent=2)
    predictions.to_csv(class_dir / "test_predictions.csv", index=False)

    best_model = cls_metrics.iloc[0]["model"]
    best_pred = predictions[f"pred_{best_model}"].to_numpy()
    best_score_column = f"score_{best_model}"
    best_score = predictions[best_score_column].to_numpy() if best_score_column in predictions.columns else None
    _plot_confusion_and_roc(y_test, best_pred, best_score, fig_dir)

    save_conditioning_visualizations(metadata, out_dir, split_df=split_df, eval_split="test")

    latent_baseline = _load_latest_latent_baseline(args.exp)
    baseline_rows = []
    if args.include_raw_mri_baseline:
        baseline_rows.append(_raw_mri_baseline(args, device))
    if latent_baseline is not None:
        baseline_rows.append(
            {
                "model": f"sampled_latent_best_{latent_baseline.get('model', 'unknown')}",
                "balanced_accuracy": latent_baseline.get("balanced_accuracy"),
                "f1": latent_baseline.get("f1"),
                "roc_auc": latent_baseline.get("roc_auc"),
                "source_run": latent_baseline.get("source_run"),
            }
        )
    pd.DataFrame(baseline_rows).to_csv(class_dir / "baseline_comparison.csv", index=False)

    best_auc = float(cls_metrics.iloc[0].get("roc_auc", np.nan))
    best_bal_acc = float(cls_metrics.iloc[0].get("balanced_accuracy", np.nan))
    unstable = bool(cls_metrics["balanced_accuracy"].std() > 0.08) if len(cls_metrics) > 1 else False
    go_pretrain = bool(np.isnan(best_auc) or best_auc < 0.65 or unstable)
    decision = {
        "best_model": best_model,
        "best_roc_auc": None if np.isnan(best_auc) else best_auc,
        "best_balanced_accuracy": best_bal_acc,
        "unstable_conditioning_signal": unstable,
        "recommendation": "go_pretrain_mri_encoder" if go_pretrain else "keep_current_conditioning",
        "go_threshold_auc": 0.65,
        "stable_threshold_balanced_accuracy": 0.70,
    }
    with open(class_dir / "go_no_go_decision.json", "w", encoding="utf-8") as file:
        json.dump(decision, file, indent=2)

    report = classification_report(y_test, best_pred, digits=4)
    with open(class_dir / "classifier_report.txt", "w", encoding="utf-8") as file:
        file.write(f"Best model: {best_model}\n\n")
        file.write(report)
    return out_dir, decision


def build_loader(split, batch_size, max_subjects=None):
    dataset = TwoDataset(
        root_MRI=config.whole_MRI,
        root_Abeta=config.whole_Abeta,
        task=task_file_for_split(split),
        stage="inference",
    )
    if max_subjects is not None:
        dataset.images = dataset.images[:max_subjects]
        dataset.length_dataset = len(dataset.images)
        dataset.len = len(dataset.images)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.numworker,
        pin_memory=(config.device == "cuda"),
    )


def decode_spect(aae, sampled_latent):
    with torch.no_grad():
        return aae.decoder(sampled_latent)


def save_subject_representation(out_dir, split, subject_id, label, sampled_latent, synthetic_spect=None):
    rep_dir = out_dir / "representations" / split
    rep_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ID": np.array(subject_id),
        "split": np.array(split),
        "label": np.array(label, dtype=np.float32),
        "sampled_latent_final": sampled_latent.astype(np.float32),
    }
    if synthetic_spect is not None:
        payload["synthetic_spect_flat"] = synthetic_spect.astype(np.float32).reshape(-1)
    out_path = rep_dir / f"{subject_id}.npz"
    np.savez_compressed(out_path, **payload)
    return out_path


def save_synthetic_nifti(out_dir, split, subject_id, synthetic_spect):
    spect_dir = out_dir / "synthetic_spect" / split
    spect_dir.mkdir(parents=True, exist_ok=True)
    reference = nib.load(resolve_nifti_path(config.whole_MRI, subject_id))
    volume = crop_volume_to_raw_shape(synthetic_spect.astype(np.float32))
    image = nib.Nifti1Image(volume, reference.affine, reference.header)
    out_path = spect_dir / f"{subject_id}.nii.gz"
    nib.save(image, out_path)
    return out_path


def sample_latents(diffusion, unet, mri_encoder, MRI, sampling_steps, eta):
    if mri_encoder is not None:
        MRI_cond = mri_encoder(MRI)
        return diffusion.sample(
            lambda x, y, t: unet(x, y, t, precomputed_condition=True),
            MRI_cond,
            sampling_steps=sampling_steps,
            eta=eta,
        )
    return diffusion.sample(
        unet,
        MRI,
        sampling_steps=sampling_steps,
        eta=eta,
    )


def subject_reconstruction_metrics(target_np, pred_np, roi_mask_np=None):
    psnr_value, ssim_value = batch_metric_scores(target_np, pred_np)
    metrics = {"PSNR": psnr_value, "SSIM": ssim_value}
    if roi_mask_np is not None:
        psnr_roi, ssim_roi = batch_metric_scores_roi(target_np, pred_np, roi_mask_np)
        metrics["PSNR_roi"] = psnr_roi
        metrics["SSIM_roi"] = ssim_roi
    return metrics


def generate_representations(args, split, aae, unet, mri_encoder, diffusion, out_dir, device, roi_mask_np=None):
    loader = build_loader(split, args.batch_size, args.max_subjects_per_split)
    rows = []
    metrics = []

    print(f"Generating {split} representations ({len(loader.dataset)} subjects)...")
    for MRI, Abeta, names, labels in tqdm(loader, desc=f"inference:{split}"):
        MRI = torch.as_tensor(np.expand_dims(MRI, axis=1), dtype=torch.float32, device=device)
        Abeta = torch.as_tensor(np.expand_dims(Abeta, axis=1), dtype=torch.float32, device=device)
        labels_np = labels.detach().cpu().numpy().reshape(-1).astype(np.float32)

        with torch.no_grad():
            sampled_latent = sample_latents(
                diffusion,
                unet,
                mri_encoder,
                MRI,
                args.sampling_steps,
                args.ddim_eta,
            )
            synthetic_spect = decode_spect(aae, sampled_latent)

        vol_shape = get_working_volume_shape()
        latent_shape = get_latent_shape()
        latent_shape_label = f"1,{','.join(map(str, latent_shape))}"
        spect_shape_label = f"1,{','.join(map(str, vol_shape))}"
        if tuple(sampled_latent.shape[1:]) != (1, *latent_shape):
            raise ValueError(f"Unexpected latent shape: {tuple(sampled_latent.shape)}")
        if tuple(synthetic_spect.shape[1:]) != (1, *vol_shape):
            raise ValueError(f"Unexpected synthetic SPECT shape: {tuple(synthetic_spect.shape)}")

        batch_size = sampled_latent.shape[0]
        for i in range(batch_size):
            subject_id = subject_id_from_name(names[i] if isinstance(names, (tuple, list)) else names)
            label = float(labels_np[i])
            rep_path_existing = out_dir / "representations" / split / f"{subject_id}.npz"
            if args.resume and rep_path_existing.exists():
                latent_np = np.load(rep_path_existing)["sampled_latent_final"]
                synthetic_np = None
                if args.save_synthetic_spect:
                    spect_existing = out_dir / "synthetic_spect" / split / f"{subject_id}.nii.gz"
                    if spect_existing.exists():
                        synthetic_np = nib.load(spect_existing).get_fdata().astype(np.float32)
                if synthetic_np is None:
                    synthetic_spect_i = decode_spect(
                        aae,
                        torch.as_tensor(latent_np, device=device).unsqueeze(0),
                    )
                    synthetic_np = synthetic_spect_i[0, 0].detach().cpu().numpy()
                target_np = Abeta[i].detach().cpu()
                pred_np = torch.as_tensor(
                    np.expand_dims(synthetic_np, axis=(0, 1)), dtype=torch.float32
                )
                metric_entry = subject_reconstruction_metrics(target_np, pred_np, roi_mask_np)
                rows.append({
                    "ID": subject_id,
                    "split": split,
                    "label": label,
                    "representation_path": str(rep_path_existing),
                    "synthetic_spect_path": str(
                        out_dir / "synthetic_spect" / split / f"{subject_id}.nii.gz"
                    )
                    if args.save_synthetic_spect
                    and (out_dir / "synthetic_spect" / split / f"{subject_id}.nii.gz").exists()
                    else "",
                    "latent_shape": latent_shape_label,
                    "synthetic_spect_shape": spect_shape_label,
                })
                metrics.append({
                    "ID": subject_id,
                    "split": split,
                    "label": label,
                    **metric_entry,
                })
                continue

            latent_np = sampled_latent[i].detach().cpu().numpy()
            synthetic_np = synthetic_spect[i, 0].detach().cpu().numpy()
            target_np = Abeta[i].detach().cpu()
            pred_np = synthetic_spect[i].detach().cpu()

            rep_path = save_subject_representation(
                out_dir,
                split,
                subject_id,
                label,
                latent_np,
                None if args.save_synthetic_spect else synthetic_np,
            )
            spect_path = ""
            if args.save_synthetic_spect:
                spect_path = str(save_synthetic_nifti(out_dir, split, subject_id, synthetic_np))

            metric_entry = subject_reconstruction_metrics(target_np, pred_np, roi_mask_np)
            rows.append({
                "ID": subject_id,
                "split": split,
                "label": label,
                "representation_path": str(rep_path),
                "synthetic_spect_path": spect_path,
                "latent_shape": latent_shape_label,
                "synthetic_spect_shape": spect_shape_label,
            })
            metrics.append({
                "ID": subject_id,
                "split": split,
                "label": label,
                **metric_entry,
            })

    return rows, metrics


def load_feature_matrix(metadata, split):
    split_rows = metadata[metadata["split"] == split].copy()
    features = []
    labels = []
    ids = []
    for row in split_rows.itertuples(index=False):
        payload = np.load(row.representation_path)
        features.append(payload["sampled_latent_final"].reshape(-1))
        labels.append(float(row.label))
        ids.append(row.ID)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), ids


def build_classical_models(y_train):
    knn_auto = min(5, len(y_train))
    if knn_auto % 2 == 0:
        knn_auto -= 1
    knn_auto = max(1, knn_auto)
    knn_k2 = min(2, len(y_train))

    def pipe(classifier):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ])

    return {
        "KNN_auto": pipe(KNeighborsClassifier(n_neighbors=knn_auto, weights="distance")),
        "KNN_K2": pipe(KNeighborsClassifier(n_neighbors=knn_k2, weights="distance")),
        "SVM_RBF": pipe(SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=42)),
        "SVM_linear": pipe(SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced", random_state=42)),
        "LogisticRegression": pipe(LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)),
    }


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


def positive_scores(model, features):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return None


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


def train_dnn(X_train, y_train, X_test, y_test, device):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    model = EmbeddingMLP(X_train_scaled.shape[1]).to(device)
    class_counts = np.bincount(y_train.astype(int), minlength=2)
    pos_weight = torch.tensor([class_counts[0] / max(class_counts[1], 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train.astype(np.float32), dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=min(16, len(train_dataset)), shuffle=True)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test.astype(np.float32), dtype=torch.float32, device=device)

    best_state = None
    best_loss = float("inf")
    bad_epochs = 0
    patience = 30
    history = []

    for epoch in range(300):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
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


def metric_row(model_name, y_true, y_pred, y_score=None, **params):
    row = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score) if y_score is not None and len(np.unique(y_true)) == 2 else np.nan,
    }
    row.update(params)
    return row


def run_classification(out_dir, device):
    metadata = pd.read_csv(out_dir / "metadata.csv", dtype={"ID": str})
    X_train, y_train, train_ids = load_feature_matrix(metadata, "train")
    X_test, y_test, test_ids = load_feature_matrix(metadata, "test")
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Classification requires both train and test representations.")

    class_dir = out_dir / "classification"
    class_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    prediction_rows = pd.DataFrame({"ID": test_ids, "label": y_test})
    for name, model in build_classical_models(y_train).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = positive_scores(model, X_test)
        rows.append(metric_row(name, y_test, pred, score, **classifier_params(model)))
        prediction_rows[f"pred_{name}"] = pred
        if score is not None:
            prediction_rows[f"score_{name}"] = score

    dnn_pred, dnn_score, history = train_dnn(X_train, y_train, X_test, y_test, device)
    rows.append(metric_row("DNN_simple", y_test, dnn_pred, dnn_score, classifier="EmbeddingMLP"))
    prediction_rows["pred_DNN_simple"] = dnn_pred
    prediction_rows["score_DNN_simple"] = dnn_score

    metrics = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    metrics.to_csv(class_dir / "classification_metrics.csv", index=False)
    prediction_rows.to_csv(class_dir / "test_predictions.csv", index=False)
    history.to_csv(class_dir / "dnn_training_history.csv", index=False)

    report = classification_report(y_test, dnn_pred, output_dict=True)
    with open(class_dir / "dnn_classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    pd.DataFrame(confusion_matrix(y_test, dnn_pred)).to_csv(class_dir / "dnn_confusion_matrix.csv", index=False)
    return metrics


def write_manifest(out_dir, args, checkpoint_aae, checkpoint_unet, mri_encoder_mode, mri_encoder_checkpoint):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": args.exp,
        "mri_encoder_mode": mri_encoder_mode,
        "checkpoints": {
            "AAE": str(checkpoint_aae),
            "Unet": str(checkpoint_unet),
            "MRI_encoder": str(mri_encoder_checkpoint) if mri_encoder_checkpoint else None,
        },
        "noiseSteps": config.noiseSteps,
        "sampling_steps": args.sampling_steps,
        "ddim_eta": args.ddim_eta,
        "device": config.device,
        "paths": {
            "whole_MRI": config.whole_MRI,
            "whole_Abeta": config.whole_Abeta,
            "latent_Abeta": config.latent_Abeta,
            "train": config.train,
            "validation": config.validation,
            "test": config.test,
        },
        "splits": args.splits,
        "batch_size": args.batch_size,
        "max_subjects_per_split": args.max_subjects_per_split,
        "save_synthetic_spect": args.save_synthetic_spect,
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    args = parse_args()
    if args.splits is None:
        args.splits = ["train"] if args.run_conditioning_audit else ["train", "test"]
    seed_torch(args.seed)
    device = config.device
    manifest = apply_manifest_to_config(args.exp)
    if manifest is not None:
        print(f"Applied run_manifest paths for {args.exp}")
    print(f"Device: {device}")
    aae, unet, checkpoint_aae, checkpoint_unet = load_models(args.exp, device)

    mri_encoder_mode = infer_mri_encoder_mode(args.exp, args.mri_encoder_mode)
    mri_encoder_ckpt_path = resolve_mri_encoder_checkpoint(
        args.exp,
        mri_encoder_mode,
        args.mri_encoder_checkpoint,
    )
    mri_encoder, mri_encoder_ckpt_resolved = load_mri_encoder(
        mri_encoder_mode,
        mri_encoder_ckpt_path,
        device,
    )
    print(f"MRI encoder mode: {mri_encoder_mode}")

    cls_head = load_cls_head_for_exp(args.exp, device)

    if args.run_conditioning_audit:
        out_dir, decision = run_conditioning_audit(args, unet, device)
        print(f"Audit output directory: {out_dir}")
        print(json.dumps(decision, indent=2))
        print(f"Done. Audit results saved in {out_dir}")
        return

    run_id = run_id_value(args.run_id)
    out_dir = inference_output_dir(args.exp, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_dir}")
    diffusion = Diffusion()
    roi_mask_np = try_load_roi_mask_numpy()
    if roi_mask_np is not None:
        print(f"ROI metrics enabled using {config.ROI_MASK_PATH}")

    metadata_rows = []
    metric_rows = []
    for split in args.splits:
        rows, metrics = generate_representations(
            args, split, aae, unet, mri_encoder, diffusion, out_dir, device, roi_mask_np=roi_mask_np
        )
        metadata_rows.extend(rows)
        metric_rows.extend(metrics)

    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(out_dir / "metadata.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out_dir / "reconstruction_metrics.csv", index=False)
    write_manifest(
        out_dir,
        args,
        checkpoint_aae,
        checkpoint_unet,
        mri_encoder_mode,
        mri_encoder_ckpt_resolved,
    )

    if "train" in args.splits and "test" in args.splits:
        metrics = run_classification(out_dir, device)
        print(metrics)
    else:
        print("Skipping classification because train and test splits were not both generated.")

    print(f"Done. Results saved in {out_dir}")


if __name__ == "__main__":
    main()
