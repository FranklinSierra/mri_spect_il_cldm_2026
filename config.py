import torch
import os

device = os.getenv("ILCLDM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
noiseSteps = int(os.getenv("ILCLDM_NOISE_STEPS", "1000"))
sampling_steps = int(os.getenv("ILCLDM_SAMPLING_STEPS", "100"))
ddim_eta = float(os.getenv("ILCLDM_DDIM_ETA", "0.0"))
latent_dim = 1
learning_rate = float(os.getenv("ILCLDM_LR", "1e-4"))
batch_size = int(os.getenv("ILCLDM_BATCH_SIZE", "8"))
numworker = 0
epochs = int(os.getenv("ILCLDM_EPOCHS", "1000"))
validation_interval = int(os.getenv("ILCLDM_VALIDATION_INTERVAL", "5"))
early_stopping_patience = int(os.getenv("ILCLDM_EARLY_STOPPING_PATIENCE", "10"))
# Minimum absolute improvement required to reset early-stopping patience.
EARLY_STOP_MIN_DELTA = float(os.getenv("ILCLDM_EARLY_STOP_MIN_DELTA", "0.0"))
# Global L2 grad clip after backward, before optimizer.step. 0 = off (baseline).
GRAD_CLIP_NORM = float(os.getenv("ILCLDM_GRAD_CLIP_NORM", "0"))
time_dim = 128
num_classes = 2
Lambda = 100

# Clamp pred_x0 in calibrated diffusion space (train predict_x0 + DDIM sample).
# Symmetric fallback; the calibrated SPECT latent is strongly left-skewed
# (train: min -13.0, p0.1 -7.6, max +4.1), so +-4 clipped 8.1% of the signal mass.
# +-10 clips 0.12%. Set _LO/_HI for an asymmetric window fitted to train stats.
PRED_X0_CLAMP = float(os.getenv("ILCLDM_PRED_X0_CLAMP", "10.0"))
_env_x0_lo = os.getenv("ILCLDM_PRED_X0_CLAMP_LO", "").strip()
_env_x0_hi = os.getenv("ILCLDM_PRED_X0_CLAMP_HI", "").strip()
PRED_X0_CLAMP_LO = float(_env_x0_lo) if _env_x0_lo else (-PRED_X0_CLAMP if PRED_X0_CLAMP > 0 else None)
PRED_X0_CLAMP_HI = float(_env_x0_hi) if _env_x0_hi else (PRED_X0_CLAMP if PRED_X0_CLAMP > 0 else None)

# Highest timestep the DDIM sampler starts from. The cosine schedule drives
# alpha_hat to 2.4e-10 at t=T-1 (amplification 1/sqrt(a_hat) = 64171x), which
# saturated 99.99% of pred_x0 on the first step. At t=990 the factor is 71x.
#
# NOTE: a wide (correct) clamp exposes epsilon shrinkage instead of masking it.
# The film_ms checkpoint under-predicts eps by ~29% (<eps_pred,eps>/<eps,eps> =
# 0.71 at every t), so pred_x0 error grows as 0.29/sqrt(alpha_hat). Measured
# latent MSE on test (var of the target = 3.90, i.e. MSE 3.90 = predicting the
# mean); every row is worse than the mean, which is a model problem, not a
# sampler one:
#     t_max=990 clamp[-10,+4.5] -> 84.0     t_max=990 clamp+-4 -> 17.4
#     t_max=900 clamp[-10,+4.5] -> 37.0     t_max=800 clamp+-4 -> 14.5
#     t_max=800 clamp[-10,+4.5] -> 16.1
#     t_max=700 clamp[-10,+4.5] ->  9.7
# Lowering t_max compensates for the shrinkage; it does not fix it. A healthy
# model should reach MSE < 3.90 at t_max=990. Do NOT tighten the clamp to make
# the numbers look better -- that is what hid the divergence for months.
SAMPLING_T_MAX = int(os.getenv("ILCLDM_SAMPLING_T_MAX", "990"))

# Label dropout: if >0, Embedding has num_classes+1 rows (null token = num_classes).
LABEL_DROPOUT_PROB = float(os.getenv("ILCLDM_LABEL_DROPOUT_PROB", "0.0"))

# EMA regime (warmup 0.9 → 0.995 is mandatory when using defaults).
EMA_BETA = float(os.getenv("ILCLDM_EMA_BETA", "0.995"))
EMA_BETA_START = float(os.getenv("ILCLDM_EMA_BETA_START", "0.9"))
# 0 → auto: max(200, 5% of estimated train steps) after step_start.
EMA_WARMUP_STEPS = int(os.getenv("ILCLDM_EMA_WARMUP_STEPS", "0"))
# 0 → auto: max(1, 5% of estimated train steps).
EMA_STEP_START = int(os.getenv("ILCLDM_EMA_STEP_START", "0"))

# UNet self-attention resolutions (comma-separated). Default (4,) is legacy/broken
# for 40³ and 24³ grids; full-brain should use 5 only.
_env_attn_res = os.getenv("ILCLDM_ATTN_RES", "4").strip()
ATTN_RES = tuple(int(x.strip()) for x in _env_attn_res.split(",") if x.strip()) or (4,)

# FiLM / AdaGN-style MRI conditioning in every ResnetBlock (off = concat-only legacy).
USE_FILM = os.getenv("ILCLDM_USE_FILM", "0") == "1"

# FiLM mode: off | global (GAP+h64→MLP bias) | spatial_ms (h64 map, gamma/beta per block).
_env_film_mode = os.getenv("ILCLDM_FILM_MODE", "off").strip().lower()
if _env_film_mode not in ("off", "global", "spatial_ms"):
    raise ValueError(
        f"ILCLDM_FILM_MODE must be off|global|spatial_ms, got {_env_film_mode!r}"
    )
FILM_MODE = _env_film_mode
if FILM_MODE == "off" and USE_FILM:
    FILM_MODE = "global"
FILM_COND_CHANNELS = int(os.getenv("ILCLDM_FILM_COND_CHANNELS", "64"))

USE_WANDB = os.getenv("ILCLDM_USE_WANDB", "0") == "1"
WANDB_PROJECT = os.getenv("ILCLDM_WANDB_PROJECT", "IL-CLDM").strip() or "IL-CLDM"
WANDB_RUN_NAME = os.getenv("ILCLDM_WANDB_RUN_NAME", "").strip()

# Classical classifiers: SVM hangs on high-D latents; off by default.
INCLUDE_SVM = os.getenv("ILCLDM_INCLUDE_SVM", "0") == "1"

# Stage-1 / image-space SPECT losses (z-score aware)
USE_WEIGHTED_RECON_LOSS = os.getenv("ILCLDM_USE_WEIGHTED_RECON", "1") == "1"
USE_TAIL_RECON_LOSS = os.getenv("ILCLDM_USE_TAIL_RECON", "1") == "1"
WEIGHTED_RECON_ALPHA = float(os.getenv("ILCLDM_WEIGHTED_ALPHA", "3.0"))
WEIGHTED_Z_THRESHOLD = float(os.getenv("ILCLDM_WEIGHTED_Z_THRESH", "1.0"))
TAIL_RECON_FRACTION = float(os.getenv("ILCLDM_TAIL_FRACTION", "0.05"))
TAIL_RECON_BETA = float(os.getenv("ILCLDM_TAIL_BETA", "1.0"))

# Stage-2 auxiliary decode loss in image space
AUX_IMAGE_LOSS = os.getenv("ILCLDM_AUX_IMAGE_LOSS", "1") == "1"
AUX_IMAGE_EVERY_N = int(os.getenv("ILCLDM_AUX_IMAGE_EVERY_N", "10"))
AUX_IMAGE_LAMBDA = float(os.getenv("ILCLDM_AUX_IMAGE_LAMBDA", "0.25"))
AUX_IMAGE_ALWAYS_HIGH_T = os.getenv("ILCLDM_AUX_IMAGE_ALWAYS_HIGH_T", "0") == "1"

# Stage-2 v2: latent-space loss and high-noise timestep sampling
LATENT_LOSS = os.getenv("ILCLDM_LATENT_LOSS", "0") == "1"
LATENT_LOSS_LAMBDA = float(os.getenv("ILCLDM_LATENT_LOSS_LAMBDA", "0.5"))

# What the UNet is trained to emit.
# epsilon: predict noise, main loss is MSE (default; existing runs).
# x0: predict the clean latent, main loss is L1. Noise MSE is not added.
_env_diffusion_objective = os.getenv("ILCLDM_DIFFUSION_OBJECTIVE", "epsilon").strip().lower()
if _env_diffusion_objective not in ("epsilon", "x0"):
    raise ValueError(
        f"ILCLDM_DIFFUSION_OBJECTIVE must be epsilon|x0, got {_env_diffusion_objective!r}"
    )
DIFFUSION_OBJECTIVE = _env_diffusion_objective
HIGH_T_SAMPLE_PROB = float(os.getenv("ILCLDM_HIGH_T_SAMPLE_PROB", "0"))
HIGH_T_THRESHOLD = int(os.getenv("ILCLDM_HIGH_T_THRESHOLD", "700"))

# Stage-2 latent calibration: diffuse in z' = (z - mean) / std, invert before the
# decoder so the AAE receives latents in its original (raw) space. Stats are the
# global mean/std over the train SPECT latents (config.latent_Abeta). Set
# ILCLDM_LATENT_MEAN / ILCLDM_LATENT_STD to pin explicit values.
LATENT_CALIBRATION = os.getenv("ILCLDM_LATENT_CALIBRATION", "1") == "1"
_env_latent_mean = os.getenv("ILCLDM_LATENT_MEAN", "").strip()
_env_latent_std = os.getenv("ILCLDM_LATENT_STD", "").strip()
LATENT_MEAN = float(_env_latent_mean) if _env_latent_mean else None
LATENT_STD = float(_env_latent_std) if _env_latent_std else None
LATENT_STATS_PATH = os.getenv("ILCLDM_LATENT_STATS_PATH", "").strip()

# Image-volume load normalization for whole_MRI / whole_Abeta (not latents).
# legacy: per-volume z-score over all voxels (MRI + SPECT).
# v2load: MRI z-score using voxels > 0 only; SPECT left as-is (region-normalized).
# minmax: clip to train percentiles 0.5/99.5 (one lo/hi per modality) and scale to [0, 1].
IMAGE_VOLUME_NORM = os.getenv("ILCLDM_IMAGE_VOLUME_NORM", "legacy").strip().lower() or "legacy"
if IMAGE_VOLUME_NORM not in ("legacy", "v2load", "minmax"):
    raise ValueError(
        f"ILCLDM_IMAGE_VOLUME_NORM must be legacy|v2load|minmax, got {IMAGE_VOLUME_NORM!r}"
    )
MINMAX_STATS_PATH = os.getenv(
    "ILCLDM_MINMAX_STATS_PATH", "data_info/minmax_rois_p05_p995.json"
).strip()
MINMAX_PERCENTILE_LO = float(os.getenv("ILCLDM_MINMAX_PERCENTILE_LO", "0.5"))
MINMAX_PERCENTILE_HI = float(os.getenv("ILCLDM_MINMAX_PERCENTILE_HI", "99.5"))

# KL on the AAE posterior toward N(0, I). The encoder then emits (mu, logvar)
# and the decoder ends in tanh. Off for every existing experiment.
KL_LOSS = os.getenv("ILCLDM_KL_LOSS", "0") == "1"
KL_LAMBDA = float(os.getenv("ILCLDM_KL_LAMBDA", "1e-3"))

# ROI patch profile (data_rois): skip IL-CLDM crop(); volumes already cropped on disk.
SKIP_CROP = os.getenv("ILCLDM_SKIP_CROP", "0") == "1"
_env_roi_volume_shape = os.getenv("ILCLDM_ROI_VOLUME_SHAPE", "").strip()
ROI_VOLUME_SHAPE = None
if _env_roi_volume_shape:
    _parts = [int(x.strip()) for x in _env_roi_volume_shape.split(",") if x.strip()]
    if len(_parts) != 3:
        raise ValueError(f"ILCLDM_ROI_VOLUME_SHAPE must have 3 integers, got {_env_roi_volume_shape!r}")
    ROI_VOLUME_SHAPE = tuple(_parts)

# Stage-2 v2: ROI-weighted aux loss (basal ganglia mask, same crop as volumes)
ROI_MASK_PATH = os.getenv("ILCLDM_ROI_MASK_PATH", "data/mask_basal_ganglia_full.nii.gz")
ROI_WEIGHT_ALPHA = float(os.getenv("ILCLDM_ROI_WEIGHT_ALPHA", "3.0"))
USE_ROI_LOSS = os.getenv("ILCLDM_USE_ROI_LOSS", "0") == "1"

# Stage-2 R3: weight the primary x0 L1 in latent space. Outside ROI = 1, inside = N.
# 0 = off (plain L1). Typical ablation value: 10.
DIFFUSION_ROI_WEIGHT = float(os.getenv("ILCLDM_DIFFUSION_ROI_WEIGHT", "0"))

# Stage-2 R4: λ(1−SSIM) on decoded SPECT vs GT when objective is x0. 0 = off.
# PASTA adds (1−SSIM) unweighted → λ=1.0. Applied every training step.
DIFFUSION_SSIM_LAMBDA = float(os.getenv("ILCLDM_DIFFUSION_SSIM_LAMBDA", "0"))

# Stage-2 v2: early stopping on validation latent MSE (else legacy PSNR+SSIM*10)
EARLY_STOP_LATENT_METRIC = os.getenv("ILCLDM_EARLY_STOP_LATENT_METRIC", "0") == "1"

# Stage-2 h05 wave: training/inference options
USE_TRAIN_LABEL = os.getenv("ILCLDM_USE_TRAIN_LABEL", "1") == "1"
UNET_ROI_CHANNEL = os.getenv("ILCLDM_UNET_ROI_CHANNEL", "0") == "1"
# Placeholder; recomputed below after DIFFUSION_FEATURE_SPACE is known.
UNET_IN_CHANNELS = 2 + int(UNET_ROI_CHANNEL)
UNET_INIT_CHECKPOINT = os.getenv("ILCLDM_UNET_INIT_CHECKPOINT", "").strip()
REGISTRY_KEY = os.getenv("ILCLDM_REGISTRY_KEY", "").strip()
PARENT_EXPERIMENT = os.getenv("ILCLDM_PARENT_EXPERIMENT", "").strip()

# Stage-2 roi3ch_cls: classification loss on x0_hat
CLS_LOSS = os.getenv("ILCLDM_CLS_LOSS", "0") == "1"
CLS_LOSS_LAMBDA = float(os.getenv("ILCLDM_CLS_LOSS_LAMBDA", "0.1"))
EARLY_STOP_CLS_METRIC = os.getenv("ILCLDM_EARLY_STOP_CLS_METRIC", "0") == "1"

exp = os.getenv("ILCLDM_EXP", "exp_1/")
whole_Abeta = os.getenv("ILCLDM_WHOLE_ABETA", "./data/whole_Abeta")
latent_Abeta = os.getenv("ILCLDM_LATENT_ABETA", "./data/latent_Abeta/")
whole_MRI = os.getenv("ILCLDM_WHOLE_MRI", "./data/whole_MRI")
latent_MRI = os.getenv("ILCLDM_LATENT_MRI", "./data/latent_MRI/")
path = os.getenv("ILCLDM_REFERENCE_NIFTI", "./data/whole_MRI/3151.nii.gz")
gpus = [0]

# Stage-2: optional path to a frozen SPECT AAE (e.g. E1 checkpoint without copying into result/<exp>/)
CHECKPOINT_AAE = os.getenv("ILCLDM_AAE_CHECKPOINT", "result/" + exp + "AAE.pth.tar")
CHECKPOINT_Unet = "result/"+exp+"Unet.pth.tar"
CHECKPOINT_MRI_ENCODER = "result/"+exp+"MRI_Encoder.pth.tar"

RUN_STAGE1_MRI_PRETRAIN = os.getenv("ILCLDM_RUN_STAGE1_MRI_PRETRAIN", "0") == "1"
RUN_STAGE1_EMBED_EVAL = os.getenv("ILCLDM_RUN_STAGE1_EMBED_EVAL", "0") == "1"
RUN_STAGE2_DIFFUSION = os.getenv("ILCLDM_RUN_STAGE2_DIFFUSION", "1") == "1"
FREEZE_MRI_ENCODER = os.getenv("ILCLDM_FREEZE_MRI_ENCODER", "0") == "1"
MRI_ENCODER_MODE = os.getenv("ILCLDM_MRI_ENCODER_MODE", "").strip().lower()
MRI_ENCODER_CHECKPOINT = os.getenv("ILCLDM_MRI_ENCODER_CHECKPOINT", CHECKPOINT_MRI_ENCODER)
MRI_ENCODER_FINETUNE_LR_SCALE = float(os.getenv("ILCLDM_MRI_ENCODER_FINETUNE_LR_SCALE", "0.25"))

# Phase-1 MRI contrastive training (recon | supcon | xmodal)
MRI_TRAINING_MODE = os.getenv("ILCLDM_MRI_TRAINING_MODE", "recon").strip().lower()
SUPCON_LAMBDA = float(os.getenv("ILCLDM_SUPCON_LAMBDA", "20"))
XMODAL_LAMBDA = float(os.getenv("ILCLDM_XMODAL_LAMBDA", "20"))
CONTRASTIVE_TEMP = float(os.getenv("ILCLDM_CONTRASTIVE_TEMP", "0.5"))
CONTRASTIVE_PROJ_DIM = int(os.getenv("ILCLDM_CONTRASTIVE_PROJ_DIM", "128"))
CONTRASTIVE_FEATURES = os.getenv("ILCLDM_CONTRASTIVE_FEATURES", "z18k").strip().lower()
MRI_INIT_CHECKPOINT = os.getenv("ILCLDM_MRI_INIT_CHECKPOINT", "").strip()
SPECT_AAE_CHECKPOINT = os.getenv("ILCLDM_SPECT_AAE_CHECKPOINT", "").strip()

# Stage-2 gated latent feature space (default = classic bottleneck z18k).
# h64: diffuse 64-ch pre-bottleneck maps; h64_gap1: channel-mean → 1ch then diffuse.
DIFFUSION_FEATURE_SPACE = os.getenv("ILCLDM_DIFFUSION_FEATURE_SPACE", "z18k").strip().lower()
if DIFFUSION_FEATURE_SPACE in ("", "z", "z18", "latent", "bottleneck"):
    DIFFUSION_FEATURE_SPACE = "z18k"
if DIFFUSION_FEATURE_SPACE not in ("z18k", "h64", "h64_gap1"):
    raise ValueError(
        f"ILCLDM_DIFFUSION_FEATURE_SPACE must be z18k|h64|h64_gap1, got {DIFFUSION_FEATURE_SPACE!r}"
    )
DIFFUSION_CHANNELS = 64 if DIFFUSION_FEATURE_SPACE == "h64" else 1
UNET_IN_CHANNELS = 2 * DIFFUSION_CHANNELS + int(UNET_ROI_CHANNEL)
# Optional bonus decode path for h64 → Conv64→1 → SPECT decoder (off by default).
DIFFUSION_H64_DECODE_BONUS = os.getenv("ILCLDM_DIFFUSION_H64_DECODE_BONUS", "0") == "1"

train = os.getenv("ILCLDM_TRAIN_FILE", "data_info/train.txt")
validation = os.getenv("ILCLDM_VALIDATION_FILE", "data_info/validation.txt")
test = os.getenv("ILCLDM_TEST_FILE", "data_info/test.txt")
