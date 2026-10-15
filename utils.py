import torch.nn as nn
import numpy as np
import torch 
import config
import random
import os
import csv
import json
import subprocess
import datetime
from pathlib import Path

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    torch.save(checkpoint, filename)

def _normalize_checkpoint_state_dict(state_dict, model_state):
    checkpoint_has_module = next(iter(state_dict)).startswith("module.")
    model_has_module = next(iter(model_state)).startswith("module.")

    if checkpoint_has_module and not model_has_module:
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    elif model_has_module and not checkpoint_has_module:
        state_dict = {
            f"module.{key}": value
            for key, value in state_dict.items()
        }

    if any("cond.layers." in key for key in model_state):
        state_dict = {
            key.replace("cond.", "cond.layers.", 1)
            if key.startswith("cond.") and not key.startswith("cond.layers.")
            else key: value
            for key, value in state_dict.items()
        }
    elif any(key.startswith("cond.") and not key.startswith("cond.layers.") for key in model_state):
        state_dict = {
            key.replace("cond.layers.", "cond.", 1)
            if key.startswith("cond.layers.")
            else key: value
            for key, value in state_dict.items()
        }
    return state_dict


def _adapt_unet_input_conv(state_dict, model_state):
    """Expand or shrink first conv input channels when loading across UNet variants."""
    weight_key = "downs.0.weight"
    if weight_key not in state_dict or weight_key not in model_state:
        return state_dict

    ckpt_w = state_dict[weight_key]
    model_w = model_state[weight_key]
    if ckpt_w.shape == model_w.shape:
        return state_dict

    if (
        ckpt_w.ndim == 5
        and model_w.ndim == 5
        and ckpt_w.shape[0] == model_w.shape[0]
        and ckpt_w.shape[2:] == model_w.shape[2:]
        and ckpt_w.shape[1] != model_w.shape[1]
    ):
        new_w = model_w.clone()
        new_w.zero_()
        in_copy = min(ckpt_w.shape[1], model_w.shape[1])
        new_w[:, :in_copy] = ckpt_w[:, :in_copy]
        adapted = dict(state_dict)
        adapted[weight_key] = new_w
        return adapted

    adapted = dict(state_dict)
    adapted.pop(weight_key, None)
    return adapted


def load_checkpoint(checkpoint_file, model, optimizer, lr, model_name=None):
    label = f" ({model_name})" if model_name else ""
    path = Path(checkpoint_file).resolve()
    print(f"=> Loading checkpoint{label}: {path}")
    checkpoint = torch.load(path, map_location=config.device)
    state_dict = checkpoint["state_dict"]
    model_state = model.state_dict()
    state_dict = _normalize_checkpoint_state_dict(state_dict, model_state)
    model.load_state_dict(state_dict)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def last_csv_metric_row(path):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def resolve_aae_resume(checkpoint_path, result_dir, run_name):
    """If ILCLDM_AAE_RESUME=1, return (start_epoch, best_average, should_load).

    AAE.pth.tar stores generator weights + Adam state, not epoch or discriminator.
    start_epoch is the 0-indexed loop index of the next epoch to train, inferred
    from the last validation CSV Epoch (1-indexed, already completed).
    """
    if os.getenv("ILCLDM_AAE_RESUME", "0") != "1":
        return 0, 0.0, False

    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        print(f"=> ILCLDM_AAE_RESUME=1 but missing {ckpt}; training from scratch")
        return 0, 0.0, False

    result_dir = Path(result_dir)
    env_epoch = os.getenv("ILCLDM_AAE_RESUME_EPOCH", "").strip()
    val_row = last_csv_metric_row(result_dir / f"{run_name}_validation.csv")
    test_row = last_csv_metric_row(result_dir / f"{run_name}_test.csv")
    if env_epoch:
        last_completed = int(env_epoch)
    elif val_row and val_row.get("Epoch"):
        last_completed = int(float(val_row["Epoch"]))
    elif test_row and test_row.get("Epoch"):
        last_completed = int(float(test_row["Epoch"]))
    else:
        last_completed = 0

    average = 0.0
    if val_row and val_row.get("PSNR"):
        average = float(val_row["PSNR"]) + float(val_row.get("SSIM") or 0.0) * 10.0

    print(
        f"=> Resuming AAE from {ckpt} after completed epoch {last_completed} "
        f"(next logged epoch {last_completed + 1}, best_score={average:.4f}). "
        "Discriminator is reinitialized (not stored in AAE.pth.tar)."
    )
    return last_completed, average, True


def load_unet_init_checkpoint(model, checkpoint_file, optimizer=None, lr=None):
    """Load UNet weights for warm-start; adapts input conv when in_channel differs."""
    path = Path(checkpoint_file)
    if not path.is_file():
        print(f"=> UNet init checkpoint not found: {path}; training from scratch")
        return False

    print(f"=> Warm-start UNet from {path}")
    checkpoint = torch.load(path, map_location=config.device)
    state_dict = checkpoint["state_dict"]
    model_state = model.state_dict()
    state_dict = _normalize_checkpoint_state_dict(state_dict, model_state)
    state_dict = _adapt_unet_input_conv(state_dict, model_state)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"=> Init load missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"=> Init load unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    if optimizer is not None and "optimizer" in checkpoint and lr is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    return True


def cls_head_checkpoint_path(result_dir=None):
    if result_dir is None:
        result_dir = Path("result") / str(config.exp).strip("/")
    return Path(result_dir) / "ClsHead.pth.tar"


def save_cls_head_checkpoint(cls_head, filename=None):
    path = Path(filename) if filename else cls_head_checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": cls_head.state_dict()}, path)
    print(f"=> Saved cls head checkpoint to {path}")


def load_cls_head_checkpoint(cls_head, filename=None):
    path = Path(filename) if filename else cls_head_checkpoint_path()
    if not path.is_file():
        return False
    resolved = path.resolve()
    print(f"=> Loading checkpoint (ClsHead): {resolved}")
    checkpoint = torch.load(resolved, map_location=config.device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cls_head.load_state_dict(state_dict, strict=True)
    return True


RUN_MANIFEST_CONFIG_KEYS = (
    "MRI_ENCODER_MODE",
    "MRI_ENCODER_CHECKPOINT",
    "CHECKPOINT_AAE",
    "LATENT_LOSS",
    "LATENT_LOSS_LAMBDA",
    "DIFFUSION_OBJECTIVE",
    "LATENT_CALIBRATION",
    "LATENT_MEAN",
    "LATENT_STD",
    "LATENT_STATS_PATH",
    "IMAGE_VOLUME_NORM",
    "MINMAX_STATS_PATH",
    "KL_LOSS",
    "KL_LAMBDA",
    "PRED_X0_CLAMP",
    "HIGH_T_SAMPLE_PROB",
    "HIGH_T_THRESHOLD",
    "AUX_IMAGE_LOSS",
    "AUX_IMAGE_EVERY_N",
    "AUX_IMAGE_LAMBDA",
    "AUX_IMAGE_ALWAYS_HIGH_T",
    "USE_ROI_LOSS",
    "ROI_MASK_PATH",
    "ROI_WEIGHT_ALPHA",
    "EARLY_STOP_LATENT_METRIC",
    "USE_TRAIN_LABEL",
    "DIFFUSION_FEATURE_SPACE",
    "DIFFUSION_CHANNELS",
    "DIFFUSION_H64_DECODE_BONUS",
    "UNET_ROI_CHANNEL",
    "UNET_IN_CHANNELS",
    "UNET_INIT_CHECKPOINT",
    "REGISTRY_KEY",
    "PARENT_EXPERIMENT",
    "train",
    "test",
    "validation",
    "latent_Abeta",
    "CLS_LOSS",
    "CLS_LOSS_LAMBDA",
    "EARLY_STOP_CLS_METRIC",
    "EARLY_STOP_MIN_DELTA",
    "PRED_X0_CLAMP",
    "PRED_X0_CLAMP_LO",
    "PRED_X0_CLAMP_HI",
    "SAMPLING_T_MAX",
    "LABEL_DROPOUT_PROB",
    "EMA_BETA",
    "EMA_BETA_START",
    "EMA_WARMUP_STEPS",
    "EMA_STEP_START",
    "ATTN_RES",
    "USE_FILM",
    "FILM_MODE",
    "FILM_COND_CHANNELS",
    "USE_WANDB",
    "INCLUDE_SVM",
    "epochs",
    "learning_rate",
    "batch_size",
    "validation_interval",
    "early_stopping_patience",
    "noiseSteps",
    "sampling_steps",
    "ddim_eta",
)


def restore_diffusion_objective(env, cfg):
    """Copy ILCLDM_DIFFUSION_OBJECTIVE from a run manifest onto live config."""
    objective = None
    if env:
        objective = env.get("ILCLDM_DIFFUSION_OBJECTIVE")
    if objective is None and cfg and "DIFFUSION_OBJECTIVE" in cfg:
        objective = cfg["DIFFUSION_OBJECTIVE"]
    if objective is None or not str(objective).strip():
        return None
    mode = str(objective).strip().lower()
    if mode not in ("epsilon", "x0"):
        raise ValueError(
            f"DIFFUSION_OBJECTIVE in run_manifest must be epsilon|x0, got {mode!r}"
        )
    config.DIFFUSION_OBJECTIVE = mode
    return mode


def _manifest_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def restore_kl_image_flags(env, cfg):
    """Restore KL and minmax settings that change the AAE and the x0 clamp."""
    env = env or {}
    cfg = cfg or {}

    def pick(env_key, cfg_key):
        if env_key in env and env.get(env_key) not in (None, ""):
            return env.get(env_key)
        if cfg_key in cfg and cfg.get(cfg_key) not in (None, ""):
            return cfg.get(cfg_key)
        return None

    kl = pick("ILCLDM_KL_LOSS", "KL_LOSS")
    if kl is not None:
        config.KL_LOSS = _manifest_flag(kl)
    kl_lambda = pick("ILCLDM_KL_LAMBDA", "KL_LAMBDA")
    if kl_lambda is not None:
        config.KL_LAMBDA = float(kl_lambda)
    stats_path = pick("ILCLDM_MINMAX_STATS_PATH", "MINMAX_STATS_PATH")
    if stats_path is not None:
        config.MINMAX_STATS_PATH = str(stats_path)
    clamp = pick("ILCLDM_PRED_X0_CLAMP", "PRED_X0_CLAMP")
    if clamp is not None:
        config.PRED_X0_CLAMP = float(clamp)
        lo = pick("ILCLDM_PRED_X0_CLAMP_LO", "PRED_X0_CLAMP_LO")
        hi = pick("ILCLDM_PRED_X0_CLAMP_HI", "PRED_X0_CLAMP_HI")
        config.PRED_X0_CLAMP_LO = float(lo) if lo is not None else -config.PRED_X0_CLAMP
        config.PRED_X0_CLAMP_HI = float(hi) if hi is not None else config.PRED_X0_CLAMP


def dump_run_manifest(result_dir):
    """Write result/<exp>/run_manifest.json describing this training run."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = ""

    gpu_name = ""
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            gpu_name = ""

    manifest = {
        "exp": str(config.exp),
        "registry_key": config.REGISTRY_KEY,
        "parent_experiment": config.PARENT_EXPERIMENT,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "seed": os.environ.get("PYTHONHASHSEED", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_name": gpu_name,
        "device": config.device,
        "config": {key: getattr(config, key) for key in RUN_MANIFEST_CONFIG_KEYS},
        "env_overrides": {
            name: value for name, value in os.environ.items() if name.startswith("ILCLDM_")
        },
    }
    path = Path(result_dir) / "run_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"=> Saved run manifest to {path}")


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
