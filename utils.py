import torch.nn as nn
import numpy as np
import torch 
import config
import random
import os
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
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.device)
    state_dict = checkpoint["state_dict"]
    model_state = model.state_dict()
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

    model_state = model.state_dict()
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

    model.load_state_dict(state_dict)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

RUN_MANIFEST_CONFIG_KEYS = (
    "MRI_ENCODER_MODE",
    "MRI_ENCODER_CHECKPOINT",
    "CHECKPOINT_AAE",
    "LATENT_LOSS",
    "LATENT_LOSS_LAMBDA",
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
    "epochs",
    "learning_rate",
    "batch_size",
    "validation_interval",
    "early_stopping_patience",
    "noiseSteps",
    "sampling_steps",
    "ddim_eta",
)


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
