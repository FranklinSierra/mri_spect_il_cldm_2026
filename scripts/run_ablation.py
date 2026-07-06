#!/usr/bin/env python3
"""Launch a stage-2 ablation experiment defined in experiments/registry.json.

Resolves the experiment's base flags + overrides from the registry, injects
them as ILCLDM_* environment variables, then runs train_LDM(). The base code
is never modified: each ablation only flips the env-driven flags read by
config.py at import time.

Usage:
    python3 scripts/run_ablation.py <registry_key> [--gpu <idx>]

Examples:
    python3 scripts/run_ablation.py abl_latloss01 --gpu 1
    python3 scripts/run_ablation.py abl_hight05 --gpu 3
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "experiments" / "registry.json"


def load_registry():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def resolve_env(registry, key):
    experiments = registry["experiments"]
    if key not in experiments:
        available = ", ".join(sorted(experiments))
        raise SystemExit(f"Unknown experiment key {key!r}. Available: {available}")
    entry = experiments[key]

    env = {}
    base_key = entry.get("base")
    if base_key:
        if base_key not in registry:
            raise SystemExit(f"Experiment {key!r} references unknown base {base_key!r}")
        env.update(registry[base_key])
    env.update(entry.get("overrides", {}))
    env["ILCLDM_EXP"] = entry["exp"].rstrip("/") + "/"
    if entry.get("parent"):
        env["ILCLDM_PARENT_EXPERIMENT"] = entry["parent"]
    if entry.get("init_checkpoint"):
        env["ILCLDM_UNET_INIT_CHECKPOINT"] = entry["init_checkpoint"]
    env["ILCLDM_REGISTRY_KEY"] = key
    return entry, env


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("key", help="Experiment key from experiments/registry.json")
    parser.add_argument("--gpu", default=None, help="GPU index for CUDA_VISIBLE_DEVICES (e.g. 0, 1, ...)")
    args = parser.parse_args()

    # Must happen before importing torch (via main/utils) so device mapping applies.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    registry = load_registry()
    entry, env = resolve_env(registry, args.key)
    for name, value in env.items():
        os.environ[name] = str(value)

    print(f"[run_ablation] key={args.key} exp={entry['exp']} run_id={entry['run_id']}")
    print(f"[run_ablation] label={entry['label']} early_stop={entry['early_stop']}")
    if entry.get("split_profile"):
        print(f"[run_ablation] split_profile={entry['split_profile']}")
    if entry.get("prerequisite"):
        print(f"[run_ablation] prerequisite: {entry['prerequisite']}")
    if args.gpu is not None:
        print(f"[run_ablation] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    for name in sorted(env):
        print(f"[run_ablation]   {name}={env[name]}")

    # Relative paths in config.py (data/, result/) assume the repo root as cwd.
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))

    from utils import seed_torch
    from main import train_LDM

    seed_torch()
    train_LDM()


if __name__ == "__main__":
    main()
