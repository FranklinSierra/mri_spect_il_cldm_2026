#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export ILCLDM_EPOCHS="${ILCLDM_EPOCHS:-1000}"
export ILCLDM_NOISE_STEPS="${ILCLDM_NOISE_STEPS:-1000}"

CODEX_SITE_PACKAGES="/home/phds_1/.local/lib/python3.12/site-packages"
if ! python3 -c "import torch" >/dev/null 2>&1 && [ -d "$CODEX_SITE_PACKAGES" ]; then
  export PYTHONPATH="${CODEX_SITE_PACKAGES}:${PYTHONPATH:-}"
fi

python3 prepare_fsierra_data.py
python3 main.py
