#!/usr/bin/env bash
# E1 MRI SupCon on pre-bottleneck flatten h64 (warm-start from mri_rois baseline).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/roi_profile_env.sh
source "$ROOT/scripts/roi_profile_env.sh"

INIT_CKPT="${INIT_CKPT:-result/exp_nm_s1_mri_rois/phase1_mri/checkpoints/MRI_AAE.pth.tar}"
EXP="${EXP:-exp_nm_s1_mri_rois_supcon_h64}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "$INIT_CKPT" ]]; then
  echo "ERROR: missing warm-start checkpoint: $INIT_CKPT" >&2
  echo "Run: bash scripts/run_e1_rois.sh" >&2
  exit 1
fi

BATCH_SIZE="${BATCH_SIZE:-8}"
SUPCON_LAMBDA="${SUPCON_LAMBDA:-20}"
CONTRASTIVE_TEMP="${CONTRASTIVE_TEMP:-0.5}"
export CUDA_VISIBLE_DEVICES="$GPU"
echo "=== E1 MRI SupCon h64 -> ${EXP} GPU=${GPU} batch=${BATCH_SIZE} lambda=${SUPCON_LAMBDA} tau=${CONTRASTIVE_TEMP} ==="
python3 code/run_phase1_mri_experiment.py \
  --exp "$EXP" \
  --training-mode supcon \
  --contrastive-features h64 \
  --batch-size "$BATCH_SIZE" \
  --supcon-lambda "$SUPCON_LAMBDA" \
  --contrastive-temp "$CONTRASTIVE_TEMP" \
  --init-checkpoint "$INIT_CKPT"

echo "Done: result/${EXP}/phase1_mri/"
