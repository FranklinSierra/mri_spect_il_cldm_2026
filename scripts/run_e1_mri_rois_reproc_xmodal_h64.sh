#!/usr/bin/env bash
# E1 MRI xmodal InfoNCE on pre-bottleneck flatten h64, ROI reproc 96×112×114.
# Isolated from exp_nm_s1_mri_rois_xmodal_h64 (105³) and from all E2 runs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/roi_profile_env.sh
source "$ROOT/scripts/roi_profile_env.sh"

export ILCLDM_IMAGE_VOLUME_NORM="${ILCLDM_IMAGE_VOLUME_NORM:-legacy}"

INIT_CKPT="${INIT_CKPT:-result/exp_nm_s1_mri_rois_reproc/phase1_mri/checkpoints/MRI_AAE.pth.tar}"
SPECT_CKPT="${SPECT_CKPT:-result/exp_nm_s1_spect_rois_reproc/AAE.pth.tar}"
EXP="${EXP:-exp_nm_s1_mri_rois_reproc_xmodal_h64}"
GPU="${CUDA_VISIBLE_DEVICES:-5}"

for f in "$INIT_CKPT" "$SPECT_CKPT"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing prerequisite: $f" >&2
    exit 1
  fi
done

BATCH_SIZE="${BATCH_SIZE:-8}"
XMODAL_LAMBDA="${XMODAL_LAMBDA:-20}"
CONTRASTIVE_TEMP="${CONTRASTIVE_TEMP:-0.5}"
export CUDA_VISIBLE_DEVICES="$GPU"
echo "=== E1 MRI xmodal h64 reproc 96³ -> ${EXP} GPU=${GPU} batch=${BATCH_SIZE} lambda=${XMODAL_LAMBDA} tau=${CONTRASTIVE_TEMP} ==="
python3 code/run_phase1_mri_experiment.py \
  --exp "$EXP" \
  --training-mode xmodal \
  --contrastive-features h64 \
  --batch-size "$BATCH_SIZE" \
  --xmodal-lambda "$XMODAL_LAMBDA" \
  --contrastive-temp "$CONTRASTIVE_TEMP" \
  --spect-checkpoint "$SPECT_CKPT" \
  --init-checkpoint "$INIT_CKPT" \
  --baseline-exp exp_nm_s1_mri_rois_reproc

echo "Done: result/${EXP}/phase1_mri/"
