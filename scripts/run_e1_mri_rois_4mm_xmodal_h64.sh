#!/usr/bin/env bash
# E1 MRI xmodal InfoNCE h64 on ROI 96³ with SPECT 4 mm frozen.
# Requires result/exp_nm_s1_spect_rois_4mm/AAE.pth.tar (run scripts/run_e1_spect_rois_4mm.sh first).
# Does not resume exp_nm_s1_mri_rois_reproc_xmodal_h64 (18 mm, aborted).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/roi_profile_env.sh
source "$ROOT/scripts/roi_profile_env.sh"

# Override ROI profile 18 mm SPECT path (must assign, not ${VAR:-default}).
export ILCLDM_WHOLE_ABETA="data_rois/whole_Abeta_4mm"
export ILCLDM_IMAGE_VOLUME_NORM="legacy"

INIT_CKPT="${INIT_CKPT:-result/exp_nm_s1_mri_rois_reproc/phase1_mri/checkpoints/MRI_AAE.pth.tar}"
SPECT_CKPT="${SPECT_CKPT:-result/exp_nm_s1_spect_rois_4mm/AAE.pth.tar}"
EXP="${EXP:-exp_nm_s1_mri_rois_4mm_xmodal_h64}"
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
echo "=== E1 MRI xmodal h64 ROI 4 mm -> ${EXP} GPU=${GPU} SPECT=${ILCLDM_WHOLE_ABETA} ckpt=${SPECT_CKPT} ==="
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
