#!/usr/bin/env bash
# Run stage-2 inference (train+test) for baseline, frozen, and finetune experiments.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-nm_stage2_v1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export ILCLDM_AAE_CHECKPOINT="${ILCLDM_AAE_CHECKPOINT:-result/exp_nm_s1_spect/AAE.pth.tar}"
RESUME_FLAG=()
if [[ "${RESUME:-1}" == "1" ]]; then
  RESUME_FLAG=(--resume)
fi
PHASE1_MRI="${ILCLDM_MRI_ENCODER_CHECKPOINT:-result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar}"

run_inference() {
  local exp="$1"
  local mode="$2"
  local mri_ckpt="${3:-}"

  echo "=== inference exp=$exp mode=$mode ==="
  local -a cmd=(
    python3 inference.py
    --exp "$exp"
    --mri-encoder-mode "$mode"
    --run-id "$RUN_ID"
    --save-synthetic-spect
    --splits train test
    "${RESUME_FLAG[@]}"
  )
  if [[ -n "$mri_ckpt" ]]; then
    cmd+=(--mri-encoder-checkpoint "$mri_ckpt")
  fi
  "${cmd[@]}"
}

run_inference exp_nm_s2_baseline raw

export ILCLDM_MRI_ENCODER_CHECKPOINT="$PHASE1_MRI"
run_inference exp_nm_s2_mri_frozen frozen "$PHASE1_MRI"

FINETUNED="result/exp_nm_s2_mri_finetune/MRI_Encoder_finetuned.pth.tar"
if [[ ! -f "$FINETUNED" ]]; then
  echo "ERROR: Missing $FINETUNED — re-train E4 first (scripts/run_experiment.sh s2_mri_finetune)." >&2
  exit 1
fi
run_inference exp_nm_s2_mri_finetune finetune "$FINETUNED"

python3 scripts/summarize_stage2_inference.py --run-id "$RUN_ID"
echo "Done. Summary in result/stage2_inference_summary/"
