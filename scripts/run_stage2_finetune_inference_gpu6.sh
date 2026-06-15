#!/usr/bin/env bash
# E4 finetune inference + classification on GPU 6 (nohup-safe).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export RUN_ID="${RUN_ID:-nm_stage2_v1}"
export ILCLDM_AAE_CHECKPOINT="${ILCLDM_AAE_CHECKPOINT:-result/exp_nm_s1_spect/AAE.pth.tar}"

FINETUNED="result/exp_nm_s2_mri_finetune/MRI_Encoder_finetuned.pth.tar"
if [[ ! -f "$FINETUNED" ]]; then
  echo "[$(date -Iseconds)] ERROR: missing $FINETUNED" >&2
  exit 1
fi

log() { echo "[$(date -Iseconds)] GPU=$CUDA_VISIBLE_DEVICES $*"; }

log "=== finetune (E4) inference + classification ==="
python3 inference.py --exp exp_nm_s2_mri_finetune --mri-encoder-mode finetune \
  --mri-encoder-checkpoint "$FINETUNED" \
  --run-id "$RUN_ID" --save-synthetic-spect --splits train test --resume

log "Finetune inference complete."
