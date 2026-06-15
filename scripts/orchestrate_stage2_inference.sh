#!/usr/bin/env bash
# Wait for encoder recovery, then run frozen + finetune inference and summary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-nm_stage2_v1}"
GPU="${CUDA_VISIBLE_DEVICES:-6}"
export RESUME="${RESUME:-1}"
export ILCLDM_AAE_CHECKPOINT="${ILCLDM_AAE_CHECKPOINT:-result/exp_nm_s1_spect/AAE.pth.tar}"
PHASE1="result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar"
FINETUNED="result/exp_nm_s2_mri_finetune/MRI_Encoder_finetuned.pth.tar"
BASELINE_PID="${1:-}"

log() { echo "[$(date -Iseconds)] $*"; }

if [[ -n "$BASELINE_PID" ]]; then
  log "Waiting for baseline inference PID $BASELINE_PID..."
  while kill -0 "$BASELINE_PID" 2>/dev/null; do sleep 120; done
  log "Baseline inference finished."
fi

export CUDA_VISIBLE_DEVICES="$GPU"
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then RESUME_ARGS=(--resume); fi

log "Starting frozen inference on GPU $GPU..."
python3 inference.py --exp exp_nm_s2_mri_frozen --mri-encoder-mode frozen \
  --mri-encoder-checkpoint "$PHASE1" --run-id "$RUN_ID" \
  --save-synthetic-spect --splits train test "${RESUME_ARGS[@]}" \
  2>&1 | tee -a inference_frozen.log

log "Waiting for finetuned MRI encoder..."
while [[ ! -f "$FINETUNED" ]]; do
  if ! pgrep -f recover_finetune_mri_encoder.py >/dev/null; then
    if [[ ! -f "$FINETUNED" ]]; then
      echo "Recovery died without producing $FINETUNED" >&2
      exit 1
    fi
  fi
  sleep 60
done
log "Found $FINETUNED"

log "Starting finetune inference..."
python3 inference.py --exp exp_nm_s2_mri_finetune --mri-encoder-mode finetune \
  --mri-encoder-checkpoint "$FINETUNED" --run-id "$RUN_ID" \
  --save-synthetic-spect --splits train test "${RESUME_ARGS[@]}" \
  2>&1 | tee -a inference_finetune.log

RUN_ID="$RUN_ID" python3 scripts/summarize_stage2_inference.py --run-id "$RUN_ID"
log "All stage-2 inference complete."
