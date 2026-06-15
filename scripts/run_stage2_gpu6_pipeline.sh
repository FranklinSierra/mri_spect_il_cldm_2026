#!/usr/bin/env bash
# Sequential stage-2 inference on GPU 6 (default).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export RUN_ID="${RUN_ID:-nm_stage2_v1}"
export RESUME="${RESUME:-1}"
export ILCLDM_AAE_CHECKPOINT="${ILCLDM_AAE_CHECKPOINT:-result/exp_nm_s1_spect/AAE.pth.tar}"

log() { echo "[$(date -Iseconds)] GPU=$CUDA_VISIBLE_DEVICES $*"; }

log "=== baseline (E3) ==="
python3 inference.py --exp exp_nm_s2_baseline --mri-encoder-mode raw \
  --run-id "$RUN_ID" --save-synthetic-spect --splits train test --resume \
  2>&1 | tee -a inference_baseline.log

log "=== frozen (E5) ==="
python3 inference.py --exp exp_nm_s2_mri_frozen --mri-encoder-mode frozen \
  --mri-encoder-checkpoint result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar \
  --run-id "$RUN_ID" --save-synthetic-spect --splits train test --resume \
  2>&1 | tee -a inference_frozen.log

FINETUNED="result/exp_nm_s2_mri_finetune/MRI_Encoder_finetuned.pth.tar"
if [[ ! -f "$FINETUNED" ]]; then
  log "ERROR: missing $FINETUNED" >&2
  exit 1
fi

log "=== finetune (E4) ==="
python3 inference.py --exp exp_nm_s2_mri_finetune --mri-encoder-mode finetune \
  --mri-encoder-checkpoint "$FINETUNED" \
  --run-id "$RUN_ID" --save-synthetic-spect --splits train test --resume \
  2>&1 | tee -a inference_finetune.log

RUN_ID="$RUN_ID" python3 scripts/summarize_stage2_inference.py --run-id "$RUN_ID"
log "Pipeline complete."
