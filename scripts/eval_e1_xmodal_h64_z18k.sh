#!/usr/bin/env bash
# Re-encode bottleneck z18k from an h64-trained MRI AAE and classify into embedding_analysis_z18k.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/roi_profile_env.sh
source "$ROOT/scripts/roi_profile_env.sh"

EXP="${EXP:-exp_nm_s1_mri_rois_xmodal_h64}"
CKPT="result/${EXP}/phase1_mri/checkpoints/MRI_AAE.pth.tar"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: missing checkpoint: $CKPT" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
echo "=== Eval z18k bottleneck of ${EXP} (no train) GPU=${GPU} ==="
python3 code/run_phase1_mri_experiment.py \
  --exp "$EXP" \
  --skip-train \
  --eval-features z18k \
  --contrastive-features z18k \
  --embeddings-name embeddings_z18k \
  --analysis-name embedding_analysis_z18k

echo "Done: result/${EXP}/phase1_mri/embedding_analysis_z18k/"
