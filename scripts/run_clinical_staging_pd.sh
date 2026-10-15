#!/usr/bin/env bash
# PD staging pipeline: text embeddings → KNN baselines → CLIP (GPU 7, nohup-safe).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU="${CUDA_VISIBLE_DEVICES:-7}"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p logs

log() { echo "[$(date -Iseconds)] GPU=$CUDA_VISIBLE_DEVICES $*"; }

log "Step 1/3: ClinicalBERT text embeddings"
python3 scripts/extract_clinical_text_embeddings.py \
  --out-root result/clinical_staging_pd \
  --gpu "$GPU" \
  > logs/clinical_staging_text_embeddings.log 2>&1

log "Step 2/3: KNN baselines (image + text)"
python3 scripts/classify_nhy_staging_pd.py \
  --out-root result/clinical_staging_pd \
  > logs/clinical_staging_baselines.log 2>&1

log "Step 3/3: CLIP v2 training + staging/fusion eval"
python3 scripts/train_clinical_radiology_clip.py \
  --out-root result/clinical_staging_pd \
  --gpu "$GPU" \
  --lambda-infonce 0.5 \
  --lambda-supcon 1.0 \
  --seed 42 \
  > logs/clinical_staging_clip_v2.log 2>&1

log "Done. Outputs under result/clinical_staging_pd/"
