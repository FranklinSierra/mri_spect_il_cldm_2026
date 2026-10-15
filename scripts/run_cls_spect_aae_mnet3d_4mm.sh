#!/usr/bin/env bash
# Mini-plan 2: MobileNetV2-3D on SPECT AAE reconstructions (ROI 4 mm). GPU 7 default.
# Does not touch E2, registry, or xmodal (GPU 5).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/roi_profile_env.sh
source "$ROOT/scripts/roi_profile_env.sh"

# Override 18 mm profile defaults (must assign, not ${VAR:-default}).
export ILCLDM_WHOLE_ABETA="data_rois/whole_Abeta_4mm"
export ILCLDM_IMAGE_VOLUME_NORM="legacy"
export ILCLDM_SKIP_CROP=1
export ILCLDM_ROI_VOLUME_SHAPE="96,112,114"

SPECT_ROOT="${SPECT_ROOT:-data_rois/whole_Abeta_4mm}"
AAE_CKPT="${AAE_CKPT:-result/exp_nm_s1_spect_rois_4mm/AAE.pth.tar}"
OUT_DIR="${OUT_DIR:-result/exp_nm_cls_spect_aae_mnet3d}"
# Mini-plan 2 is pinned to GPU 7; override only via CLS_GPU (ignore ambient CUDA_VISIBLE_DEVICES).
GPU="${CLS_GPU:-7}"
export CUDA_VISIBLE_DEVICES="$GPU"

BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-15}"

if [[ ! -d "$SPECT_ROOT" ]]; then
  echo "ERROR: missing SPECT 4 mm dir: $SPECT_ROOT" >&2
  exit 1
fi
if [[ "$SPECT_ROOT" != *"_4mm"* ]]; then
  echo "ERROR: refusing non-4mm SPECT path $SPECT_ROOT" >&2
  exit 1
fi
if [[ ! -f "$AAE_CKPT" ]]; then
  echo "ERROR: missing AAE checkpoint: $AAE_CKPT" >&2
  exit 1
fi
if [[ "$AAE_CKPT" != *"spect_rois_4mm"* ]]; then
  echo "ERROR: refusing AAE ckpt (expected spect_rois_4mm): $AAE_CKPT" >&2
  exit 1
fi

echo "=== MobileNetV2-3D SPECT recon 4 mm -> ${OUT_DIR} GPU=${GPU} SPECT=${SPECT_ROOT} AAE=${AAE_CKPT} ==="
python3 scripts/train_spect_mobilenet3d_aae.py \
  --spect-root "$SPECT_ROOT" \
  --aae-checkpoint "$AAE_CKPT" \
  --out-dir "$OUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE"

echo "Done: ${OUT_DIR}/best.pt"
