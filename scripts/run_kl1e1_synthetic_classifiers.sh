#!/usr/bin/env bash
# Train the six synthetic-SPECT MobileNet classifiers for kl1e1 (sequential, GPU 0).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SYN="$ROOT/result/exp_nm_s2_frozen_calib_roi3ch_uni_rois_reproc_x0_minmax_kl1e1/inference/calib_roi3ch_uni_rois_reproc_x0_minmax_kl1e1/synthetic_spect"
CLS="$ROOT/result/exp_nm_s2_frozen_calib_roi3ch_uni_rois_reproc_x0_minmax_kl1e1/inference/calib_roi3ch_uni_rois_reproc_x0_minmax_kl1e1/classification"
META="$ROOT/data_info/data_info.csv"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR" "$CLS"

TRAIN_DIR="$SYN/train"
TEST_DIR="$SYN/test"

if [[ ! -d "$TRAIN_DIR" || ! -d "$TEST_DIR" ]]; then
  echo "ERROR: missing synthetic SPECT dirs under $SYN" >&2
  exit 1
fi

n_train=$(find "$TRAIN_DIR" -maxdepth 1 \( -name '*.nii.gz' -o -name '*.nii' \) | wc -l)
n_test=$(find "$TEST_DIR" -maxdepth 1 \( -name '*.nii.gz' -o -name '*.nii' \) | wc -l)
echo "=== kl1e1 synthetic classifiers ==="
echo "GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "train=$n_train test=$n_test"
echo "out=$CLS"

run_one() {
  local name="$1"
  shift
  local log="$LOG_DIR/cls_kl1e1_${name}.log"
  echo
  echo ">>> START $name  $(date -Is)"
  echo ">>> log $log"
  # shellcheck disable=SC2068
  python3 "$@" 2>&1 | tee "$log"
  echo ">>> DONE  $name  $(date -Is)"
}

# 1) 2D ImageNet
run_one fromImageNetModel \
  scripts/train_synthetic_spect_mobilenet2d.py \
  --train-dir "$TRAIN_DIR" \
  --test-dir "$TEST_DIR" \
  --out-dir "$CLS/fromImageNetModel" \
  --meta-csv "$META" \
  --z-min 68 --z-max 87 \
  --epochs 60 --batch-size 8 --lr 0.0001 --weight-decay 0.0001 \
  --patience 15 --val-fraction 0.15 --seed 42 \
  --num-workers 4 --image-size 224

# 2) 2D finetune from real SPECT
run_one fromPretrainedSpect \
  scripts/train_synthetic_spect_mobilenet2d.py \
  --train-dir "$TRAIN_DIR" \
  --test-dir "$TEST_DIR" \
  --out-dir "$CLS/fromPretrainedSpect" \
  --meta-csv "$META" \
  --z-min 68 --z-max 87 \
  --epochs 60 --batch-size 8 --lr 0.0001 --weight-decay 0.0001 \
  --patience 15 --val-fraction 0.15 --seed 42 \
  --num-workers 4 --image-size 224 \
  --init-checkpoint "$ROOT/data_rois/classifiers/SPECT/2D/best_mobilenet_v2.pt"

# 3) 2.5D ImageNet
run_one fromImageNetModel2.5D \
  scripts/train_synthetic_spect_mobilenet3d.py \
  --train-dir "$TRAIN_DIR" \
  --test-dir "$TEST_DIR" \
  --out-dir "$CLS/fromImageNetModel2.5D" \
  --meta-csv "$META" \
  --val-fraction 0.15 --epochs 100 --batch-size 2 \
  --lr 0.0001 --weight-decay 0.0001 --patience 15 --seed 42 \
  --num-workers 0 --z-min 68 --z-max 87

# 4) 2.5D finetune from real SPECT
run_one fromPretrainedSpect2.5D \
  scripts/train_synthetic_spect_mobilenet3d.py \
  --train-dir "$TRAIN_DIR" \
  --test-dir "$TEST_DIR" \
  --out-dir "$CLS/fromPretrainedSpect2.5D" \
  --meta-csv "$META" \
  --val-fraction 0.15 --epochs 100 --batch-size 2 \
  --lr 0.0001 --weight-decay 0.0001 --patience 15 --seed 42 \
  --num-workers 0 --z-min 68 --z-max 87 \
  --init-checkpoint "$ROOT/data_rois/classifiers/SPECT/2.5D/best.pt"

# 5) 3D ImageNet
run_one fromImageNetModel3D \
  scripts/train_synthetic_spect_mobilenet3d.py \
  --train-dir "$TRAIN_DIR" \
  --test-dir "$TEST_DIR" \
  --out-dir "$CLS/fromImageNetModel3D" \
  --meta-csv "$META" \
  --val-fraction 0.15 --epochs 100 --batch-size 2 \
  --lr 0.0001 --weight-decay 0.0001 --patience 15 --seed 42 \
  --num-workers 0

# 6) 3D finetune from real SPECT
run_one fromPretrainedSpect3D \
  scripts/train_synthetic_spect_mobilenet3d.py \
  --train-dir "$TRAIN_DIR" \
  --test-dir "$TEST_DIR" \
  --out-dir "$CLS/fromPretrainedSpect3D" \
  --meta-csv "$META" \
  --val-fraction 0.15 --epochs 100 --batch-size 2 \
  --lr 0.0001 --weight-decay 0.0001 --patience 15 --seed 42 \
  --num-workers 0 \
  --init-checkpoint "$ROOT/data_rois/classifiers/SPECT/3D/best.pt"

echo
echo "=== ALL SIX RUNS COMPLETE $(date -Is) ==="
