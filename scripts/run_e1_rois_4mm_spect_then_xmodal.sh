#!/usr/bin/env bash
# Sequential GPU 5 pipeline: E1 SPECT ROI 4 mm, then MRI xmodal h64.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

bash "$ROOT/scripts/run_e1_spect_rois_4mm.sh"
bash "$ROOT/scripts/run_e1_mri_rois_4mm_xmodal_h64.sh"
