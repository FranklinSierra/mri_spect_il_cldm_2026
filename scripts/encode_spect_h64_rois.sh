#!/usr/bin/env bash
# Encode SPECT E1 ROI → data_rois/latent_Abeta_h64/ (+ stats). Does NOT train E2.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/roi_profile_env.sh
source "$ROOT/scripts/roi_profile_env.sh"

GPU="${CUDA_VISIBLE_DEVICES:-7}"
AAE="${SPECT_AAE_CKPT:-result/exp_nm_s1_spect_rois/AAE.pth.tar}"
export CUDA_VISIBLE_DEVICES="$GPU"

if [[ ! -f "$AAE" ]]; then
  echo "ERROR: missing SPECT AAE: $AAE" >&2
  exit 1
fi

echo "=== Encode SPECT h64 (GPU=${GPU}) → data_rois/latent_Abeta_h64/ ==="
python3 scripts/encode_spect_h64_rois.py --aae-checkpoint "$AAE"
echo "Done."
