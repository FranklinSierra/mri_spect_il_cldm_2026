#!/usr/bin/env bash
# E1 SPECT + MRI on data_rois (ROI patches, SKIP_CROP=1).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/roi_profile_env.sh
source "$ROOT/scripts/roi_profile_env.sh"

export ILCLDM_EXP="${ILCLDM_EXP:-exp_nm_s1_spect_rois/}"
export ILCLDM_RUN_STAGE1_MRI_PRETRAIN=0
export ILCLDM_RUN_STAGE1_EMBED_EVAL=0
export ILCLDM_RUN_STAGE2_DIFFUSION=0

echo "=== E1 SPECT (data_rois) -> ${ILCLDM_EXP} ==="
python3 main.py

echo "=== E1 MRI (data_rois) -> exp_nm_s1_mri_rois ==="
python3 code/run_phase1_mri_experiment.py --exp exp_nm_s1_mri_rois

echo "=== Latent stats (train split) ==="
python3 -c "
import config
from dataset import get_latent_stats
mean, std = get_latent_stats()
print(f'latent_stats: mean={mean:.6f} std={std:.6f} path={config.LATENT_STATS_PATH}')
"

echo "E1 ROI complete."
