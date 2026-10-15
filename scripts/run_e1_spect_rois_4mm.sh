#!/usr/bin/env bash
# E1 SPECT AAE on ROI 4 mm (data_rois/whole_Abeta_4mm). Does not train MRI.
# Isolated from 18 mm (whole_Abeta / spect_rois_reproc / latent_Abeta_reproc) and from E2.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/roi_profile_env.sh
source "$ROOT/scripts/roi_profile_env.sh"

# Override ROI profile 18 mm defaults (must assign, not ${VAR:-default}).
export ILCLDM_WHOLE_ABETA="data_rois/whole_Abeta_4mm"
export ILCLDM_LATENT_ABETA="data_rois/latent_Abeta_4mm/"
export ILCLDM_LATENT_STATS_PATH="data_info/latent_stats_latent_Abeta_rois_4mm.json"
export ILCLDM_IMAGE_VOLUME_NORM="legacy"
export ILCLDM_EXP="exp_nm_s1_spect_rois_4mm/"
export ILCLDM_RUN_STAGE1_MRI_PRETRAIN=0
export ILCLDM_RUN_STAGE1_EMBED_EVAL=0
export ILCLDM_RUN_STAGE2_DIFFUSION=0
# Warm-start generator + Adam from AAE.pth.tar; epoch inferred from validation CSV.
# Discriminator is not in the checkpoint and is reinitialized.
export ILCLDM_AAE_RESUME="${ILCLDM_AAE_RESUME:-1}"

GPU="${CUDA_VISIBLE_DEVICES:-5}"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "${ILCLDM_LATENT_ABETA}"

if [[ ! -d "$ILCLDM_WHOLE_ABETA" ]]; then
  echo "ERROR: missing SPECT 4 mm dir: $ILCLDM_WHOLE_ABETA" >&2
  exit 1
fi
if [[ "$ILCLDM_WHOLE_ABETA" == *"/whole_Abeta" && "$ILCLDM_WHOLE_ABETA" != *"_4mm"* ]]; then
  echo "ERROR: refusing 18 mm path $ILCLDM_WHOLE_ABETA" >&2
  exit 1
fi

echo "=== E1 SPECT ROI 4 mm -> ${ILCLDM_EXP} GPU=${GPU} SPECT=${ILCLDM_WHOLE_ABETA} latents=${ILCLDM_LATENT_ABETA} ==="
python3 main.py

echo "=== Latent stats (train split) ==="
python3 -c "
import config
from dataset import get_latent_stats
mean, std = get_latent_stats()
print(f'latent_stats: mean={mean:.6f} std={std:.6f} path={config.LATENT_STATS_PATH}')
"

echo "Done: result/${ILCLDM_EXP}AAE.pth.tar"
