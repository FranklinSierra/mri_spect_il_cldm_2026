#!/usr/bin/env bash
# Source ROI data_rois env for E1 / E2 (does not run training).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ILCLDM_SKIP_CROP=1
export ILCLDM_WHOLE_MRI="${ILCLDM_WHOLE_MRI:-data_rois/whole_MRI}"
export ILCLDM_WHOLE_ABETA="${ILCLDM_WHOLE_ABETA:-data_rois/whole_Abeta}"
export ILCLDM_LATENT_ABETA="${ILCLDM_LATENT_ABETA:-data_rois/latent_Abeta/}"
export ILCLDM_LATENT_MRI="${ILCLDM_LATENT_MRI:-data_rois/latent_MRI/}"
export ILCLDM_ROI_VOLUME_SHAPE="${ILCLDM_ROI_VOLUME_SHAPE:-105,102,102}"
export ILCLDM_REFERENCE_NIFTI="${ILCLDM_REFERENCE_NIFTI:-data_rois/whole_MRI/3102.nii.gz}"
export ILCLDM_ROI_MASK_PATH="${ILCLDM_ROI_MASK_PATH:-data_rois/roi_mask_unet.nii.gz}"
export ILCLDM_LATENT_STATS_PATH="${ILCLDM_LATENT_STATS_PATH:-data_info/latent_stats_latent_Abeta_rois.json}"
