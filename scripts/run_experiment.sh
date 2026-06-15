#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_experiment.sh <mode> [exp_name]

Modes:
  s1_spect         Train stage-1 SPECT AAE and encode latent_Abeta
  s1_mri           Run phase-1 MRI autoencoder pipeline script
  s2_baseline      Train diffusion with raw MRI conditioner (no pretrained MRI encoder)
  s2_mri_frozen    Train diffusion with pretrained MRI encoder frozen
  s2_mri_frozen_v2 Stage 2 frozen v2 (latent loss, high-t sampling, ROI aux, latent early stop)
  s2_mri_finetune  Train diffusion with pretrained MRI encoder trainable
  abl              Run a stage-2 ablation from experiments/registry.json (arg: registry key)
  preflight        Run preflight checks + smoke test

Examples:
  scripts/run_experiment.sh s1_spect exp_nm_s1_spect
  scripts/run_experiment.sh s1_mri exp_nm_s1_mri
  scripts/run_experiment.sh s2_mri_frozen exp_nm_s2_mri_frozen
  scripts/run_experiment.sh s2_mri_frozen_v2 exp_nm_s2_mri_frozen_v2
  scripts/run_experiment.sh abl abl_latloss01
EOF
}

mode="${1:-}"
exp_name="${2:-}"

if [[ -z "$mode" ]]; then
  usage
  exit 1
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

run_preflight() {
  python3 scripts/preflight_data.py
  python3 smoke_test_fsierra.py
}

case "$mode" in
  preflight)
    run_preflight
    ;;

  s1_spect)
    : "${exp_name:=exp_nm_s1_spect}"
    export ILCLDM_EXP="${exp_name}/"
    export ILCLDM_RUN_STAGE1_MRI_PRETRAIN=0
    export ILCLDM_RUN_STAGE1_EMBED_EVAL=0
    export ILCLDM_RUN_STAGE2_DIFFUSION=0
    python3 main.py
    ;;

  s1_mri)
    : "${exp_name:=exp_nm_s1_mri}"
    python3 code/run_phase1_mri_experiment.py --exp "$exp_name"
    ;;

  s2_baseline)
    : "${exp_name:=exp_nm_s2_baseline}"
    export ILCLDM_EXP="${exp_name}/"
    export ILCLDM_MRI_ENCODER_MODE=raw
    python3 -c "from utils import seed_torch; from main import train_LDM; seed_torch(); train_LDM()"
    ;;

  s2_mri_frozen)
    : "${exp_name:=exp_nm_s2_mri_frozen}"
    export ILCLDM_EXP="${exp_name}/"
    export ILCLDM_MRI_ENCODER_MODE=frozen
    export ILCLDM_MRI_ENCODER_CHECKPOINT="${ILCLDM_MRI_ENCODER_CHECKPOINT:-result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar}"
    python3 -c "from utils import seed_torch; from main import train_LDM; seed_torch(); train_LDM()"
    ;;

  s2_mri_frozen_v2)
    : "${exp_name:=exp_nm_s2_mri_frozen_v2}"
    export ILCLDM_EXP="${exp_name}/"
    export ILCLDM_MRI_ENCODER_MODE=frozen
    export ILCLDM_MRI_ENCODER_CHECKPOINT="${ILCLDM_MRI_ENCODER_CHECKPOINT:-result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar}"
    export ILCLDM_AAE_CHECKPOINT="${ILCLDM_AAE_CHECKPOINT:-result/exp_nm_s1_spect/AAE.pth.tar}"
    export ILCLDM_LATENT_LOSS=1
    export ILCLDM_LATENT_LOSS_LAMBDA=0.5
    export ILCLDM_AUX_IMAGE_EVERY_N=1
    export ILCLDM_AUX_IMAGE_LAMBDA=0.5
    export ILCLDM_AUX_IMAGE_ALWAYS_HIGH_T=1
    export ILCLDM_HIGH_T_SAMPLE_PROB=0.5
    export ILCLDM_HIGH_T_THRESHOLD=700
    export ILCLDM_ROI_MASK_PATH="${ILCLDM_ROI_MASK_PATH:-data/mask_basal_ganglia.nii.gz}"
    export ILCLDM_ROI_WEIGHT_ALPHA=3.0
    export ILCLDM_USE_ROI_LOSS=1
    export ILCLDM_EARLY_STOP_LATENT_METRIC=1
    python3 -c "from utils import seed_torch; from main import train_LDM; seed_torch(); train_LDM()"
    ;;

  abl)
    : "${exp_name:?usage: scripts/run_experiment.sh abl <registry_key>}"
    python3 scripts/run_ablation.py "$exp_name"
    ;;

  s2_mri_finetune)
    : "${exp_name:=exp_nm_s2_mri_finetune}"
    export ILCLDM_EXP="${exp_name}/"
    export ILCLDM_MRI_ENCODER_MODE=finetune
    export ILCLDM_MRI_ENCODER_CHECKPOINT="${ILCLDM_MRI_ENCODER_CHECKPOINT:-result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar}"
    python3 -c "from utils import seed_torch; from main import train_LDM; seed_torch(); train_LDM()"
    ;;

  *)
    echo "Unknown mode: $mode" >&2
    usage
    exit 1
    ;;
esac
