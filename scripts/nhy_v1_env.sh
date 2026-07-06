#!/usr/bin/env bash
# Source before E1 SPECT, E1 MRI, E3 NHY stage-2 ablations, or inference on the NHY split.
# Usage: source scripts/nhy_v1_env.sh

export ILCLDM_TRAIN_FILE=data_info/splits/nhy_v1/train.txt
export ILCLDM_TEST_FILE=data_info/splits/nhy_v1/test.txt
export ILCLDM_VALIDATION_FILE=data_info/splits/nhy_v1/validation.txt
export ILCLDM_LATENT_ABETA=data/latent_Abeta_nhy_v1/
