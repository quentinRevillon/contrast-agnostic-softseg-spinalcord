#!/bin/bash
# Full pipeline: train + evaluate.
# Calls train_contrast_agnostic.sh (steps 1–5) then evaluate_csa_local.sh (CSA + Dice).
# All dataset/model configuration is defined in train_contrast_agnostic.sh.
#
# Usage:
#   bash run_all.sh

set -e

PATH_REPO="/home/quentinr/contrast-agnostic-softseg-spinalcord"

# ====================================
# CONFIGURATION (must match train_contrast_agnostic.sh)
# ====================================

PATH_DATA_BASE="/home/quentinr/datasets_contrast_agnostic_retraining"
PATH_NNUNET_RESULTS="/home/quentinr/nnunet-v2/nnUNet_results"
DATASET_NUMBER=2000
DATASET_NAME="ContrastAgnosticScCrop"
NNUNET_PLANS_FILE="nnUNetPlans"
MODEL_VERSION="sc-crop-v2"
EVAL_JOBS=4

PATH_MODEL="${PATH_NNUNET_RESULTS}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/nnUNetTrainer__${NNUNET_PLANS_FILE}__3d_fullres"
PATH_SPINE_GENERIC="${PATH_DATA_BASE}/data-multi-subject"


# ====================================
# TRAINING (steps 1–5, auto-resumes)
# ====================================

bash ${PATH_REPO}/scripts/train_contrast_agnostic.sh


# ====================================
# EVALUATION (CSA + Dice on spine-generic test set)
# ====================================

echo "-----------------------------------"
echo "Fetching disc labels from git-annex ..."
echo "-----------------------------------"
cd ${PATH_SPINE_GENERIC}
git annex get derivatives/labels/*/anat/*label-discs*.nii.gz
cd -

echo "-----------------------------------"
echo "Starting evaluation ..."
echo "Model   : ${PATH_MODEL}"
echo "Version : ${MODEL_VERSION}"
echo "Data    : ${PATH_SPINE_GENERIC}"
echo "-----------------------------------"

bash ${PATH_REPO}/scripts/evaluate_csa_local.sh \
    --model-folder  ${PATH_MODEL} \
    --model-version ${MODEL_VERSION} \
    --data-path     ${PATH_SPINE_GENERIC} \
    --jobs          ${EVAL_JOBS}
