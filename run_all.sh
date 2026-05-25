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
EVAL_JOBS=1

PATH_MODEL="${PATH_NNUNET_RESULTS}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/nnUNetTrainer__${NNUNET_PLANS_FILE}__3d_fullres"
PATH_SPINE_GENERIC="${PATH_DATA_BASE}/data-multi-subject"


# ====================================
# TRAINING (steps 1–5, auto-resumes, skipped if already complete)
# ====================================

CHECKPOINT="${PATH_MODEL}/fold_0/checkpoint_final.pth"
if [ -f "${CHECKPOINT}" ]; then
    echo "Training already complete (checkpoint_final.pth found), skipping."
else
    bash ${PATH_REPO}/scripts/train_contrast_agnostic.sh
fi


# ====================================
# ONNX EXPORT (skipped if already done)
# ====================================

PATH_ONNX_DIR="${PATH_MODEL}/onnx"
PATH_ONNX="${PATH_ONNX_DIR}/nnunet_seg.onnx"
PATH_PLANS_ONNX="${PATH_ONNX_DIR}/plans.json"

if [ -f "${PATH_ONNX}" ]; then
    echo "ONNX model already exists, skipping export."
else
    echo "-----------------------------------"
    echo "Exporting ONNX model ..."
    echo "-----------------------------------"
    conda run -n contrast_agnostic python ${PATH_REPO}/nnUnet/export_nnunet_to_onnx.py \
        --model-folder ${PATH_MODEL} \
        --output       ${PATH_ONNX}
    cp ${PATH_MODEL}/plans.json ${PATH_PLANS_ONNX}
fi


# ====================================
# EVALUATION (CSA + Dice on spine-generic test set)
# ====================================

echo "-----------------------------------"
echo "Fetching disc labels from git-annex ..."
echo "-----------------------------------"
cd ${PATH_SPINE_GENERIC}
git annex get derivatives/labels/*/*/*.nii.gz
cd -

echo "-----------------------------------"
echo "Starting evaluation ..."
echo "Model   : ${PATH_ONNX}"
echo "Version : ${MODEL_VERSION}"
echo "Data    : ${PATH_SPINE_GENERIC}"
echo "-----------------------------------"

bash ${PATH_REPO}/scripts/evaluate_csa_local.sh \
    --onnx          ${PATH_ONNX} \
    --plans         ${PATH_PLANS_ONNX} \
    --model-version ${MODEL_VERSION} \
    --data-path     ${PATH_SPINE_GENERIC} \
    --jobs          ${EVAL_JOBS}
