#!/bin/bash
# ====================================
# ONLY VARIABLES TO CHANGE
# ====================================
DATASET_NUMBER=4000
N_SUBJECTS=""   # leave empty for all subjects, set e.g. 10 for a quick test

# ====================================
# DERIVED PATHS (do not edit)
# ====================================
PATH_REPO="/home/quentinr/contrast-agnostic-softseg-spinalcord"
DATASET_NAME="ContrastAgnosticScCrop"
PATH_NNUNET_RAW="/home/quentinr/nnunet-v2/nnUNet_raw"
PATH_NNUNET_RESULTS="/home/quentinr/nnunet-v2/nnUNet_results"
export PATH="/home/quentinr/spinalcordtoolbox/bin:${PATH}"

DATASET_DIR="${PATH_NNUNET_RAW}/Dataset${DATASET_NUMBER}_${DATASET_NAME}"
CHECKPOINT="${PATH_NNUNET_RESULTS}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
OUTPUT_DIR="/home/quentinr/qc_results_dataset${DATASET_NUMBER}"

export CUDA_VISIBLE_DEVICES=0

python ${PATH_REPO}/scripts/generate_qc_report.py \
    --dataset-dir ${DATASET_DIR} \
    --checkpoint  ${CHECKPOINT} \
    --output-dir  ${OUTPUT_DIR} \
    ${N_SUBJECTS:+--n-subjects ${N_SUBJECTS}}

python ${PATH_REPO}/scripts/plot_metrics.py \
    --metrics    ${OUTPUT_DIR}/metrics.json \
    --output-dir ${OUTPUT_DIR}/plots
