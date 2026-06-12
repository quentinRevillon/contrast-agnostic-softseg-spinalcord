#!/bin/bash
# ====================================
# QC report for the NO-CROP model (Dataset7001), mirroring run_qc_report.sh.
# Runs the no-crop nnUNet predictions ONCE on the test set, then builds a QC
# report comparing gt / seg_v3 (sct_deepseg) / seg_nocrop, plus a metrics.json
# with dice_v3 + dice_nocrop per subject.
# ====================================

set -e

# ====================================
# ONLY VARIABLES TO CHANGE
# ====================================
DATASET_NUMBER=7001
N_SUBJECTS=""   # leave empty for all subjects, set e.g. 10 for a quick test
CHECKPOINT=${CHECKPOINT:-checkpoint_best.pth}

# ====================================
# DERIVED PATHS (do not edit)
# ====================================
PATH_REPO="/home/quentinr/contrast-agnostic-softseg-spinalcord"
DATASET_NAME="ContrastAgnosticNoCrop"
PATH_NNUNET_RAW="/home/quentinr/nnunet-v2/nnUNet_raw"
PATH_NNUNET_RESULTS="/home/quentinr/nnunet-v2/nnUNet_results"

export nnUNet_raw=${PATH_NNUNET_RAW}
export nnUNet_preprocessed="/home/quentinr/nnunet-v2/nnUNet_preprocessed"
export nnUNet_results=${PATH_NNUNET_RESULTS}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export PATH="/home/quentinr/spinalcordtoolbox/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=0
export TORCHDYNAMO_DISABLE=1

DATASET_DIR="${PATH_NNUNET_RAW}/Dataset${DATASET_NUMBER}_${DATASET_NAME}"
MODEL_DIR="${PATH_NNUNET_RESULTS}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/nnUNetTrainer__nnUNetPlans__3d_fullres"
PATH_IMAGES_TS="${DATASET_DIR}/imagesTs"
PREDICTIONS_DIR="${MODEL_DIR}/predictions_test_fold0_${CHECKPOINT%.pth}"
OUTPUT_DIR="/home/quentinr/qc_results_dataset${DATASET_NUMBER}"

# ====================================
# STEP 1 — no-crop nnUNet predictions on the test set (skip if already done)
# ====================================
if [ ! -d "${PREDICTIONS_DIR}" ] || [ -z "$(ls ${PREDICTIONS_DIR}/*.nii.gz 2>/dev/null)" ]; then
    echo "Running no-crop nnUNet predictions -> ${PREDICTIONS_DIR}"
    nnUNetv2_predict \
        -d ${DATASET_NUMBER} \
        -i ${PATH_IMAGES_TS} \
        -o ${PREDICTIONS_DIR} \
        -f 0 \
        -tr nnUNetTrainer \
        -p nnUNetPlans \
        -c 3d_fullres \
        -chk ${CHECKPOINT}
else
    echo "Reusing existing predictions in ${PREDICTIONS_DIR}"
fi

# ====================================
# STEP 2 — QC report + metrics
# ====================================
python ${PATH_REPO}/scripts/generate_qc_report_nocrop.py \
    --dataset-dir     ${DATASET_DIR} \
    --predictions-dir ${PREDICTIONS_DIR} \
    --output-dir      ${OUTPUT_DIR} \
    ${N_SUBJECTS:+--n-subjects ${N_SUBJECTS}}

# ====================================
# STEP 3 — plots (no-crop variant: reads dice_nocrop / dice_v3)
# ====================================
python ${PATH_REPO}/scripts/plot_metrics_nocrop.py \
    --metrics    ${OUTPUT_DIR}/metrics.json \
    --output-dir ${OUTPUT_DIR}/plots

echo "Done. QC: ${OUTPUT_DIR}/qc/index.html   Metrics: ${OUTPUT_DIR}/metrics.json   Plots: ${OUTPUT_DIR}/plots"
