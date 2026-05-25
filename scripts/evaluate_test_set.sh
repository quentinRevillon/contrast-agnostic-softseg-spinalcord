#!/bin/bash
# Run nnUNet inference + Dice evaluation on the test set.
#
# Uses the standard nnUNet pipeline:
#   1. nnUNetv2_predict  — inference on imagesTs/ (already sc_crop-preprocessed)
#   2. nnUNetv2_evaluate_folder — Dice vs labelsTs/
#
# Skipped if test_predictions/summary.json already exists.
#
# Usage (standalone):
#   conda activate contrast_agnostic
#   bash scripts/evaluate_test_set.sh \
#       --dataset-number 2000 \
#       --dataset-name   ContrastAgnosticScCrop \
#       --results-dir    ~/nnunet-v2/nnUNet_results \
#       --raw-dir        ~/nnunet-v2/nnUNet_raw
#
# Usage (from run_all.sh — variables already set):
#   bash scripts/evaluate_test_set.sh \
#       --dataset-number ${DATASET_NUMBER} \
#       --dataset-name   ${DATASET_NAME} \
#       --results-dir    ${PATH_NNUNET_RESULTS} \
#       --raw-dir        ${PATH_NNUNET_RAW}

set -e

# ==============================
# PARSE ARGUMENTS
# ==============================

DATASET_NUMBER=""
DATASET_NAME=""
PATH_NNUNET_RESULTS=""
PATH_NNUNET_RAW=""
FOLD=0
PLANS_FILE="nnUNetPlans"

while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset-number) DATASET_NUMBER="$2"; shift 2 ;;
    --dataset-name)   DATASET_NAME="$2";   shift 2 ;;
    --results-dir)    PATH_NNUNET_RESULTS="$2"; shift 2 ;;
    --raw-dir)        PATH_NNUNET_RAW="$2";     shift 2 ;;
    --fold)           FOLD="$2";           shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$DATASET_NUMBER" || -z "$DATASET_NAME" || -z "$PATH_NNUNET_RESULTS" || -z "$PATH_NNUNET_RAW" ]]; then
  echo "Usage: bash $0 --dataset-number N --dataset-name NAME --results-dir DIR --raw-dir DIR"
  exit 1
fi

PATH_MODEL="${PATH_NNUNET_RESULTS}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/nnUNetTrainer__${PLANS_FILE}__3d_fullres"
PATH_IMAGES_TS="${PATH_NNUNET_RAW}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/imagesTs"
PATH_LABELS_TS="${PATH_NNUNET_RAW}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/labelsTs"
PATH_DATASET_JSON="${PATH_NNUNET_RAW}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/dataset.json"
PATH_PRED="${PATH_MODEL}/test_predictions"
PATH_SUMMARY="${PATH_PRED}/summary.json"

# ==============================
# PREDICT
# ==============================

if [[ -f "${PATH_SUMMARY}" ]]; then
  echo "Test predictions already exist (summary.json found), skipping."
else
  echo "-----------------------------------"
  echo "Running nnUNet test set inference ..."
  echo "Model  : ${PATH_MODEL}"
  echo "Images : ${PATH_IMAGES_TS}"
  echo "Output : ${PATH_PRED}"
  echo "-----------------------------------"

  nnUNetv2_predict \
      -i  ${PATH_IMAGES_TS} \
      -o  ${PATH_PRED} \
      -d  ${DATASET_NUMBER} \
      -f  ${FOLD} \
      -c  3d_fullres

  echo "-----------------------------------"
  echo "Computing Dice ..."
  echo "-----------------------------------"

  nnUNetv2_evaluate_folder \
      ${PATH_LABELS_TS} \
      ${PATH_PRED} \
      -djfile ${PATH_DATASET_JSON} \
      -pfile  ${PATH_MODEL}/plans.json

  echo "-----------------------------------"
  echo "Done. Results → ${PATH_SUMMARY}"
  echo "-----------------------------------"
fi
