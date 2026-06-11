#!/bin/bash
# This script runs nnUNet inference on the test set and evaluates Dice scores.
# It assumes training is complete and the test set (imagesTs/labelsTs) is already
# in the nnUNet_raw folder (populated by train_contrast_agnostic.sh step 3).
#
# Usage:
#   bash scripts/test_contrast_agnostic.sh
#   CHECKPOINT=checkpoint_best bash scripts/test_contrast_agnostic.sh  (default: checkpoint_final)

set -e

# ====================================
# VARIABLES
# ====================================

PATH_NNUNET_RAW="/home/quentinr/nnunet-v2/nnUNet_raw"
PATH_NNUNET_RESULTS="/home/quentinr/nnunet-v2/nnUNet_results"

export nnUNet_raw=${PATH_NNUNET_RAW}
export nnUNet_preprocessed="/home/quentinr/nnunet-v2/nnUNet_preprocessed"
export nnUNet_results=${PATH_NNUNET_RESULTS}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

DATASET_NAME="ContrastAgnosticScCrop"
DATASET_NUMBER=7000
NNUNET_TRAINER="nnUNetTrainer"
NNUNET_PLANS_FILE="nnUNetPlans"
configurations=("3d_fullres")
folds=(0)
cuda_visible_devices=0
CHECKPOINT=${CHECKPOINT:-checkpoint_best.pth}

PATH_IMAGES_TS="${PATH_NNUNET_RAW}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/imagesTs"
PATH_LABELS_TS="${PATH_NNUNET_RAW}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/labelsTs"
PATH_DATASET_JSON="${PATH_NNUNET_RAW}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/dataset.json"

# ====================================
# INFERENCE ON TEST SET
# ====================================

for configuration in ${configurations[@]}; do
    for fold in ${folds[@]}; do

        PATH_PREDICTIONS="${PATH_NNUNET_RESULTS}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/nnUNetTrainer__nnUNetPlans__${configuration}/predictions_test_fold${fold}_${CHECKPOINT%.pth}"
        PATH_PLANS_JSON="${PATH_NNUNET_RESULTS}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/nnUNetTrainer__nnUNetPlans__${configuration}/plans.json"

        echo "-------------------------------------------"
        echo "Running inference on test set"
        echo "  Configuration : ${configuration}"
        echo "  Fold          : ${fold}"
        echo "  Checkpoint    : ${CHECKPOINT}"
        echo "  Output        : ${PATH_PREDICTIONS}"
        echo "-------------------------------------------"

        CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_predict \
            -d ${DATASET_NUMBER} \
            -i ${PATH_IMAGES_TS} \
            -o ${PATH_PREDICTIONS} \
            -f ${fold} \
            -tr ${NNUNET_TRAINER} \
            -p ${NNUNET_PLANS_FILE} \
            -c ${configuration} \
            -chk ${CHECKPOINT}

        echo "-------------------------------------------"
        echo "Evaluating predictions (Dice)"
        echo "-------------------------------------------"

        nnUNetv2_evaluate_folder \
            ${PATH_LABELS_TS} \
            ${PATH_PREDICTIONS} \
            -djfile ${PATH_DATASET_JSON} \
            -pfile ${PATH_PLANS_JSON}

        echo "-------------------------------------------"
        echo "Done. Results saved in ${PATH_PREDICTIONS}/summary.json"
        echo "-------------------------------------------"

    done
done
