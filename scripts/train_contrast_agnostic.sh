#!/bin/bash
# Training and evaluation pipeline for contrast-agnostic v3.0.
#
# Steps:
#   1. Clone datasets from git-annex
#   2. Create datalists (JSON with image/label pairs)
#   3. Convert to nnUNet format (GT bbox crop + RPI reorientation)
#   4. nnUNet plan and preprocess
#   5. nnUNet training
#   6. PyTorch inference on GT-crop test set (oracle upper bound)
#   7. PyTorch inference on sc_crop test set (realistic detection pipeline)
#   8. Export trained model to ONNX
#   9. ONNX benchmark on GT-crop test set (comparable to step 6)
#  10. ONNX benchmark on sc_crop test set (comparable to step 7)
#
# Set START_STEP to skip completed steps. If starting from step 3+, set PATH_OUT_DATALISTS manually.


# Define (full) path to the contrast-agnostic repository
PATH_REPO="/home/quentinr/contrast-agnostic-softseg-spinalcord"

# Step to start from (1–10)
START_STEP=8


#  [SKIP] sc_crop detection failed: /home/quentinr/datasets_contrast_agnostic_retraining/canproco/sub-cal175/ses-M0/anat/sub-cal175_ses-M0_STIR.nii.gz
# ====================================
# VARIABLES FOR DATASET CREATION
# ====================================

# Set seed for reproducibility. Note seed=50 was used train the contrast-agnostic model 
# If you're using a different seed, note that you cannot use the predefined random dataset splits (more info below)
SEED=50

DATASETS=("data-multi-subject" "basel-mp2rage" "canproco" \
            "lumbar-epfl" "lumbar-vanderbilt" "dcm-brno" "dcm-zurich" "dcm-zurich-lesions" "dcm-zurich-lesions-20231115" \
            "sci-paris" "sci-zurich" "sci-colorado" "sct-testing-large" \
            "site_006" "site_007"
            )

PATH_DATA_BASE="/home/quentinr/datasets_contrast_agnostic_retraining"
PATH_OUT_DATALISTS="/home/quentinr/datalists/20250421-temp"
PATH_INCLUDE_SUBJECTS=${PATH_REPO}/subjects_to_include.yml


# ====================================
# VARIABLES FOR NNUNET TRAINING
# ====================================

PATH_NNUNET_RAW="/home/quentinr/nnunet-v2/nnUNet_raw"
PATH_NNUNET_RESULTS="/home/quentinr/nnunet-v2/nnUNet_results"

export nnUNet_raw=${PATH_NNUNET_RAW}
export nnUNet_preprocessed="/home/quentinr/nnunet-v2/nnUNet_preprocessed"
export nnUNet_results=${PATH_NNUNET_RESULTS}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

DATASET_NAME="TempContrastAgnosticCropped"
DATASET_NUMBER=1000
NNUNET_TRAINER="nnUNetTrainer"
NNUNET_PLANS_FILE="nnUNetPlans"
configurations=("3d_fullres")
folds=(0)
cuda_visible_devices=0


# ====================================
# STEP 1 — CLONE DATASETS
# ====================================

if [ ${START_STEP} -le 1 ]; then
    for dataset in ${DATASETS[@]}; do
        if [[ ${dataset} == site_* ]]; then
            if [[ ! -d "${PATH_DATA_BASE}/${dataset}" ]]; then
                echo "Dataset ${dataset} not found in ${PATH_DATA_BASE}, please download manually"
                exit 1
            fi
        else
            python ${PATH_REPO}/nnUnet/01_clone_dataset.py \
                --ofolder ${PATH_DATA_BASE} \
                --dataset ${dataset} \
                --path-datasplits ${PATH_REPO}/datasplits
        fi
    done
fi


# ====================================
# STEP 2 — CREATE DATALISTS
# ====================================

if [ ${START_STEP} -le 2 ]; then
    for dataset in ${DATASETS[@]}; do
        python ${PATH_REPO}/nnUnet/02_create_msd_data.py \
            --seed ${SEED} \
            --path-data ${PATH_DATA_BASE}/${dataset} \
            --path-out ${PATH_OUT_DATALISTS} \
            --include ${PATH_INCLUDE_SUBJECTS} \
            --use-predefined-splits \
            --path-datasplits ${PATH_REPO}/datasplits
    done
    echo "Datalists created in ${PATH_OUT_DATALISTS}"
fi


# ====================================
# STEP 3 — CONVERT TO NNUNET FORMAT
# ====================================

if [ ${START_STEP} -le 3 ]; then
    # Without cropping (original pipeline):
    # python ${PATH_REPO}/nnUnet/03_convert_msd_to_nnunet_reorient.py \
    #     --input ${PATH_OUT_DATALISTS} \
    #     --output ${PATH_NNUNET_RAW} \
    #     --taskname ${DATASET_NAME} \
    #     --tasknumber ${DATASET_NUMBER} \
    #     --workers 8

    # With SC bbox cropping (asymmetric per-face padding in mm):
    python ${PATH_REPO}/nnUnet/03_convert_msd_to_nnunet_reorient_cropped.py \
        --input ${PATH_OUT_DATALISTS} \
        --output ${PATH_NNUNET_RAW} \
        --taskname ${DATASET_NAME} \
        --tasknumber ${DATASET_NUMBER} \
        --pad-left 20 --pad-right 20 \
        --pad-anterior 30 --pad-posterior 30 \
        --pad-superior 40 --pad-inferior 40 \
        --workers 8
fi


# ====================================
# STEP 4 — NNUNET PREPROCESSING
# ====================================

if [ ${START_STEP} -le 4 ]; then
    for configuration in ${configurations[@]}; do
        nnUNetv2_plan_and_preprocess -d ${DATASET_NUMBER} --verify_dataset_integrity -c ${configuration}
    done
fi


# ====================================
# STEP 5 — NNUNET TRAINING
# ====================================

if [ ${START_STEP} -le 5 ]; then
    for configuration in ${configurations[@]}; do
        for fold in ${folds[@]}; do
            start=$(date +%s)
            CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_train ${DATASET_NUMBER} \
                $configuration $fold -tr ${NNUNET_TRAINER} -p ${NNUNET_PLANS_FILE}
            end=$(date +%s)
            runtime=$((end-start))
            echo "Fold $fold | $configuration | $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
        done
    done
    echo "Model saved in ${PATH_NNUNET_RESULTS}/Dataset${DATASET_NUMBER}_${DATASET_NAME}"
fi


# ====================================
# STEP 6 — TEST SET EVALUATION (oracle crop)
# ====================================
# Runs inference on imagesTs (preprocessed with GT bbox crop, same as training)
# and evaluates Dice. This is the "oracle" upper bound: crop coordinates are
# derived from the GT label, not a detection model.

if [ ${START_STEP} -le 6 ]; then
    MODEL_DIR="${PATH_NNUNET_RESULTS}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/nnUNetTrainer__${NNUNET_PLANS_FILE}__${configurations[0]}"
    PATH_TEST_PRED="${MODEL_DIR}/test_predictions"
    mkdir -p ${PATH_TEST_PRED}

    CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_predict \
        -i ${PATH_NNUNET_RAW}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/imagesTs \
        -o ${PATH_TEST_PRED} \
        -d ${DATASET_NUMBER} \
        -f ${folds[0]} \
        -c ${configurations[0]} \
        -tr ${NNUNET_TRAINER} \
        -p ${NNUNET_PLANS_FILE}

    nnUNetv2_evaluate_folder \
        ${PATH_NNUNET_RAW}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/labelsTs \
        ${PATH_TEST_PRED} \
        -djfile ${PATH_NNUNET_RAW}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/dataset.json \
        -pfile ${MODEL_DIR}/plans.json

    echo "Oracle test Dice saved to ${PATH_TEST_PRED}/summary.json"
fi


# ====================================
# STEP 7 — TEST SET EVALUATION (sc_crop pipeline)
# ====================================
# Runs the realistic inference pipeline on the original (uncropped) test images:
# sc_crop detects the SC bbox → crop → reorient RPI → nnUNet → Dice vs GT.
# Outputs coverage, dice_within_crop, and dice_global (penalises missed SC).
# Requires sc_crop installed in a separate conda env (SC_CROP_ENV).

SC_CROP_ENV="sc_crop"

if [ ${START_STEP} -le 7 ]; then
    MODEL_DIR="${PATH_NNUNET_RESULTS}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/nnUNetTrainer__${NNUNET_PLANS_FILE}__${configurations[0]}"
    PATH_SC_CROP_EVAL="${MODEL_DIR}/test_sc_crop"

    python ${PATH_REPO}/nnUnet/04_evaluate_with_sc_crop.py \
        --dataset-folder ${PATH_NNUNET_RAW}/Dataset${DATASET_NUMBER}_${DATASET_NAME} \
        --model-folder ${MODEL_DIR} \
        --output-dir ${PATH_SC_CROP_EVAL} \
        --pad-left 20 --pad-right 20 \
        --pad-anterior 30 --pad-posterior 30 \
        --pad-superior 40 --pad-inferior 40 \
        --sc-crop-env ${SC_CROP_ENV} \
        --use-gpu

    echo "sc_crop evaluation results saved to ${PATH_SC_CROP_EVAL}/metrics_sc_crop.csv"
fi


# ====================================
# STEP 8 — EXPORT MODEL TO ONNX
# ====================================
# Exports the trained network weights to ONNX (no nnunetv2 required at inference time).
# Output: ${MODEL_DIR}/onnx/nnunet_seg.onnx

if [ ${START_STEP} -le 8 ]; then
    MODEL_DIR="${PATH_NNUNET_RESULTS}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/nnUNetTrainer__${NNUNET_PLANS_FILE}__${configurations[0]}"
    ONNX_DIR="${MODEL_DIR}/onnx"
    mkdir -p ${ONNX_DIR}

    python ${PATH_REPO}/nnUnet/export_nnunet_to_onnx.py \
        --model-folder ${MODEL_DIR} \
        --fold ${folds[0]} \
        --output ${ONNX_DIR}/nnunet_seg.onnx

    cp ${MODEL_DIR}/plans.json ${ONNX_DIR}/plans.json
    echo "ONNX model saved to ${ONNX_DIR}/nnunet_seg.onnx"
fi


# ====================================
# STEP 9 — ONNX BENCHMARK (GT-crop)
# ====================================
# ONNX inference on imagesTs (GT bbox crop, same as step 6).
# Directly comparable to test_predictions/summary.json.

if [ ${START_STEP} -le 9 ]; then
    MODEL_DIR="${PATH_NNUNET_RESULTS}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/nnUNetTrainer__${NNUNET_PLANS_FILE}__${configurations[0]}"
    ONNX_DIR="${MODEL_DIR}/onnx"

    python ${PATH_REPO}/nnUnet/05_benchmark_onnx.py \
        --images-dir ${PATH_NNUNET_RAW}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/imagesTs \
        --labels-dir ${PATH_NNUNET_RAW}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/labelsTs \
        --model      ${ONNX_DIR}/nnunet_seg.onnx \
        --plans      ${ONNX_DIR}/plans.json \
        --output     ${ONNX_DIR}/benchmark_gt_crop.csv

    echo "ONNX GT-crop results saved to ${ONNX_DIR}/benchmark_gt_crop.csv"
fi


# ====================================
# STEP 10 — ONNX BENCHMARK (sc_crop)
# ====================================
# ONNX inference via sc_crop detection pipeline on original test images.
# Directly comparable to test_sc_crop/metrics_sc_crop.csv (step 7).

if [ ${START_STEP} -le 10 ]; then
    MODEL_DIR="${PATH_NNUNET_RESULTS}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/nnUNetTrainer__${NNUNET_PLANS_FILE}__${configurations[0]}"
    ONNX_DIR="${MODEL_DIR}/onnx"

    python ${PATH_REPO}/nnUnet/05_benchmark_onnx_sc_crop.py \
        --dataset-folder ${PATH_NNUNET_RAW}/Dataset${DATASET_NUMBER}_${DATASET_NAME} \
        --model          ${ONNX_DIR}/nnunet_seg.onnx \
        --plans          ${ONNX_DIR}/plans.json \
        --output         ${ONNX_DIR}/benchmark_sc_crop.csv \
        --pad-rl 20 --pad-ap 30 --pad-si 40 \
        --sc-crop-env    ${SC_CROP_ENV}

    echo "ONNX sc_crop results saved to ${ONNX_DIR}/benchmark_sc_crop.csv"
fi
