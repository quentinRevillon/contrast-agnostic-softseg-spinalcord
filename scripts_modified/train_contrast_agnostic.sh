#!/bin/bash
# Training script for contrast-agnostic v3.0 + YOLO crop augmentation (Dataset998).
#
# Pipeline:
#   1. Clone datasets from git-annex          → already done, COMMENTED OUT
#   2. Create MSD datalists (JSON)             → already done, COMMENTED OUT
#   3. Convert to nnUNet format + SC bbox crop → run 03_convert_msd_to_nnunet_reorient_cropped.py
#   4. nnUNet plan and preprocess              → run nnUNetv2_plan_and_preprocess
#   5. Train with nnUNetTrainer_CropAugmentation
#
# Dataset999: full images, no SC crop (original pipeline, already processed)
# Dataset998: SC bbox cropped to GT + 10mm padding (this script)
#
# Usage (via set_slot):
#   set_slot 0 /bin/bash /home/quentinr/contrast-agnostic-softseg-spinalcord/scripts_modified/train_contrast_agnostic.sh


# ====================================
# ENVIRONMENT
# ====================================

PATH_REPO="/home/quentinr/contrast-agnostic-softseg-spinalcord"
cd ${PATH_REPO}

PYTHON=/home/quentinr/.conda/envs/contrast_agnostic/bin/python
export PATH=/home/quentinr/.conda/envs/contrast_agnostic/bin:/home/quentinr/spinalcordtoolbox/bin:$PATH
export nnUNet_raw=/home/quentinr/nnunet-v2/nnUNet_raw
export nnUNet_preprocessed=/home/quentinr/nnunet-v2/nnUNet_preprocessed
export nnUNet_results=/home/quentinr/nnunet-v2/nnUNet_results
export TORCHDYNAMO_DISABLE=1

cuda_visible_devices=0
export CUDA_VISIBLE_DEVICES=${cuda_visible_devices}


# ====================================
# VARIABLES FOR DATASET CREATION
# ====================================

SEED=50

# Datasets used for contrast-agnostic v3.0:
# https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/releases/tag/v3.0
DATASETS=("data-multi-subject" "basel-mp2rage" "canproco" \
            "lumbar-epfl" "lumbar-vanderbilt" "dcm-brno" "dcm-zurich" "dcm-zurich-lesions" "dcm-zurich-lesions-20231115" \
            "sci-paris" "sci-zurich" "sci-colorado" "sct-testing-large" \
            "site_006" "site_007"
            )

PATH_DATA_BASE="/home/quentinr/datasets_contrast_agnostic_retraining"
PATH_OUT_DATALISTS="/home/quentinr/datalists/20250421-temp"   # datalists already generated here
PATH_INCLUDE_SUBJECTS=${PATH_REPO}/subjects_to_include.yml


# ====================================
# VARIABLES FOR NNUNET TRAINING
# ====================================

PATH_NNUNET_RAW="/home/quentinr/nnunet-v2/nnUNet_raw"
PATH_NNUNET_RESULTS="/home/quentinr/nnunet-v2/nnUNet_results"

DATASET_NAME="TempContrastAgnosticCropped"
DATASET_NUMBER=998

SC_CROP_PADDING_MM=10.0

# NNUNET_TRAINER="nnUNetTrainer_CropAugmentation"
NNUNET_TRAINER="nnUNetTrainer"   # baseline without crop augmentation

NNUNET_PLANS_FILE="nnUNetPlans"
configurations=("3d_fullres")
folds=(0)


# ====================================
# PHASE 1 — CLONE DATASETS  (already done)
# ====================================

# for dataset in ${DATASETS[@]}; do
#     if [[ ${dataset} == site_* ]]; then
#         if [[ ! -d "${PATH_DATA_BASE}/${dataset}" ]]; then
#             echo "No dataset found in ${PATH_DATA_BASE}/${dataset}, please download manually"
#             exit 1
#         fi
#     else
#         ${PYTHON} ${PATH_REPO}/nnUnet_modified/01_clone_dataset.py \
#             --ofolder ${PATH_DATA_BASE} \
#             --dataset ${dataset} \
#             --path-datasplits ${PATH_REPO}/datasplits
#     fi
# done


# ====================================
# PHASE 2 — CREATE MSD DATALISTS  (already done)
# ====================================

for dataset in ${DATASETS[@]}; do
    ${PYTHON} ${PATH_REPO}/nnUnet_modified/02_create_msd_data.py \
        --seed ${SEED} \
        --path-data ${PATH_DATA_BASE}/${dataset} \
        --path-out ${PATH_OUT_DATALISTS} \
        --include ${PATH_INCLUDE_SUBJECTS} \
        --use-predefined-splits \
        --path-datasplits ${PATH_REPO}/datasplits
done


# ====================================
# PHASE 3 — CONVERT TO NNUNET FORMAT + SC BBOX CROP
# ====================================
# Produces Dataset998_TempContrastAgnosticCropped in nnUNet_raw.
# Dataset999 (full images, already processed) is left untouched.
# Takes ~8-10h with 8 workers.

echo "-----------------------------------"
echo "Converting datalists to nnUNet format with SC bbox crop (${SC_CROP_PADDING_MM}mm padding)..."
echo "-----------------------------------"

${PYTHON} ${PATH_REPO}/nnUnet_modified/03_convert_msd_to_nnunet_reorient_cropped.py \
    --input ${PATH_OUT_DATALISTS} \
    --output ${PATH_NNUNET_RAW} \
    --taskname ${DATASET_NAME} \
    --tasknumber ${DATASET_NUMBER} \
    --padding-mm ${SC_CROP_PADDING_MM} \
    --workers 8

echo "-----------------------------------"
echo "Conversion done. Dataset: ${PATH_NNUNET_RAW}/Dataset${DATASET_NUMBER}_${DATASET_NAME}"
echo "-----------------------------------"


# ====================================
# PHASE 4 — NNUNET PLAN AND PREPROCESS
# ====================================

for configuration in ${configurations[@]}; do
    echo "-----------------------------------"
    echo "Planning and preprocessing for ${configuration}..."
    echo "-----------------------------------"
    nnUNetv2_plan_and_preprocess -d ${DATASET_NUMBER} --verify_dataset_integrity -c ${configuration}
done

echo "-----------------------------------"
echo "Preprocessing done. Starting training..."
echo "-----------------------------------"


# ====================================
# PHASE 5 — TRAINING
# ====================================

for configuration in ${configurations[@]}; do
    for fold in ${folds[@]}; do

        echo "-------------------------------------------"
        echo "Training fold $fold, config $configuration, trainer ${NNUNET_TRAINER}..."
        echo "-------------------------------------------"

        start=$(date +%s)

        nnUNetv2_train ${DATASET_NUMBER} $configuration $fold -tr ${NNUNET_TRAINER} -p ${NNUNET_PLANS_FILE}

        end=$(date +%s)
        runtime=$((end-start))
        echo "Duration: $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
        echo "Model saved in ${PATH_NNUNET_RESULTS}/Dataset${DATASET_NUMBER}_${DATASET_NAME}"
    done
done
