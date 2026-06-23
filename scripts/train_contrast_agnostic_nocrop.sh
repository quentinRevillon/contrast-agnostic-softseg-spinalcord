#!/bin/bash
# ============================================================================
# NO-CROP ABLATION — train the full-volume twin of the contrast-agnostic model.
#
# Trains nnUNet on FULL volumes (no sc-crop) so it can be compared against the
# cropped model (Dataset7000) in a controlled way. This is a self-contained
# mirror of train_contrast_agnostic.sh: it builds its OWN datalists, converts
# (WITHOUT cropping) into Dataset7001, runs its OWN nnUNet preprocessing, and
# trains.
#
# WHY THE SPLIT IS STILL IDENTICAL TO v4 (the key point of the ablation):
#   the train/val/test split is PREDEFINED in datasplits/*.yaml
#   (--use-predefined-splits), version-pinned via dataset_version_commit.
#   It is NOT random. Regenerating the datalists therefore reproduces exactly
#   the same split as the cropped run — the ONLY difference between Dataset7000
#   and Dataset7001 is the absence of cropping at conversion (step 3).
#
# Steps: 1=clone  2=datalists  3=convert(NO crop)  4=preprocess  5=train
# Usage:
#   bash scripts/train_contrast_agnostic_nocrop.sh              (auto-resume)
#   START_STEP=3 bash scripts/train_contrast_agnostic_nocrop.sh (skip clone/datalists)
# Run inside tmux/screen; launch heavy steps via set_slot.
# ============================================================================

set -e

PATH_REPO="/home/quentinr/contrast-agnostic-softseg-spinalcord"

# ====================================
# VARIABLES FOR DATASET CREATION  (identical to train_contrast_agnostic.sh)
# ====================================
SEED=50

DATASETS=("data-multi-subject" "basel-mp2rage" "canproco" \
            "lumbar-epfl" "lumbar-vanderbilt" "dcm-brno" "dcm-zurich" "dcm-zurich-lesions" "dcm-zurich-lesions-20231115" \
            "sci-paris" "sci-zurich" "sci-colorado" "sct-testing-large" \
            "site_006" "site_007"
            )

PATH_DATA_BASE="/home/quentinr/datasets_contrast_agnostic_retraining"

# Datalists go in their own dated folder. Content is split-identical to v4
# because the split is predefined (see header). Name is "-nocrop" only to keep
# the folder distinct on disk; the JSON inside are the same predefined splits.
folder_name=$(date +"%Y%m%d")-nocrop
PATH_OUT_DATALISTS="/home/quentinr/datalists/${folder_name}"

PATH_INCLUDE_SUBJECTS=${PATH_REPO}/subjects_to_include.yml

# ====================================
# VARIABLES FOR NNUNET TRAINING
# ====================================
PATH_NNUNET_RAW="/home/quentinr/nnunet-v2/nnUNet_raw"
PATH_NNUNET_RESULTS="/home/quentinr/nnunet-v2/nnUNet_results"

export nnUNet_raw=${PATH_NNUNET_RAW}
export nnUNet_preprocessed="/home/quentinr/nnunet-v2/nnUNet_preprocessed"
export nnUNet_results=${PATH_NNUNET_RESULTS}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=3
export PATH="/home/quentinr/spinalcordtoolbox/bin:${PATH}"

# No-crop dataset identity (DIFFERENT from the cropped Dataset7000)
DATASET_NAME="ContrastAgnosticNoCrop"
DATASET_NUMBER=7001

NNUNET_TRAINER="nnUNetTrainer"
NNUNET_PLANS_FILE="nnUNetPlans"
configurations=("3d_fullres")
folds=(0)
cuda_visible_devices=0

# Disable torch.compile (torch.inductor/Triton) — required for sm_120+ GPUs with older PyTorch
export TORCHDYNAMO_DISABLE=1

# ====================================
# STEP CONTROL — auto-detects where to resume, or override with START_STEP
# Steps: 1=clone  2=datalists  3=convert(no-crop)  4=preprocess  5=train
# ====================================
END_STEP=${END_STEP:-5}

# Re-run the no-crop conversion from scratch: the Dataset7001 raw dir already
# exists (partial), which would make auto-resume skip step 3 and preprocess an
# incomplete dataset. Force START_STEP=3 to redo convert -> preprocess -> train.
# Remove this line to restore auto-resume.
START_STEP=${START_STEP:-3}

if [ -z "${START_STEP}" ]; then
    NNUNET_RESULTS_DIR="${PATH_NNUNET_RESULTS}/Dataset${DATASET_NUMBER}_${DATASET_NAME}"
    NNUNET_PREPROCESSED_DIR="${nnUNet_preprocessed}/Dataset${DATASET_NUMBER}_${DATASET_NAME}"
    NNUNET_RAW_DIR="${PATH_NNUNET_RAW}/Dataset${DATASET_NUMBER}_${DATASET_NAME}"

    if [ -d "${NNUNET_RESULTS_DIR}/nnUNetTrainer__nnUNetPlans__${configurations[0]}" ]; then
        START_STEP=5
    elif [ -f "${NNUNET_PREPROCESSED_DIR}/nnUNetPlans.json" ]; then
        START_STEP=5
    elif [ -d "${NNUNET_RAW_DIR}" ]; then
        START_STEP=4
    elif [ -d "${PATH_OUT_DATALISTS}" ] && [ -n "$(ls ${PATH_OUT_DATALISTS}/*.json 2>/dev/null)" ]; then
        START_STEP=3
    elif [ -d "${PATH_DATA_BASE}/${DATASETS[0]}" ]; then
        START_STEP=2
    else
        START_STEP=1
    fi
    echo "-----------------------------------"
    echo "Auto-detected START_STEP=${START_STEP}"
    echo "-----------------------------------"
fi


# ====================================
# STEP 1 — CLONE DATASETS
# ====================================
if [ ${START_STEP} -le 1 ] && [ ${END_STEP} -ge 1 ]; then

    cd ${PATH_REPO}
    for dataset in ${DATASETS[@]}; do
        if [[ ${dataset} == site_* ]]; then
            if [[ ! -d "${PATH_DATA_BASE}/${dataset}" ]]; then
                echo "No dataset found in ${PATH_DATA_BASE}/${dataset}, please download the praxis dataset manually"
                exit
            fi
        else
            echo "Cloning ${dataset} dataset from git-annex ..."
            python ${PATH_REPO}/nnUnet/01_clone_dataset.py \
                --ofolder ${PATH_DATA_BASE} \
                --dataset ${dataset} \
                --path-datasplits ${PATH_REPO}/datasplits
        fi
    done
    echo "STEP 1 done — all datasets cloned."

fi


# ====================================
# STEP 2 — CREATE DATALISTS (predefined splits → identical split to v4)
# ====================================
if [ ${START_STEP} -le 2 ] && [ ${END_STEP} -ge 2 ]; then

    cd ${PATH_REPO}
    for dataset in ${DATASETS[@]}; do
        echo "Creating datalist for ${dataset} ..."
        python ${PATH_REPO}/nnUnet/02_create_msd_data.py \
            --seed ${SEED} \
            --path-data ${PATH_DATA_BASE}/${dataset} \
            --path-out ${PATH_OUT_DATALISTS} \
            --include ${PATH_INCLUDE_SUBJECTS} \
            --use-predefined-splits \
            --path-datasplits ${PATH_REPO}/datasplits
    done
    echo "STEP 2 done — datalists in ${PATH_OUT_DATALISTS}"

fi


# ====================================
# STEP 3 — CONVERT TO NNUNET FORMAT (NO cropping)
# ====================================
if [ ${START_STEP} -le 3 ] && [ ${END_STEP} -ge 3 ]; then

    echo "Converting the datalists to nnUNetv2 format WITHOUT cropping ..."
    # NOTE: no-crop = full-FOV volumes, so each per-image SCT registration is heavy
    # (~4-5 GB RAM). ANTs threads are capped to 1 via ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS,
    # so each worker uses ~1 core. 10 workers ~= 10 cores / ~50 GB RAM: fast without
    # re-saturating CPU/swap on this 48-core / 125 GB server.
    python ${PATH_REPO}/nnUnet/03_convert_msd_to_nnunet_reorient_nocrop.py \
        --input ${PATH_OUT_DATALISTS} \
        --output ${PATH_NNUNET_RAW} \
        --taskname ${DATASET_NAME} \
        --tasknumber ${DATASET_NUMBER} \
        --workers 10
    echo "STEP 3 done — no-crop dataset at ${PATH_NNUNET_RAW}/Dataset${DATASET_NUMBER}_${DATASET_NAME}."

fi


# ====================================
# STEP 4 — NNUNET PREPROCESSING
# ====================================
if [ ${START_STEP} -le 4 ] && [ ${END_STEP} -ge 4 ]; then

    for configuration in ${configurations[@]}; do
        echo "Verifying dataset integrity and preprocessing for ${configuration} ..."
        echo "NOTE: nnUNet self-configures for FULL volumes here (larger patch/spacing"
        echo "than the cropped dataset). This is expected — it is the no-crop baseline."
        nnUNetv2_plan_and_preprocess -d ${DATASET_NUMBER} --verify_dataset_integrity -c ${configuration}
    done
    echo "STEP 4 done — preprocessing completed."

fi


# ====================================
# STEP 5 — NNUNET TRAINING
# ====================================
if [ ${START_STEP} -le 5 ] && [ ${END_STEP} -ge 5 ]; then

for configuration in ${configurations[@]}; do
    for fold in ${folds[@]}; do

        echo "Training (no-crop) on Fold $fold, Configuration $configuration ..."
        start=$(date +%s)

        CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_train ${DATASET_NUMBER} \
                                $configuration $fold -tr ${NNUNET_TRAINER} -p ${NNUNET_PLANS_FILE}

        end=`date +%s`
        runtime=$((end-start))
        echo "~~~"
        echo "Ran on:      `uname -nsr`"
        echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
        echo "~~~"
    done
done

echo "STEP 5 done — no-crop training completed."
echo "Model: ${PATH_NNUNET_RESULTS}/Dataset${DATASET_NUMBER}_${DATASET_NAME}"

fi
