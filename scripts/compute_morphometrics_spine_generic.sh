#!/bin/bash
# Post-training script for computing morphometrics on spine-generic dataset using lifelong learning contrast-agnostic 
# spinal cord segmentation model. 
# This script is one of the steps in the automated GitHub actions for computing spinal cord morphometrics (CSA)
# 
# This script performs the following tasks:
# 1. Resolves the model checkpoint: a local fold_N/checkpoint_best.pth, or a release asset URL
#    (zipped nnUNet model folder) that is downloaded and unzipped.
# 2. Runs a batch analysis (sct_run_batch) to compute the spinal cord cross-sectional area (CSA) on a
# mini-batch of test subjects obtained as input. Inference uses sc-crop (YOLO crop) + nnUNet.
# 3. Moves the logs/ and results/ to the an output folder
#
# Usage:
#   bash compute_morphometrics_spine_generic.sh "<subjects>" "<checkpoint_path_or_release_url>"

# Exit immediately if a command exits with a non-zero status
set -e

# ==============================
# DEFINE GLOBAL VARIABLES
# ==============================

# get current working directory before doing anything else
CWD=${PWD}

# get list of subjects as input
TEST_SUBJECTS=($1)
echo "Running analysis on ${TEST_SUBJECTS[@]}"

# Path to the output folder; the data, model, results, etc. will be stored in this folder
PATH_OUTPUT="csa-analysis"

# Model checkpoint to run inference with. Accepts either:
#   - a local path to fold_N/checkpoint_best.pth (local testing), or
#   - a release asset URL pointing to the zipped nnUNet model folder (GitHub Actions).
MODEL_INPUT=$2
echo "Using model: ${MODEL_INPUT}"

# Number of parallel processes to run (choose a smaller number as inference is run only on 1 gpu)
NUM_WORKERS=4

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

echo "=============================="
echo "Resolving model checkpoint ..."
echo "=============================="

if [[ -f "${MODEL_INPUT}" ]]; then
    # Local checkpoint path
    PATH_CHECKPOINT="${MODEL_INPUT}"
    MODEL_VERSION="local"
else
    # Release asset URL: parse version from the tag, download and unzip the model folder
    MODEL_VERSION=$(echo "${MODEL_INPUT}" | sed -E 's#.*/download/([^/]+)/.*#\1#')
    curl -L -o model.zip "${MODEL_INPUT}"
    unzip -o model.zip -d model_dir
    PATH_CHECKPOINT=$(find model_dir -name checkpoint_best.pth | head -1)
fi

# sct_run_batch runs compute_csa.sh from a different working directory, so the checkpoint
# path must be absolute (sc-segment-pt resolves dataset.json/plans.json relative to it).
PATH_CHECKPOINT=$(realpath "${PATH_CHECKPOINT}")

echo "Model version  : ${MODEL_VERSION}"
echo "Checkpoint path: ${PATH_CHECKPOINT}"

# ==============================
# RUN BATCH ANALYSIS
# NOTE: this section piggybacks on the sct_run_batch argument provided by SCT
# ==============================

echo "=============================="
echo "Running batch analysis ..."
echo "=============================="

# Run batch processing
path_out_run_batch=${PATH_OUTPUT}/batch_processing_results
echo ${path_out_run_batch}

sct_run_batch -path-data data-multi-subject \
    -path-output ${path_out_run_batch} \
    -jobs ${NUM_WORKERS} \
    -script scripts/compute_csa.sh \
    -script-args "${MODEL_VERSION} ${PATH_CHECKPOINT}" \
    -include-list ${TEST_SUBJECTS[@]}


echo "=============================="
echo "Copying log and results folders to ${CWD}/logs_results ..."
echo "=============================="

mkdir -p ${CWD}/logs_results
cp -r ${path_out_run_batch}/log ${CWD}/logs_results
cp -r ${path_out_run_batch}/results ${CWD}/logs_results
# NOTE: this copying is done so that it is easy to find these folders outside of the script to be uploaded by GH Actions

# Go back to the current working directory at the beginning
cd ${CWD}
