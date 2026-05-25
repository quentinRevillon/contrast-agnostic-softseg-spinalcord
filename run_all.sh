#!/bin/bash
# ============================================================================
# run_all.sh — Full reproducible pipeline: train + evaluate
#
# Steps:
#   0. Check prerequisites (SCT, conda env, git-annex) + install ANIMA
#   1. Clone datasets from git-annex
#   2. Create datalists
#   3. Convert to nnUNet format with sc_crop detection-based crop
#   4. nnUNet plan & preprocess
#   5. nnUNet training
#   6. Download spine-generic test data (49 subjects)
#   7. Run CSA evaluation (sct_run_batch → csa_c2c3.csv)
#   8. Run Dice evaluation (ANIMA → Dice, Jaccard, Hausdorff)
#
# Usage:
#   bash run_all.sh                  # run all steps
#   START_STEP=6 bash run_all.sh     # resume from step 6
#   START_STEP=6 END_STEP=8 bash run_all.sh
#
# ============================================================================

set -e
trap "echo; echo '>>> Interrupted. Re-run with START_STEP=$(cat .last_completed_step 2>/dev/null || echo 0) to resume.'; exit 1" INT

# ============================================================================
# ── CONFIGURATION — edit these paths before running ─────────────────────────
# ============================================================================

PATH_REPO="$(cd "$(dirname "$0")" && pwd)"

# Datasets to train on (full list for v3.0 reproduction; keep small for quick tests)
DATASETS=("data-multi-subject" "basel-mp2rage" "canproco" \
          "lumbar-epfl" "lumbar-vanderbilt" "dcm-brno" "dcm-zurich" \
          "dcm-zurich-lesions" "dcm-zurich-lesions-20231115" \
          "sci-paris" "sci-zurich" "sci-colorado" "sct-testing-large" \
          "site_006" "site_007")

PATH_DATA_BASE="/home/quentinr/datasets_contrast_agnostic_retraining"
PATH_OUT_DATALISTS="/home/quentinr/datalists/$(date +%Y%m%d)-sc-crop"
PATH_INCLUDE_SUBJECTS="${PATH_REPO}/subjects_to_include.yml"

PATH_NNUNET_RAW="/home/quentinr/nnunet-v2/nnUNet_raw"
PATH_NNUNET_PREPROCESSED="/home/quentinr/nnunet-v2/nnUNet_preprocessed"
PATH_NNUNET_RESULTS="/home/quentinr/nnunet-v2/nnUNet_results"

DATASET_NAME="ContrastAgnosticScCrop"
DATASET_NUMBER=1000
NNUNET_TRAINER="nnUNetTrainer"
NNUNET_PLANS="nnUNetPlans"
CONFIGURATION="3d_fullres"
FOLD=0
CUDA_DEVICE=0

MODEL_VERSION="sc-crop-v1"   # label for results folders

# Padding around sc_crop detected bbox (mm)
PAD_RL=10
PAD_AP=15
PAD_SI=30

# Step range (override via env: START_STEP=3 END_STEP=5 bash run_all.sh)
START_STEP=${START_STEP:-0}
END_STEP=${END_STEP:-8}

# Conda environment
CONDA_ENV="contrast_agnostic"

# ============================================================================
# ── HELPERS ──────────────────────────────────────────────────────────────────
# ============================================================================

export nnUNet_raw=${PATH_NNUNET_RAW}
export nnUNet_preprocessed=${PATH_NNUNET_PREPROCESSED}
export nnUNet_results=${PATH_NNUNET_RESULTS}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

log()  { echo; echo "══════════════════════════════════════════════════════"; echo "  $*"; echo "══════════════════════════════════════════════════════"; }
done_() { echo "  ✓ $*"; echo "$1" > .last_completed_step; }

run_step() {
    local step=$1
    [[ ${step} -lt ${START_STEP} ]] && return 0
    [[ ${step} -gt ${END_STEP}   ]] && return 0
    return 1   # meaning: execute this step
}

# ============================================================================
# ── STEP 0 — Prerequisites + ANIMA install ───────────────────────────────────
# ============================================================================

if ! run_step 0; then
    log "STEP 0 — Checking prerequisites"

    # SCT
    if ! command -v sct_image &>/dev/null; then
        echo "ERROR: SCT (sct_image) not found. Install from https://spinalcordtoolbox.com"
        exit 1
    fi
    echo "  SCT:       $(sct_version 2>/dev/null | head -1)"

    # conda env
    if ! conda run -n ${CONDA_ENV} python -c "import nnunetv2" &>/dev/null; then
        echo "ERROR: conda env '${CONDA_ENV}' not found or missing nnunetv2."
        echo "  Create it with: conda create -n ${CONDA_ENV} python=3.9 && pip install nnunetv2"
        exit 1
    fi
    echo "  conda env: ${CONDA_ENV} ✓"

    # sc_crop
    if ! conda run -n ${CONDA_ENV} python -c "import sc_crop" &>/dev/null; then
        echo "ERROR: sc_crop not installed in ${CONDA_ENV}."
        echo "  Install: pip install git+https://github.com/ivadomed/sc-crop.git"
        exit 1
    fi
    echo "  sc_crop:   $(conda run -n ${CONDA_ENV} python -c 'import sc_crop; print(sc_crop.__version__)')"

    # git-annex
    if ! command -v git-annex &>/dev/null; then
        echo "ERROR: git-annex not found. Install with: conda install -c conda-forge git-annex"
        exit 1
    fi
    echo "  git-annex: $(git-annex version | head -1)"

    # ANIMA — install if missing
    ANIMA_BIN="${HOME}/anima/Anima-Binaries-4.2"
    if [[ ! -f "${ANIMA_BIN}/animaSegPerfAnalyzer" ]]; then
        echo ""
        echo "  ANIMA not found — installing to ~/anima/ ..."
        mkdir -p ~/anima && cd ~/anima
        wget -q --show-progress \
            https://github.com/Inria-Visages/Anima-Public/releases/download/v4.2/Anima-Ubuntu-4.2.zip \
            -O Anima-Ubuntu-4.2.zip
        unzip -q Anima-Ubuntu-4.2.zip
        rm Anima-Ubuntu-4.2.zip
        cd ${PATH_REPO}
        echo "  ANIMA installed."
    else
        echo "  ANIMA:     ${ANIMA_BIN}/animaSegPerfAnalyzer ✓"
    fi

    # Write ANIMA config
    mkdir -p ~/.anima
    cat > ~/.anima/config.txt << EOF
[anima-scripts]
anima = ${HOME}/anima/Anima-Binaries-4.2/
anima-scripts-public-root = ${HOME}/anima/Anima-Scripts-Public/
extra-data-root = ${HOME}/anima/Anima-Scripts-Data-Public/
EOF
    echo "  ~/.anima/config.txt written."

    done_ 0
fi


# ============================================================================
# ── STEP 1 — Clone datasets ───────────────────────────────────────────────────
# ============================================================================

if ! run_step 1; then
    log "STEP 1 — Clone datasets"
    for dataset in "${DATASETS[@]}"; do
        if [[ ${dataset} == site_* ]]; then
            [[ ! -d "${PATH_DATA_BASE}/${dataset}" ]] && \
                { echo "ERROR: PRAXIS dataset ${dataset} not found in ${PATH_DATA_BASE}. Download manually."; exit 1; }
        else
            conda run -n ${CONDA_ENV} python ${PATH_REPO}/nnUnet/01_clone_dataset.py \
                --ofolder ${PATH_DATA_BASE} \
                --dataset ${dataset} \
                --path-datasplits ${PATH_REPO}/datasplits
        fi
    done
    done_ 1
fi


# ============================================================================
# ── STEP 2 — Create datalists ─────────────────────────────────────────────────
# ============================================================================

if ! run_step 2; then
    log "STEP 2 — Create datalists"
    mkdir -p ${PATH_OUT_DATALISTS}
    for dataset in "${DATASETS[@]}"; do
        conda run -n ${CONDA_ENV} python ${PATH_REPO}/nnUnet/02_create_msd_data.py \
            --seed 50 \
            --path-data  ${PATH_DATA_BASE}/${dataset} \
            --path-out   ${PATH_OUT_DATALISTS} \
            --include    ${PATH_INCLUDE_SUBJECTS} \
            --use-predefined-splits \
            --path-datasplits ${PATH_REPO}/datasplits
    done
    echo "  Datalists saved to ${PATH_OUT_DATALISTS}"
    done_ 2
fi


# ============================================================================
# ── STEP 3 — Convert to nnUNet format (sc_crop detection-based crop) ──────────
# ============================================================================

if ! run_step 3; then
    log "STEP 3 — Convert to nnUNet format with sc_crop crop"
    conda run -n ${CONDA_ENV} python ${PATH_REPO}/nnUnet/03_convert_msd_to_nnunet_reorient_sc_crop.py \
        --input      ${PATH_OUT_DATALISTS} \
        --output     ${PATH_NNUNET_RAW} \
        --taskname   ${DATASET_NAME} \
        --tasknumber ${DATASET_NUMBER} \
        --workers    8 \
        --pad-rl     ${PAD_RL} \
        --pad-ap     ${PAD_AP} \
        --pad-si     ${PAD_SI} \
        --skip-failed
    done_ 3
fi


# ============================================================================
# ── STEP 4 — nnUNet plan & preprocess ────────────────────────────────────────
# ============================================================================

if ! run_step 4; then
    log "STEP 4 — nnUNet plan & preprocess"
    conda run -n ${CONDA_ENV} \
        nnUNetv2_plan_and_preprocess -d ${DATASET_NUMBER} \
            --verify_dataset_integrity -c ${CONFIGURATION}
    done_ 4
fi


# ============================================================================
# ── STEP 5 — nnUNet training ──────────────────────────────────────────────────
# ============================================================================

if ! run_step 5; then
    log "STEP 5 — nnUNet training (fold ${FOLD})"
    start=$(date +%s)
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} conda run -n ${CONDA_ENV} \
        nnUNetv2_train ${DATASET_NUMBER} ${CONFIGURATION} ${FOLD} \
            -tr ${NNUNET_TRAINER} -p ${NNUNET_PLANS}
    runtime=$(( $(date +%s) - start ))
    echo "  Training duration: $(($runtime/3600))h $((($runtime/60)%60))min"
    done_ 5
fi


# ============================================================================
# ── STEP 6 — Download spine-generic test data (49 subjects) ──────────────────
# ============================================================================

if ! run_step 6; then
    log "STEP 6 — Download spine-generic test data"
    # Run from repo root so data-multi-subject lands there (sct_run_batch expects it)
    cd ${PATH_REPO}
    if [[ ! -d "data-multi-subject" ]]; then
        bash scripts/download_spine_generic_test_data.sh
    else
        echo "  data-multi-subject already present, skipping download."
    fi
    done_ 6
fi


# ============================================================================
# ── STEP 7 — CSA evaluation (sct_run_batch → csa_c2c3.csv) ──────────────────
# ============================================================================

if ! run_step 7; then
    log "STEP 7 — CSA evaluation (49 subjects, 6 contrasts)"

    MODEL_DIR="${PATH_NNUNET_RESULTS}/Dataset${DATASET_NUMBER}_${DATASET_NAME}/nnUNetTrainer__${NNUNET_PLANS}__${CONFIGURATION}"

    cd ${PATH_REPO}
    conda run -n ${CONDA_ENV} \
        bash scripts/evaluate_csa_local.sh \
            --model-folder  ${MODEL_DIR} \
            --model-version ${MODEL_VERSION} \
            --data-path     ${PATH_REPO}/data-multi-subject \
            --jobs          4

    echo "  CSA results: logs_results_${MODEL_VERSION}/results/csa_c2c3.csv"
    done_ 7
fi


# ============================================================================
# ── STEP 8 — Dice / ANIMA metrics ────────────────────────────────────────────
# ============================================================================
# Note: ANIMA is run by evaluate_csa_local.sh at the end of step 7.
# This step generates the summary report and prints final numbers.

if ! run_step 8; then
    log "STEP 8 — Dice summary"

    ANIMA_LOG="${PATH_REPO}/logs_results_${MODEL_VERSION}/anima_metrics/log_spine-generic.txt"
    if [[ -f "${ANIMA_LOG}" ]]; then
        echo ""
        echo "  ── Dice results (${MODEL_VERSION}) ──────────────────────"
        cat ${ANIMA_LOG}
        echo "  ─────────────────────────────────────────────────────"
    else
        echo "  WARNING: ${ANIMA_LOG} not found."
        echo "  ANIMA may not have been installed or evaluation failed."
        echo "  Re-run step 7 after installing ANIMA (step 0)."
    fi
    done_ 8
fi


# ============================================================================
# ── DONE ─────────────────────────────────────────────────────────────────────
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Pipeline complete — ${MODEL_VERSION}"
echo ""
echo "  CSA  : logs_results_${MODEL_VERSION}/results/csa_c2c3.csv"
echo "  Dice : logs_results_${MODEL_VERSION}/anima_metrics/log_spine-generic.txt"
echo ""
echo "  To compare CSA across versions:"
echo "    python csa_generate_figures/analyse_csa_across_releases.py \\"
echo "        --path-results logs_results_*"
echo "════════════════════════════════════════════════════════"
