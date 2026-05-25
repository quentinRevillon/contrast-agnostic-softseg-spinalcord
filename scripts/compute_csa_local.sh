#!/bin/bash
#
# Compute the CSA of a locally-trained model on the spine-generic test dataset.
# Identical to compute_csa.sh but uses run_inference_single_subject.py with a local nnUNet
# model folder instead of sct_deepseg (which requires a published release).
#
# To be used within evaluate_csa_local.sh.
#
# Author: Quentin Revillon (adapted from compute_csa.sh by Naga Karthik)
#

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

echo "Retrieved variables from the caller sct_run_batch:"
echo "PATH_DATA: ${PATH_DATA}"
echo "PATH_DATA_PROCESSED: ${PATH_DATA_PROCESSED}"
echo "PATH_RESULTS: ${PATH_RESULTS}"
echo "PATH_LOG: ${PATH_LOG}"
echo "PATH_QC: ${PATH_QC}"

# Variable passed by `sct_run_batch -script-args`
SUBJECT=$1
MODEL_VERSION=$2
PATH_NNUNET_SCRIPT=$3   # path to run_inference_single_subject.py
PATH_NNUNET_MODEL=$4    # path to the nnUNet model folder (contains fold_0/, plans.json, dataset.json)

echo "SUBJECT: ${SUBJECT}"
echo "MODEL_VERSION: ${MODEL_VERSION}"
echo "PATH_NNUNET_SCRIPT: ${PATH_NNUNET_SCRIPT}"
echo "PATH_NNUNET_MODEL: ${PATH_NNUNET_MODEL}"

# ------------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS  (identical to compute_csa.sh)
# ------------------------------------------------------------------------------

label_vertebrae(){
  local file="$1"
  local contrast="$2"

  FILESEG="${file}_softseg_bin"
  FILELABEL="${file}_discs"

  sct_label_utils -i ${FILESEG}.nii.gz -disc ${FILELABEL}.nii.gz -o ${FILESEG}_labeled.nii.gz
}

copy_gt_disc_labels(){
  local file="$1"
  local type="$2"
  local contrast="$3"

  if [[ $contrast == "T1w" ]] || [[ $contrast == "T2w" ]]; then
    file_name="${file%%_*}_${contrast}_label-discs_dlabel"
  elif [[ $contrast == "T2star" ]]; then
    file_name="${file%%_*}_${contrast}_label-discs_desc-warp_dlabel"
  elif [[ $contrast == "MTon" ]]; then
    file_name="${file%%_*}_flip-1_mt-on_MTS_label-discs_desc-warp_dlabel"
  elif [[ $contrast == "MToff" ]]; then
    file_name="${file%%_*}_flip-2_mt-off_MTS_label-discs_desc-warp_dlabel"
  elif [[ $contrast == "DWI" ]]; then
    file_name="${file%%_*}_rec-average_dwi_label-discs_desc-warp_dlabel"
  fi
  FILEDISCLABELS="${PATH_DATA}/derivatives/labels/${SUBJECT}/${type}/${file_name}.nii.gz"
  echo ""
  echo "Looking for manual disc labels: $FILEDISCLABELS"
  if [[ -e $FILEDISCLABELS ]]; then
      echo "Found! Copying ..."
      rsync -avzh $FILEDISCLABELS ${file}_discs.nii.gz
  else
      echo "File ${FILEDISCLABELS} does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: Manual Disc Labels ${FILEDISCLABELS} does not exist. Exiting."
      exit 1
  fi
}

copy_gt_softseg_bin(){
  local file="$1"
  local type="$2"
  FILESEG="${PATH_DATA}/derivatives/labels_softseg_bin/${SUBJECT}/${type}/${file}_desc-softseg_label-SC_seg.nii.gz"
  echo ""
  echo "Looking for manual segmentation: $FILESEG"
  if [[ -e $FILESEG ]]; then
      echo "Found! Copying ..."
      rsync -avzh $FILESEG ${file}_softseg_bin.nii.gz
  else
      echo "File ${FILESEG} does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: Manual Segmentation ${FILESEG} does not exist. Exiting."
      exit 1
  fi
}

# ------------------------------------------------------------------------------
# Segment spinal cord using local nnUNet model (no sct_deepseg release needed)
# ------------------------------------------------------------------------------
segment_sc(){
  local file="$1"
  local file_gt_vert_label="$2"
  local model_basename="$3"
  local contrast="$4"

  FILESEG="${file%%_*}_${contrast}_seg_sc-crop"

  start_time=$(date +%s)

  # Run SC segmentation: sc_crop detection → nnUNet → reproject to original space
  python ${PATH_NNUNET_SCRIPT} \
      -i ${file}.nii.gz \
      -o ${FILESEG}.nii.gz \
      -path-model ${PATH_NNUNET_MODEL} \
      -pred-type sc

  end_time=$(date +%s)
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # QC
  sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # Compute CSA at C2-C3
  sct_process_segmentation -i ${FILESEG}.nii.gz -vert 2:3 -vertfile ${file_gt_vert_label}_labeled.nii.gz \
      -o $PATH_RESULTS/csa_c2c3.csv -append 1
}

# ------------------------------------------------------------------------------
# SCRIPT STARTS HERE  (identical to compute_csa.sh from here)
# ------------------------------------------------------------------------------
start=`date +%s`

sct_check_dependencies -short

cd $PATH_DATA_PROCESSED

rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/* .
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/dwi/* .

contrasts="T1w T2w T2star flip-1_mt-on_MTS flip-2_mt-off_MTS rec-average_dwi"

for contrast in ${contrasts}; do

  if [[ $contrast == "rec-average_dwi" ]]; then
    type="dwi"
  else
    type="anat"
  fi

  cd ${PATH_DATA_PROCESSED}/${SUBJECT}/${type}

  file="${SUBJECT}_${contrast}"

  if [[ ! -e ${file}.nii.gz ]]; then
      echo "File ${file}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: File ${file}.nii.gz does not exist. Exiting."
      exit 1
  fi

  if [[ $contrast == "flip-1_mt-on_MTS" ]]; then
    contrast="MTon"
  elif [[ $contrast == "flip-2_mt-off_MTS" ]]; then
    contrast="MToff"
  elif [[ $contrast == "rec-average_dwi" ]]; then
    contrast="DWI"
  fi

  # Compute CSA of GT masks
  copy_gt_softseg_bin "${file}" "${type}"
  copy_gt_disc_labels "${file}" "${type}" "${contrast}"
  label_vertebrae ${file} 't2'

  FILEBIN="${file%%_*}_${contrast}_softseg_bin"
  if [[ "${file}_softseg_bin.nii.gz" != "${FILEBIN}.nii.gz" ]]; then
    mv ${file}_softseg_bin.nii.gz ${FILEBIN}.nii.gz
  fi

  sct_qc -i ${file}.nii.gz -s ${FILEBIN}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

  sct_process_segmentation -i ${FILEBIN}.nii.gz -vert 2:3 -vertfile ${file}_softseg_bin_labeled.nii.gz \
      -o $PATH_RESULTS/csa_c2c3.csv -append 1

  # Compute CSA of model predictions
  segment_sc ${file} "${file}_softseg_bin" ${MODEL_VERSION} ${contrast}

done

end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
