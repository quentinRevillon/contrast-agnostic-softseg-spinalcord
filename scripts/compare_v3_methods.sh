#!/bin/bash
# Compare sct_deepseg spinalcord vs infer_pt on a single image.
# Dice between the two predictions should be ~1.0 if equivalent.

IMAGE="/home/quentinr/datasets_contrast_agnostic_retraining/site_006/sub-mon111/anat/sub-mon111_acq-sag_run-01_T2w.nii.gz"
OUT_SCT_CPU="/tmp/seg_v3_sct_cpu.nii.gz"
OUT_SCT_GPU="/tmp/seg_v3_sct_gpu.nii.gz"

echo "--- sct_deepseg spinalcord (CPU, install originale) ---"
/home/quentinr/spinalcordtoolbox/bin/sct_deepseg spinalcord -i ${IMAGE} -o ${OUT_SCT_CPU}

echo "--- sct_deepseg spinalcord (GPU, install GPU) ---"
CUDA_VISIBLE_DEVICES=0 SCT_USE_GPU=1 \
    /home/quentinr/spinalcordtoolbox-gpu/bin/sct_deepseg spinalcord -i ${IMAGE} -o ${OUT_SCT_GPU}

echo "--- Dice comparison (CPU vs GPU) ---"
python -c "
import nibabel as nib, numpy as np
a = np.asarray(nib.load('${OUT_SCT_CPU}').dataobj).astype(bool)
b = np.asarray(nib.load('${OUT_SCT_GPU}').dataobj).astype(bool)
dice = 2 * (a & b).sum() / (a.sum() + b.sum())
print(f'Dice CPU vs GPU : {dice:.4f}')
print(f'Voxels CPU : {a.sum()}')
print(f'Voxels GPU : {b.sum()}')
"
