#!/bin/bash
# Compare sct_deepseg spinalcord vs infer_pt on a single image.
# Dice between the two predictions should be ~1.0 if equivalent.

IMAGE="/home/quentinr/datasets_contrast_agnostic_retraining/site_006/sub-mon111/anat/sub-mon111_acq-sag_run-01_T2w.nii.gz"
V3_CHECKPOINT="/home/quentinr/spinalcordtoolbox/data/deepseg_models/model_seg_sc_contrast_agnostic_nnunet/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
OUT_SCT="/tmp/seg_v3_sct.nii.gz"
OUT_PT="/tmp/seg_v3_pt.nii.gz"

export PATH="/home/quentinr/spinalcordtoolbox/bin:${PATH}"

echo "--- sct_deepseg spinalcord ---"
sct_deepseg spinalcord -i ${IMAGE} -o ${OUT_SCT}

echo "--- infer_pt (GPU) ---"
python -c "
import nibabel as nib
from nnunet_onnx.inference import infer_pt
seg = infer_pt(nib.load('${IMAGE}'), '${V3_CHECKPOINT}', device='cuda')
nib.save(seg, '${OUT_PT}')
print('Done.')
"

echo "--- Dice comparison ---"
python -c "
import nibabel as nib, numpy as np
a = np.asarray(nib.load('${OUT_SCT}').dataobj).astype(bool)
b = np.asarray(nib.load('${OUT_PT}').dataobj).astype(bool)
dice = 2 * (a & b).sum() / (a.sum() + b.sum())
print(f'Dice sct_deepseg vs infer_pt : {dice:.4f}')
print(f'Voxels sct_deepseg : {a.sum()}')
print(f'Voxels infer_pt    : {b.sum()}')
"
