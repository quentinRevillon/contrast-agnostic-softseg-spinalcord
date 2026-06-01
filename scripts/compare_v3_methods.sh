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

echo "--- SCT inference pipeline (GPU) ---"
python -c "
import sys, tempfile, shutil, torch
sys.path.insert(0, '/home/quentinr/spinalcordtoolbox')
from spinalcordtoolbox.deepseg.nnunet import create_nnunet_from_plans
from spinalcordtoolbox.deepseg.inference import segment_nnunet

V3_MODEL_DIR = '/home/quentinr/spinalcordtoolbox/data/deepseg_models/model_seg_sc_contrast_agnostic_nnunet'
device = torch.device('cuda')
predictor = create_nnunet_from_plans(V3_MODEL_DIR, device)
tmpdir = tempfile.mkdtemp()
fnames_out, _ = segment_nnunet('${IMAGE}', tmpdir, predictor, device)
shutil.copy(fnames_out[0], '${OUT_PT}')
shutil.rmtree(tmpdir)
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
