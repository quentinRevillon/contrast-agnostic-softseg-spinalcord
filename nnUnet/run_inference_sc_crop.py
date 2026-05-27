"""
Run inference on a single subject using a nnUNet model trained with sc_crop preprocessing.

Pipeline:
  1. Reorient input to RPI
  2. sc_crop detects the spinal cord bounding box and crops the volume
  3. nnUNet inference runs on the cropped volume (PyTorch checkpoint)
  4. Segmentation is restored to the original full-image space with sc_crop.uncrop()
  5. Reorient output back to the original orientation

This is the training-time evaluation equivalent of run_inference.py (ONNX).
Use this script during development (before ONNX export) for CSA/Dice evaluation.

Usage:
    python run_inference_sc_crop.py \
        -i sub-001_T2w.nii.gz \
        -o sub-001_T2w_seg.nii.gz \
        -path-model /path/to/nnUNetTrainer__nnUNetPlans__3d_fullres \
        -pred-type sc

Author: Quentin Revillon (adapted from run_inference_single_subject.py by Jan Valosek / Naga Karthik)
"""

import os
import shutil
import argparse
import glob
import subprocess
import time
import tempfile

import numpy as np
import nibabel as nib
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from sc_crop import detect, crop, uncrop


def get_parser():
    parser = argparse.ArgumentParser(description='nnUNet inference with sc_crop detection and full-space restoration')
    parser.add_argument('-i',   required=True, help='Input image. Example: sub-001_T2w.nii.gz')
    parser.add_argument('-o',   required=True, help='Output segmentation. Example: sub-001_T2w_seg.nii.gz')
    parser.add_argument('-path-model', required=True, type=str,
                        help='Path to nnUNet model folder (contains fold_0/, plans.json, dataset.json)')
    parser.add_argument('-pred-type', choices=['sc', 'lesion'], default='sc',
                        help='Type of prediction: sc (spinal cord) or lesion. Default: sc')
    parser.add_argument('-use-gpu', action='store_true', default=False,
                        help='Use GPU for inference. Default: False (CPU)')
    parser.add_argument('-use-best-checkpoint', action='store_true', default=False,
                        help='Use checkpoint_best.pth instead of checkpoint_final.pth')
    parser.add_argument('-tile-step-size', default=0.5, type=float,
                        help='Tile step size for sliding window inference. Default: 0.5')
    parser.add_argument('--pad-rl', type=float, default=10.0, help='sc_crop R-L padding mm (default: 10)')
    parser.add_argument('--pad-ap', type=float, default=15.0, help='sc_crop A-P padding mm (default: 15)')
    parser.add_argument('--pad-si', type=float, default=30.0, help='sc_crop S-I padding mm (default: 30)')
    return parser


def get_orientation(fname):
    return subprocess.check_output(
        f"sct_image -i {fname} -header | grep -E 'qform_[xyz]' | "
        "awk '{printf \"%s\", substr($2, 1, 1)}'",
        shell=True,
    ).decode('utf-8')


def main():
    args = get_parser().parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── Step 1: reorient input to RPI ─────────────────────────────────────
        fname_rpi = os.path.join(tmpdir, 'input_rpi.nii.gz')
        shutil.copyfile(args.i, fname_rpi)

        orig_orientation = get_orientation(fname_rpi)
        if orig_orientation != 'RPI':
            os.system(f'sct_image -i {fname_rpi} -setorient RPI -o {fname_rpi}')

        # ── Step 2: sc_crop detect + crop ─────────────────────────────────────
        print('Running sc_crop detection...')
        bbox     = detect(fname_rpi, pad_left=args.pad_rl, pad_right=args.pad_rl,
                          pad_anterior=args.pad_ap, pad_posterior=args.pad_ap,
                          pad_superior=args.pad_si, pad_inferior=args.pad_si)
        img_rpi  = nib.load(fname_rpi)
        crop_img = crop(img_rpi, bbox)
        print(f'Cropped: {img_rpi.shape} → {crop_img.shape}')

        fname_crop = os.path.join(tmpdir, 'input_crop.nii.gz')
        nib.save(crop_img, fname_crop)

        # ── Step 3: nnUNet inference on cropped volume ─────────────────────────
        folds_avail = [int(f.split('_')[-1]) for f in os.listdir(args.path_model) if f.startswith('fold_')]
        print(f'Running nnUNet inference on folds {folds_avail}...')

        tmpdir_pred = os.path.join(tmpdir, 'nnunet_pred')
        os.makedirs(tmpdir_pred)

        start = time.time()
        predictor = nnUNetPredictor(
            tile_step_size=args.tile_step_size,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=args.use_gpu,
            device=torch.device('cuda') if args.use_gpu else torch.device('cpu'),
            verbose=False,
            verbose_preprocessing=False,
        )
        predictor.initialize_from_trained_model_folder(
            args.path_model,
            use_folds=folds_avail,
            checkpoint_name='checkpoint_best.pth' if args.use_best_checkpoint else 'checkpoint_final.pth',
        )
        predictor.predict_from_files(
            list_of_lists_or_source_folder=[[fname_crop]],
            output_folder_or_list_of_truncated_output_files=tmpdir_pred,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=4,
            num_processes_segmentation_export=4,
        )
        print(f'Inference done in {time.time() - start:.1f}s')

        # ── Step 4: restore to full image space ───────────────────────────────
        pred_file = glob.glob(os.path.join(tmpdir_pred, '*.nii.gz'))[0]
        seg_crop  = nib.load(pred_file)
        seg_full  = uncrop(seg_crop, bbox)

        # Binarize
        seg_data = np.asarray(seg_full.dataobj)
        seg_bin  = (seg_data > 0).astype(np.uint8) if args.pred_type == 'sc' else (seg_data == 2).astype(np.uint8)
        seg_out  = nib.Nifti1Image(seg_bin, seg_full.affine, seg_full.header)

        # ── Step 5: reorient back to original orientation ─────────────────────
        fname_seg_rpi = os.path.join(tmpdir, 'seg_rpi.nii.gz')
        nib.save(seg_out, fname_seg_rpi)
        if orig_orientation != 'RPI':
            os.system(f'sct_image -i {fname_seg_rpi} -setorient {orig_orientation} -o {fname_seg_rpi}')

        shutil.copyfile(fname_seg_rpi, args.o)

    print(f'Segmentation saved → {args.o}')


if __name__ == '__main__':
    main()
