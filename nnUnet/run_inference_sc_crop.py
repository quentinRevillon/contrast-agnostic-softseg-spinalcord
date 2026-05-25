"""
Run inference on a single subject using a nnUNet model trained with sc_crop preprocessing.

Pipeline:
  1. sc_crop detects the spinal cord bounding box (YOLO, no GT mask required)
  2. Image is cropped to the detected bbox
  3. nnUNet inference runs on the cropped volume (PyTorch checkpoint)
  4. Segmentation is reprojected into the original full-image space

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
import time
import tempfile

import numpy as np
import nibabel as nib
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from sc_crop import run as sc_crop_run


def get_parser():
    parser = argparse.ArgumentParser(description='nnUNet inference with sc_crop detection and full-space reprojection')
    parser.add_argument('-i', required=True, help='Input image. Example: sub-001_T2w.nii.gz')
    parser.add_argument('-o', required=True, help='Output segmentation. Example: sub-001_T2w_seg.nii.gz')
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


def add_suffix(fname, suffix):
    stem, ext = os.path.splitext(fname)
    if ext == '.gz':
        stem, ext2 = os.path.splitext(stem)
        ext = ext2 + ext
    return stem + suffix + ext


def restore_to_full_space(seg_cropped: nib.Nifti1Image, original_img: nib.Nifti1Image,
                          xmin, xmax, ymin, ymax, zmin, zmax) -> nib.Nifti1Image:
    """Paste cropped segmentation back into the full original image space."""
    full = np.zeros(original_img.shape[:3], dtype=np.uint8)
    seg_arr = np.asarray(seg_cropped.dataobj).astype(np.uint8)
    full[xmin:xmax + 1, ymin:ymax + 1, zmin:zmax + 1] = seg_arr
    return nib.Nifti1Image(full, original_img.affine, original_img.header)


def main():
    args = get_parser().parse_args()

    fname_in  = args.i
    fname_out = args.o

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── Step 1: reorient input to RPI ─────────────────────────────────────
        fname_rpi = os.path.join(tmpdir, 'input_rpi.nii.gz')
        shutil.copyfile(fname_in, fname_rpi)

        import subprocess
        result = subprocess.run(['sct_image', '-i', fname_rpi, '-header'],
                                capture_output=True, text=True)
        orig_orientation = subprocess.check_output(
            f"sct_image -i {fname_rpi} -header | grep -E 'qform_[xyz]' | "
            "awk '{printf \"%s\", substr($2, 1, 1)}'",
            shell=True
        ).decode('utf-8')

        if orig_orientation != 'RPI':
            os.system(f'sct_image -i {fname_rpi} -setorient RPI -o {fname_rpi}')

        original_img = nib.load(fname_rpi)

        # ── Step 2: sc_crop detection → crop ──────────────────────────────────
        print('Running sc_crop detection...')
        result = sc_crop_run(
            fname_rpi,
            padding_rl_mm=args.pad_rl,
            padding_ap_mm=args.pad_ap,
            padding_si_mm=(args.pad_si, args.pad_si),
        )
        xmin, xmax = result['xmin'], result['xmax']
        ymin, ymax = result['ymin'], result['ymax']
        zmin, zmax = result['zmin'], result['zmax']

        data = np.asarray(original_img.dataobj)
        affine = original_img.affine.copy()
        affine[:3, 3] = (original_img.affine @ np.array([xmin, ymin, zmin, 1.0]))[:3]
        cropped_img = nib.Nifti1Image(data[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1], affine)

        fname_crop = os.path.join(tmpdir, 'input_crop.nii.gz')
        nib.save(cropped_img, fname_crop)
        print(f'Cropped: {original_img.shape} → {cropped_img.shape}')

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

        pred_file = glob.glob(os.path.join(tmpdir_pred, '*.nii.gz'))[0]
        seg_crop = nib.load(pred_file)

        # ── Step 4: reproject to original full-image space ─────────────────────
        print('Reprojecting segmentation to original space...')
        seg_full = restore_to_full_space(seg_crop, original_img, xmin, xmax, ymin, ymax, zmin, zmax)

        # Binarize (keep SC class)
        seg_data = np.asarray(seg_full.dataobj)
        if args.pred_type == 'sc':
            seg_bin = (seg_data > 0).astype(np.uint8)
        else:
            seg_bin = (seg_data == 2).astype(np.uint8)

        seg_out = nib.Nifti1Image(seg_bin, seg_full.affine, seg_full.header)

        # Reorient back to original orientation
        if orig_orientation != 'RPI':
            fname_seg_rpi = os.path.join(tmpdir, 'seg_rpi.nii.gz')
            nib.save(seg_out, fname_seg_rpi)
            os.system(f'sct_image -i {fname_seg_rpi} -setorient {orig_orientation} -o {fname_seg_rpi}')
            seg_out = nib.load(fname_seg_rpi)

        nib.save(seg_out, fname_out)

    print(f'Segmentation saved → {fname_out}')


if __name__ == '__main__':
    main()
