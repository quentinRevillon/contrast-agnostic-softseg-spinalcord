"""
Evaluate nnUNet segmentation model using automatic SC crop (sc_crop) on the original test images.

For each test image, applies sc_crop (detection-based bounding box) to crop the original image,
runs nnUNet inference, and computes three metrics:

  - coverage:        |GT_crop| / |GT_total|  — fraction of SC voxels captured by the crop
  - dice_within_crop: standard Dice computed inside the cropped region (what the model sees)
  - dice_global:     2 * |pred ∩ GT_crop| / (|pred_crop| + |GT_total|)
                     penalises crops that miss SC voxels — false negatives include all SC
                     outside the crop, even if segmentation inside is perfect

Pipeline (three phases):
  Phase 1 — sc_crop on all images → cropped images in <output-dir>/crops_tmp/
             GT crops saved to <gt-dir>/
  Phase 2 — nnUNet batch inference on crops_tmp/ → predictions in <pred-dir>/
  Phase 3 — compute metrics, write CSV, run nnUNetv2_evaluate_folder

Predictions are saved to <pred-dir>/ and matching GT crops to <gt-dir>/, both in sc_crop+RPI
space, so that nnUNetv2_evaluate_folder can generate summary.json against the GT crops.

Requires:
  - nnUNet conda env (contrast_agnostic) active
  - sc_crop in a separate conda env (--sc-crop-env, default: sc_crop)
  - SCT tools (sct_crop_image, sct_image) in PATH

Usage:
    python 04_evaluate_with_sc_crop.py
        --dataset-folder /path/to/nnUNet_raw/Dataset1000_TempContrastAgnosticCropped
        --model-folder /path/to/nnUNet_results/.../nnUNetTrainer__nnUNetPlans__3d_fullres
        --output-dir /path/to/output
        [--pred-dir /path/to/pred]        default: <model-folder>/test_sc_crop
        [--gt-dir /path/to/gt]            default: <model-folder>/labelsTs_sc_crop
        [--pad-left 20] [--pad-right 20]
        [--pad-anterior 30] [--pad-posterior 30]
        [--pad-superior 40] [--pad-inferior 40]
        [--sc-crop-env sc_crop]
        [--use-gpu]

Output:
    <output-dir>/metrics_sc_crop.csv  — coverage, dice_within_crop, dice_global per image
    <pred-dir>/TempContrastAgnosticCropped_NNN.nii.gz  — nnUNet predictions in sc_crop+RPI space
    <gt-dir>/TempContrastAgnosticCropped_NNN.nii.gz    — GT crops in sc_crop+RPI space
    <pred-dir>/summary.json  — from nnUNetv2_evaluate_folder
    Prints mean ± std summary to stdout

Author: Quentin Revillon
"""

import os
import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import tqdm
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate nnUNet model with sc_crop preprocessing on test images')
    parser.add_argument('--dataset-folder', required=True,
                        help='Path to the nnUNet raw dataset folder (e.g. .../Dataset1000_TempContrastAgnosticCropped)')
    parser.add_argument('--model-folder', required=True,
                        help='Path to the nnUNet results trainer folder (e.g. .../nnUNetTrainer__nnUNetPlans__3d_fullres)')
    parser.add_argument('--output-dir', required=True, help='Folder to save the results CSV')
    parser.add_argument('--pred-dir', default=None,
                        help='Folder for prediction .nii.gz files (default: <model-folder>/test_sc_crop)')
    parser.add_argument('--gt-dir', default=None,
                        help='Folder for GT crop .nii.gz files (default: <model-folder>/labelsTs_sc_crop)')
    parser.add_argument('--pad-left',      type=float, default=20.0)
    parser.add_argument('--pad-right',     type=float, default=20.0)
    parser.add_argument('--pad-anterior',  type=float, default=30.0)
    parser.add_argument('--pad-posterior', type=float, default=30.0)
    parser.add_argument('--pad-superior',  type=float, default=40.0)
    parser.add_argument('--pad-inferior',  type=float, default=40.0)
    parser.add_argument('--sc-crop-env', default='sc_crop')
    parser.add_argument('--use-gpu', action='store_true', default=False)
    return parser.parse_args()


def build_test_mapping(conversion_dict_path):
    """Return {orig_image_path: (orig_label_path, nnunet_id)} for test images only."""
    with open(conversion_dict_path) as f:
        d = json.load(f)

    id_to_orig_label = {}
    for orig, nnunet in d.items():
        if 'labelsTs' in nnunet:
            img_id = Path(nnunet).stem.replace('.nii', '')
            id_to_orig_label[img_id] = orig

    mapping = {}
    for orig, nnunet in d.items():
        if 'imagesTs' in nnunet:
            img_id = Path(nnunet).stem.replace('.nii', '').replace('_0000', '')
            orig_label = id_to_orig_label.get(img_id)
            assert orig_label is not None, f"No label found for {nnunet} (id={img_id})"
            mapping[orig] = (orig_label, img_id)

    return mapping


def parse_bbox_txt(bbox_file):
    """Parse sc_crop bbox.txt and return (xmin, xmax, ymin, ymax, zmin, zmax)."""
    with open(bbox_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            return tuple(int(p) for p in parts)
    raise ValueError(f"No bbox data found in {bbox_file}")


def run_sc_crop(image_path, output_dir, pad_left, pad_right, pad_anterior, pad_posterior,
                pad_superior, pad_inferior, sc_crop_env):
    """Run sc_crop CLI. Returns path to bbox.txt, or None if detection failed."""
    stem = Path(image_path).name.replace('.nii.gz', '').replace('.nii', '')
    bbox_file = os.path.join(output_dir, f"{stem}_bbox.txt")
    cmd = (f"conda run -n {sc_crop_env} sc_crop -i {image_path} -o {bbox_file} "
           f"--padding-rl '{pad_left} {pad_right}' "
           f"--padding-ap '{pad_anterior} {pad_posterior}' "
           f"--padding-si '{pad_superior} {pad_inferior}'")
    ret = os.system(cmd)
    return bbox_file if ret == 0 else None


def crop_and_reorient_rpi(image_path, xmin, xmax, ymin, ymax, zmin, zmax, output_path):
    """Crop image to bbox (inclusive voxel indices) and reorient to RPI."""
    native_tmp = str(output_path).replace('.nii.gz', '_native.nii.gz')
    crop_cmd = (f"sct_crop_image -i {image_path} "
                f"-xmin {xmin} -xmax {xmax} -ymin {ymin} -ymax {ymax} -zmin {zmin} -zmax {zmax} "
                f"-o {native_tmp}")
    assert os.system(crop_cmd) == 0, f"sct_crop_image failed on {image_path}"
    assert os.system(f"sct_image -i {native_tmp} -setorient RPI -o {output_path}") == 0
    os.remove(native_tmp)


def compute_metrics(pred_arr, gt_crop_arr, gt_total_count):
    """Compute coverage, dice_within_crop, and dice_global."""
    pred_bin = pred_arr > 0.5
    gt_bin   = gt_crop_arr > 0.5

    gt_crop_count = int(gt_bin.sum())
    coverage = gt_crop_count / gt_total_count if gt_total_count > 0 else 1.0

    intersection = int((pred_bin & gt_bin).sum())
    pred_count   = int(pred_bin.sum())

    denom_within = pred_count + gt_crop_count
    dice_within  = 2 * intersection / denom_within if denom_within > 0 else 1.0

    denom_global = pred_count + gt_total_count
    dice_global  = 2 * intersection / denom_global if denom_global > 0 else 1.0

    return coverage, dice_within, dice_global


def main():
    args = parse_args()

    output_dir   = Path(args.output_dir)
    model_folder = Path(args.model_folder)
    pred_dir = Path(args.pred_dir) if args.pred_dir else model_folder / "test_sc_crop"
    gt_dir   = Path(args.gt_dir)   if args.gt_dir   else model_folder / "labelsTs_sc_crop"
    crops_dir = output_dir / "crops_tmp"

    for d in (output_dir, pred_dir, gt_dir, crops_dir):
        d.mkdir(parents=True, exist_ok=True)

    conversion_dict_path = Path(args.dataset_folder) / 'conversion_dict.json'
    test_mapping = build_test_mapping(conversion_dict_path)
    print(f"Found {len(test_mapping)} test images")
    print(f"Predictions → {pred_dir}")
    print(f"GT crops    → {gt_dir}")

    # ── Phase 1 : sc_crop + crop all images ──────────────────────────────────
    print("\n── Phase 1 : sc_crop + crop ──")
    failed     = set()   # nnunet_ids for which sc_crop failed
    gt_counts  = {}      # nnunet_id → total SC voxel count in original image
    bbox_tmpdir = tempfile.mkdtemp(prefix="sc_crop_bbox_")

    for orig_img, (orig_label, nnunet_id) in tqdm.tqdm(test_mapping.items(), desc="sc_crop"):
        out_stem = f"TempContrastAgnosticCropped_{nnunet_id}"
        gt_counts[nnunet_id] = int((np.asarray(nib.load(orig_label).dataobj) > 0.5).sum())

        bbox_file = run_sc_crop(orig_img, bbox_tmpdir,
                                args.pad_left, args.pad_right,
                                args.pad_anterior, args.pad_posterior,
                                args.pad_superior, args.pad_inferior,
                                args.sc_crop_env)
        if bbox_file is None:
            print(f"  [SKIP] sc_crop failed: {orig_img}")
            failed.add(nnunet_id)
            continue

        xmin, xmax, ymin, ymax, zmin, zmax = parse_bbox_txt(bbox_file)
        crop_and_reorient_rpi(orig_img,   xmin, xmax, ymin, ymax, zmin, zmax,
                              crops_dir / f"{out_stem}_0000.nii.gz")
        crop_and_reorient_rpi(orig_label, xmin, xmax, ymin, ymax, zmin, zmax,
                              gt_dir / f"{out_stem}.nii.gz")

    shutil.rmtree(bbox_tmpdir)
    print(f"Phase 1 done — {len(test_mapping) - len(failed)} crops, {len(failed)} failed")

    # ── Phase 2 : nnUNet batch inference ─────────────────────────────────────
    print("\n── Phase 2 : nnUNet inference ──")
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    folds = sorted(int(f.split('_')[-1]) for f in os.listdir(args.model_folder) if f.startswith('fold_'))
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=args.use_gpu,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        args.model_folder,
        use_folds=folds,
        checkpoint_name='checkpoint_final.pth',
    )
    print(f"Model loaded (folds={folds}, device={device})")

    predictor.predict_from_files(
        list_of_lists_or_source_folder=str(crops_dir),
        output_folder_or_list_of_truncated_output_files=str(pred_dir),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=3,
        num_processes_segmentation_export=3,
    )

    shutil.rmtree(crops_dir)
    print("Phase 2 done")

    # ── Phase 3 : metrics ─────────────────────────────────────────────────────
    print("\n── Phase 3 : metrics ──")
    results = []
    for orig_img, (orig_label, nnunet_id) in test_mapping.items():
        out_name = f"TempContrastAgnosticCropped_{nnunet_id}.nii.gz"
        if nnunet_id in failed:
            results.append({'image': orig_img, 'coverage': 0.0, 'dice_within_crop': 0.0,
                             'dice_global': 0.0, 'sc_crop_failed': True})
            continue
        pred_arr    = np.asarray(nib.load(pred_dir / out_name).dataobj)
        gt_crop_arr = np.asarray(nib.load(gt_dir   / out_name).dataobj)
        coverage, dice_within, dice_global = compute_metrics(pred_arr, gt_crop_arr, gt_counts[nnunet_id])
        results.append({'image': orig_img, 'coverage': coverage,
                        'dice_within_crop': dice_within, 'dice_global': dice_global,
                        'sc_crop_failed': False})

    csv_path = output_dir / 'metrics_sc_crop.csv'
    fieldnames = ['image', 'coverage', 'dice_within_crop', 'dice_global', 'sc_crop_failed']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    n_failed = len(failed)
    n_ok = len(results) - n_failed
    if n_failed:
        print(f"Warning: sc_crop failed on {n_failed}/{len(results)} images (recorded as 0)")
    for metric in ['coverage', 'dice_within_crop', 'dice_global']:
        vals = [r[metric] for r in results]
        print(f"  {metric} (N={n_ok} ok + {n_failed} failed=0): {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"\nResults saved to {csv_path}")

    # nnUNetv2_evaluate_folder → summary.json
    print("\nRunning nnUNetv2_evaluate_folder …")
    subprocess.run([
        "nnUNetv2_evaluate_folder",
        str(gt_dir),
        str(pred_dir),
        "-djfile", str(Path(args.dataset_folder) / "dataset.json"),
        "-pfile",  str(model_folder / "plans.json"),
    ], check=True)
    print(f"summary.json saved to {pred_dir}/summary.json")


if __name__ == '__main__':
    main()
