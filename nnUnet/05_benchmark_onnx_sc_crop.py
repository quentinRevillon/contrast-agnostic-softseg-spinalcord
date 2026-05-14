"""
Benchmark ONNX segmentation using the sc_crop detection pipeline on original test images.

Mirrors 04_evaluate_with_sc_crop.py (PyTorch) for a fair ONNX vs PyTorch comparison.
For each test image:
  1. sc_crop detection (conda env sc_crop) → bounding box in original space
  2. Crop + reorient to RPI (pure Python, no SCT required)
  3. nnUNet ONNX preprocessing (crop_to_nonzero → zscore → resample) + sliding-window inference
  4. Postprocess: resample back → pad to sc_crop shape → metrics

Metrics (same definition as 04_evaluate_with_sc_crop.py):
  - coverage:          |GT_crop| / |GT_total|  — fraction of SC voxels captured by sc_crop
  - dice_within_crop:  Dice(pred, GT_crop)      — model quality inside the crop
  - dice_global:       2 * |pred ∩ GT_crop| / (|pred| + |GT_total|)
                        penalises crops that miss SC voxels

Timing columns: sc_crop_time_s, preprocess_time_s, inference_time_s, total_time_s

Inputs  : conversion_dict.json (original image ↔ nnunet ID), nnunet_seg.onnx, plans.json
Output  : CSV with one row per test image

Usage:
    conda activate contrast_agnostic
    python nnUnet/05_benchmark_onnx_sc_crop.py \
        --dataset-folder /path/to/nnUNet_raw/Dataset1000_TempContrastAgnosticCropped \
        --model          /path/to/nnunet_seg.onnx \
        --plans          /path/to/plans.json \
        --output         results_onnx_sc_crop.csv \
        [--n 5]

Author: Quentin Revillon
"""

import argparse
import csv
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import onnxruntime as ort
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.transform import resize as sk_resize


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset-folder', required=True,
                        help='Path to nnUNet_raw dataset folder (contains conversion_dict.json)')
    parser.add_argument('--model',   required=True, help='Path to nnunet_seg.onnx')
    parser.add_argument('--plans',   required=True, help='Path to plans.json')
    parser.add_argument('--output',  required=True, help='Output CSV path')
    parser.add_argument('--pad-rl',  type=float, default=20.0, help='sc_crop padding left+right (mm)')
    parser.add_argument('--pad-ap',  type=float, default=30.0, help='sc_crop padding ant+post (mm)')
    parser.add_argument('--pad-si',  type=float, default=40.0, help='sc_crop padding sup+inf (mm)')
    parser.add_argument('--sc-crop-env', default='sc_crop', help='Conda env with sc_crop installed')
    parser.add_argument('--n', type=int, default=None, help='Limit to first N test images')
    return parser.parse_args()


# ── Mapping ───────────────────────────────────────────────────────────────────

def build_test_mapping(conversion_dict_path):
    """Return list of (orig_img, orig_label, nnunet_id) for test images only."""
    with open(conversion_dict_path) as f:
        d = json.load(f)

    id_to_orig_label = {}
    for orig, nnunet in d.items():
        if 'labelsTs' in nnunet:
            img_id = Path(nnunet).stem.replace('.nii', '')
            id_to_orig_label[img_id] = orig

    mapping = []
    for orig, nnunet in sorted(d.items()):
        if 'imagesTs' in nnunet:
            img_id = Path(nnunet).stem.replace('.nii', '').replace('_0000', '')
            orig_label = id_to_orig_label.get(img_id)
            assert orig_label is not None, f'No label for {nnunet}'
            mapping.append((orig, orig_label, img_id))
    return mapping


# ── sc_crop detection ─────────────────────────────────────────────────────────

def run_sc_crop(image_path, pad_rl, pad_ap, pad_si, sc_crop_env):
    """Run sc_crop and return (xmin, xmax, ymin, ymax, zmin, zmax)."""
    bbox_file = tempfile.mktemp(suffix='_bbox.txt')
    cmd = (f"conda run -n {sc_crop_env} sc_crop -i {image_path} -o {bbox_file} "
           f"--padding-rl '{pad_rl} {pad_rl}' "
           f"--padding-ap '{pad_ap} {pad_ap}' "
           f"--padding-si '{pad_si} {pad_si}'")
    assert subprocess.call(cmd, shell=True) == 0, f'sc_crop failed: {cmd}'
    with open(bbox_file) as f:
        for line in f:
            if not line.startswith('#'):
                bbox = tuple(int(v) for v in line.split())
                os.remove(bbox_file)
                return bbox
    raise ValueError(f'No bbox in {bbox_file}')


# ── Reorientation ─────────────────────────────────────────────────────────────

def reorient_to_rpi(img):
    target  = axcodes2ornt(('R', 'P', 'I'))
    current = io_orientation(img.affine)
    return img.as_reoriented(ornt_transform(current, target))


def crop_nifti(nib_img, xmin, xmax, ymin, ymax, zmin, zmax):
    """Crop a NIfTI image to the given inclusive voxel indices and update affine."""
    data    = nib_img.get_fdata(dtype=np.float32)
    cropped = data[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]
    affine  = nib_img.affine.copy()
    affine[:3, 3] = (nib_img.affine @ np.array([xmin, ymin, zmin, 1.0]))[:3]
    return nib.Nifti1Image(cropped, affine)


# ── nnUNet preprocessing ──────────────────────────────────────────────────────

def crop_to_nonzero(data):
    mask = binary_fill_holes(data != 0)
    bbox = []
    for ax in range(3):
        proj    = np.any(mask, axis=tuple(i for i in range(3) if i != ax))
        indices = np.where(proj)[0]
        bbox.append([int(indices[0]), int(indices[-1]) + 1] if len(indices) > 0
                    else [0, data.shape[ax]])
    return data[tuple(slice(b[0], b[1]) for b in bbox)], bbox


def zscore_normalize(data):
    return ((data - data.mean()) / max(float(data.std()), 1e-8)).astype(np.float32)


def resample(data, src_shape, dst_shape, order):
    if tuple(src_shape) == tuple(dst_shape):
        return data.astype(np.float32)
    return sk_resize(data.astype(float), dst_shape,
                     order=order, mode='edge', anti_aliasing=False).astype(np.float32)


def compute_new_shape(src_shape, src_spacing, tgt_spacing):
    return tuple(int(round(s * o / t))
                 for s, o, t in zip(src_shape, src_spacing, tgt_spacing))


# ── Sliding window ────────────────────────────────────────────────────────────

def make_gaussian_map(patch_size):
    tmp = np.zeros(patch_size, dtype=np.float64)
    tmp[tuple(i // 2 for i in patch_size)] = 1
    g = gaussian_filter(tmp, [i / 8 for i in patch_size], mode='constant', cval=0)
    g /= g.max()
    g[g == 0] = g[g > 0].min()
    return g.astype(np.float32)


def sliding_steps(img_s, patch_s, step=0.5):
    if img_s <= patch_s:
        return [0]
    n   = int(np.ceil((img_s - patch_s) / (patch_s * step))) + 1
    mv  = img_s - patch_s
    act = mv / (n - 1) if n > 1 else 99999
    return [int(np.round(act * i)) for i in range(n)]


def sliding_window_inference(data, session, patch_size, in_name, out_name):
    D, H, W    = data.shape
    pd, ph, pw = patch_size

    pad_d = max(0, pd - D); d0 = pad_d // 2; d1 = pad_d - d0
    pad_h = max(0, ph - H); h0 = pad_h // 2; h1 = pad_h - h0
    pad_w = max(0, pw - W); w0 = pad_w // 2; w1 = pad_w - w0
    if pad_d or pad_h or pad_w:
        data = np.pad(data, ((d0, d1), (h0, h1), (w0, w1)), mode='constant')
    Dp, Hp, Wp = data.shape

    gauss = make_gaussian_map(patch_size)
    accum = np.zeros((Dp, Hp, Wp), np.float32)
    wmap  = np.zeros((Dp, Hp, Wp), np.float32)

    for dz in sliding_steps(Dp, pd):
        for dy in sliding_steps(Hp, ph):
            for dx in sliding_steps(Wp, pw):
                patch  = data[dz:dz+pd, dy:dy+ph, dx:dx+pw][np.newaxis, np.newaxis]
                logits = session.run([out_name], {in_name: patch})[0][0]
                exp    = np.exp(logits - logits.max(axis=0, keepdims=True))
                prob1  = exp[1] / exp.sum(axis=0)
                accum[dz:dz+pd, dy:dy+ph, dx:dx+pw] += prob1 * gauss
                wmap [dz:dz+pd, dy:dy+ph, dx:dx+pw] += gauss

    return (accum / wmap)[d0:d0+D, h0:h0+H, w0:w0+W]


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred_arr, gt_crop_arr, gt_total_count):
    pred_bin     = pred_arr > 0.5
    gt_bin       = gt_crop_arr > 0.5
    gt_crop_count = int(gt_bin.sum())
    coverage     = gt_crop_count / gt_total_count if gt_total_count > 0 else 1.0
    intersection = int((pred_bin & gt_bin).sum())
    pred_count   = int(pred_bin.sum())
    denom_within = pred_count + gt_crop_count
    dice_within  = 2 * intersection / denom_within if denom_within > 0 else 1.0
    denom_global = pred_count + gt_total_count
    dice_global  = 2 * intersection / denom_global if denom_global > 0 else 1.0
    return coverage, dice_within, dice_global


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    plans      = json.load(open(args.plans))
    cfg        = plans['configurations']['3d_fullres']
    target_sp  = cfg['spacing']
    patch_size = cfg['patch_size']

    session  = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    in_name  = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    dataset_folder = Path(args.dataset_folder)
    test_mapping   = build_test_mapping(dataset_folder / 'conversion_dict.json')
    if args.n:
        test_mapping = test_mapping[:args.n]
    print(f'Running ONNX sc_crop pipeline on {len(test_mapping)} images...')
    print(f'Target spacing : {target_sp}  patch_size : {patch_size}')

    rows = []
    for orig_img, orig_label, nnunet_id in test_mapping:
        t0 = time.perf_counter()

        # ── Phase 1: sc_crop detection + crop + RPI ───────────────────────────
        xmin, xmax, ymin, ymax, zmin, zmax = run_sc_crop(
            orig_img, args.pad_rl, args.pad_ap, args.pad_si, args.sc_crop_env)

        lbl_nib = nib.load(orig_label)
        gt_total_count = int((np.asarray(lbl_nib.dataobj) > 0.5).sum())

        img_nib     = nib.load(orig_img)
        img_crop_rpi = reorient_to_rpi(crop_nifti(img_nib, xmin, xmax, ymin, ymax, zmin, zmax))
        lbl_crop_rpi = reorient_to_rpi(crop_nifti(lbl_nib, xmin, xmax, ymin, ymax, zmin, zmax))

        t_sc = time.perf_counter() - t0

        # ── Phase 2: nnUNet preprocessing ─────────────────────────────────────
        t_pre0 = time.perf_counter()

        data    = img_crop_rpi.get_fdata().transpose((2, 1, 0)).astype(np.float32)
        spacing = [float(v) for v in img_crop_rpi.header.get_zooms()[:3][::-1]]
        shape_rpi = data.shape

        data_crop, nnunet_bbox = crop_to_nonzero(data)
        shape_crop = data_crop.shape
        data_norm  = zscore_normalize(data_crop)
        shape_rs   = compute_new_shape(shape_crop, spacing, target_sp)
        data_rs    = resample(data_norm, shape_crop, shape_rs, order=3)

        t_pre = time.perf_counter() - t_pre0

        # ── Phase 3: ONNX inference ────────────────────────────────────────────
        t_inf0 = time.perf_counter()
        prob   = sliding_window_inference(data_rs, session, patch_size, in_name, out_name)
        t_inf  = time.perf_counter() - t_inf0

        # ── Postprocess: resample back → pad to sc_crop+RPI shape ─────────────
        pred_crop = (resample((prob > 0.5).astype(np.float32), shape_rs, shape_crop, order=0) > 0.5).astype(np.uint8)
        pred_zyx  = np.zeros(shape_rpi, np.uint8)
        pred_zyx[tuple(slice(b[0], b[1]) for b in nnunet_bbox)] = pred_crop
        pred_arr  = pred_zyx.transpose((2, 1, 0))  # [z,y,x] → [x,y,z] = RPI

        gt_crop_arr = lbl_crop_rpi.get_fdata()
        coverage, dice_within, dice_global = compute_metrics(pred_arr, gt_crop_arr, gt_total_count)

        t_total = time.perf_counter() - t0
        rows.append({
            'image_id':          nnunet_id,
            'coverage':          round(coverage,    6),
            'dice_within_crop':  round(dice_within, 6),
            'dice_global':       round(dice_global, 6),
            'sc_crop_time_s':    round(t_sc,   2),
            'preprocess_time_s': round(t_pre,  3),
            'inference_time_s':  round(t_inf,  3),
            'total_time_s':      round(t_total, 3),
        })
        print(f'  {nnunet_id}  cov={coverage:.3f}  dice={dice_within:.4f}  '
              f'inf={t_inf:.1f}s  sc_crop={t_sc:.1f}s  total={t_total:.1f}s')

    # ── Write CSV ─────────────────────────────────────────────────────────────
    fieldnames = ['image_id', 'coverage', 'dice_within_crop', 'dice_global',
                  'sc_crop_time_s', 'preprocess_time_s', 'inference_time_s', 'total_time_s']
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    coverages  = [r['coverage']         for r in rows]
    dices_w    = [r['dice_within_crop']  for r in rows]
    dices_g    = [r['dice_global']       for r in rows]
    infs       = [r['inference_time_s']  for r in rows]
    totals     = [r['total_time_s']      for r in rows]

    print()
    print(f'N images         : {len(rows)}')
    print(f'Coverage         : {np.mean(coverages):.4f} ± {np.std(coverages):.4f}')
    print(f'Dice within crop : {np.mean(dices_w):.4f} ± {np.std(dices_w):.4f}')
    print(f'Dice global      : {np.mean(dices_g):.4f} ± {np.std(dices_g):.4f}')
    print(f'Inference time   : {np.mean(infs):.2f}s ± {np.std(infs):.2f}s')
    print(f'Total time       : {np.mean(totals):.2f}s ± {np.std(totals):.2f}s')
    print(f'CSV saved        : {args.output}')


if __name__ == '__main__':
    main()
