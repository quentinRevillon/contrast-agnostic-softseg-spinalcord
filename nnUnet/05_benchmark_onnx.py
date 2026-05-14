"""
Run ONNX inference on the GT-crop test images (imagesTs) and record Dice + timing.

NOTE: imagesTs are cropped using the GT spinal cord bounding box (not sc_crop detection).
      This script benchmarks ONNX on the same crops used to train/test nnUNet.
      For the sc_crop detection pipeline, use 05_benchmark_onnx_sc_crop.py instead.

Inputs  : imagesTs/ (GT-bbox cropped + RPI), labelsTs/, nnunet_seg.onnx, plans.json
Output  : CSV with columns: image_id, dice, inference_time_s, preprocess_time_s, total_time_s

Usage:
    conda activate contrast_agnostic
    python nnUnet/05_benchmark_onnx.py \
        --images-dir  /path/to/imagesTs \
        --labels-dir  /path/to/labelsTs \
        --model       /path/to/nnunet_seg.onnx \
        --plans       /path/to/plans.json \
        --output      results_onnx_gt_crop.csv

Author: Quentin Revillon
"""

import argparse
import csv
import glob
import json
import time

import nibabel as nib
import numpy as np
import onnxruntime as ort
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.transform import resize as sk_resize


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--labels-dir', required=True)
    parser.add_argument('--model',   required=True)
    parser.add_argument('--plans',   required=True)
    parser.add_argument('--output',  required=True, help='Output CSV path')
    parser.add_argument('--n',       type=int, default=None, help='Limit to first N images')
    return parser.parse_args()


# ── Preprocessing (matches nnUNet DefaultPreprocessor exactly) ────────────────

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
    mean, std = data.mean(), data.std()
    return ((data - mean) / max(float(std), 1e-8)).astype(np.float32)


def resample(data, src_shape, dst_shape, order):
    if tuple(src_shape) == tuple(dst_shape):
        return data.astype(np.float32)
    return sk_resize(data.astype(float), dst_shape,
                     order=order, mode='edge', anti_aliasing=False).astype(np.float32)


def target_shape(src_shape, src_spacing, tgt_spacing):
    return tuple(int(round(s * o / t))
                 for s, o, t in zip(src_shape, src_spacing, tgt_spacing))


# ── Sliding window (matches nnUNet exactly: symmetric pad + Gaussian) ─────────

def gaussian_map(patch_size):
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

    gauss = gaussian_map(patch_size)
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


def dice(pred, gt):
    p, g  = pred > 0.5, gt > 0.5
    denom = p.sum() + g.sum()
    return float(2 * (p & g).sum() / denom) if denom > 0 else 1.0


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
    gauss    = gaussian_map(patch_size)  # precompute once

    img_paths = sorted(glob.glob(args.images_dir + '/*_0000.nii.gz'))
    if args.n:
        img_paths = img_paths[:args.n]
    print(f'Running ONNX on {len(img_paths)} images...')

    rows = []
    for img_path in img_paths:
        img_id = img_path.split('/')[-1].replace('_0000.nii.gz', '')
        gt_path = args.labels_dir + '/' + img_id + '.nii.gz'

        # ── Preprocessing ────────────────────────────────────────────────────
        t_pre0 = time.perf_counter()

        img          = nib.load(img_path)
        data         = img.get_fdata().transpose((2, 1, 0)).astype(np.float32)
        spacing      = [float(v) for v in img.header.get_zooms()[:3][::-1]]
        shape_orig   = data.shape

        data_crop, bbox = crop_to_nonzero(data)
        shape_crop      = data_crop.shape
        data_norm       = zscore_normalize(data_crop)
        shape_rs        = target_shape(shape_crop, spacing, target_sp)
        data_rs         = resample(data_norm, shape_crop, shape_rs, order=3)

        t_pre = time.perf_counter() - t_pre0

        # ── Inference ────────────────────────────────────────────────────────
        t_inf0 = time.perf_counter()
        prob   = sliding_window_inference(data_rs, session, patch_size, in_name, out_name)
        t_inf  = time.perf_counter() - t_inf0

        # ── Postprocessing ───────────────────────────────────────────────────
        pred_crop = resample((prob > 0.5).astype(np.float32), shape_rs, shape_crop, order=0) > 0.5
        pred_full = np.zeros(shape_orig, np.uint8)
        pred_full[tuple(slice(b[0], b[1]) for b in bbox)] = pred_crop.astype(np.uint8)
        pred_xyz  = pred_full.transpose((2, 1, 0))

        gt  = np.asarray(nib.load(gt_path).dataobj)
        d   = dice(pred_xyz, gt)

        rows.append({
            'image_id':         img_id,
            'dice':             round(d, 6),
            'inference_time_s': round(t_inf, 3),
            'preprocess_time_s': round(t_pre, 3),
            'total_time_s':     round(t_pre + t_inf, 3),
        })
        print(f'  {img_id}  dice={d:.4f}  inf={t_inf:.1f}s  pre={t_pre:.2f}s')

    # ── Write CSV ─────────────────────────────────────────────────────────────
    fieldnames = ['image_id', 'dice', 'inference_time_s', 'preprocess_time_s', 'total_time_s']
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    dices = [r['dice'] for r in rows]
    infs  = [r['inference_time_s'] for r in rows]
    print()
    print(f'N images     : {len(rows)}')
    print(f'Dice mean±std: {np.mean(dices):.4f} ± {np.std(dices):.4f}')
    print(f'Dice median  : {np.median(dices):.4f}')
    print(f'Inf time mean: {np.mean(infs):.2f}s ± {np.std(infs):.2f}s')
    print(f'CSV saved    : {args.output}')


if __name__ == '__main__':
    main()
