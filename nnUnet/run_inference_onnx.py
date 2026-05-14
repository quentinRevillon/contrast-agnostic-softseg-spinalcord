"""
Standalone nnUNet ONNX inference for spinal cord segmentation.

Takes a NIfTI image (already cropped to the spinal cord region and reoriented to RPI
by sc_crop), runs segmentation using the exported ONNX model, and saves the binary mask.

No nnunetv2 required — only nibabel, scipy, numpy, onnxruntime.

Dependencies:
    pip install nibabel scipy numpy onnxruntime

Usage:
    # Step 1 — crop with sc_crop (installs via: pip install sc-crop)
    sc_crop -i image.nii.gz --crop -o image_crop.nii.gz

    # Step 2 — segment
    python run_inference_onnx.py \
        -i image_crop.nii.gz \
        -o image_seg.nii.gz \
        --model nnunet_seg.onnx \
        --plans plans.json

Author: Quentin Revillon
"""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import onnxruntime as ort
from scipy.ndimage import gaussian_filter, zoom


def parse_args():
    parser = argparse.ArgumentParser(
        description='Spinal cord segmentation via nnUNet ONNX (standalone, no nnunetv2 required)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-i', required=True, help='Input NIfTI image (.nii or .nii.gz), RPI orientation')
    parser.add_argument('-o', required=True, help='Output segmentation mask (.nii.gz)')
    parser.add_argument('--model', required=True, help='Path to nnunet_seg.onnx')
    parser.add_argument('--plans', required=True, help='Path to plans.json from the nnUNet model folder')
    parser.add_argument('--tile-step', type=float, default=0.5,
                        help='Sliding window step as fraction of patch size (0.5 = 50%% overlap)')
    parser.add_argument('--threads', type=int, default=None,
                        help='ONNX Runtime intra-op threads (default: auto)')
    return parser.parse_args()


def load_plans(plans_path):
    plans = json.load(open(plans_path))
    cfg = plans['configurations']['3d_fullres']
    return {
        'target_spacing': cfg['spacing'],       # [z, y, x] in mm
        'patch_size':     cfg['patch_size'],     # [D, H, W]
    }


def get_voxel_spacing(img):
    """Return (z, y, x) voxel spacing in mm — reversed to match nnUNet's SimpleITK convention."""
    return [float(z) for z in img.header.get_zooms()[:3][::-1]]


def resample_volume(data, orig_spacing, target_spacing, order=3):
    """Resample 3D array to target spacing using scipy zoom."""
    zoom_factors = [o / t for o, t in zip(orig_spacing, target_spacing)]
    if all(abs(z - 1.0) < 1e-4 for z in zoom_factors):
        return data, zoom_factors
    resampled = zoom(data, zoom_factors, order=order, prefilter=order > 1)
    return resampled, zoom_factors


def zscore_normalize(data):
    """ZScore normalisation on all voxels (matches nnUNet ZScoreNormalization with use_mask_for_norm=False)."""
    mean = data.mean()
    std  = data.std()
    return ((data - mean) / max(std, 1e-8)).astype(np.float32)


def make_gaussian_map(patch_size, sigma_scale=1.0 / 8):
    """Gaussian importance map matching nnUNet's compute_gaussian exactly."""
    tmp = np.zeros(patch_size, dtype=np.float64)
    center = tuple(i // 2 for i in patch_size)
    tmp[center] = 1
    sigmas = [i * sigma_scale for i in patch_size]
    g = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    g = g / g.max()
    # Replace zeros with minimum nonzero value (nnUNet avoids NaNs this way)
    min_nonzero = g[g > 0].min()
    g[g == 0] = min_nonzero
    return g.astype(np.float32)


def compute_steps(image_size, patch_size, tile_step):
    """Compute sliding window start positions matching nnUNet's compute_steps_for_sliding_window."""
    steps = []
    for img_s, patch_s in zip(image_size, patch_size):
        if img_s <= patch_s:
            steps.append([0])
            continue
        num_steps = int(np.ceil((img_s - patch_s) / (patch_s * tile_step))) + 1
        max_step = img_s - patch_s
        if num_steps > 1:
            actual_step = max_step / (num_steps - 1)
        else:
            actual_step = 99999
        steps.append([int(np.round(actual_step * i)) for i in range(num_steps)])
    return steps


def sliding_window_inference(data, session, patch_size, tile_step):
    """
    Sliding window inference with Gaussian weighting, matching nnUNet exactly.
    data:    float32 array (D, H, W) in [z, y, x], already normalised
    returns: float32 array (D, H, W), probability of class 1 (spinal cord)
    """
    D, H, W = data.shape
    pd, ph, pw = patch_size

    # Pad so image is at least patch size in every dimension
    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)
    if pad_d or pad_h or pad_w:
        data = np.pad(data, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
    Dp, Hp, Wp = data.shape

    gauss        = make_gaussian_map(patch_size)
    accumulator  = np.zeros((Dp, Hp, Wp), dtype=np.float32)
    weight_map   = np.zeros((Dp, Hp, Wp), dtype=np.float32)
    input_name   = session.get_inputs()[0].name
    output_name  = session.get_outputs()[0].name

    steps_d, steps_h, steps_w = compute_steps((Dp, Hp, Wp), patch_size, tile_step)

    for dz in steps_d:
        for dy in steps_h:
            for dx in steps_w:
                patch = data[dz:dz+pd, dy:dy+ph, dx:dx+pw]
                inp   = patch[np.newaxis, np.newaxis].astype(np.float32)
                out   = session.run([output_name], {input_name: inp})[0]  # (1, 2, D, H, W)
                logits = out[0]  # (2, D, H, W)
                exp    = np.exp(logits - logits.max(axis=0, keepdims=True))
                prob1  = exp[1] / exp.sum(axis=0)
                accumulator[dz:dz+pd, dy:dy+ph, dx:dx+pw] += prob1 * gauss
                weight_map [dz:dz+pd, dy:dy+ph, dx:dx+pw] += gauss

    prob = (accumulator / weight_map)[:D, :H, :W]
    return prob


def main():
    args = parse_args()
    t0 = time.perf_counter()

    plans = load_plans(args.plans)
    target_spacing = plans['target_spacing']
    patch_size     = plans['patch_size']
    print(f'Target spacing : {target_spacing}')
    print(f'Patch size     : {patch_size}')

    # Load image — transpose to [z, y, x] to match nnUNet's internal convention
    img = nib.load(args.i)
    data = img.get_fdata().transpose((2, 1, 0)).astype(np.float32)
    orig_spacing = get_voxel_spacing(img)  # [z, y, x]
    print(f'Input shape    : {data.shape}  spacing={orig_spacing}  (z,y,x)')
    t_load = time.perf_counter()

    # Resample to target spacing
    data_rs, zoom_factors = resample_volume(data, orig_spacing, target_spacing, order=3)
    t_resample = time.perf_counter()
    print(f'Resampled      : {data_rs.shape}  ({t_resample - t_load:.2f}s)')

    # Normalise
    data_norm = zscore_normalize(data_rs)
    t_norm = time.perf_counter()

    # ONNX Runtime session
    sess_options = ort.SessionOptions()
    if args.threads:
        sess_options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(args.model, sess_options=sess_options,
                                   providers=['CPUExecutionProvider'])
    t_model = time.perf_counter()
    print(f'Model loaded   : ({t_model - t_norm:.2f}s)')

    # Sliding window inference
    prob = sliding_window_inference(data_norm, session, patch_size, args.tile_step)
    t_infer = time.perf_counter()
    print(f'Inference      : ({t_infer - t_model:.2f}s)')

    # Threshold → binary mask
    pred_rs = (prob > 0.5).astype(np.uint8)

    # Resample prediction back to original spacing
    inv_zoom = [1.0 / z for z in zoom_factors]
    pred = zoom(pred_rs.astype(np.float32), inv_zoom, order=0)
    # Match original shape exactly
    pred = pred[:data.shape[0], :data.shape[1], :data.shape[2]]
    pad_back = [(0, max(0, s - p)) for s, p in zip(data.shape, pred.shape)]
    if any(p[1] > 0 for p in pad_back):
        pred = np.pad(pred, pad_back, mode='constant')
    pred = pred.astype(np.uint8)
    t_post = time.perf_counter()

    # Transpose back to nibabel [x, y, z] convention before saving
    pred = pred.transpose((2, 1, 0))

    # Save
    out_img = nib.Nifti1Image(pred, img.affine, img.header)
    out_img.set_data_dtype(np.uint8)
    nib.save(out_img, args.o)
    t_save = time.perf_counter()

    print()
    print('─' * 40)
    print(f'  Load + resample : {t_resample - t0:.2f}s')
    print(f'  Normalise       : {t_norm - t_resample:.2f}s')
    print(f'  Model load      : {t_model - t_norm:.2f}s')
    print(f'  Inference       : {t_infer - t_model:.2f}s')
    print(f'  Post + save     : {t_save - t_infer:.2f}s')
    print(f'  Total           : {t_save - t0:.2f}s')
    print('─' * 40)
    print(f'Saved → {args.o}')


if __name__ == '__main__':
    main()
