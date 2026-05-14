"""
Standalone nnUNet ONNX inference for spinal cord segmentation.

Takes a NIfTI image (already cropped to the spinal cord region and reoriented to RPI
by sc_crop), runs segmentation using the exported ONNX model, and saves the binary mask.

Preprocessing pipeline matches nnUNet exactly:
  load → transpose [x,y,z]→[z,y,x] → crop_to_nonzero → zscore_normalize → resample
  → sliding_window_inference → resample_seg_back → pad_back → transpose [z,y,x]→[x,y,z]

No nnunetv2 required — only nibabel, scipy, numpy, onnxruntime, scikit-image.

Dependencies:
    pip install nibabel scipy numpy onnxruntime scikit-image

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

import nibabel as nib
import numpy as np
import onnxruntime as ort
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.transform import resize as sk_resize


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
        'target_spacing': cfg['spacing'],   # [z, y, x] in mm
        'patch_size':     cfg['patch_size'], # [D, H, W]
    }


def get_voxel_spacing_zyx(img):
    """Return (z, y, x) spacing — reversed from nibabel [x,y,z] to match nnUNet's convention."""
    return [float(v) for v in img.header.get_zooms()[:3][::-1]]


# ── Preprocessing steps matching nnUNet DefaultPreprocessor exactly ───────────

def crop_to_nonzero(data):
    """
    Crop (D, H, W) array to bounding box of nonzero voxels.
    Returns (cropped_data, bbox) where bbox = [[d0,d1],[h0,h1],[w0,w1]].
    Matches nnUNet's crop_to_nonzero with binary_fill_holes.
    """
    mask = binary_fill_holes(data != 0)
    bbox = []
    for ax in range(3):
        # project onto this axis
        proj = np.any(mask, axis=tuple(i for i in range(3) if i != ax))
        indices = np.where(proj)[0]
        if len(indices) == 0:
            bbox.append([0, data.shape[ax]])
        else:
            bbox.append([int(indices[0]), int(indices[-1]) + 1])
    slicer = tuple(slice(b[0], b[1]) for b in bbox)
    return data[slicer], bbox


def pad_back(pred_cropped, bbox, full_shape):
    """Pad prediction back to full_shape using the stored bbox."""
    out = np.zeros(full_shape, dtype=pred_cropped.dtype)
    slicer = tuple(slice(b[0], b[1]) for b in bbox)
    out[slicer] = pred_cropped
    return out


def zscore_normalize(data):
    """ZScore on all voxels — matches nnUNet ZScoreNormalization with use_mask_for_norm=False."""
    mean = data.mean()
    std  = data.std()
    return ((data - mean) / max(float(std), 1e-8)).astype(np.float32)


def resample(data, orig_shape, target_shape, order):
    """Resample using skimage.transform.resize with mode='edge' — matches nnUNet exactly."""
    if tuple(orig_shape) == tuple(target_shape):
        return data.astype(np.float32)
    return sk_resize(data.astype(float), target_shape,
                     order=order, mode='edge', anti_aliasing=False).astype(np.float32)


def compute_new_shape(orig_shape, orig_spacing, target_spacing):
    return tuple(int(round(s * o / t)) for s, o, t in zip(orig_shape, orig_spacing, target_spacing))


# ── Sliding window matching nnUNet ────────────────────────────────────────────

def make_gaussian_map(patch_size, sigma_scale=1.0 / 8):
    """Bell-shaped Gaussian importance map — matches nnUNet's compute_gaussian."""
    tmp = np.zeros(patch_size, dtype=np.float64)
    tmp[tuple(i // 2 for i in patch_size)] = 1
    g = gaussian_filter(tmp, [i * sigma_scale for i in patch_size], mode='constant', cval=0)
    g /= g.max()
    g[g == 0] = g[g > 0].min()
    return g.astype(np.float32)


def compute_steps(image_size, patch_size, tile_step):
    """Uniformly distributed start positions — matches nnUNet's compute_steps_for_sliding_window."""
    steps = []
    for img_s, patch_s in zip(image_size, patch_size):
        if img_s <= patch_s:
            steps.append([0])
            continue
        n = int(np.ceil((img_s - patch_s) / (patch_s * tile_step))) + 1
        max_val = img_s - patch_s
        actual = max_val / (n - 1) if n > 1 else 99999
        steps.append([int(np.round(actual * i)) for i in range(n)])
    return steps


def sliding_window_inference(data, session, patch_size, tile_step):
    """Sliding window with Gaussian weighting — matches nnUNet exactly."""
    D, H, W = data.shape
    pd, ph, pw = patch_size

    # Symmetric padding matches nnUNet's pad_nd_image (pad equally on both sides)
    pad_d = max(0, pd - D); d0 = pad_d // 2; d1 = pad_d - d0
    pad_h = max(0, ph - H); h0 = pad_h // 2; h1 = pad_h - h0
    pad_w = max(0, pw - W); w0 = pad_w // 2; w1 = pad_w - w0
    if pad_d or pad_h or pad_w:
        data = np.pad(data, ((d0, d1), (h0, h1), (w0, w1)), mode='constant')
    Dp, Hp, Wp = data.shape

    gauss       = make_gaussian_map(patch_size)
    accum       = np.zeros((Dp, Hp, Wp), dtype=np.float32)
    weight_map  = np.zeros((Dp, Hp, Wp), dtype=np.float32)
    in_name     = session.get_inputs()[0].name
    out_name    = session.get_outputs()[0].name

    for dz in compute_steps((Dp, Hp, Wp), patch_size, tile_step)[0]:
        for dy in compute_steps((Dp, Hp, Wp), patch_size, tile_step)[1]:
            for dx in compute_steps((Dp, Hp, Wp), patch_size, tile_step)[2]:
                patch  = data[dz:dz+pd, dy:dy+ph, dx:dx+pw][np.newaxis, np.newaxis]
                logits = session.run([out_name], {in_name: patch})[0][0]  # (2, D, H, W)
                exp    = np.exp(logits - logits.max(axis=0, keepdims=True))
                prob1  = exp[1] / exp.sum(axis=0)
                accum      [dz:dz+pd, dy:dy+ph, dx:dx+pw] += prob1 * gauss
                weight_map [dz:dz+pd, dy:dy+ph, dx:dx+pw] += gauss

    return (accum / weight_map)[d0:d0+D, h0:h0+H, w0:w0+W]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0 = time.perf_counter()

    plans        = load_plans(args.plans)
    target_sp    = plans['target_spacing']
    patch_size   = plans['patch_size']
    print(f'Target spacing : {target_sp}  patch_size : {patch_size}')

    # 1. Load + transpose to [z, y, x]
    img          = nib.load(args.i)
    data         = img.get_fdata().transpose((2, 1, 0)).astype(np.float32)
    orig_spacing = get_voxel_spacing_zyx(img)
    shape_before_crop = data.shape
    print(f'Loaded         : {data.shape}  spacing={orig_spacing}  (z,y,x)')
    t_load = time.perf_counter()

    # 2. crop_to_nonzero  (nnUNet step 1)
    data_cropped, bbox = crop_to_nonzero(data)
    shape_after_crop = data_cropped.shape
    print(f'Cropped        : {shape_after_crop}  bbox={bbox}')

    # 3. zscore_normalize  (nnUNet step 2 — BEFORE resampling)
    data_norm = zscore_normalize(data_cropped)

    # 4. Resample to target spacing  (nnUNet step 3)
    new_shape = compute_new_shape(shape_after_crop, orig_spacing, target_sp)
    data_rs   = resample(data_norm, shape_after_crop, new_shape, order=3)
    t_pre = time.perf_counter()
    print(f'Resampled      : {data_rs.shape}  ({t_pre - t_load:.2f}s)')

    # 5. Load ONNX model
    sess_options = ort.SessionOptions()
    if args.threads:
        sess_options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(args.model, sess_options=sess_options,
                                   providers=['CPUExecutionProvider'])
    t_model = time.perf_counter()
    print(f'Model loaded   : ({t_model - t_pre:.2f}s)')

    # 6. Sliding window inference
    prob  = sliding_window_inference(data_rs, session, patch_size, args.tile_step)
    pred_rs = (prob > 0.5).astype(np.float32)
    t_infer = time.perf_counter()
    print(f'Inference      : ({t_infer - t_model:.2f}s)')

    # 7. Resample seg back to cropped shape  (order=0, matches nnUNet seg resampling)
    pred_cropped = resample(pred_rs, new_shape, shape_after_crop, order=0)
    pred_cropped = (pred_cropped > 0.5).astype(np.uint8)

    # 8. Pad back to full shape
    pred_zyx = pad_back(pred_cropped, bbox, shape_before_crop)

    # 9. Transpose back to nibabel [x, y, z] and save
    pred = pred_zyx.transpose((2, 1, 0)).astype(np.uint8)
    nib.save(nib.Nifti1Image(pred, img.affine, img.header), args.o)
    t_end = time.perf_counter()

    print()
    print('─' * 40)
    print(f'  Preprocess  : {t_pre - t_load:.2f}s')
    print(f'  Model load  : {t_model - t_pre:.2f}s')
    print(f'  Inference   : {t_infer - t_model:.2f}s')
    print(f'  Postprocess : {t_end - t_infer:.2f}s')
    print(f'  Total       : {t_end - t0:.2f}s')
    print('─' * 40)
    print(f'Saved → {args.o}')


if __name__ == '__main__':
    main()
