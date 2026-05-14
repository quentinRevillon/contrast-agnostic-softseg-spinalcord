"""
Standalone nnUNet ONNX inference for spinal cord segmentation.

Replicates the full training pipeline exactly:
  1. sc_crop detection (conda env sc_crop) → bounding box in native space
  2. Crop image with bbox + reorient crop to RPI
  3. nnUNet ONNX inference:
       crop_to_nonzero → zscore_normalize → resample → sliding_window_inference
       → resample_seg_back → pad_back
  4. Reorient segmentation back to original orientation
  5. Pad back to full image space

Input: any NIfTI image (any orientation, not pre-cropped).
Output: binary segmentation mask in the same space/orientation as the input.

No nnunetv2 required — only nibabel, scipy, numpy, onnxruntime, scikit-image.
sc_crop must be installed in the conda env specified by --sc-crop-env (default: sc_crop).

Dependencies:
    pip install nibabel scipy numpy onnxruntime scikit-image
    conda activate sc_crop && pip install sc-crop

Usage:
    python nnUnet/run_inference_onnx.py -i image.nii.gz -o image_seg.nii.gz

Model files expected in ~/nnunet_contrast_agnostic/:
    nnunet_seg.onnx   — exported with export_nnunet_to_onnx.py
    plans.json        — from the nnUNet model folder

Author: Quentin Revillon
"""

import argparse
import json
import os
import subprocess
import time

import nibabel as nib
import numpy as np
import onnxruntime as ort
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.transform import resize as sk_resize


def parse_args():
    parser = argparse.ArgumentParser(
        description='Spinal cord segmentation via nnUNet ONNX (standalone, no nnunetv2 required)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _model_dir = os.path.expanduser('~/nnunet_contrast_agnostic')
    parser.add_argument('-i', required=True, help='Input NIfTI image (.nii or .nii.gz), any orientation')
    parser.add_argument('-o', required=True, help='Output segmentation mask (.nii.gz)')
    parser.add_argument('--model', default=os.path.join(_model_dir, 'nnunet_seg.onnx'),
                        help='Path to nnunet_seg.onnx')
    parser.add_argument('--plans', default=os.path.join(_model_dir, 'plans.json'),
                        help='Path to plans.json')
    parser.add_argument('--pad-rl', type=float, default=20.0, help='sc_crop padding left/right (mm)')
    parser.add_argument('--pad-ap', type=float, default=30.0, help='sc_crop padding anterior/posterior (mm)')
    parser.add_argument('--pad-si', type=float, default=40.0, help='sc_crop padding superior/inferior (mm)')
    parser.add_argument('--sc-crop-env', default='sc_crop', help='Conda env where sc_crop is installed')
    parser.add_argument('--tile-step', type=float, default=0.5,
                        help='Sliding window step as fraction of patch size')
    parser.add_argument('--threads', type=int, default=None,
                        help='ONNX Runtime intra-op threads (default: auto)')
    return parser.parse_args()


def load_plans(plans_path):
    plans = json.load(open(plans_path))
    cfg = plans['configurations']['3d_fullres']
    return {
        'target_spacing': cfg['spacing'],
        'patch_size':     cfg['patch_size'],
    }


def get_voxel_spacing_zyx(img):
    return [float(v) for v in img.header.get_zooms()[:3][::-1]]


# ── Reorientation ─────────────────────────────────────────────────────────────

def reorient_to_rpi(img):
    target  = axcodes2ornt(('R', 'P', 'I'))
    current = io_orientation(img.affine)
    return img.as_reoriented(ornt_transform(current, target))


def reorient_back(img_rpi, original_ornt):
    rpi_ornt = axcodes2ornt(('R', 'P', 'I'))
    return img_rpi.as_reoriented(ornt_transform(rpi_ornt, original_ornt))


# ── sc_crop detection + crop ──────────────────────────────────────────────────

def _parse_bbox_txt(bbox_file):
    with open(bbox_file) as f:
        for line in f:
            if not line.startswith('#'):
                xmin, xmax, ymin, ymax, zmin, zmax = (int(v) for v in line.split())
                return xmin, xmax, ymin, ymax, zmin, zmax
    raise ValueError(f'No bbox data in {bbox_file}')


def detect_and_crop(img_path, pad_rl, pad_ap, pad_si, sc_crop_env):
    """Run sc_crop in its conda env, parse bbox, return (cropped_rpi_img, bbox, orig_ornt, img).

    The bbox is expressed in the original image voxel space (inclusive indices).
    """
    import tempfile, os
    bbox_file = tempfile.mktemp(suffix='_bbox.txt')
    cmd = (f"conda run -n {sc_crop_env} sc_crop -i {img_path} -o {bbox_file} "
           f"--padding-rl '{pad_rl} {pad_rl}' "
           f"--padding-ap '{pad_ap} {pad_ap}' "
           f"--padding-si '{pad_si} {pad_si}'")
    assert subprocess.call(cmd, shell=True) == 0, f'sc_crop failed: {cmd}'
    xmin, xmax, ymin, ymax, zmin, zmax = _parse_bbox_txt(bbox_file)
    os.remove(bbox_file)
    print(f'sc_crop bbox : x=[{xmin},{xmax}] y=[{ymin},{ymax}] z=[{zmin},{zmax}]')

    img      = nib.load(img_path)
    orig_ornt = io_orientation(img.affine)
    data     = img.get_fdata(dtype=np.float32)
    cropped  = data[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]

    # Update affine: translate origin to the crop corner
    affine        = img.affine.copy()
    affine[:3, 3] = (img.affine @ np.array([xmin, ymin, zmin, 1.0]))[:3]
    crop_nii      = nib.Nifti1Image(cropped, affine)

    crop_rpi = reorient_to_rpi(crop_nii)
    bbox     = (xmin, xmax, ymin, ymax, zmin, zmax)
    return crop_rpi, bbox, orig_ornt, img


# ── nnUNet preprocessing matching DefaultPreprocessor exactly ─────────────────

def crop_to_nonzero(data):
    mask = binary_fill_holes(data != 0)
    bbox = []
    for ax in range(3):
        proj    = np.any(mask, axis=tuple(i for i in range(3) if i != ax))
        indices = np.where(proj)[0]
        bbox.append([int(indices[0]), int(indices[-1]) + 1] if len(indices) > 0
                    else [0, data.shape[ax]])
    return data[tuple(slice(b[0], b[1]) for b in bbox)], bbox


def pad_back(pred_cropped, bbox, full_shape):
    out    = np.zeros(full_shape, dtype=pred_cropped.dtype)
    slicer = tuple(slice(b[0], b[1]) for b in bbox)
    out[slicer] = pred_cropped
    return out


def zscore_normalize(data):
    mean = data.mean()
    std  = data.std()
    return ((data - mean) / max(float(std), 1e-8)).astype(np.float32)


def resample(data, orig_shape, target_shape, order):
    if tuple(orig_shape) == tuple(target_shape):
        return data.astype(np.float32)
    return sk_resize(data.astype(float), target_shape,
                     order=order, mode='edge', anti_aliasing=False).astype(np.float32)


def compute_new_shape(orig_shape, orig_spacing, target_spacing):
    return tuple(int(round(s * o / t))
                 for s, o, t in zip(orig_shape, orig_spacing, target_spacing))


# ── Sliding window matching nnUNet ────────────────────────────────────────────

def make_gaussian_map(patch_size, sigma_scale=1.0 / 8):
    tmp = np.zeros(patch_size, dtype=np.float64)
    tmp[tuple(i // 2 for i in patch_size)] = 1
    g = gaussian_filter(tmp, [i * sigma_scale for i in patch_size], mode='constant', cval=0)
    g /= g.max()
    g[g == 0] = g[g > 0].min()
    return g.astype(np.float32)


def compute_steps(image_size, patch_size, tile_step):
    steps = []
    for img_s, patch_s in zip(image_size, patch_size):
        if img_s <= patch_s:
            steps.append([0])
            continue
        n      = int(np.ceil((img_s - patch_s) / (patch_s * tile_step))) + 1
        max_val = img_s - patch_s
        actual  = max_val / (n - 1) if n > 1 else 99999
        steps.append([int(np.round(actual * i)) for i in range(n)])
    return steps


def sliding_window_inference(data, session, patch_size, tile_step):
    """Sliding window with Gaussian weighting and symmetric padding — matches nnUNet exactly."""
    D, H, W = data.shape
    pd, ph, pw = patch_size

    # Symmetric padding matches nnUNet's pad_nd_image (pad equally on both sides)
    pad_d = max(0, pd - D); d0 = pad_d // 2; d1 = pad_d - d0
    pad_h = max(0, ph - H); h0 = pad_h // 2; h1 = pad_h - h0
    pad_w = max(0, pw - W); w0 = pad_w // 2; w1 = pad_w - w0
    if pad_d or pad_h or pad_w:
        data = np.pad(data, ((d0, d1), (h0, h1), (w0, w1)), mode='constant')
    Dp, Hp, Wp = data.shape

    gauss      = make_gaussian_map(patch_size)
    accum      = np.zeros((Dp, Hp, Wp), dtype=np.float32)
    weight_map = np.zeros((Dp, Hp, Wp), dtype=np.float32)
    in_name    = session.get_inputs()[0].name
    out_name   = session.get_outputs()[0].name

    for dz in compute_steps((Dp, Hp, Wp), patch_size, tile_step)[0]:
        for dy in compute_steps((Dp, Hp, Wp), patch_size, tile_step)[1]:
            for dx in compute_steps((Dp, Hp, Wp), patch_size, tile_step)[2]:
                patch  = data[dz:dz+pd, dy:dy+ph, dx:dx+pw][np.newaxis, np.newaxis]
                logits = session.run([out_name], {in_name: patch})[0][0]
                exp    = np.exp(logits - logits.max(axis=0, keepdims=True))
                prob1  = exp[1] / exp.sum(axis=0)
                accum      [dz:dz+pd, dy:dy+ph, dx:dx+pw] += prob1 * gauss
                weight_map [dz:dz+pd, dy:dy+ph, dx:dx+pw] += gauss

    return (accum / weight_map)[d0:d0+D, h0:h0+H, w0:w0+W]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0   = time.perf_counter()

    plans      = load_plans(args.plans)
    target_sp  = plans['target_spacing']
    patch_size = plans['patch_size']
    print(f'Target spacing : {target_sp}  patch_size : {patch_size}')

    # 1. sc_crop detection + crop + reorient to RPI
    crop_rpi, bbox, orig_ornt, img_orig = detect_and_crop(
        args.i, args.pad_rl, args.pad_ap, args.pad_si, args.sc_crop_env)
    t_crop = time.perf_counter()
    print(f'Crop+RPI       : {crop_rpi.shape}  ({t_crop - t0:.2f}s)')

    # 2. Load RPI crop + transpose to nnUNet [z, y, x]
    data         = crop_rpi.get_fdata().transpose((2, 1, 0)).astype(np.float32)
    orig_spacing = get_voxel_spacing_zyx(crop_rpi)
    shape_before_crop = data.shape
    print(f'Loaded         : {data.shape}  spacing={orig_spacing}  (z,y,x)')

    # 3. crop_to_nonzero
    data_cropped, nnunet_bbox = crop_to_nonzero(data)
    shape_after_crop = data_cropped.shape
    print(f'Cropped        : {shape_after_crop}  bbox={nnunet_bbox}')

    # 4. zscore_normalize  (BEFORE resampling — matches nnUNet DefaultPreprocessor)
    data_norm = zscore_normalize(data_cropped)

    # 5. Resample to target spacing
    new_shape = compute_new_shape(shape_after_crop, orig_spacing, target_sp)
    data_rs   = resample(data_norm, shape_after_crop, new_shape, order=3)
    t_pre = time.perf_counter()
    print(f'Resampled      : {data_rs.shape}  ({t_pre - t_crop:.2f}s)')

    # 6. Load ONNX model
    sess_options = ort.SessionOptions()
    if args.threads:
        sess_options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(args.model, sess_options=sess_options,
                                   providers=['CPUExecutionProvider'])
    t_model = time.perf_counter()
    print(f'Model loaded   : ({t_model - t_pre:.2f}s)')

    # 7. Sliding window inference
    prob    = sliding_window_inference(data_rs, session, patch_size, args.tile_step)
    pred_rs = (prob > 0.5).astype(np.float32)
    t_infer = time.perf_counter()
    print(f'Inference      : ({t_infer - t_model:.2f}s)')

    # 8. Resample seg back to cropped shape (order=0)
    pred_cropped = resample(pred_rs, new_shape, shape_after_crop, order=0)
    pred_cropped = (pred_cropped > 0.5).astype(np.uint8)

    # 9. Pad back to RPI crop shape and transpose [z,y,x] → [x,y,z]
    pred_zyx       = pad_back(pred_cropped, nnunet_bbox, shape_before_crop)
    pred_rpi_arr   = pred_zyx.transpose((2, 1, 0)).astype(np.uint8)

    # 10. Reorient segmentation back to original orientation
    pred_rpi_nii   = nib.Nifti1Image(pred_rpi_arr, crop_rpi.affine)
    pred_orig_crop = reorient_back(pred_rpi_nii, orig_ornt)

    # 11. Pad back to full image space
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    pred_full = np.zeros(img_orig.shape[:3], dtype=np.uint8)
    pred_full[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1] = pred_orig_crop.get_fdata().astype(np.uint8)

    nib.save(nib.Nifti1Image(pred_full, img_orig.affine, img_orig.header), args.o)
    t_end = time.perf_counter()

    print()
    print('─' * 40)
    print(f'  sc_crop     : {t_crop - t0:.2f}s')
    print(f'  Preprocess  : {t_pre - t_crop:.2f}s')
    print(f'  Model load  : {t_model - t_pre:.2f}s')
    print(f'  Inference   : {t_infer - t_model:.2f}s')
    print(f'  Postprocess : {t_end - t_infer:.2f}s')
    print(f'  Total       : {t_end - t0:.2f}s')
    print('─' * 40)
    print(f'Saved → {args.o}')


if __name__ == '__main__':
    main()
