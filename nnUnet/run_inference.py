"""
Standalone spinal cord segmentation inference.

Supports three inference modes via --mode:
  onnx     — ONNX Runtime (no nnunetv2, CPU only, no TTA)   [default]
  pt       — PyTorch nnUNet predictor, no mirroring (TTA off)
  pt-tta   — PyTorch nnUNet predictor, with mirroring (TTA on, 8× slower)

Pipeline (all modes):
  1. sc_crop detection → crop in original orientation  (sc_crop.detect_and_crop)
  2. Reorient to RPI
  3. nnUNet inference (ONNX or PyTorch depending on --mode)
  4. Reorient segmentation back to original orientation
  5. Restore to full image space                        (sc_crop.restore_segmentation)

Input : any NIfTI image (any orientation, not pre-cropped).
Output: binary segmentation mask in the same space/orientation as the input.

ONNX mode requires : nibabel, scipy, numpy, onnxruntime, scikit-image, sc-crop
PT modes require   : nnunetv2, torch (+ the above)

Usage:
    # Download all model files (first use only — nnUNet + sc_crop)
    python nnUnet/run_inference.py download

    # Full image (sc_crop runs internally)
    python nnUnet/run_inference.py -i image.nii.gz -o seg.nii.gz

    # Pre-cropped image (skip sc_crop)
    python nnUnet/run_inference.py -i image_crop.nii.gz -o seg.nii.gz --pre-cropped

    # PyTorch without TTA
    python nnUnet/run_inference.py -i image.nii.gz -o seg.nii.gz --mode pt

    # PyTorch with TTA (mirroring)
    python nnUnet/run_inference.py -i image.nii.gz -o seg.nii.gz --mode pt-tta

nnUNet model files are downloaded to ~/nnunet_contrast_agnostic/.
sc_crop models are cached in ~/.cache/sc_crop/ (auto-download on first use).

Resampling matches nnUNet's resample_data_or_seg_to_shape exactly:
  - Isotropic images  : skimage.transform.resize with order=3/0, mode='edge'
  - Anisotropic images: skimage.transform.resize in-plane + scipy.ndimage.map_coordinates
                        on the thick axis with order_z=0 (nearest-neighbour), per plans.json

Author: Quentin Revillon
"""

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request

_MODEL_DIR = os.path.expanduser('~/nnunet_contrast_agnostic')
_RELEASE_URL = 'https://github.com/quentinRevillon/contrast-agnostic-softseg-spinalcord/releases/download/v0.0.2'
_ASSETS = [
    ('nnunet_seg.onnx',        'nnunet_seg.onnx'),
    ('plans.json',             'plans.json'),
    ('dataset.json',           'dataset.json'),
    ('checkpoint_final.pth',   'fold_0/checkpoint_final.pth'),
]


def download_models(model_dir=_MODEL_DIR):
    os.makedirs(os.path.join(model_dir, 'fold_0'), exist_ok=True)
    for asset_name, rel_path in _ASSETS:
        dest = os.path.join(model_dir, rel_path)
        if os.path.exists(dest):
            print(f'  already exists: {dest}')
            continue
        url = f'{_RELEASE_URL}/{asset_name}'
        print(f'  downloading {asset_name} ...', flush=True)
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()
    print(f'Models ready in {model_dir}')
    print('(sc_crop models download automatically on first use)')


def _progress(count, block_size, total_size):
    pct = min(int(count * block_size * 100 / total_size), 100) if total_size > 0 else 0
    print(f'\r  {pct}%', end='', flush=True)


import nibabel as nib
import numpy as np
import onnxruntime as ort
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from scipy.ndimage import binary_fill_holes, gaussian_filter, map_coordinates
from skimage.transform import resize as sk_resize

_ANISO_THRESHOLD = 3


def parse_args():
    if len(sys.argv) > 1 and sys.argv[1] == 'download':
        download_models()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description='Spinal cord segmentation via nnUNet (ONNX or PyTorch)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-i', required=True, help='Input NIfTI image (.nii or .nii.gz), any orientation')
    parser.add_argument('-o', required=True, help='Output segmentation mask (.nii.gz)')
    parser.add_argument('--mode', default='onnx', choices=['onnx', 'pt', 'pt-tta'],
                        help='Inference mode: onnx (no nnunetv2), pt (PyTorch, no TTA), pt-tta (PyTorch + mirroring)')
    # ONNX-mode args
    parser.add_argument('--model', default=os.path.join(_MODEL_DIR, 'nnunet_seg.onnx'),
                        help='[onnx] Path to nnunet_seg.onnx')
    parser.add_argument('--plans', default=os.path.join(_MODEL_DIR, 'plans.json'),
                        help='[onnx] Path to plans.json')
    parser.add_argument('--tile-step', type=float, default=0.5,
                        help='[onnx] Sliding window step as fraction of patch size')
    parser.add_argument('--threads', type=int, default=None,
                        help='[onnx] ONNX Runtime intra-op threads (default: auto)')
    # PT-mode args
    parser.add_argument('--model-folder', default=_MODEL_DIR,
                        help='[pt/pt-tta] Path to model folder (must contain plans.json, dataset.json, fold_0/)')
    parser.add_argument('--fold', type=int, default=0,
                        help='[pt/pt-tta] Fold index')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'],
                        help='[pt/pt-tta] Inference device')
    # sc_crop args
    parser.add_argument('--pre-cropped', action='store_true',
                        help='Input image is already cropped around the SC — skip sc_crop detection')
    parser.add_argument('--pad-rl', type=float, default=20.0, help='sc_crop padding left/right (mm)')
    parser.add_argument('--pad-ap', type=float, default=30.0, help='sc_crop padding anterior/posterior (mm)')
    parser.add_argument('--pad-si', type=float, default=40.0, help='sc_crop padding superior/inferior (mm)')
    # Output control
    parser.add_argument('--time', action='store_true',
                        help='Print per-step timing breakdown')
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


# ── Reorientation helpers ─────────────────────────────────────────────────────

def reorient_to_rpi(img):
    target  = axcodes2ornt(('R', 'P', 'I'))
    current = io_orientation(img.affine)
    return img.as_reoriented(ornt_transform(current, target))


def reorient_back(img_rpi, original_ornt):
    rpi_ornt = axcodes2ornt(('R', 'P', 'I'))
    return img_rpi.as_reoriented(ornt_transform(rpi_ornt, original_ornt))


# ── sc_crop detection + crop ──────────────────────────────────────────────────

def detect_and_crop(img_path, pad_rl, pad_ap, pad_si):
    """Detect SC, crop in original orientation, reorient to RPI. Returns (crop_rpi, ctx, orig_ornt)."""
    from sc_crop import detect_and_crop as _sc_detect_and_crop
    crop_nii, ctx = _sc_detect_and_crop(
        img_path,
        padding_rl_mm=pad_rl,
        padding_ap_mm=pad_ap,
        padding_si_mm=pad_si,
    )
    xmin, xmax = ctx['xmin'], ctx['xmax']
    ymin, ymax = ctx['ymin'], ctx['ymax']
    zmin, zmax = ctx['zmin'], ctx['zmax']
    print(f'sc_crop bbox : x=[{xmin},{xmax}] y=[{ymin},{ymax}] z=[{zmin},{zmax}]')
    orig_ornt = io_orientation(crop_nii.affine)
    crop_rpi  = reorient_to_rpi(crop_nii)
    return crop_rpi, ctx, orig_ornt


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


def pad_back(pred_cropped, bbox, full_shape):
    out    = np.zeros(full_shape, dtype=pred_cropped.dtype)
    slicer = tuple(slice(b[0], b[1]) for b in bbox)
    out[slicer] = pred_cropped
    return out


def zscore_normalize(data):
    mean = data.mean()
    std  = data.std()
    return ((data - mean) / max(float(std), 1e-8)).astype(np.float32)


def _get_lowres_axis(spacing):
    """Return indices where spacing == max(spacing) — the anisotropic axis."""
    return np.where(max(spacing) / np.array(spacing) == 1)[0]


def _do_separate_z(spacing):
    return (max(spacing) / min(spacing)) > _ANISO_THRESHOLD


def _determine_sep_z_axis(current_spacing, new_spacing):
    """Mirrors nnUNet determine_do_sep_z_and_axis with force_separate_z=None."""
    if _do_separate_z(current_spacing):
        axis = _get_lowres_axis(current_spacing)
    elif _do_separate_z(new_spacing):
        axis = _get_lowres_axis(new_spacing)
    else:
        return False, None
    if len(axis) in (2, 3):   # ambiguous — fall back to isotropic resampling
        return False, None
    return True, int(axis[0])


def resample(data, new_shape, current_spacing, new_spacing, order=3, order_z=0):
    """Resample 3D data matching nnUNet's resample_data_or_seg_to_shape exactly.

    For isotropic images uses skimage.transform.resize (order=3 / mode='edge').
    For anisotropic images resamples the thick axis separately with
    scipy.ndimage.map_coordinates at order_z=0, matching plans.json config.

    Args:
        data:            3D numpy array in ZYX order.
        new_shape:       Target shape as (Z, Y, X) tuple/array.
        current_spacing: Voxel spacing of data in ZYX order.
        new_spacing:     Target voxel spacing in ZYX order.
        order:           Interpolation order for in-plane resampling (default 3).
        order_z:         Interpolation order for the anisotropic axis (default 0).
    """
    old_shape = np.array(data.shape)
    new_shape  = np.array(new_shape)

    if np.all(old_shape == new_shape):
        return data.astype(np.float32)

    do_sep, axis = _determine_sep_z_axis(current_spacing, new_spacing)
    data_f = data.astype(float)

    if not do_sep:
        return sk_resize(data_f, new_shape, order=order,
                         mode='edge', anti_aliasing=False).astype(np.float32)

    # ── Separate Z path (anisotropic image) ──────────────────────────────────
    if axis == 0:
        new_shape_2d = new_shape[1:]
    elif axis == 1:
        new_shape_2d = new_shape[[0, 2]]
    else:
        new_shape_2d = new_shape[:-1]

    # Step 1: resample in-plane with order=3
    tmp_shape = new_shape.copy()
    tmp_shape[axis] = old_shape[axis]
    reshaped = np.zeros(tmp_shape, dtype=float)
    for idx in range(old_shape[axis]):
        if axis == 0:
            reshaped[idx] = sk_resize(data_f[idx], new_shape_2d, order=order,
                                      mode='edge', anti_aliasing=False)
        elif axis == 1:
            reshaped[:, idx] = sk_resize(data_f[:, idx], new_shape_2d, order=order,
                                         mode='edge', anti_aliasing=False)
        else:
            reshaped[:, :, idx] = sk_resize(data_f[:, :, idx], new_shape_2d, order=order,
                                            mode='edge', anti_aliasing=False)

    # Step 2: resample anisotropic axis with order_z using map_coordinates
    if old_shape[axis] == new_shape[axis]:
        return reshaped.astype(np.float32)

    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
    orig_rows, orig_cols, orig_dim = reshaped.shape

    row_scale = float(orig_rows) / rows
    col_scale = float(orig_cols) / cols
    dim_scale = float(orig_dim)  / dim

    map_r, map_c, map_d = np.mgrid[:rows, :cols, :dim]
    map_r = row_scale * (map_r + 0.5) - 0.5
    map_c = col_scale * (map_c + 0.5) - 0.5
    map_d = dim_scale * (map_d + 0.5) - 0.5

    coord_map = np.array([map_r, map_c, map_d])
    return map_coordinates(reshaped, coord_map, order=order_z, mode='nearest').astype(np.float32)


def compute_new_shape(orig_shape, orig_spacing, target_spacing):
    return tuple(int(round(s * o / t))
                 for s, o, t in zip(orig_shape, orig_spacing, target_spacing))


# ── Sliding window (ONNX) ─────────────────────────────────────────────────────

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
        n       = int(np.ceil((img_s - patch_s) / (patch_s * tile_step))) + 1
        max_val = img_s - patch_s
        actual  = max_val / (n - 1) if n > 1 else 99999
        steps.append([int(np.round(actual * i)) for i in range(n)])
    return steps


def sliding_window_inference(data, session, patch_size, tile_step):
    """Sliding window with Gaussian weighting — accumulates logits, matches nnUNet exactly.

    nnUNet accumulates logits (not softmax probabilities) across overlapping patches and
    divides by the Gaussian weight map once at the end. Applying softmax before accumulation
    (as averaging probabilities) gives different results from averaging logits then softmax.

    Returns averaged logits of shape (2, D, H, W).
    """
    D, H, W = data.shape
    pd, ph, pw = patch_size

    # Pad so every spatial dim is at least the patch size (same convention as pad_nd_image)
    pad_d = max(0, pd - D); d0 = pad_d // 2; d1 = pad_d - d0
    pad_h = max(0, ph - H); h0 = pad_h // 2; h1 = pad_h - h0
    pad_w = max(0, pw - W); w0 = pad_w // 2; w1 = pad_w - w0
    if pad_d or pad_h or pad_w:
        data = np.pad(data, ((d0, d1), (h0, h1), (w0, w1)), mode='constant')
    Dp, Hp, Wp = data.shape

    gauss      = make_gaussian_map(patch_size)                            # (pd, ph, pw)
    n_classes  = 2
    accum      = np.zeros((n_classes, Dp, Hp, Wp), dtype=np.float32)     # logit accumulator
    weight_map = np.zeros((Dp, Hp, Wp), dtype=np.float32)
    in_name    = session.get_inputs()[0].name
    out_name   = session.get_outputs()[0].name

    for dz in compute_steps((Dp, Hp, Wp), patch_size, tile_step)[0]:
        for dy in compute_steps((Dp, Hp, Wp), patch_size, tile_step)[1]:
            for dx in compute_steps((Dp, Hp, Wp), patch_size, tile_step)[2]:
                patch  = data[dz:dz+pd, dy:dy+ph, dx:dx+pw][np.newaxis, np.newaxis]
                logits = session.run([out_name], {in_name: patch})[0][0]  # (2, pd, ph, pw)
                accum      [:, dz:dz+pd, dy:dy+ph, dx:dx+pw] += logits * gauss
                weight_map [dz:dz+pd, dy:dy+ph, dx:dx+pw]    += gauss

    avg_logits = (accum / weight_map)[:, d0:d0+D, h0:h0+H, w0:w0+W]  # (2, D, H, W)
    return avg_logits


# ── ONNX inference ────────────────────────────────────────────────────────────

def infer_onnx(crop_rpi, model_path, plans_path, tile_step, threads):
    plans      = load_plans(plans_path)
    target_sp  = plans['target_spacing']
    patch_size = plans['patch_size']

    data         = crop_rpi.get_fdata().transpose((2, 1, 0)).astype(np.float32)
    orig_spacing = get_voxel_spacing_zyx(crop_rpi)
    shape_before = data.shape

    data_cropped, nnunet_bbox = crop_to_nonzero(data)
    data_norm                 = zscore_normalize(data_cropped)
    new_shape                 = compute_new_shape(data_cropped.shape, orig_spacing, target_sp)
    data_rs                   = resample(data_norm, new_shape, orig_spacing, target_sp, order=3, order_z=0)

    sess_options = ort.SessionOptions()
    if threads:
        sess_options.intra_op_num_threads = threads
    session = ort.InferenceSession(model_path, sess_options=sess_options,
                                   providers=['CPUExecutionProvider'])

    # sliding window → averaged logits (2, Z, Y, X) at target_spacing
    avg_logits = sliding_window_inference(data_rs, session, patch_size, tile_step)

    # resample logits back to original spacing with order=1 (resampling_fn_probabilities_kwargs)
    # then apply softmax and threshold — matches nnUNet's convert_predicted_logits_to_segmentation
    logits_back = np.stack([
        resample(avg_logits[c], data_cropped.shape, target_sp, orig_spacing, order=1, order_z=0)
        for c in range(avg_logits.shape[0])
    ])  # (2, Z_orig, Y_orig, X_orig)
    exp          = np.exp(logits_back - logits_back.max(axis=0, keepdims=True))
    prob1        = exp[1] / exp.sum(axis=0)                 # softmax class 1
    pred_cropped = (prob1 > 0.5).astype(np.uint8)
    pred_zyx     = pad_back(pred_cropped, nnunet_bbox, shape_before)
    return pred_zyx.transpose((2, 1, 0)).astype(np.uint8)


# ── PyTorch inference ─────────────────────────────────────────────────────────

def infer_pt(crop_rpi, model_folder, fold, device_str, use_mirroring):
    import glob
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device(device_str)
    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                        use_mirroring=use_mirroring,
                        perform_everything_on_device=(device_str != 'cpu'),
                        device=device, verbose=False, verbose_preprocessing=False,
                        allow_tqdm=False)
    p.initialize_from_trained_model_folder(model_folder, use_folds=[fold],
                                           checkpoint_name='checkpoint_final.pth')

    crop_tmp = tempfile.mktemp(suffix='_0000.nii.gz')
    nib.save(crop_rpi, crop_tmp)
    tmpdir = tempfile.mkdtemp()
    p.predict_from_files([[crop_tmp]], tmpdir, save_probabilities=False, overwrite=True,
                         num_processes_preprocessing=1, num_processes_segmentation_export=1)
    pred_arr = np.asarray(nib.load(glob.glob(tmpdir + '/*.nii.gz')[0]).dataobj)
    os.remove(crop_tmp)
    return pred_arr.astype(np.uint8)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0   = time.perf_counter()

    # 1. sc_crop detection + crop + reorient to RPI  (or load pre-cropped image directly)
    if args.pre_cropped:
        img_orig  = nib.load(args.i)
        orig_ornt = io_orientation(img_orig.affine)
        crop_rpi  = reorient_to_rpi(img_orig)
        ctx       = None
    else:
        crop_rpi, ctx, orig_ornt = detect_and_crop(args.i, args.pad_rl, args.pad_ap, args.pad_si)
    t_crop = time.perf_counter()

    # 2. Inference
    if args.mode == 'onnx':
        pred_rpi_arr = infer_onnx(crop_rpi, args.model, args.plans, args.tile_step, args.threads)
    else:
        pred_rpi_arr = infer_pt(crop_rpi, args.model_folder, args.fold, args.device,
                                use_mirroring=(args.mode == 'pt-tta'))
    t_infer = time.perf_counter()

    # 3. Reorient segmentation back to original orientation
    pred_rpi_nii = nib.Nifti1Image(pred_rpi_arr, crop_rpi.affine)
    pred_crop    = reorient_back(pred_rpi_nii, orig_ornt)

    if ctx is not None:
        # 4. Restore to full image space
        from sc_crop import restore_segmentation as _sc_restore
        pred_out = _sc_restore(pred_crop, ctx)
    else:
        pred_out = pred_crop

    nib.save(pred_out, args.o)
    t_end = time.perf_counter()

    if args.time:
        print()
        print('─' * 40)
        print(f'  sc_crop   : {t_crop - t0:.2f}s')
        print(f'  inference : {t_infer - t_crop:.2f}s')
        print(f'  postproc  : {t_end - t_infer:.2f}s')
        print(f'  total     : {t_end - t0:.2f}s')
        print('─' * 40)

    print(f'Saved → {args.o}')


if __name__ == '__main__':
    main()
