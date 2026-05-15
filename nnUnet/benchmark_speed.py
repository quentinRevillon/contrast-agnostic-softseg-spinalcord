"""
Compare inference speed and Dice of three pipelines on a single image.

Pipelines:
  1. ONNX    — sc_crop + ONNX inference (no mirroring)
  2. PT-noTTA — sc_crop + PyTorch inference (no mirroring)
  3. PT-TTA   — sc_crop + PyTorch inference (mirroring, 8× TTA)

Requires:
  - onnxruntime, nibabel, scipy, scikit-image  (for ONNX pipeline)
  - nnunetv2, torch                            (for PT pipelines)
  - sc_crop in conda env --sc-crop-env         (default: sc_crop)

Usage:
    python nnUnet/benchmark_speed.py \
        -i /path/to/image.nii.gz \
        --label /path/to/label.nii.gz \
        --model-folder /path/to/nnUNetTrainer__nnUNetPlans__3d_fullres \
        --onnx /path/to/nnunet_seg.onnx \
        --plans /path/to/plans.json

Author: Quentin Revillon
"""

import argparse
import json
import os
import subprocess
import tempfile
import time

import nibabel as nib
import numpy as np
import onnxruntime as ort
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.transform import resize as sk_resize


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', required=True, help='Input NIfTI image (original, not pre-cropped)')
    parser.add_argument('--label', default=None, help='GT label NIfTI (optional, for Dice)')
    parser.add_argument('--model-folder', required=True,
                        help='Path to nnUNetTrainer__nnUNetPlans__3d_fullres folder')
    parser.add_argument('--onnx', required=True, help='Path to nnunet_seg.onnx')
    parser.add_argument('--plans', required=True, help='Path to plans.json')
    parser.add_argument('--sc-crop-env', default='sc_crop', help='Conda env with sc_crop')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    return parser.parse_args()


# ── sc_crop ───────────────────────────────────────────────────────────────────

def run_sc_crop(image_path, sc_crop_env):
    bbox_file = tempfile.mktemp(suffix='_bbox.txt')
    cmd = f"conda run -n {sc_crop_env} sc_crop -i {image_path} -o {bbox_file}"
    assert subprocess.call(cmd, shell=True) == 0, f'sc_crop failed: {cmd}'
    with open(bbox_file) as f:
        for line in f:
            if not line.startswith('#'):
                bbox = tuple(int(v) for v in line.split())
                os.remove(bbox_file)
                return bbox
    raise ValueError('No bbox in sc_crop output')


def crop_and_reorient_rpi(img_path, xmin, xmax, ymin, ymax, zmin, zmax):
    img = nib.load(img_path)
    orig_ornt = io_orientation(img.affine)
    data = img.get_fdata(dtype=np.float32)
    cropped = data[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]
    affine = img.affine.copy()
    affine[:3, 3] = (img.affine @ np.array([xmin, ymin, zmin, 1.0]))[:3]
    crop_nii = nib.Nifti1Image(cropped, affine)
    target = axcodes2ornt(('R', 'P', 'I'))
    current = io_orientation(crop_nii.affine)
    return crop_nii.as_reoriented(ornt_transform(current, target)), orig_ornt, img


# ── ONNX inference ────────────────────────────────────────────────────────────

def onnx_infer(crop_rpi, session, target_sp, patch_size):
    data = crop_rpi.get_fdata().transpose((2, 1, 0)).astype(np.float32)
    spacing = [float(v) for v in crop_rpi.header.get_zooms()[:3][::-1]]

    mask = binary_fill_holes(data != 0)
    bbox = []
    for ax in range(3):
        proj = np.any(mask, axis=tuple(i for i in range(3) if i != ax))
        idx = np.where(proj)[0]
        bbox.append([int(idx[0]), int(idx[-1]) + 1] if len(idx) > 0 else [0, data.shape[ax]])
    data_crop = data[tuple(slice(b[0], b[1]) for b in bbox)]

    mean, std = data_crop.mean(), data_crop.std()
    data_norm = ((data_crop - mean) / max(float(std), 1e-8)).astype(np.float32)

    shape_rs = tuple(int(round(s * o / t)) for s, o, t in zip(data_crop.shape, spacing, target_sp))
    if tuple(data_crop.shape) != shape_rs:
        data_rs = sk_resize(data_norm.astype(float), shape_rs, order=3, mode='edge',
                            anti_aliasing=False).astype(np.float32)
    else:
        data_rs = data_norm

    pd, ph, pw = patch_size
    D, H, W = data_rs.shape
    pad_d = max(0, pd-D); d0 = pad_d//2; d1 = pad_d-d0
    pad_h = max(0, ph-H); h0 = pad_h//2; h1 = pad_h-h0
    pad_w = max(0, pw-W); w0 = pad_w//2; w1 = pad_w-w0
    if pad_d or pad_h or pad_w:
        data_rs = np.pad(data_rs, ((d0, d1), (h0, h1), (w0, w1)), mode='constant')
    Dp, Hp, Wp = data_rs.shape

    tmp = np.zeros(patch_size, dtype=np.float64)
    tmp[tuple(i // 2 for i in patch_size)] = 1
    gauss = gaussian_filter(tmp, [i / 8 for i in patch_size], mode='constant', cval=0)
    gauss /= gauss.max(); gauss[gauss == 0] = gauss[gauss > 0].min()
    gauss = gauss.astype(np.float32)

    def steps(img_s, patch_s):
        if img_s <= patch_s: return [0]
        n = int(np.ceil((img_s - patch_s) / (patch_s * 0.5))) + 1
        mv = img_s - patch_s
        act = mv / (n - 1) if n > 1 else 99999
        return [int(np.round(act * i)) for i in range(n)]

    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    accum = np.zeros((Dp, Hp, Wp), np.float32)
    wmap  = np.zeros((Dp, Hp, Wp), np.float32)
    for dz in steps(Dp, pd):
        for dy in steps(Hp, ph):
            for dx in steps(Wp, pw):
                patch = data_rs[dz:dz+pd, dy:dy+ph, dx:dx+pw][np.newaxis, np.newaxis]
                logits = session.run([out_name], {in_name: patch})[0][0]
                exp = np.exp(logits - logits.max(axis=0, keepdims=True))
                prob1 = exp[1] / exp.sum(axis=0)
                accum[dz:dz+pd, dy:dy+ph, dx:dx+pw] += prob1 * gauss
                wmap [dz:dz+pd, dy:dy+ph, dx:dx+pw] += gauss
    prob = (accum / wmap)[d0:d0+D, h0:h0+H, w0:w0+W]

    if tuple(data_crop.shape) != shape_rs:
        pred_crop = sk_resize((prob > 0.5).astype(float), data_crop.shape, order=0,
                              mode='edge', anti_aliasing=False) > 0.5
    else:
        pred_crop = prob > 0.5
    pred_zyx = np.zeros(data.shape, np.uint8)
    pred_zyx[tuple(slice(b[0], b[1]) for b in bbox)] = pred_crop.astype(np.uint8)
    return pred_zyx.transpose((2, 1, 0))


# ── PyTorch inference ─────────────────────────────────────────────────────────

def pt_infer(crop_path, model_folder, fold, device_str, use_mirroring, tmpdir):
    import torch
    import glob
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    device = torch.device(device_str)
    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                        use_mirroring=use_mirroring,
                        perform_everything_on_device=(device_str != 'cpu'),
                        device=device, verbose=False, verbose_preprocessing=False,
                        allow_tqdm=False)
    p.initialize_from_trained_model_folder(model_folder, use_folds=[fold],
                                           checkpoint_name='checkpoint_final.pth')
    p.predict_from_files([[crop_path]], tmpdir, save_probabilities=False, overwrite=True,
                         num_processes_preprocessing=1, num_processes_segmentation_export=1)
    pred_file = glob.glob(tmpdir + '/*.nii.gz')[0]
    return np.asarray(nib.load(pred_file).dataobj)


def dice(a, b):
    a, b = a > 0.5, b > 0.5
    denom = a.sum() + b.sum()
    return float(2 * (a & b).sum() / denom) if denom > 0 else 1.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    plans = json.load(open(args.plans))
    cfg = plans['configurations']['3d_fullres']
    target_sp = cfg['spacing']
    patch_size = cfg['patch_size']

    gt = np.asarray(nib.load(args.label).dataobj) if args.label else None

    print(f'Image  : {args.i}')
    print(f'Label  : {args.label or "none"}')
    print()

    # ── sc_crop (shared across all pipelines) ─────────────────────────────────
    print('Running sc_crop...')
    t0 = time.perf_counter()
    bbox = run_sc_crop(args.i, args.sc_crop_env)
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    crop_rpi, orig_ornt, img_orig = crop_and_reorient_rpi(args.i, xmin, xmax, ymin, ymax, zmin, zmax)
    t_sc = time.perf_counter() - t0
    print(f'  sc_crop : {t_sc:.2f}s  bbox={bbox}')
    print()

    # Save crop to temp file for PT pipelines
    crop_tmp = tempfile.mktemp(suffix='_0000.nii.gz')
    nib.save(crop_rpi, crop_tmp)

    results = {}

    # ── Pipeline 1 : ONNX ─────────────────────────────────────────────────────
    session = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    t1 = time.perf_counter()
    pred_onnx = onnx_infer(crop_rpi, session, target_sp, patch_size)
    t_onnx = time.perf_counter() - t1
    d_onnx = dice(pred_onnx, gt) if gt is not None else float('nan')
    results['ONNX (no TTA)'] = {'sc_crop': t_sc, 'infer': t_onnx, 'total': t_sc + t_onnx, 'dice': d_onnx}

    # ── Pipeline 2 : PT no TTA ────────────────────────────────────────────────
    tmpdir2 = tempfile.mkdtemp()
    t2 = time.perf_counter()
    pred_pt_nomir = pt_infer(crop_tmp, args.model_folder, args.fold, args.device,
                              use_mirroring=False, tmpdir=tmpdir2)
    t_pt_nomir = time.perf_counter() - t2
    d_pt_nomir = dice(pred_pt_nomir, gt) if gt is not None else float('nan')
    results['PT (no TTA)'] = {'sc_crop': t_sc, 'infer': t_pt_nomir, 'total': t_sc + t_pt_nomir, 'dice': d_pt_nomir}

    # ── Pipeline 3 : PT + TTA ─────────────────────────────────────────────────
    tmpdir3 = tempfile.mkdtemp()
    t3 = time.perf_counter()
    pred_pt_mir = pt_infer(crop_tmp, args.model_folder, args.fold, args.device,
                            use_mirroring=True, tmpdir=tmpdir3)
    t_pt_mir = time.perf_counter() - t3
    d_pt_mir = dice(pred_pt_mir, gt) if gt is not None else float('nan')
    results['PT (TTA)'] = {'sc_crop': t_sc, 'infer': t_pt_mir, 'total': t_sc + t_pt_mir, 'dice': d_pt_mir}

    os.remove(crop_tmp)

    # ── Summary ───────────────────────────────────────────────────────────────
    print('%-20s  %10s  %10s  %10s  %6s' % ('Pipeline', 'sc_crop', 'infer', 'total', 'Dice'))
    print('-' * 65)
    for name, r in results.items():
        print('%-20s  %9.2fs  %9.2fs  %9.2fs  %.4f' % (
            name, r['sc_crop'], r['infer'], r['total'], r['dice']))
    print()
    ref_total = results['PT (TTA)']['total']
    for name, r in results.items():
        speedup = ref_total / r['total'] if r['total'] > 0 else float('nan')
        print(f"  {name}: {speedup:.1f}× faster than PT+TTA")


if __name__ == '__main__':
    main()
