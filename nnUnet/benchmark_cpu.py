"""
CPU inference benchmark: sct_deepseg vs sc_crop + ONNX nnUNet pipeline.

For each input image:
  Method A — SCT contrast-agnostic:
    sct_deepseg -task seg_sc_contrast_agnostic  (full volume, CPU)

  Method B — sc_crop + ONNX (this work):
    sc_crop detection → crop → ONNX nnUNet → restore to full space (CPU)

Dice vs GT is computed when --gt is provided.
Both methods run on CPU. GPU is disabled via CUDA_VISIBLE_DEVICES="".

Usage:
    conda activate contrast_agnostic
    python nnUnet/benchmark_cpu.py \
        -i  sub-001_T2w.nii.gz \
        -gt sub-001_T2w_seg.nii.gz \
        --model /path/to/nnunet_seg.onnx \
        --plans /path/to/plans.json

    # Multiple images (one per line in a text file):
    python nnUnet/benchmark_cpu.py \
        --image-list images.txt \
        --gt-list    labels.txt \
        --model /path/to/nnunet_seg.onnx \
        --plans /path/to/plans.json \
        --output benchmark_results.csv

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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    _onnx_dir = os.path.expanduser('~/nnunet-v2/nnUNet_results/'
                                   'Dataset1000_TempContrastAgnosticCropped/'
                                   'nnUNetTrainer__nnUNetPlans__3d_fullres/onnx')
    parser = argparse.ArgumentParser(
        description='CPU benchmark: sct_deepseg vs sc_crop + ONNX nnUNet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input — single or list
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('-i', '--image', help='Single input NIfTI image (.nii.gz)')
    grp.add_argument('--image-list', help='Text file with one image path per line')

    parser.add_argument('-gt', '--gt', default=None,
                        help='Ground-truth segmentation for Dice (single image mode)')
    parser.add_argument('--gt-list', default=None,
                        help='Text file with one GT path per line (list mode)')
    parser.add_argument('--model', default=os.path.join(_onnx_dir, 'nnunet_seg.onnx'),
                        help='Path to nnunet_seg.onnx')
    parser.add_argument('--plans', default=os.path.join(_onnx_dir, 'plans.json'),
                        help='Path to plans.json')
    parser.add_argument('--output', default='benchmark_cpu_results.csv',
                        help='Output CSV file')
    return parser.parse_args()


# ── Dice ──────────────────────────────────────────────────────────────────────

def dice(pred_path, gt_path):
    """Compute Dice score between two binary NIfTI masks (reprojected to GT space)."""
    pred = nib.load(pred_path).get_fdata() > 0.5
    gt   = nib.load(gt_path).get_fdata()   > 0.5
    assert pred.shape == gt.shape, (
        f'Shape mismatch pred {pred.shape} vs gt {gt.shape}\n'
        f'pred: {pred_path}\ngt:   {gt_path}'
    )
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2 * intersection / denom) if denom > 0 else 1.0


# ── Method A — sct_deepseg ────────────────────────────────────────────────────

def run_sct_deepseg(image_path, out_path):
    """Run sct_deepseg contrast-agnostic on CPU. Returns elapsed seconds."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    cmd = (
        f'sct_deepseg spinalcord -i {image_path} -o {out_path}'
    )
    t0 = time.perf_counter()
    ret = subprocess.run(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.perf_counter() - t0
    assert ret.returncode == 0, (
        f'sct_deepseg failed (code {ret.returncode}):\n{ret.stderr.decode()}'
    )
    return elapsed


# ── Method B — sc_crop + ONNX ─────────────────────────────────────────────────

def run_sc_crop_onnx(image_path, out_path, model, plans):
    """Run sc_crop + ONNX nnUNet pipeline on CPU. Returns (sc_crop_s, onnx_s, total_s)."""
    script = Path(__file__).parent / 'run_inference.py'
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    cmd = (
        f'python {script} -i {image_path} -o {out_path} '
        f'--mode onnx --model {model} --plans {plans} --time'
    )
    t0 = time.perf_counter()
    ret = subprocess.run(cmd, shell=True, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    total = time.perf_counter() - t0
    assert ret.returncode == 0, (
        f'run_inference.py (ONNX) failed (code {ret.returncode}):\n{ret.stderr.decode()}'
    )
    stdout = ret.stdout.decode()
    sc_crop_s, onnx_s = _parse_timing(stdout)
    return sc_crop_s, onnx_s, total


def _parse_timing(stdout):
    """Extract sc_crop and inference times from run_inference.py --time output.

    Expected format (inside a ─── box):
      sc_crop   : 1.02s
      inference : 10.44s
    """
    sc_crop_s = onnx_s = None
    for line in stdout.splitlines():
        stripped = line.strip()
        # Match "sc_crop   : 1.02s" but not "sc_crop bbox : x=[...]"
        if stripped.startswith('sc_crop') and ':' in stripped:
            val = stripped.split(':', 1)[1].strip()
            if val.endswith('s') and val[:-1].replace('.', '', 1).isdigit():
                sc_crop_s = float(val.rstrip('s'))
        elif stripped.startswith('inference') and ':' in stripped:
            onnx_s = float(stripped.split(':')[1].strip().rstrip('s'))
    assert sc_crop_s is not None and onnx_s is not None, (
        f'Could not parse timing from output:\n{stdout}'
    )
    return sc_crop_s, onnx_s


# ── Per-image benchmark ───────────────────────────────────────────────────────

def benchmark_one(image_path, gt_path, model, plans):
    """Run both methods on one image. Returns a result dict."""
    image_path = str(image_path)
    name = Path(image_path).name

    print(f'\n{"─"*60}')
    print(f'Image : {name}')
    print(f'GT    : {Path(gt_path).name if gt_path else "none"}')

    row = {'image': name}

    with tempfile.TemporaryDirectory() as tmp:
        # Method A — sct_deepseg
        out_sct = os.path.join(tmp, 'seg_sct.nii.gz')
        print('  [A] sct_deepseg (CPU)...', end='', flush=True)
        sct_time = run_sct_deepseg(image_path, out_sct)
        print(f' {sct_time:.1f}s')
        row['sct_time_s'] = round(sct_time, 2)
        if gt_path:
            row['sct_dice'] = round(dice(out_sct, gt_path), 4)
            print(f'      Dice = {row["sct_dice"]:.4f}')

        # Method B — sc_crop + ONNX
        out_onnx = os.path.join(tmp, 'seg_onnx.nii.gz')
        print('  [B] sc_crop + ONNX (CPU)...', end='', flush=True)
        sc_crop_s, onnx_s, total_s = run_sc_crop_onnx(image_path, out_onnx, model, plans)
        print(f' sc_crop={sc_crop_s:.1f}s  onnx={onnx_s:.1f}s  total={total_s:.1f}s')
        row['sc_crop_time_s']     = round(sc_crop_s, 2)
        row['onnx_infer_time_s']  = round(onnx_s, 2)
        row['sc_crop_onnx_total_s'] = round(total_s, 2)
        if gt_path:
            row['sc_crop_onnx_dice'] = round(dice(out_onnx, gt_path), 4)
            print(f'      Dice = {row["sc_crop_onnx_dice"]:.4f}')

    if gt_path:
        speedup = row['sct_time_s'] / row['sc_crop_onnx_total_s']
        print(f'  Speedup sc_crop+ONNX vs sct_deepseg: ×{speedup:.1f}')

    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    assert os.path.isfile(args.model), f'ONNX model not found: {args.model}'
    assert os.path.isfile(args.plans), f'plans.json not found: {args.plans}'

    # Build list of (image, gt_or_None) pairs
    if args.image:
        pairs = [(args.image, args.gt)]
    else:
        images = Path(args.image_list).read_text().splitlines()
        gts    = (Path(args.gt_list).read_text().splitlines()
                  if args.gt_list else [None] * len(images))
        assert len(images) == len(gts), 'image-list and gt-list must have the same number of lines'
        pairs = list(zip(images, gts))

    rows = []
    for image_path, gt_path in pairs:
        rows.append(benchmark_one(image_path, gt_path, args.model, args.plans))

    # Write CSV
    fieldnames = sorted({k for r in rows for k in r})
    fieldnames = ['image'] + [f for f in fieldnames if f != 'image']
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print(f'\n{"═"*60}')
    print(f'  Results saved → {args.output}')
    if len(rows) > 1:
        avg = lambda key: np.mean([r[key] for r in rows if key in r])
        print(f'  N = {len(rows)} images')
        print(f'  sct_deepseg        : {avg("sct_time_s"):.1f}s mean  |  Dice {avg("sct_dice"):.4f}' if 'sct_dice' in rows[0] else f'  sct_deepseg        : {avg("sct_time_s"):.1f}s mean')
        print(f'  sc_crop + ONNX     : {avg("sc_crop_onnx_total_s"):.1f}s mean  |  Dice {avg("sc_crop_onnx_dice"):.4f}' if 'sc_crop_onnx_dice' in rows[0] else f'  sc_crop + ONNX     : {avg("sc_crop_onnx_total_s"):.1f}s mean')
        print(f'  Speedup (mean)     : ×{avg("sct_time_s") / avg("sc_crop_onnx_total_s"):.1f}')
    print(f'{"═"*60}')


if __name__ == '__main__':
    main()
