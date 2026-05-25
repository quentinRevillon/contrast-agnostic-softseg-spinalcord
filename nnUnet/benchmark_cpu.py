"""
CPU inference benchmark: sct_deepseg vs sc_crop + ONNX vs sc_crop + PyTorch.

For each input image:
  Method A — SCT contrast-agnostic:
    sct_deepseg spinalcord  (full volume, CPU)

  Method B — sc_crop + ONNX (this work, CPU-only deployment):
    sc_crop detection → crop → ONNX nnUNet → restore to full space

  Method C — sc_crop + PyTorch (this work, GPU or CPU):
    sc_crop detection → crop → PyTorch nnUNet → restore to full space

Dice vs GT is computed when --gt is provided.
GPU is disabled for methods A and B (CUDA_VISIBLE_DEVICES="").
Method C uses CPU by default; pass --pt-use-gpu to enable GPU.

Usage:
    conda activate contrast_agnostic
    python nnUnet/benchmark_cpu.py \
        -i  sub-001_T2w.nii.gz \
        -gt sub-001_T2w_seg.nii.gz \
        --model      /path/to/nnunet_seg.onnx \
        --plans      /path/to/plans.json \
        --model-folder /path/to/nnUNetTrainer__nnUNetPlans__3d_fullres

    # Multiple images (one per line in a text file):
    python nnUnet/benchmark_cpu.py \
        --image-list images.txt \
        --gt-list    labels.txt \
        --model      /path/to/nnunet_seg.onnx \
        --plans      /path/to/plans.json \
        --model-folder /path/to/nnUNetTrainer__nnUNetPlans__3d_fullres \
        --output benchmark_results.csv

Author: Quentin Revillon
"""

import argparse
import csv
import os
import subprocess
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np


# ── CLI ───────────────────────────────────────────────────────────────────────

_RESULTS_DIR = os.path.expanduser(
    '~/nnunet-v2/nnUNet_results/Dataset1000_TempContrastAgnosticCropped/'
    'nnUNetTrainer__nnUNetPlans__3d_fullres'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='CPU benchmark: sct_deepseg vs sc_crop+ONNX vs sc_crop+PT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('-i', '--image', help='Single input NIfTI image (.nii.gz)')
    grp.add_argument('--image-list', help='Text file with one image path per line')

    parser.add_argument('-gt', '--gt', default=None,
                        help='GT segmentation for Dice (single image mode)')
    parser.add_argument('--gt-list', default=None,
                        help='Text file with one GT path per line (list mode)')

    # ONNX model
    parser.add_argument('--model',
                        default=os.path.join(_RESULTS_DIR, 'onnx', 'nnunet_seg.onnx'),
                        help='Path to nnunet_seg.onnx (Method B)')
    parser.add_argument('--plans',
                        default=os.path.join(_RESULTS_DIR, 'onnx', 'plans.json'),
                        help='Path to plans.json (Method B)')

    # PyTorch model
    parser.add_argument('--model-folder', default=_RESULTS_DIR,
                        help='Path to nnUNetTrainer__nnUNetPlans__3d_fullres folder (Method C)')
    parser.add_argument('--pt-use-gpu', action='store_true', default=False,
                        help='Enable GPU for Method C (sc_crop + PyTorch). Default: CPU')

    parser.add_argument('--output', default='benchmark_cpu_results.csv',
                        help='Output CSV file')
    return parser.parse_args()


# ── Dice ──────────────────────────────────────────────────────────────────────

def dice(pred_path, gt_path):
    """Dice between two binary NIfTI masks (must share the same voxel space)."""
    pred = nib.load(pred_path).get_fdata() > 0.5
    gt   = nib.load(gt_path).get_fdata()   > 0.5
    assert pred.shape == gt.shape, (
        f'Shape mismatch pred {pred.shape} vs gt {gt.shape} — '
        f'pred: {pred_path}  gt: {gt_path}'
    )
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2 * intersection / denom) if denom > 0 else 1.0


# ── Method A — sct_deepseg ────────────────────────────────────────────────────

def run_sct_deepseg(image_path, out_path):
    """Run sct_deepseg spinalcord on CPU. Returns elapsed seconds."""
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
    cmd = f'sct_deepseg spinalcord -i {image_path} -o {out_path}'
    t0  = time.perf_counter()
    ret = subprocess.run(cmd, shell=True, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.perf_counter() - t0
    assert ret.returncode == 0, (
        f'sct_deepseg failed (code {ret.returncode}):\n{ret.stderr.decode()}'
    )
    return elapsed


# ── Methods B & C — run_inference.py ─────────────────────────────────────────

def run_sc_crop_inference(image_path, out_path, mode, model=None, plans=None,
                          model_folder=None, use_gpu=False):
    """Run run_inference.py in the given mode. Returns (sc_crop_s, infer_s, total_s)."""
    script = Path(__file__).parent / 'run_inference.py'
    env    = {**os.environ, 'CUDA_VISIBLE_DEVICES': '' if not use_gpu else '0'}

    if mode == 'onnx':
        extra = f'--model {model} --plans {plans}'
    else:
        extra = f'--model-folder {model_folder}'

    cmd = f'python {script} -i {image_path} -o {out_path} --mode {mode} {extra} --time'
    t0  = time.perf_counter()
    ret = subprocess.run(cmd, shell=True, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    total = time.perf_counter() - t0
    assert ret.returncode == 0, (
        f'run_inference.py --mode {mode} failed (code {ret.returncode}):\n{ret.stderr.decode()}'
    )
    sc_crop_s, infer_s = _parse_timing(ret.stdout.decode())
    return sc_crop_s, infer_s, total


def _parse_timing(stdout):
    """Extract (sc_crop_s, inference_s) from run_inference.py --time stdout.

    Expected lines (inside ─── box):
      sc_crop   : 1.02s
      inference : 10.44s
    """
    sc_crop_s = infer_s = None
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith('sc_crop') and ':' in s:
            val = s.split(':', 1)[1].strip()
            # Reject "sc_crop bbox : x=[...]" — only accept numeric values
            if val.endswith('s') and val[:-1].replace('.', '', 1).isdigit():
                sc_crop_s = float(val[:-1])
        elif s.startswith('inference') and ':' in s:
            infer_s = float(s.split(':', 1)[1].strip().rstrip('s'))
    assert sc_crop_s is not None and infer_s is not None, (
        f'Could not parse timing from output:\n{stdout}'
    )
    return sc_crop_s, infer_s


# ── Per-image benchmark ───────────────────────────────────────────────────────

def benchmark_one(image_path, gt_path, args):
    image_path = str(image_path)
    name = Path(image_path).name

    print(f'\n{"─"*62}')
    print(f'Image : {name}')
    print(f'GT    : {Path(gt_path).name if gt_path else "none"}')

    row = {'image': name}

    with tempfile.TemporaryDirectory() as tmp:

        # Method A — sct_deepseg
        out_a = os.path.join(tmp, 'seg_sct.nii.gz')
        print('  [A] sct_deepseg (CPU)...', end='', flush=True)
        row['sct_time_s'] = round(run_sct_deepseg(image_path, out_a), 2)
        print(f' {row["sct_time_s"]:.1f}s')
        if gt_path:
            row['sct_dice'] = round(dice(out_a, gt_path), 4)
            print(f'      Dice = {row["sct_dice"]:.4f}')

        # Method B — sc_crop + ONNX
        out_b = os.path.join(tmp, 'seg_onnx.nii.gz')
        print('  [B] sc_crop + ONNX (CPU)...', end='', flush=True)
        sc_s, onnx_s, total_b = run_sc_crop_inference(
            image_path, out_b, 'onnx',
            model=args.model, plans=args.plans,
        )
        row['sc_crop_time_s']       = round(sc_s, 2)
        row['onnx_infer_time_s']    = round(onnx_s, 2)
        row['sc_crop_onnx_total_s'] = round(total_b, 2)
        print(f' sc_crop={sc_s:.1f}s  onnx={onnx_s:.1f}s  total={total_b:.1f}s')
        if gt_path:
            row['sc_crop_onnx_dice'] = round(dice(out_b, gt_path), 4)
            print(f'      Dice = {row["sc_crop_onnx_dice"]:.4f}')

        # Method C — sc_crop + PyTorch
        out_c = os.path.join(tmp, 'seg_pt.nii.gz')
        device_label = 'GPU' if args.pt_use_gpu else 'CPU'
        print(f'  [C] sc_crop + PyTorch ({device_label})...', end='', flush=True)
        sc_s2, pt_s, total_c = run_sc_crop_inference(
            image_path, out_c, 'pt',
            model_folder=args.model_folder,
            use_gpu=args.pt_use_gpu,
        )
        row['pt_infer_time_s']   = round(pt_s, 2)
        row['sc_crop_pt_total_s'] = round(total_c, 2)
        print(f' sc_crop={sc_s2:.1f}s  pt={pt_s:.1f}s  total={total_c:.1f}s')
        if gt_path:
            row['sc_crop_pt_dice'] = round(dice(out_c, gt_path), 4)
            print(f'      Dice = {row["sc_crop_pt_dice"]:.4f}')

    # Speedups vs sct_deepseg
    print(f'  Speedup ONNX vs sct: ×{row["sct_time_s"] / row["sc_crop_onnx_total_s"]:.1f}')
    print(f'  Speedup PT   vs sct: ×{row["sct_time_s"] / row["sc_crop_pt_total_s"]:.1f}')

    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    assert os.path.isfile(args.model),        f'ONNX model not found: {args.model}'
    assert os.path.isfile(args.plans),        f'plans.json not found: {args.plans}'
    assert os.path.isdir(args.model_folder),  f'PT model folder not found: {args.model_folder}'

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
        rows.append(benchmark_one(image_path, gt_path, args))

    # Write CSV
    fieldnames = ['image'] + sorted(k for k in {k for r in rows for k in r} if k != 'image')
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print(f'\n{"═"*62}')
    print(f'  Results saved → {args.output}')
    if len(rows) > 1:
        avg = lambda k: np.mean([r[k] for r in rows if k in r])
        has_dice = 'sct_dice' in rows[0]
        n = len(rows)
        print(f'  N = {n} images')
        print(f'  [A] sct_deepseg        : {avg("sct_time_s"):.1f}s'
              + (f'  |  Dice {avg("sct_dice"):.4f}' if has_dice else ''))
        print(f'  [B] sc_crop + ONNX     : {avg("sc_crop_onnx_total_s"):.1f}s'
              + (f'  |  Dice {avg("sc_crop_onnx_dice"):.4f}' if has_dice else '')
              + f'  (×{avg("sct_time_s") / avg("sc_crop_onnx_total_s"):.1f} faster)')
        print(f'  [C] sc_crop + PyTorch  : {avg("sc_crop_pt_total_s"):.1f}s'
              + (f'  |  Dice {avg("sc_crop_pt_dice"):.4f}' if has_dice else '')
              + f'  (×{avg("sct_time_s") / avg("sc_crop_pt_total_s"):.1f} faster)')
    print(f'{"═"*62}')


if __name__ == '__main__':
    main()
