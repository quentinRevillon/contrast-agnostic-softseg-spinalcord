"""
Find the 3 (subject, contrast) pairs with highest Dice variance across all
paddings (including baseline=-1) and save all intermediate files for inspection
in FSLeyes.

Reuses all functions from eval_seg_robustness_with_cropping_nnunet.py.

Output structure per case:
    images/{subject}_{contrast}/
        original_image.nii.gz         → symlink to raw image
        original_gt.nii.gz            → symlink to GT mask
        pred_baseline_fullspace.nii.gz
        image_crop_pad{N}mm.nii.gz    (cropped image, one per crop padding)
        gt_crop_pad{N}mm.nii.gz       (cropped GT, one per crop padding)
        pred_crop_pad{N}mm.nii.gz     (inference in cropped space)
        pred_fullspace_pad{N}mm.nii.gz (inference embedded in original space)

Usage (single contrast for quick check):
    python eval_seg_robustness_with_cropping/save_worst_cases.py \
        --csv results_crop_robustness.csv \
        --data-root /home/quentinr/data/data-multi-subject \
        --contrast T1w \
        --device cpu

Full run (all contrasts, top-3):
    python eval_seg_robustness_with_cropping/save_worst_cases.py \
        --csv results_crop_robustness.csv \
        --data-root /home/quentinr/data/data-multi-subject \
        --device cuda
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import nibabel.orientations as nib_ornt
import numpy as np
import pandas as pd
import torch

# reuse all pipeline functions from the evaluation script
sys.path.insert(0, str(Path(__file__).parent))
from eval_seg_robustness_with_cropping_nnunet import (
    GT_SUFFIX,
    MODEL_DIR,
    crop_nii,
    dice,
    embed_pred,
    gt_bbox,
    load_predictor,
    padded_bbox,
    reorient_nii,
    run_prediction,
    voxel_sizes,
)

OUT_DIR = Path(__file__).parent / "images"


def find_worst_cases(csv_path: Path, contrast_filter, n: int = 3):
    """Return top-n (subject, contrast) pairs per contrast by Dice variance across all paddings.
    If contrast_filter is set, returns top-n for that contrast only."""
    df = pd.read_csv(csv_path)
    if contrast_filter:
        df = df[df["contrast"] == contrast_filter]
    var = df.groupby(["subject", "contrast"])["dice"].var().reset_index()
    var = var.rename(columns={"dice": "dice_var"})
    worst = var.groupby("contrast", group_keys=False).apply(lambda g: g.nlargest(n, "dice_var"))
    print(f"\nTop-{n} per contrast (highest Dice variance across all paddings):")
    print(worst.to_string(index=False))
    return list(zip(worst["subject"], worst["contrast"]))


def find_paths(data_root: Path, subject: str, contrast: str) -> tuple[Path, Path]:
    img_path = data_root / subject / "anat" / f"{subject}_{contrast}.nii.gz"
    gt_path = data_root / "derivatives" / "labels_softseg_bin" / subject / "anat" / f"{subject}_{contrast}{GT_SUFFIX}"
    assert img_path.exists(), f"Image not found: {img_path}"
    assert gt_path.exists(), f"GT not found: {gt_path}"
    return img_path, gt_path


def save_case(subject: str, contrast: str, img_path: Path, gt_path: Path,
              predictor, paddings: list[float], csv_dice: dict, device) -> None:
    out = OUT_DIR / f"{subject}_{contrast}"
    out.mkdir(parents=True, exist_ok=True)

    # symlinks to originals
    for name, src in [("original_image.nii.gz", img_path), ("original_gt.nii.gz", gt_path)]:
        link = out / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(src.resolve())

    img_nii = nib.load(img_path)
    gt_nii = nib.load(gt_path)
    gt_data = gt_nii.get_fdata()
    vox = voxel_sizes(img_nii.affine)
    bbox_tight = gt_bbox(gt_data)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        for pad_mm in paddings:
            if pad_mm == -1:
                # baseline: full image, no crop
                pred_data = run_prediction(predictor, img_nii, tmpdir)
                pred_nii = nib.Nifti1Image(pred_data, img_nii.affine)
                nib.save(pred_nii, out / "pred_baseline_fullspace.nii.gz")

                d_computed = round(float(dice(pred_data, gt_data)), 4)
                d_expected = csv_dice.get(pad_mm)
                print(f"  baseline  dice={d_computed:.4f}  (csv={d_expected:.4f})  {'OK' if abs(d_computed - d_expected) < 0.001 else 'MISMATCH'}")

            else:
                label = f"pad{int(pad_mm)}mm"
                bbox = padded_bbox(bbox_tight, np.round(pad_mm / vox).astype(int), img_nii.shape)

                # cropped image and GT
                img_crop = crop_nii(img_nii, bbox)
                gt_crop = crop_nii(gt_nii, bbox)
                nib.save(img_crop, out / f"image_crop_{label}.nii.gz")
                nib.save(gt_crop, out / f"gt_crop_{label}.nii.gz")

                # inference in cropped space
                pred_crop_data = run_prediction(predictor, img_crop, tmpdir)
                pred_crop_nii = nib.Nifti1Image(pred_crop_data, img_crop.affine)
                nib.save(pred_crop_nii, out / f"pred_crop_{label}.nii.gz")

                # inference embedded back in full image space
                pred_full_data = embed_pred(pred_crop_data, img_nii.shape, bbox)
                pred_full_nii = nib.Nifti1Image(pred_full_data, img_nii.affine)
                nib.save(pred_full_nii, out / f"pred_fullspace_{label}.nii.gz")

                d_computed = round(float(dice(pred_full_data, gt_data)), 4)
                d_expected = csv_dice.get(pad_mm)
                print(f"  {label:<12} dice={d_computed:.4f}  (csv={d_expected:.4f})  {'OK' if abs(d_computed - d_expected) < 0.001 else 'MISMATCH'}")

    print(f"  → saved to {out}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model-dir", default=MODEL_DIR, type=Path)
    parser.add_argument("--contrast", default=None, type=str,
                        help="Restrict to one contrast for quick testing (e.g. T1w).")
    parser.add_argument("--n-cases", default=3, type=int,
                        help="Number of worst cases to save (default: 3).")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading predictor on {device}...")
    predictor = load_predictor(args.model_dir, device)

    worst = find_worst_cases(args.csv, args.contrast, args.n_cases)

    df = pd.read_csv(args.csv)

    for subject, contrast in worst:
        print(f"\n[{subject} / {contrast}]")
        img_path, gt_path = find_paths(args.data_root, subject, contrast)

        sub_df = df[(df["subject"] == subject) & (df["contrast"] == contrast)]
        paddings = sorted(sub_df["padding_mm"].tolist(), key=lambda x: (x == -1, x))
        csv_dice = dict(zip(sub_df["padding_mm"], sub_df["dice"]))

        save_case(subject, contrast, img_path, gt_path, predictor, paddings, csv_dice, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
