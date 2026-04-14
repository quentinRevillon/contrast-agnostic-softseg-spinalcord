"""
Evaluate SC segmentation robustness to GT-bbox cropping.

For each subject in the frozen test set and each padding level, crops the MRI
around the 3D GT bounding box + padding_mm, runs sct_deepseg, embeds the
prediction back in the original space, and computes Dice against GT.

Padding -1 = no crop (full image, baseline).

Usage:
    python cropping_YOLO/eval_crop_robustness.py \
        --data-root ~/data/data-multi-subject \
        --output results_crop_robustness.csv \
        [--paddings 0 5 10 20 40 -1] \
        [--sct-bin /path/to/sct_deepseg]
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

GT_SUFFIX = "_desc-softseg_label-SC_seg.nii.gz"
TEST_SPLIT = (
    Path(__file__).parent.parent
    / "scripts"
    / "spine_generic_test_split_for_csa_drift_monitoring.yaml"
)


def load_test_subjects(yaml_path):
    return yaml.safe_load(yaml_path.read_text())["test"]


def discover_contrasts(data_root, subject):
    """Return [(contrast_stem, img_path, gt_path)] from GT files present on disk."""
    gt_dir = data_root / "derivatives" / "labels_softseg_bin" / subject / "anat"
    pairs = []
    for gt_path in sorted(gt_dir.glob(f"{subject}_*{GT_SUFFIX}")):
        contrast = gt_path.name[len(subject) + 1 : -len(GT_SUFFIX)]
        img_path = data_root / subject / "anat" / f"{subject}_{contrast}.nii.gz"
        assert img_path.exists(), f"Image not found: {img_path}"
        pairs.append((contrast, img_path, gt_path))
    return pairs


def voxel_sizes(affine):
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def gt_bbox(mask_data):
    """Tight (lo, hi) bounding box per axis of nonzero voxels."""
    coords = np.argwhere(mask_data > 0.5)
    return [(int(coords[:, i].min()), int(coords[:, i].max()) + 1) for i in range(3)]


def padded_bbox(bbox, padding_vox, shape):
    return [(max(0, lo - p), min(s, hi + p)) for (lo, hi), p, s in zip(bbox, padding_vox, shape)]


def crop_nii(nii, bbox):
    data = nii.get_fdata()[tuple(slice(lo, hi) for lo, hi in bbox)]
    offset = np.array([lo for lo, _ in bbox] + [0], dtype=float)
    new_affine = nii.affine.copy()
    new_affine[:, 3] = nii.affine @ offset
    return nib.Nifti1Image(data, new_affine)


def embed_pred(pred_data, orig_shape, bbox):
    full = np.zeros(orig_shape, dtype=pred_data.dtype)
    full[tuple(slice(lo, hi) for lo, hi in bbox)] = pred_data
    return full


def dice(pred, gt):
    pred_bin = pred > 0.5
    gt_bin = gt > 0.5
    return 2 * (pred_bin & gt_bin).sum() / (pred_bin.sum() + gt_bin.sum())


def run_deepseg(img_path, out_path, sct_bin):
    subprocess.run(
        [sct_bin, "spinalcord", "-i", str(img_path), "-o", str(out_path), "-v", "0"],
        check=True,
    )


def evaluate_one(subject, contrast, img_path, gt_path, paddings, sct_bin, tmpdir):
    img_nii = nib.load(img_path)
    gt_data = nib.load(gt_path).get_fdata()
    vox = voxel_sizes(img_nii.affine)
    bbox_tight = gt_bbox(gt_data)

    rows = []
    for pad_mm in paddings:
        if pad_mm == -1:
            input_nii, bbox = img_nii, None
        else:
            bbox = padded_bbox(bbox_tight, np.round(pad_mm / vox).astype(int), img_nii.shape)
            input_nii = crop_nii(img_nii, bbox)

        stem = f"{subject}_{contrast}_pad{pad_mm}"
        img_tmp = tmpdir / f"{stem}.nii.gz"
        pred_tmp = tmpdir / f"{stem}_pred.nii.gz"
        nib.save(input_nii, img_tmp)
        run_deepseg(img_tmp, pred_tmp, sct_bin)

        pred_data = nib.load(pred_tmp).get_fdata()
        if bbox is not None:
            pred_data = embed_pred(pred_data, img_nii.shape, bbox)

        d = round(float(dice(pred_data, gt_data)), 4)
        rows.append({"subject": subject, "contrast": contrast, "padding_mm": pad_mm, "dice": d})
        print(f"  {contrast:<25} pad={str(pad_mm):>3} mm  dice={d:.4f}")

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", default="results_crop_robustness.csv", type=Path)
    parser.add_argument("--paddings", nargs="+", type=float, default=[-1, 0, 5, 10, 20, 40],
                        help="Padding in mm around GT bbox. -1 = no crop (baseline).")
    parser.add_argument("--sct-bin", default=shutil.which("sct_deepseg") or
                        "/home/quentinr/spinalcordtoolbox/bin/sct_deepseg")
    args = parser.parse_args()

    subjects = load_test_subjects(TEST_SPLIT)

    all_rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for i, subject in enumerate(subjects):
            print(f"[{i+1}/{len(subjects)}] {subject}")
            for contrast, img_path, gt_path in discover_contrasts(args.data_root, subject):
                rows = evaluate_one(
                    subject, contrast, img_path, gt_path,
                    args.paddings, args.sct_bin, tmpdir,
                )
                all_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv(args.output, index=False)
    print(f"\nSaved {len(all_rows)} rows → {args.output}")


if __name__ == "__main__":
    main()
