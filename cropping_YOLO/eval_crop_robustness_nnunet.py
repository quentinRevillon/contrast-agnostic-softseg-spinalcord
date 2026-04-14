"""
Evaluate SC segmentation robustness to GT-bbox cropping (nnUNetv2, GPU/CPU).

Identical experiment to eval_crop_robustness.py but uses nnUNetPredictor
directly (no sct_deepseg subprocess), enabling GPU inference. The predictor
is loaded once before the loop for efficiency.

Requires nnunetv2 installed in the current environment:
    pip install nnunetv2

Padding -1 = no crop (full image, baseline).

Usage:
    python cropping_YOLO/eval_crop_robustness_nnunet.py \
        --data-root ~/data/data-multi-subject \
        --model-dir /home/quentinr/spinalcordtoolbox/data/deepseg_models/model_seg_sc_contrast_agnostic_nnunet/nnUNetTrainer__nnUNetPlans__3d_fullres \
        --output results_crop_robustness.csv \
        [--paddings 0 5 10 20 40 -1] \
        [--device cuda]
"""

import argparse
import contextlib
import os
import tempfile
from pathlib import Path

import nibabel as nib
import nibabel.orientations as nib_ornt
import numpy as np
import pandas as pd
import yaml

# nnUNet requires these env vars at import time (dummy values silence warnings)
os.environ.setdefault("nnUNet_raw", "/tmp")
os.environ.setdefault("nnUNet_preprocessed", "/tmp")
os.environ.setdefault("nnUNet_results", "/tmp")

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402

GT_SUFFIX = "_desc-softseg_label-SC_seg.nii.gz"
MODEL_DIR = (
    Path("/home/quentinr/spinalcordtoolbox/data/deepseg_models")
    / "model_seg_sc_contrast_agnostic_nnunet"
    / "nnUNetTrainer__nnUNetPlans__3d_fullres"
)
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


def load_predictor(model_dir, device):
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=(device.type == "cuda"),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=[int(f.name.split("_")[1]) for f in sorted(model_dir.glob("fold_*"))],
        checkpoint_name="checkpoint_best.pth",
    )
    return predictor


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


def reorient_nii(nii, target):
    """Reorient NIfTI to target axcodes tuple, return (reoriented_nii, orig_axcodes)."""
    orig = nib_ornt.aff2axcodes(nii.affine)
    if orig == target:
        return nii, orig
    transform = nib_ornt.ornt_transform(
        nib_ornt.axcodes2ornt(orig), nib_ornt.axcodes2ornt(target)
    )
    data = nib_ornt.apply_orientation(nii.get_fdata(), transform)
    affine = nii.affine @ nib_ornt.inv_ornt_aff(transform, nii.shape)
    return nib.Nifti1Image(data, affine), orig


def embed_pred(pred_data, orig_shape, bbox):
    full = np.zeros(orig_shape, dtype=pred_data.dtype)
    full[tuple(slice(lo, hi) for lo, hi in bbox)] = pred_data
    return full


def dice(pred, gt):
    pred_bin = pred > 0.5
    gt_bin = gt > 0.5
    return 2 * (pred_bin & gt_bin).sum() / (pred_bin.sum() + gt_bin.sum())


def run_prediction(predictor, nii, tmpdir):
    """Reorient to RPI, predict with nnUNet, reorient back. Returns pred array in original space."""
    rpi_nii, orig_axcodes = reorient_nii(nii, ("R", "P", "I"))

    img_tmp = tmpdir / "input.nii.gz"
    pred_dir = tmpdir / "pred"
    pred_dir.mkdir(exist_ok=True)
    nib.save(rpi_nii, img_tmp)

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            predictor.predict_from_files(
                [[str(img_tmp)]],
                [str(pred_dir / "pred.nii.gz")],
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )

    pred_nii = nib.load(pred_dir / "pred.nii.gz")
    pred_orig, _ = reorient_nii(pred_nii, orig_axcodes)
    return pred_orig.get_fdata()


def evaluate_one(subject, contrast, img_path, gt_path, paddings, predictor, tmpdir, vol_idx, n_vols):
    img_nii = nib.load(img_path)
    gt_data = nib.load(gt_path).get_fdata()
    vox = voxel_sizes(img_nii.affine)
    bbox_tight = gt_bbox(gt_data)

    rows = []
    n_pads = len(paddings)
    for k, pad_mm in enumerate(paddings):
        pad_label = "baseline" if pad_mm == -1 else f"{int(pad_mm):>2}mm"
        print(f"  [vol {vol_idx}/{n_vols}] {contrast:<25}  pad {k+1}/{n_pads} ({pad_label}) ...", end=" ", flush=True)

        if pad_mm == -1:
            input_nii, bbox = img_nii, None
        else:
            bbox = padded_bbox(bbox_tight, np.round(pad_mm / vox).astype(int), img_nii.shape)
            input_nii = crop_nii(img_nii, bbox)

        pred_data = run_prediction(predictor, input_nii, tmpdir)
        if bbox is not None:
            pred_data = embed_pred(pred_data, img_nii.shape, bbox)

        d = round(float(dice(pred_data, gt_data)), 4)
        rows.append({"subject": subject, "contrast": contrast, "padding_mm": pad_mm, "dice": d})
        print(f"dice={d:.4f}")

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model-dir", default=MODEL_DIR, type=Path)
    parser.add_argument("--output", default="results_crop_robustness.csv", type=Path)
    parser.add_argument("--paddings", nargs="+", type=float, default=[-1, 0, 5, 10, 20, 40],
                        help="Padding in mm around GT bbox. -1 = no crop (baseline).")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading predictor on {device}...")
    predictor = load_predictor(args.model_dir, device)

    subjects = load_test_subjects(TEST_SPLIT)

    # count total volumes upfront for progress display
    all_contrasts = [
        (subject, contrast, img_path, gt_path)
        for subject in subjects
        for contrast, img_path, gt_path in discover_contrasts(args.data_root, subject)
    ]
    n_vols = len(all_contrasts)
    n_subjects = len(subjects)
    print(f"Found {n_vols} volumes across {n_subjects} subjects — {len(args.paddings)} paddings each = {n_vols * len(args.paddings)} inferences\n")

    all_rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        prev_subject = None
        subj_num = 0
        for vol_idx, (subject, contrast, img_path, gt_path) in enumerate(all_contrasts, 1):
            if subject != prev_subject:
                subj_num += 1
                print(f"[subject {subj_num}/{n_subjects}] {subject}")
                prev_subject = subject
            rows = evaluate_one(
                subject, contrast, img_path, gt_path,
                args.paddings, predictor, tmpdir, vol_idx, n_vols,
            )
            all_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv(args.output, index=False)
    print(f"\nSaved {len(all_rows)} rows → {args.output}")


if __name__ == "__main__":
    main()
