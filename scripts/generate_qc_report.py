"""
Generate QC reports comparing v4 (sc-crop + nnUNet) vs v3 (sct_deepseg spinalcord)
on the test set, with crop QC included.

Three QC groups in a single index.html:
  - crop   : cropped image + cropped GT — verifies sc_crop didn't cut the SC
  - seg_v4 : original image + v4 prediction
  - seg_v3 : original image + v3 prediction (sct_deepseg spinalcord)

Also produces:
  - metrics.json : Dice v3/v4 per subject + aggregate stats + crop QC per subject
  - run.log      : full stdout log

Usage:
    python scripts/generate_qc_report.py \
        --dataset-dir /path/to/nnUNet_raw/Dataset4000_ContrastAgnosticScCrop \
        --checkpoint   /path/to/fold_0/checkpoint_best.pth \
        --output-dir   /path/to/qc_results

Author: Quentin Revillon
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from sc_crop import detect, check_label_crop

# SCT model directory — create_nnunet_from_plans selects the checkpoint
# exactly as sct_deepseg does (checkpoint_final.pth first, checkpoint_best.pth as fallback)
V3_MODEL_DIR = (
    "/home/quentinr/spinalcordtoolbox/data/deepseg_models"
    "/model_seg_sc_contrast_agnostic_nnunet"
)

_CONTRAST_PATTERN = (
    r'.*(T1w|T2w|acq-sagthor_T2w|acq-sagcerv_T2w|acq-sagstir_T2w|acq-ax_T2w'
    r'|T2star|PSIR|STIR|UNIT1|flip-1_mt-on_MTS|flip-2_mt-off_MTS'
    r'|acq-MTon_MTR|acq-dwiMean_dwi|rec-average_dwi|acq-T1w_MTR).*'
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True,
                        help="Path to nnUNet_raw/DatasetXXXX_... folder")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to fold_0/checkpoint_best.pth")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write seg_v4/, seg_v3/, qc/, metrics.json, run.log")
    parser.add_argument("--n-subjects", type=int, default=None,
                        help="Process only the first N subjects (for testing)")
    return parser.parse_args()


class Logger:
    def __init__(self, path: Path):
        self._f = open(path, "w")

    def log(self, msg: str = "") -> None:
        print(msg)
        self._f.write(msg + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def run(cmd: list, logger: Logger) -> None:
    logger.log(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def subject_name(path: str) -> str:
    return Path(path).name.replace(".nii.gz", "").replace(".nii", "")


def dataset_name(path: str) -> str:
    parts = Path(path).parts
    for i, p in enumerate(parts):
        if "datasets_contrast_agnostic" in p and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def contrast_name(path: str) -> str:
    match = re.search(_CONTRAST_PATTERN, path)
    return match.group(1) if match else "unknown"


def infer_v3_gpu(orig_image: str, seg_v3: Path) -> None:
    """Run v3 inference on GPU using SCT's exact inference pipeline.

    Replicates sct_deepseg spinalcord behaviour (predict_single_npy_array +
    manual [2,1,0] transpose) but on GPU instead of CPU.
    """
    import sys
    import tempfile
    import torch
    sys.path.insert(0, "/home/quentinr/spinalcordtoolbox")
    from spinalcordtoolbox.deepseg.nnunet import create_nnunet_from_plans
    from spinalcordtoolbox.deepseg.inference import segment_nnunet

    v3_model_dir = V3_MODEL_DIR
    device = torch.device("cuda")
    predictor = create_nnunet_from_plans(v3_model_dir, device)

    tmpdir = tempfile.mkdtemp()
    fnames_out, _ = segment_nnunet(path_img=orig_image, tmpdir=tmpdir,
                                   predictor=predictor, device=device)
    import shutil
    shutil.copy(fnames_out[0], seg_v3)
    shutil.rmtree(tmpdir)


def dice(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_bin   = gt   > 0
    pred_bin = pred > 0
    denom    = gt_bin.sum() + pred_bin.sum()
    return float(2 * np.logical_and(gt_bin, pred_bin).sum() / denom) if denom > 0 else 0.0


def build_test_pairs(dataset_dir: Path) -> list[dict]:
    conv = json.loads((dataset_dir / "conversion_dict.json").read_text())

    images_ts = {k: v for k, v in conv.items() if "/imagesTs/" in v}
    labels_ts  = {k: v for k, v in conv.items() if "/labelsTs/"  in v}

    label_by_id = {}
    for orig, crop in labels_ts.items():
        id_ = Path(crop).name.replace(".nii.gz", "")
        label_by_id[id_] = (orig, crop)

    pairs = []
    for orig_img, crop_img in images_ts.items():
        id_ = Path(crop_img).name.replace("_0000.nii.gz", "")
        orig_lbl, crop_lbl = label_by_id[id_]
        pairs.append({
            "orig_image": orig_img,
            "crop_image": crop_img,
            "orig_label": orig_lbl,
            "crop_label": crop_lbl,
        })

    return sorted(pairs, key=lambda x: x["crop_image"])


def main():
    args        = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    qc_dir      = output_dir / "qc"
    seg_v4_dir  = output_dir / "seg_v4"
    seg_v3_dir  = output_dir / "seg_v3"

    assert not qc_dir.exists(), \
        f"{qc_dir} already exists — delete it or use a different --output-dir"

    for d in [qc_dir, seg_v4_dir, seg_v3_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = Logger(output_dir / "run.log")
    pairs  = build_test_pairs(dataset_dir)
    if args.n_subjects:
        pairs = pairs[:args.n_subjects]
    logger.log(f"Processing {len(pairs)} test subjects")

    subjects_metrics = []

    for i, p in enumerate(pairs, 1):
        subj     = subject_name(p["orig_image"])
        dset     = dataset_name(p["orig_image"])
        contrast = contrast_name(p["orig_image"])
        seg_v4 = seg_v4_dir / f"{subj}_seg_v4.nii.gz"
        seg_v3 = seg_v3_dir / f"{subj}_seg_v3.nii.gz"

        logger.log(f"\n[{i}/{len(pairs)}] {subj} ({dset})")

        # Crop QC — detect on original image, check GT label
        bbox     = detect(p["orig_image"])
        crop_qc  = check_label_crop(nib.load(p["orig_label"]), bbox)
        logger.log(f"  crop_ok={crop_qc['ok']}  "
                   f"voxels_before={crop_qc['voxels_before']}  "
                   f"voxels_after={crop_qc['voxels_after']}")

        # Crop QC — visual
        run(["sct_qc",
             "-i", p["crop_image"], "-s", p["crop_label"],
             "-p", "sct_deepseg_sc",
             "-qc", str(qc_dir), "-qc-subject", f"{dset}/{subj}", "-qc-dataset", "crop"],
            logger)

        # v4 inference
        run(["sc-segment-pt",
             "-i", p["orig_image"], "-o", str(seg_v4),
             "--checkpoint", args.checkpoint,
             "--device", "cuda"],
            logger)

        # v3 inference — GPU, no sc-crop (replicates sct_deepseg internal pipeline)
        logger.log(f"  $ infer_v3_gpu {p['orig_image']}")
        infer_v3_gpu(p["orig_image"], seg_v3)

        # Dice
        gt   = np.asarray(nib.load(p["orig_label"]).dataobj)
        d_v4 = dice(gt, np.asarray(nib.load(seg_v4).dataobj))
        d_v3 = dice(gt, np.asarray(nib.load(seg_v3).dataobj))
        logger.log(f"  dice_v4={d_v4:.4f}  dice_v3={d_v3:.4f}")

        subjects_metrics.append({
            "subject":        subj,
            "dataset":        dset,
            "contrast":       contrast,
            "dice_v4":        round(d_v4, 4),
            "dice_v3":        round(d_v3, 4),
            "crop_ok":        crop_qc["ok"],
            "voxels_before":  crop_qc["voxels_before"],
            "voxels_after":   crop_qc["voxels_after"],
        })

        # QC v4
        run(["sct_qc",
             "-i", p["orig_image"], "-s", str(seg_v4),
             "-p", "sct_deepseg_sc",
             "-qc", str(qc_dir), "-qc-subject", f"{dset}/{subj}", "-qc-dataset", "seg_v4"],
            logger)

        # QC v3
        run(["sct_qc",
             "-i", p["orig_image"], "-s", str(seg_v3),
             "-p", "sct_deepseg_sc",
             "-qc", str(qc_dir), "-qc-subject", f"{dset}/{subj}", "-qc-dataset", "seg_v3"],
            logger)

    # Aggregate metrics
    def _stats(values):
        a = np.array(values)
        return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4), "n": len(a)}

    dices_v4    = [s["dice_v4"] for s in subjects_metrics]
    dices_v3    = [s["dice_v3"] for s in subjects_metrics]
    crop_ok     = [s for s in subjects_metrics if     s["crop_ok"]]
    crop_failed = [s for s in subjects_metrics if not s["crop_ok"]]

    metrics = {
        "aggregate": {
            "n_total":                    len(subjects_metrics),
            "n_bad_crops":                len(crop_failed),
            "dice_v4":                    _stats(dices_v4),
            "dice_v3":                    _stats(dices_v3),
            "dice_v4_crop_ok":            _stats([s["dice_v4"] for s in crop_ok])     if crop_ok     else None,
            "dice_v4_crop_failed":        _stats([s["dice_v4"] for s in crop_failed]) if crop_failed else None,
            "dice_v3_crop_ok":            _stats([s["dice_v3"] for s in crop_ok])     if crop_ok     else None,
            "dice_v3_crop_failed":        _stats([s["dice_v3"] for s in crop_failed]) if crop_failed else None,
        },
        "subjects": subjects_metrics,
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=4))

    logger.log(f"\n{'='*50}")
    logger.log(f"Dice v4 : {metrics['aggregate']['dice_v4']['mean']} ± {metrics['aggregate']['dice_v4']['std']}")
    logger.log(f"Dice v3 : {metrics['aggregate']['dice_v3']['mean']} ± {metrics['aggregate']['dice_v3']['std']}")
    if crop_ok:
        logger.log(f"Dice v4 (crop ok)     : {metrics['aggregate']['dice_v4_crop_ok']['mean']} ± {metrics['aggregate']['dice_v4_crop_ok']['std']}  (n={metrics['aggregate']['dice_v4_crop_ok']['n']})")
    if crop_failed:
        logger.log(f"Dice v4 (crop failed) : {metrics['aggregate']['dice_v4_crop_failed']['mean']} ± {metrics['aggregate']['dice_v4_crop_failed']['std']}  (n={metrics['aggregate']['dice_v4_crop_failed']['n']})")
    logger.log(f"Bad crops: {len(crop_failed)}/{len(subjects_metrics)}")
    logger.log(f"QC report: {qc_dir}/index.html")
    logger.log(f"Metrics  : {metrics_path}")
    logger.close()


if __name__ == "__main__":
    main()
