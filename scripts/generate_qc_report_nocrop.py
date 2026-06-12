"""
Generate a QC report for the NO-CROP model (Dataset7001) on the test set,
mirroring generate_qc_report.py but without any cropping.

Three QC groups in a single index.html, all in the nnUNet test space (RPI, full
volume):
  - gt        : test image + ground-truth label (labelsTs, already cleaned/binarized)
  - seg_v3    : test image + sct_deepseg spinalcord prediction (deployed baseline)
  - seg_nocrop: test image + the no-crop nnUNet prediction (Dataset7001)

v3 is kept as a shared ANCHOR with the cropped report: the same subjects and the
same v3 model appear in both, so the cropped report's dice_v4 and this report's
dice_nocrop can be compared per subject / per contrast (sanity-checked by dice_v3
matching across the two reports).

The no-crop predictions are NOT computed here — run them once with
nnUNetv2_predict (see run_qc_report_nocrop.sh) and pass the folder via
--predictions-dir. This avoids reloading the nnUNet model per subject.

Also produces:
  - metrics.json : Dice v3/nocrop per subject + aggregate stats (NO crop QC)
  - run.log      : full stdout log

Usage:
    python scripts/generate_qc_report_nocrop.py \
        --dataset-dir     /path/to/nnUNet_raw/Dataset7001_ContrastAgnosticNoCrop \
        --predictions-dir /path/to/predictions_test_fold0_checkpoint_best \
        --output-dir      /path/to/qc_results_dataset7001

Author: Quentin Revillon
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

SCT_GPU_BIN = "/home/quentinr/spinalcordtoolbox-gpu/bin/sct_deepseg"

_CONTRAST_PATTERN = (
    r'.*(T1w|T2w|acq-sagthor_T2w|acq-sagcerv_T2w|acq-sagstir_T2w|acq-ax_T2w'
    r'|T2star|PSIR|STIR|UNIT1|flip-1_mt-on_MTS|flip-2_mt-off_MTS'
    r'|acq-MTon_MTR|acq-dwiMean_dwi|rec-average_dwi|acq-T1w_MTR).*'
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True,
                        help="Path to nnUNet_raw/Dataset7001_... folder (no-crop)")
    parser.add_argument("--predictions-dir", required=True,
                        help="Folder with the no-crop nnUNet predictions ({case_id}.nii.gz)")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write seg_v3/, qc/, metrics.json, run.log")
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


def run(cmd: list, logger: Logger, env: dict | None = None) -> None:
    logger.log(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env={**os.environ, **(env or {})})


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


def infer_v3_gpu(image: str, seg_v3: Path, logger: Logger) -> None:
    """Run v3 inference via sct_deepseg spinalcord on GPU (no built-in QC).

    Runs on the nnUNet test image (RPI full volume), the same space as the GT and
    the no-crop prediction, so all Dice scores are computed on aligned grids.
    """
    run([SCT_GPU_BIN, "spinalcord",
         "-i", image, "-o", str(seg_v3)],
        logger,
        env={"SCT_USE_GPU": "1", "CUDA_VISIBLE_DEVICES": "0",
             "TORCHDYNAMO_DISABLE": "1"})


def dice(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_bin   = gt   > 0
    pred_bin = pred > 0
    denom    = gt_bin.sum() + pred_bin.sum()
    return float(2 * np.logical_and(gt_bin, pred_bin).sum() / denom) if denom > 0 else 0.0


def build_test_pairs(dataset_dir: Path) -> list[dict]:
    """Build test pairs from conversion_dict.json.

    Returns, per test subject: the original image/label paths (for naming) and
    the nnUNet test image/label paths (imagesTs/labelsTs, used for QC and Dice),
    plus the nnUNet case id (to locate the no-crop prediction).
    """
    conv = json.loads((dataset_dir / "conversion_dict.json").read_text())

    images_ts = {k: v for k, v in conv.items() if "/imagesTs/" in v}
    labels_ts = {k: v for k, v in conv.items() if "/labelsTs/" in v}

    label_by_id = {}
    for orig, ts in labels_ts.items():
        id_ = Path(ts).name.replace(".nii.gz", "")
        label_by_id[id_] = (orig, ts)

    pairs = []
    for orig_img, ts_img in images_ts.items():
        id_ = Path(ts_img).name.replace("_0000.nii.gz", "")
        orig_lbl, ts_lbl = label_by_id[id_]
        pairs.append({
            "orig_image": orig_img,
            "ts_image":   ts_img,
            "orig_label": orig_lbl,
            "ts_label":   ts_lbl,
            "case_id":    id_,
        })

    return sorted(pairs, key=lambda x: x["ts_image"])


def main():
    args        = parse_args()
    dataset_dir = Path(args.dataset_dir)
    pred_dir    = Path(args.predictions_dir)
    output_dir  = Path(args.output_dir)
    qc_dir      = output_dir / "qc"
    seg_v3_dir  = output_dir / "seg_v3"

    assert not qc_dir.exists(), \
        f"{qc_dir} already exists — delete it or use a different --output-dir"

    for d in [qc_dir, seg_v3_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = Logger(output_dir / "run.log")
    pairs  = build_test_pairs(dataset_dir)
    if args.n_subjects:
        pairs = pairs[:args.n_subjects]
    logger.log(f"Processing {len(pairs)} test subjects (no-crop)")

    subjects_metrics = []

    for i, p in enumerate(pairs, 1):
        subj     = subject_name(p["orig_image"])
        dset     = dataset_name(p["orig_image"])
        contrast = contrast_name(p["orig_image"])
        ts_image = p["ts_image"]
        gt_path  = p["ts_label"]
        seg_nocrop = pred_dir / f"{p['case_id']}.nii.gz"
        seg_v3     = seg_v3_dir / f"{subj}_seg_v3.nii.gz"

        logger.log(f"\n[{i}/{len(pairs)}] {subj} ({dset})")

        assert seg_nocrop.exists(), \
            f"Missing no-crop prediction {seg_nocrop} — run nnUNetv2_predict first"

        # v3 inference on the test image (same RPI full space as GT / no-crop pred)
        infer_v3_gpu(ts_image, seg_v3, logger)

        # Dice vs GT (labelsTs — already cleaned to largest component + binarized)
        gt        = np.asarray(nib.load(gt_path).dataobj)
        d_v3      = dice(gt, np.asarray(nib.load(seg_v3).dataobj))
        d_nocrop  = dice(gt, np.asarray(nib.load(seg_nocrop).dataobj))
        logger.log(f"  dice_nocrop={d_nocrop:.4f}  dice_v3={d_v3:.4f}")

        subjects_metrics.append({
            "subject":     subj,
            "dataset":     dset,
            "contrast":    contrast,
            "dice_nocrop": round(d_nocrop, 4),
            "dice_v3":     round(d_v3, 4),
        })

        # QC GT
        run(["sct_qc",
             "-i", ts_image, "-s", str(gt_path),
             "-p", "sct_deepseg_sc",
             "-qc", str(qc_dir), "-qc-subject", f"{dset}/{subj}", "-qc-dataset", "gt"],
            logger)

        # QC v3
        run(["sct_qc",
             "-i", ts_image, "-s", str(seg_v3),
             "-p", "sct_deepseg_sc",
             "-qc", str(qc_dir), "-qc-subject", f"{dset}/{subj}", "-qc-dataset", "seg_v3"],
            logger)

        # QC no-crop
        run(["sct_qc",
             "-i", ts_image, "-s", str(seg_nocrop),
             "-p", "sct_deepseg_sc",
             "-qc", str(qc_dir), "-qc-subject", f"{dset}/{subj}", "-qc-dataset", "seg_nocrop"],
            logger)

    # Aggregate metrics
    def _stats(values):
        a = np.array(values)
        return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4), "n": len(a)}

    dices_nocrop = [s["dice_nocrop"] for s in subjects_metrics]
    dices_v3     = [s["dice_v3"]     for s in subjects_metrics]

    metrics = {
        "aggregate": {
            "n_total":     len(subjects_metrics),
            "dice_nocrop": _stats(dices_nocrop),
            "dice_v3":     _stats(dices_v3),
        },
        "subjects": subjects_metrics,
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=4))

    logger.log(f"\n{'='*50}")
    logger.log(f"Dice no-crop : {metrics['aggregate']['dice_nocrop']['mean']} ± {metrics['aggregate']['dice_nocrop']['std']}")
    logger.log(f"Dice v3      : {metrics['aggregate']['dice_v3']['mean']} ± {metrics['aggregate']['dice_v3']['std']}")
    logger.log(f"QC report: {qc_dir}/index.html")
    logger.log(f"Metrics  : {metrics_path}")
    logger.close()


if __name__ == "__main__":
    main()
