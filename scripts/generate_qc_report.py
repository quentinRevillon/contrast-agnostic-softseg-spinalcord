"""
Generate QC reports comparing v4 (sc-crop + nnUNet) vs v3 (sct_deepseg spinalcord)
on the test set, with crop QC included.

Three QC groups in a single index.html:
  - crop   : cropped image + cropped GT — verifies sc_crop didn't cut the SC
  - seg_v4 : original image + v4 prediction
  - seg_v3 : original image + v3 prediction (sct_deepseg spinalcord)

Usage:
    python scripts/generate_qc_report.py \
        --dataset-dir /path/to/nnUNet_raw/Dataset4000_ContrastAgnosticScCrop \
        --checkpoint   /path/to/fold_0/checkpoint_best.pth \
        --output-dir   /path/to/qc_results

Author: Quentin Revillon
"""

import argparse
import json
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True,
                        help="Path to nnUNet_raw/DatasetXXXX_... folder")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to fold_0/checkpoint_best.pth")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write seg_v4/, seg_v3/, qc/")
    return parser.parse_args()


def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def subject_name(path: str) -> str:
    """Extract filename stem for -qc-subject (e.g. sub-C001_UNIT1)."""
    return Path(path).name.replace(".nii.gz", "").replace(".nii", "")


def dataset_name(path: str) -> str:
    """Extract dataset folder name from original BIDS path."""
    parts = Path(path).parts
    for i, p in enumerate(parts):
        if "datasets_contrast_agnostic" in p and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def build_test_pairs(dataset_dir: Path) -> list[dict]:
    """Build (orig_image, crop_image, orig_label, crop_label) for each test subject."""
    conv = json.loads((dataset_dir / "conversion_dict.json").read_text())

    images_ts = {k: v for k, v in conv.items() if "/imagesTs/" in v}
    labels_ts  = {k: v for k, v in conv.items() if "/labelsTs/"  in v}

    # index labels by numeric ID (ContrastAgnosticScCrop_001)
    label_by_id = {}
    for orig, crop in labels_ts.items():
        id_ = Path(crop).name.replace(".nii.gz", "")
        label_by_id[id_] = (orig, crop)

    pairs = []
    for orig_img, crop_img in images_ts.items():
        id_ = Path(crop_img).name.replace("_0000.nii.gz", "")
        orig_lbl, crop_lbl = label_by_id[id_]
        pairs.append({
            "orig_image":  orig_img,
            "crop_image":  crop_img,
            "orig_label":  orig_lbl,
            "crop_label":  crop_lbl,
        })

    return sorted(pairs, key=lambda x: x["crop_image"])


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    qc_dir      = output_dir / "qc"
    seg_v4_dir  = output_dir / "seg_v4"
    seg_v3_dir  = output_dir / "seg_v3"

    for d in [qc_dir, seg_v4_dir, seg_v3_dir]:
        d.mkdir(parents=True, exist_ok=True)

    pairs = build_test_pairs(dataset_dir)
    print(f"Found {len(pairs)} test subjects")

    for i, p in enumerate(pairs, 1):
        subj   = subject_name(p["orig_image"])
        dset   = dataset_name(p["orig_image"])
        seg_v4 = seg_v4_dir / f"{subj}_seg_v4.nii.gz"
        seg_v3 = seg_v3_dir / f"{subj}_seg_v3.nii.gz"

        print(f"\n[{i}/{len(pairs)}] {subj} ({dset})")

        # 1. Crop QC
        run(["sct_qc",
             "-i", p["crop_image"], "-s", p["crop_label"],
             "-p", "sct_deepseg_sc",
             "-qc", str(qc_dir), "-qc-subject", subj, "-qc-dataset", f"crop_{dset}"])

        # 2. v4 inference (sc-crop + nnUNet, GPU)
        run(["sc-segment-pt",
             "-i", p["orig_image"], "-o", str(seg_v4),
             "--checkpoint", args.checkpoint,
             "--device", "cuda"])

        # 3. v3 inference (sct_deepseg spinalcord)
        run(["sct_deepseg", "spinalcord",
             "-i", p["orig_image"], "-o", str(seg_v3)])

        # 4. QC v4
        run(["sct_qc",
             "-i", p["orig_image"], "-s", str(seg_v4),
             "-p", "sct_deepseg_sc",
             "-qc", str(qc_dir), "-qc-subject", subj, "-qc-dataset", f"seg_v4_{dset}"])

        # 5. QC v3
        run(["sct_qc",
             "-i", p["orig_image"], "-s", str(seg_v3),
             "-p", "sct_deepseg_sc",
             "-qc", str(qc_dir), "-qc-subject", subj, "-qc-dataset", f"seg_v3_{dset}"])

    print(f"\nDone. QC report: {qc_dir}/index.html")


if __name__ == "__main__":
    main()
