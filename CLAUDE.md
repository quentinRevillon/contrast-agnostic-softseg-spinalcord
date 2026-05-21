# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All work runs in the `contrast_agnostic` conda environment:

```bash
conda activate contrast_agnostic
```

For training/PT inference, nnUNetv2 (`v2.5.1`) must also be installed. ONNX inference only needs `nibabel scipy numpy onnxruntime scikit-image` plus `sc-crop`.

Set nnUNet paths for training/preprocessing:
```bash
export nnUNet_raw=/home/quentinr/nnunet-v2/nnUNet_raw
export nnUNet_preprocessed=/home/quentinr/nnunet-v2/nnUNet_preprocessed
export nnUNet_results=/home/quentinr/nnunet-v2/nnUNet_results
```

## Common commands

```bash
# Download models (first use)
python nnUnet/run_inference.py download
sc_crop download

# Run inference — ONNX (CPU, recommended)
python nnUnet/run_inference.py -i image.nii.gz -o seg.nii.gz --time

# Run inference — PyTorch, no TTA
python nnUnet/run_inference.py -i image.nii.gz -o seg.nii.gz --mode pt --device cpu

# Run inference — PyTorch + TTA (mirroring, ~8× slower)
python nnUnet/run_inference.py -i image.nii.gz -o seg.nii.gz --mode pt-tta --device cpu

# Skip sc_crop detection (pre-cropped image)
python nnUnet/run_inference.py -i image_crop.nii.gz -o seg.nii.gz --pre-cropped

# Export trained model to ONNX
python nnUnet/export_nnunet_to_onnx.py \
    --model-folder $nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres \
    --output nnunet_seg.onnx

# Run full training pipeline (set START_STEP/END_STEP inside the script first)
bash scripts/train_contrast_agnostic.sh

# Evaluate with sc_crop on test set
python nnUnet/04_evaluate_with_sc_crop.py \
    --dataset-folder $nnUNet_raw/Dataset1000_TempContrastAgnosticCropped \
    --model-folder $nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres
```

## Architecture

### Inference pipeline (`nnUnet/run_inference.py`)

Single entry point for all inference. Three modes share the same pre/post-processing:

1. **sc_crop** — YOLO-based detection crops the image to a bounding box around the SC (`sc_crop.run()`), imported directly from the `sc_crop` package
2. **Reorient** → RPI
3. **nnUNet preprocessing** — crop-to-nonzero, z-score normalize, resample to `plans.json` spacing
4. **Inference** — sliding window (ONNX) or `nnUNetPredictor` (PT/PT-TTA)
5. **Postprocess** — resample back, reorient to original, pad to full image space

Models are stored in `~/nnunet_contrast_agnostic/` (set by `_MODEL_DIR`). The `download` subcommand fetches all assets from the GitHub release.

The nnUNet env-var warnings (`nnUNet_raw is not defined`) in PT mode are expected during standalone inference — those variables are only needed for training.

### Training pipeline (`scripts/train_contrast_agnostic.sh`)

10-step pipeline controlled by `START_STEP`/`END_STEP`:

| Step | Action |
|------|--------|
| 1 | Clone datasets via git-annex from `PATH_DATA_BASE` |
| 2 | Create MSD datalists (JSON) |
| 3 | Convert to nnUNet format with GT-bbox crop + RPI reorientation |
| 4 | nnUNet plan & preprocess |
| 5 | nnUNet training (fold 0, Dataset1000) |
| 6 | PT inference on GT-crop test set (oracle upper bound) |
| 7 | PT inference on sc_crop test set (realistic pipeline) |
| 8 | Export to ONNX |
| 9 | ONNX benchmark on GT-crop test set |
| 10 | ONNX benchmark on sc_crop test set |

Dataset: `Dataset1000_TempContrastAgnosticCropped` — 13 datasets, 2563 training subjects, seed=50.

### Release workflow

On every published release, the GitHub Actions workflow (`.github/workflows/run_morphometric_analysis.yml`) automatically:
- Downloads the spine-generic test split (n=49 subjects, 6 contrasts)
- Runs CSA computation in parallel batches
- Uploads `csa_c2c3__model_<tag>.csv` to the release
- Generates violin plots comparing morphometric drift across all releases

**To publish a new release:** tag HEAD, attach `nnunet_seg.onnx`, `plans.json`, `dataset.json`, `fold_0/checkpoint_final.pth` as release assets, then publish.

### Key files

| File | Role |
|------|------|
| `nnUnet/run_inference.py` | Standalone inference + model download |
| `nnUnet/export_nnunet_to_onnx.py` | Export PyTorch → ONNX |
| `nnUnet/04_evaluate_with_sc_crop.py` | Batch evaluation with sc_crop detection |
| `nnUnet/05_benchmark_onnx_sc_crop.py` | ONNX speed/accuracy benchmark |
| `scripts/train_contrast_agnostic.sh` | End-to-end training pipeline |
| `configs/train_all.yaml` | Training configuration |
| `datasplits/` | Per-dataset train/val/test splits (seed=50) |
| `subjects_to_include.yml` | Inclusion list across datasets |
| `exclude_from_training.yml` | Exclusion list |
