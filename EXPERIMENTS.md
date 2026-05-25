# Experiment reproducibility log

Detailed record of environment, scripts, implementation, and exact commands for each experiment.

---

## Environment

| Component | Version |
|---|---|
| Repository | `quentinRevillon/contrast-agnostic-softseg-spinalcord` |
| Conda env | `contrast_agnostic` (Python 3.9) |
| PyTorch | 2.8.0+cu128 |
| nnunetv2 | 2.5.2 |
| sc_crop | **0.0.5** |
| onnxruntime | 1.19.2 |
| nibabel | 5.3.3 |
| numpy | 1.26.4 |
| SCT | git-master `e7d8b48` |

```bash
export nnUNet_raw=/home/quentinr/nnunet-v2/nnUNet_raw
export nnUNet_preprocessed=/home/quentinr/nnunet-v2/nnUNet_preprocessed
export nnUNet_results=/home/quentinr/nnunet-v2/nnUNet_results
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
```

---

## Experiment 1 — sc-crop (v1)

### Versioning

| | |
|---|---|
| Branch | `sc-crop` · HEAD `ffef761` |
| Tag | `v0.0.2` → commit `91a6142` |
| sc_crop | `0.0.5` |

### Dataset

| | |
|---|---|
| ID | `Dataset1000_TempContrastAgnosticCropped` |
| Datalist folder | `/home/quentinr/datalists/20260512-temp` |
| Datasets (12) | data-multi-subject, basel-mp2rage, canproco, lumbar-epfl, lumbar-vanderbilt, dcm-zurich-lesions-20231115, sci-paris, sci-zurich, sci-colorado, sct-testing-large, site_006, site_007 |
| Missing | dcm-brno, dcm-zurich, dcm-zurich-lesions *(glob bug — fixed in v2)* |
| Train + val | 2 563 |
| Test | 510 |
| Split seed | 50 |

---

### Step 1 — sc_crop preprocessing (`nnUnet/03_convert_msd_to_nnunet_reorient_sc_crop.py`)

**Script**: `nnUnet/03_convert_msd_to_nnunet_reorient_sc_crop.py`

```bash
python nnUnet/03_convert_msd_to_nnunet_reorient_sc_crop.py \
    --input    /home/quentinr/datalists/20260512-temp \
    --output   $nnUNet_raw \
    --taskname TempContrastAgnosticCropped \
    --tasknumber 1000 \
    --workers 8 \
    --pad-rl 20 --pad-ap 30 --pad-si 40
```

**What it does — per image:**

1. `sct_image -setorient RPI` — reorient image and label to RPI
2. `sc_crop.run(img_rpi, padding_rl_mm=20, padding_ap_mm=30, padding_si_mm=(40,40))` — YOLO-based SC detection → returns `{xmin, xmax, ymin, ymax, zmin, zmax}` in voxel coordinates
3. `nibabel` crop of image and label to the bbox
4. `sct_register_multimodal -identity 1` — align label to image (header match)
5. `sct_maths -bin 0.5` — binarize label

Output: `$nnUNet_raw/Dataset1000_TempContrastAgnosticCropped/imagesTr/`, `labelsTr/`, `imagesTs/`, `labelsTs/`

**sc_crop internals (v0.0.5):**
- Reorients image to LAS internally, resamples to isotropic 1 mm in S-I
- Runs YOLOv8 on each axial slice (2D) → detects SC bounding box per slice
- Aggregates across slices → 3D bbox in original voxel space
- `use_onnx=True` by default → runs fully on CPU via ONNX Runtime

---

### Step 2 — nnUNet preprocessing + training

```bash
# Preprocessing
nnUNetv2_plan_and_preprocess -d 1000 --verify_dataset_integrity -c 3d_fullres

# Training
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1000 3d_fullres 0 \
    -tr nnUNetTrainer -p nnUNetPlans
```

| | |
|---|---|
| Trainer | `nnUNetTrainer` (default) |
| Configuration | `3d_fullres` |
| Fold | 0 |
| Epochs | 1 000 |
| Duration | ~8 h GPU |
| Checkpoint | `fold_0/checkpoint_final.pth` |

---

### Step 3 — ONNX export (`nnUnet/export_nnunet_to_onnx.py`)

**Script**: `nnUnet/export_nnunet_to_onnx.py` · branch `sc-crop` · commit `ffef761`

```bash
python nnUnet/export_nnunet_to_onnx.py \
    --model-folder $nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres \
    --output       $nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres/onnx/nnunet_seg.onnx
```

**What it does:**

1. Loads the trained nnUNet via `nnUNetPredictor.initialize_from_trained_model_folder()` on CPU
2. Extracts the `PlainConvUNet` network (`predictor.network`) — the neural network only, without nnUNet's preprocessing/postprocessing wrappers
3. Creates a dummy input of shape `(1, 1, D, H, W)` matching `plans.json` patch size
4. Exports with `torch.onnx.export(..., opset_version=14)` with dynamic batch axis

**What is NOT exported** (handled separately at inference time):
- Resampling to target spacing
- Z-score normalization
- Crop-to-nonzero
- Sliding window aggregation with Gaussian weighting

These are reimplemented in pure numpy/scipy in `nnUnet/run_inference.py`.

Output:
```
onnx/
  nnunet_seg.onnx   — exported PlainConvUNet weights (~50 MB)
  plans.json        — copied from model folder (patch size, target spacing)
```

---

### Step 4 — PyTorch evaluation (`nnUnet/04_evaluate_with_sc_crop.py`)

**Script**: `nnUnet/04_evaluate_with_sc_crop.py` · branch `sc-crop` · commit `35f88ce`

```bash
python nnUnet/04_evaluate_with_sc_crop.py \
    --dataset-folder $nnUNet_raw/Dataset1000_TempContrastAgnosticCropped \
    --model-folder   $nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres \
    --output-dir     $nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres/test_sc_crop \
    --pad-left 20 --pad-right 20 --pad-anterior 30 --pad-posterior 30 \
    --pad-superior 40 --pad-inferior 40
```

**What it does — 3 phases:**

**Phase 1** — sc_crop on all test images:
- Reads original test images from `conversion_dict.json`
- Calls `sc_crop.run()` on each → crops image and GT label to SC bbox
- Saves cropped images to `crops_tmp/` and cropped GT to `labelsTs_sc_crop/`

**Phase 2** — nnUNet PyTorch batch inference on crops:
- `nnUNetPredictor.predict_from_files(crops_tmp/, test_sc_crop/)`
- Uses `checkpoint_final.pth`, fold 0, TTA disabled (no mirroring)

**Phase 3** — metrics:
- `coverage = |GT_crop| / |GT_total|` — fraction of SC voxels captured by sc_crop
- `dice_within_crop = Dice(pred, GT_crop)` — model quality inside crop
- `dice_global = 2 * |pred ∩ GT_crop| / (|pred| + |GT_total|)` — penalises missed SC voxels
- `nnUNetv2_evaluate_folder` → `summary.json`

Output files:
```
test_sc_crop/
  metrics_sc_crop.csv   ← per-image: coverage, dice_within_crop, dice_global
  summary.json          ← nnUNet eval in crop space (Dice mean/std)
  TempContrastAgnosticCropped_NNN.nii.gz  ← predictions in sc_crop+RPI space
```

**Results**: Dice global mean = **0.9531** · summary.json mean = **0.9537**

---

### Step 5 — ONNX evaluation (`nnUnet/05_benchmark_onnx_sc_crop.py`)

**Script**: `nnUnet/05_benchmark_onnx_sc_crop.py` · branch `sc-crop` · commit `35f88ce`

```bash
python nnUnet/05_benchmark_onnx_sc_crop.py \
    --dataset-folder $nnUNet_raw/Dataset1000_TempContrastAgnosticCropped \
    --model   $nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres/onnx/nnunet_seg.onnx \
    --plans   $nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres/onnx/plans.json \
    --output  $nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres/onnx/benchmark_sc_crop.csv
```

**What it does — per test image:**

1. `sc_crop.run()` → SC bbox in original space (~3 s/image)
2. Crop + reorient to RPI (nibabel, no SCT)
3. ONNX preprocessing (reimplemented in numpy — matches nnUNet exactly):
   - `crop_to_nonzero` — remove background padding
   - `zscore_normalize` — subtract mean, divide by std
   - `resample` (order=3, scipy) — to target spacing from `plans.json`
4. Sliding-window inference with Gaussian weighting (step = 0.5 × patch)
5. Postprocess: resample back (order=0) → pad to sc_crop shape → threshold 0.5
6. Metrics: same `coverage`, `dice_within_crop`, `dice_global` as PyTorch eval + timings

Output files:
```
onnx/
  benchmark_sc_crop.csv              ← per-image: coverage, dice_global, sc_crop_time_s, inference_time_s, total_time_s
  benchmark_sc_crop_summary.json     ← mean/std: coverage, dice, timing
  benchmark_gt_crop.csv              ← same with GT crop instead of sc_crop detection
  benchmark_gt_crop_summary.json
```

**Results**: Dice global mean = **0.9448** · SC coverage = **99.87%** · total time = **6.57 s ± 2.97 s**

---

## Experiment 2 — sc-crop-v2 (in progress)

### Versioning

| | |
|---|---|
| Branch | `sc-crop-v2` · HEAD `887a10c` |
| sc_crop | `0.0.5` |

### Dataset

| | |
|---|---|
| ID | `Dataset2000_ContrastAgnosticScCrop` |
| Datalist folder | `/home/quentinr/datalists/20260524-sc-crop` |
| Datasets (15) | idem v1 + **dcm-brno, dcm-zurich, dcm-zurich-lesions** |
| Train + val | 2 944 |
| Test | 510 |
| Split seed | 50 |

### Scripts

| Step | Script | Command |
|---|---|---|
| 1. Clone datasets | `nnUnet/01_clone_dataset.py` | called by `train_contrast_agnostic.sh` |
| 2. Create datalists | `nnUnet/02_create_msd_data.py` | called by `train_contrast_agnostic.sh` |
| 3. sc_crop conversion | `nnUnet/03_convert_msd_to_nnunet_reorient_sc_crop.py` | called by `train_contrast_agnostic.sh` |
| 4. Preprocessing | `nnUNetv2_plan_and_preprocess` | called by `train_contrast_agnostic.sh` |
| 5. Training | `nnUNetv2_train` | called by `train_contrast_agnostic.sh` |
| 6. Evaluation | `scripts/evaluate_csa_local.sh` | calls `compute_csa_local.sh` + ANIMA |

One-click:
```bash
bash run_all.sh   # = train_contrast_agnostic.sh + evaluate_csa_local.sh
```

### sc_crop preprocessing parameters (v2 — tighter padding than v1)

```bash
python nnUnet/03_convert_msd_to_nnunet_reorient_sc_crop.py \
    --input    /home/quentinr/datalists/20260524-sc-crop \
    --output   $nnUNet_raw \
    --taskname ContrastAgnosticScCrop \
    --tasknumber 2000 \
    --workers 8 \
    --pad-rl 10 --pad-ap 15 --pad-si 30
```

---

## CPU Speed benchmark

### Versioning

| | |
|---|---|
| Branch | `sc-crop-v2` · commit `2d42742` |
| Model | Exp 1 ONNX checkpoint (`Dataset1000_TempContrastAgnosticCropped`) |
| Script | `nnUnet/benchmark_cpu.py` |
| sc_crop | `0.0.5` |

### Test image

```
sub-beijingPrisma04_T2w.nii.gz
  path  : datasets_contrast_agnostic_retraining/data-multi-subject/sub-beijingPrisma04/anat/
  shape : (64, 320, 320)
  GT    : derivatives/labels_softseg_bin/sub-beijingPrisma04/anat/
            sub-beijingPrisma04_T2w_desc-softseg_label-SC_seg.nii.gz
```

### What each method calls

**[A] `sct_deepseg spinalcord`**
```bash
sct_deepseg spinalcord -i image.nii.gz -o seg_sct.nii.gz
```
- SCT built-in contrast-agnostic model (also nnUNet-based)
- Runs on the **full uncropped volume**
- CPU forced via `CUDA_VISIBLE_DEVICES=""`

**[B] sc_crop + ONNX** — `nnUnet/run_inference.py --mode onnx`
```bash
python nnUnet/run_inference.py -i image.nii.gz -o seg_onnx.nii.gz \
    --mode onnx --model onnx/nnunet_seg.onnx --plans onnx/plans.json --time
```
Pipeline (all in Python, no SCT, no nnunetv2):
1. `sc_crop.detect_and_crop()` → SC bbox + cropped image in original orientation
2. Reorient crop to RPI (nibabel)
3. `crop_to_nonzero` → `zscore_normalize` → `resample` to target spacing
4. ONNX sliding-window inference (Gaussian weighting, step=0.5)
5. Resample back → threshold 0.5 → pad to full image space
6. `sc_crop.restore_segmentation()` → segmentation in original orientation + space

**[C] sc_crop + PyTorch** — `nnUnet/run_inference.py --mode pt`
```bash
python nnUnet/run_inference.py -i image.nii.gz -o seg_pt.nii.gz \
    --mode pt --model-folder nnUNetTrainer__nnUNetPlans__3d_fullres --time
```
Same sc_crop detection (step 1–2), then nnUNetPredictor for inference (no TTA).

### Reproduce

```bash
python nnUnet/benchmark_cpu.py \
    -i  datasets_contrast_agnostic_retraining/data-multi-subject/sub-beijingPrisma04/anat/sub-beijingPrisma04_T2w.nii.gz \
    -gt datasets_contrast_agnostic_retraining/data-multi-subject/derivatives/labels_softseg_bin/sub-beijingPrisma04/anat/sub-beijingPrisma04_T2w_desc-softseg_label-SC_seg.nii.gz \
    --model        $nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres/onnx/nnunet_seg.onnx \
    --plans        $nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres/onnx/plans.json \
    --model-folder $nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres \
    --output benchmark_cpu_results.csv
```

### Results

| Method | Script | Time (CPU) | Dice vs GT |
|---|---|---|---|
| [A] `sct_deepseg spinalcord` | SCT built-in | 56 s | 0.9663 |
| [B] sc_crop + ONNX | `run_inference.py --mode onnx` | **11 s** | 0.9623 |
| [C] sc_crop + PyTorch | `run_inference.py --mode pt` | 25 s | **0.9678** |
