# Experiment reproducibility log

This file records the exact environment, scripts, and commands needed to reproduce each experiment.

---

## Environment

| Component | Version |
|---|---|
| Repository | `quentinRevillon/contrast-agnostic-softseg-spinalcord` |
| Conda env | `contrast_agnostic` |
| Python | 3.9 |
| PyTorch | 2.8.0+cu128 |
| nnunetv2 | 2.5.2 |
| sc_crop | **0.0.5** |
| onnxruntime | 1.19.2 |
| nibabel | 5.3.3 |
| numpy | 1.26.4 |
| SCT | git-master `e7d8b48` |

nnUNet environment variables:
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
| Branch | `sc-crop` |
| HEAD commit | `ffef761` |
| Tag | `v0.0.2` (points to commit `91a6142` on `sc-crop`) |
| sc_crop | `0.0.5` |

### Dataset

| | |
|---|---|
| ID | `Dataset1000_TempContrastAgnosticCropped` |
| Datalist folder | `/home/quentinr/datalists/20260512-temp` |
| Datasets (12) | data-multi-subject, basel-mp2rage, canproco, lumbar-epfl, lumbar-vanderbilt, dcm-zurich-lesions-20231115, sci-paris, sci-zurich, sci-colorado, sct-testing-large, site_006, site_007 |
| Missing | dcm-brno, dcm-zurich, dcm-zurich-lesions *(glob bug — fixed in v2)* |
| Train + val | 2 563 |
| Test | 466 |
| Split seed | 50 |

### sc_crop preprocessing parameters

```bash
python nnUnet/03_convert_msd_to_nnunet_reorient_sc_crop.py \
    --input  /home/quentinr/datalists/20260512-temp \
    --output /home/quentinr/nnunet-v2/nnUNet_raw \
    --taskname TempContrastAgnosticCropped \
    --tasknumber 1000 \
    --workers 8 \
    --pad-rl 20 --pad-ap 30 --pad-si 40
```

### Training command

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1000 3d_fullres 0 \
    -tr nnUNetTrainer -p nnUNetPlans
```

| | |
|---|---|
| Trainer | `nnUNetTrainer` |
| Configuration | `3d_fullres` |
| Fold | 0 |
| Epochs | 1 000 |
| Duration | ~8 h |

### ONNX export

```bash
python nnUnet/export_nnunet_to_onnx.py \
    --model-folder $nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres \
    --output       $nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres/onnx
```

ONNX model path:
```
$nnUNet_results/Dataset1000_TempContrastAgnosticCropped/
  nnUNetTrainer__nnUNetPlans__3d_fullres/onnx/
    nnunet_seg.onnx
    plans.json
```

### Evaluation — PyTorch sc_crop end-to-end (N=466 test volumes)

Script: `nnUnet/04_evaluate_with_sc_crop.py` (branch `sc-crop`, commit `35f88ce`)

```bash
python nnUnet/04_evaluate_with_sc_crop.py \
    --dataset-folder $nnUNet_raw/Dataset1000_TempContrastAgnosticCropped \
    --model-folder   $nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres \
    --pad-left 20 --pad-right 20 --pad-anterior 30 --pad-posterior 30 \
    --pad-superior 40 --pad-inferior 40
```

Results written to:
```
$nnUNet_results/Dataset1000_TempContrastAgnosticCropped/
  nnUNetTrainer__nnUNetPlans__3d_fullres/test_sc_crop/
    metrics_sc_crop.csv          ← per-image: coverage, dice_within_crop, dice_global
    summary.json                 ← nnUNet standard eval (Dice in crop space)
```

Overall Dice (dice_global, full image space): **0.9531**  
Overall Dice (summary.json, crop space): **0.9537**

### Evaluation — ONNX sc_crop end-to-end (N=466 test volumes)

Script: `nnUnet/05_benchmark_onnx_sc_crop.py` (branch `sc-crop`, commit `35f88ce`)

```bash
python nnUnet/05_benchmark_onnx_sc_crop.py \
    --dataset-folder $nnUNet_raw/Dataset1000_TempContrastAgnosticCropped \
    --model   $nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres/onnx/nnunet_seg.onnx \
    --plans   $nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres/onnx/plans.json \
    --output  benchmark_sc_crop.csv
```

Results written to:
```
$nnUNet_results/Dataset1000_TempContrastAgnosticCropped/
  nnUNetTrainer__nnUNetPlans__3d_fullres/onnx/
    benchmark_sc_crop.csv              ← per-image: coverage, dice_global, timing
    benchmark_sc_crop_summary.json     ← Dice mean/std, SC coverage, timing stats
    benchmark_gt_crop.csv              ← same metrics with GT crop instead of sc_crop
    benchmark_gt_crop_summary.json
```

Overall Dice (dice_global, full image space): **0.9448**

---

## Experiment 2 — sc-crop-v2 (in progress)

### Versioning

| | |
|---|---|
| Branch | `sc-crop-v2` |
| HEAD commit | `2d42742` |
| sc_crop | `0.0.5` |

### Dataset

| | |
|---|---|
| ID | `Dataset2000_ContrastAgnosticScCrop` |
| Datalist folder | `/home/quentinr/datalists/20260524-sc-crop` |
| Datasets (15) | data-multi-subject, basel-mp2rage, canproco, lumbar-epfl, lumbar-vanderbilt, **dcm-brno**, **dcm-zurich**, **dcm-zurich-lesions**, dcm-zurich-lesions-20231115, sci-paris, sci-zurich, sci-colorado, sct-testing-large, site_006, site_007 |
| Train + val | 2 944 |
| Test | 510 |
| Split seed | 50 |

### sc_crop preprocessing parameters

```bash
python nnUnet/03_convert_msd_to_nnunet_reorient_sc_crop.py \
    --input  /home/quentinr/datalists/20260524-sc-crop \
    --output /home/quentinr/nnunet-v2/nnUNet_raw \
    --taskname ContrastAgnosticScCrop \
    --tasknumber 2000 \
    --workers 8 \
    --pad-rl 10 --pad-ap 15 --pad-si 30
```

### Scripts

| Step | Script |
|---|---|
| 1–3. Clone + datalists + sc_crop conversion | `scripts/train_contrast_agnostic.sh` |
| 4–5. Preprocessing + training | `scripts/train_contrast_agnostic.sh` (continued) |
| Full pipeline (train + evaluate) | `run_all.sh` |
| Evaluation (CSA + Dice) | `scripts/evaluate_csa_local.sh` |
| Per-subject CSA computation | `scripts/compute_csa_local.sh` |
| ANIMA Dice metrics | `anima_metrics/compute_anima_metrics_spine_generic.py` |

One-click:
```bash
bash run_all.sh
```

Step by step:
```bash
bash scripts/train_contrast_agnostic.sh   # steps 1–5
bash scripts/evaluate_csa_local.sh        # evaluation
```

`train_contrast_agnostic.sh` differs from `main` by only 3 lines (sc_crop preprocessing — see `PROGRESS.md`).

### Training command (launched by train_contrast_agnostic.sh)

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 2000 3d_fullres 0 \
    -tr nnUNetTrainer -p nnUNetPlans
```

---

## CPU Speed benchmark

### Versioning

| | |
|---|---|
| Branch | `sc-crop-v2` · commit `2d42742` |
| Model used | Exp 1 ONNX checkpoint (`Dataset1000_TempContrastAgnosticCropped`) |
| sc_crop | `0.0.5` |

### Test image

```
/home/quentinr/datasets_contrast_agnostic_retraining/data-multi-subject/
  sub-beijingPrisma04/anat/sub-beijingPrisma04_T2w.nii.gz
    shape : (64, 320, 320)
    GT    : derivatives/labels_softseg_bin/sub-beijingPrisma04/anat/
              sub-beijingPrisma04_T2w_desc-softseg_label-SC_seg.nii.gz
```

### Script

`nnUnet/benchmark_cpu.py` (branch `sc-crop-v2`, commit `2d42742`)

```bash
python nnUnet/benchmark_cpu.py \
    -i  /home/quentinr/datasets_contrast_agnostic_retraining/data-multi-subject/sub-beijingPrisma04/anat/sub-beijingPrisma04_T2w.nii.gz \
    -gt /home/quentinr/datasets_contrast_agnostic_retraining/data-multi-subject/derivatives/labels_softseg_bin/sub-beijingPrisma04/anat/sub-beijingPrisma04_T2w_desc-softseg_label-SC_seg.nii.gz \
    --model        $nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres/onnx/nnunet_seg.onnx \
    --plans        $nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres/onnx/plans.json \
    --model-folder $nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres \
    --output benchmark_cpu_results.csv
```

### Results

| Method | Time (CPU) | Dice vs GT | Speedup |
|---|---|---|---|
| [A] `sct_deepseg spinalcord` (full volume) | 56 s | 0.9663 | — |
| [B] sc_crop + ONNX *(no nnunetv2)* | **11 s** | 0.9623 | **×4.9** |
| [C] sc_crop + PyTorch *(no TTA)* | 25 s | **0.9678** | ×2.2 |

Breakdown [B]:

| Step | Time |
|---|---|
| sc_crop detection (YOLO ONNX) | ~1 s |
| ONNX sliding-window inference | ~10 s |
| **Total** | **~11 s** |
