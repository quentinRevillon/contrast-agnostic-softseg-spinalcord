# sc-crop integration — Progress & Results

> Last updated: 2026-05-25 · Presentation: 2026-05-25

---

## Goal

Integrate `sc_crop` (YOLO-based SC detector) into the contrast-agnostic v3.0 training **and**
inference pipeline so that train and test volumes are cropped to the same tight bounding box
around the spinal cord.

**Why?**
- **Train/test consistency** — no domain shift between cropped inference and full-volume training
- **Faster inference** — smaller input volumes → less compute
- **Deployment-ready** — sc_crop ONNX + nnUNet ONNX runs on CPU with no nnunetv2 dependency

**Code change** — 3 functional lines in `scripts/train_contrast_agnostic.sh`:
```bash
# Original (commented out):
# python nnUnet/03_convert_msd_to_nnunet_reorient.py --input ... --output ...

# sc_crop (active):
python nnUnet/03_convert_msd_to_nnunet_reorient_sc_crop.py \
    --input ... --output ... --pad-rl 10 --pad-ap 15 --pad-si 30
```

---

## Experiment 1 — sc-crop (v1)

| Parameter | Value |
|---|---|
| Branch | `sc-crop` · commit `ffef761` |
| Dataset ID | `Dataset1000_TempContrastAgnosticCropped` |
| Datasets | 12 *(dcm-brno, dcm-zurich, dcm-zurich-lesions absent — glob bug)* |
| Train + val volumes | 2 563 |
| Test volumes | 466 |
| sc_crop padding | RL 20 mm · AP 30 mm · SI 40 mm |
| Training | nnUNetTrainer · 3d_fullres · fold 0 · 1 000 epochs · ~8 h GPU |

### Dice vs paper baseline (~0.950)

| Configuration | Dice | ΔDice |
|---|---|---|
| Paper v3.0 (baseline) | 0.9500 | — |
| PyTorch · GT crop · TTA | **0.9586** | +0.86 pt |
| PyTorch · GT crop · no TTA | 0.9574 | +0.74 pt |
| PyTorch · sc_crop end-to-end | **0.9537** | +0.37 pt |
| ONNX · GT crop · CPU | 0.9487 | −0.13 pt |
| ONNX · sc_crop end-to-end · CPU | **0.9448** | −0.52 pt |

### Per-dataset Dice — ONNX sc_crop end-to-end (N=466)

| Dataset | N | Dice |
|---|---|---|
| data-multi-subject | 294 | 0.9563 |
| canproco | 85 | 0.9515 |
| sct-testing-large | 36 | 0.9599 |
| sci-zurich | 14 | 0.8958 |
| basel-mp2rage | 11 | 0.9547 |
| sci-colorado | 8 | 0.9313 |
| lumbar-vanderbilt | 6 | 0.9395 |
| dcm-zurich-lesions-20231115 | 5 | 0.9608 |
| lumbar-epfl | 3 | 0.9360 |
| sci-paris | 2 | 0.9607 |
| site_006 / site_007 | 2 | 0.9419 |
| **OVERALL** | **466** | **0.9448** |

### Inference speed — ONNX · CPU · N=466

| Step | Mean ± std |
|---|---|
| sc_crop detection | ~3.0 s |
| ONNX nnUNet inference | 3.67 s ± 2.31 s |
| **Total end-to-end** | **6.57 s ± 2.97 s** |

---

## Experiment 2 — sc-crop-v2 (in progress)

| Parameter | Value |
|---|---|
| Branch | `sc-crop-v2` · commit `2d42742` |
| Dataset ID | `Dataset2000_ContrastAgnosticScCrop` |
| Datasets | **15** (all included — glob bug fixed) |
| Train + val volumes | 2 944 |
| Test volumes | 510 |
| sc_crop padding | RL 10 mm · AP 15 mm · SI 30 mm |
| Training | nnUNetTrainer · 3d_fullres · fold 0 · 1 000 epochs · ~8 h GPU |

### Changes vs v1

| | v1 | v2 |
|---|---|---|
| Datasets | 12 | **15** (+dcm-brno, dcm-zurich, dcm-zurich-lesions) |
| Train+val volumes | 2 563 | **2 944** (+15%) |
| Test volumes | 466 | **510** (+9%) |
| Datalist folder | `20260512-temp` | `20260524-sc-crop` |
| sc_crop padding | RL20/AP30/SI40 mm | RL10/AP15/SI30 mm |

### Current status

| Step | Status |
|---|---|
| 1. Clone datasets | ✅ done |
| 2. Create datalists | ✅ done |
| 3. sc_crop conversion | 🔄 **83.6%** (2 460 / 2 944 imagesTr) |
| 4. nnUNet preprocessing | ⏳ pending |
| 5. Training (~8 h) | ⏳ pending |
| 6. Evaluation | ⏳ pending |

---

## CPU Speed benchmark

Single image: `sub-beijingPrisma04_T2w` (T2w · shape 64×320×320 · **CPU only**).  
Model used: `Dataset1000_TempContrastAgnosticCropped` (Exp 1 ONNX checkpoint).

| Method | Time (CPU) | Dice vs GT | Speedup vs [A] |
|---|---|---|---|
| [A] `sct_deepseg spinalcord` (SCT built-in, full volume) | 56 s | 0.9663 | — |
| [B] sc_crop + ONNX *(no nnunetv2)* | **11 s** | 0.9623 | **×4.9** |
| [C] sc_crop + PyTorch *(no TTA)* | 25 s | **0.9678** | ×2.2 |

Breakdown for [B] sc_crop + ONNX:

| Step | Time |
|---|---|
| sc_crop detection (YOLO ONNX) | ~1 s |
| ONNX nnUNet sliding-window inference | ~10 s |
| **Total** | **~11 s** |

Reproduce with:
```bash
python nnUnet/benchmark_cpu.py \
    -i image.nii.gz -gt seg_gt.nii.gz \
    --model        ~/nnunet-v2/nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres/onnx/nnunet_seg.onnx \
    --plans        ~/nnunet-v2/nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres/onnx/plans.json \
    --model-folder ~/nnunet-v2/nnUNet_results/Dataset1000_TempContrastAgnosticCropped/nnUNetTrainer__nnUNetPlans__3d_fullres
```

---

## Key takeaways

| | |
|---|---|
| Dice (PyTorch, end-to-end) | **0.9537** — on par with paper (0.9500) |
| Dice (ONNX, end-to-end, CPU) | **0.9448** — −0.52 pt vs paper |
| CPU speed vs sct_deepseg | **×4.9** faster (ONNX) · ×2.2 (PyTorch) |
| SC detection coverage | **99.87%** on 466 test volumes |
| Code change in training script | **3 lines** |
| New files added | `03_convert_msd_to_nnunet_reorient_sc_crop.py` · `run_inference.py` · `run_inference_sc_crop.py` · `benchmark_cpu.py` |
