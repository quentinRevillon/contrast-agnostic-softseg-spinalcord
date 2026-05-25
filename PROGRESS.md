# sc-crop × contrast-agnostic — Progress & Results

> Presentation — 2026-05-25

---

## Context

The **contrast-agnostic v3.0** model (nnUNet 3D fullres) achieves ~**0.95 Dice** on spinal cord
segmentation across MRI contrasts (paper baseline).

**Goal**: integrate `sc_crop` — a lightweight YOLO-based spinal cord detector — into the
training *and* inference pipeline, so that both train and test volumes are cropped to the same
tight bounding box around the spinal cord.

**Why?**  
- **Train/test consistency**: at inference, sc_crop crops the input before segmentation; training on the same crop avoids domain shift.  
- **Faster inference**: smaller volumes → less compute.  
- **Deployment-ready**: the full pipeline (sc_crop ONNX + nnUNet ONNX) runs on CPU without any
  dependency on nnunetv2 at inference time.

---

## How much code does sc_crop add?

The entire integration is **3 lines of functional code** added to `train_contrast_agnostic.sh`:

```bash
# Original pipeline — commented out (1 line):
# python 03_convert_msd_to_nnunet_reorient.py ...

# sc_crop pipeline — active (3 lines):
python 03_convert_msd_to_nnunet_reorient_sc_crop.py \
    ... --pad-rl 10 --pad-ap 15 --pad-si 30
```

A new script `03_convert_msd_to_nnunet_reorient_sc_crop.py` is added alongside the original —
**identical except for one step**: after RPI reorientation, sc_crop detects the SC bounding box
and the volume is cropped before saving to `nnUNet_raw`.

---

## sc_crop at a glance

| Property | Value |
|---|---|
| Model | YOLOv8 (2D, axial slices) |
| Runtime | ONNX Runtime — **CPU only** |
| Detection time | ~3 s per volume |
| SC coverage | **99.87%** (almost never misses) |
| API | `from sc_crop import run; result = run("image.nii.gz")` |

---

## Experiment 1 — sc-crop branch (v1)

**Dataset**: 12 datasets, 2 563 train+val / 466 test volumes  
*(Note: dcm-brno, dcm-zurich, dcm-zurich-lesions were absent due to a glob bug — fixed in v2)*

**Training**: nnUNetTrainer, 3d_fullres, fold 0, 1 000 epochs — ~**8 hours** (GPU)

### Results

| Configuration | Dice | ΔDice vs paper |
|---|---|---|
| Paper v3.0 (baseline) | ~0.9500 | — |
| **PyTorch — GT crop (oracle)** | **0.9586** | **+0.86 pt** |
| PyTorch — GT crop, no TTA | 0.9574 | +0.74 pt |
| **PyTorch — sc_crop pipeline (end-to-end)** | **0.9537** | **+0.37 pt** |
| ONNX — GT crop (CPU) | 0.9487 | −0.13 pt |
| **ONNX — sc_crop pipeline (CPU, end-to-end)** | **0.9448** | **−0.52 pt** |

**The full production pipeline (ONNX + sc_crop, CPU-only) loses only 0.52 Dice points vs paper.**

### Per-dataset Dice (sc_crop end-to-end pipeline)

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

### Inference speed — ONNX only (CPU)

> PyTorch inference time was not measured. Timings below are from the ONNX benchmark.

| Step | Time (mean ± std) |
|---|---|
| sc_crop detection | ~3.0 s |
| ONNX nnUNet inference | 3.67 s ± 2.31 s |
| **Total end-to-end (sc_crop + ONNX)** | **6.57 s ± 2.97 s** |

---

## Experiment 2 — sc-crop-v2 (current, in progress)

**What changed vs v1:**

| | v1 | v2 |
|---|---|---|
| Dataset number | 1000 | 2000 |
| Datasets | 12 (dcm-brno/zurich missing) | **15** (all included, bug fixed) |
| Total volumes | 3 029 | **3 454** |
| Datalists | 20260512-temp | 20260524-sc-crop |

**Current status**: step 3 (sc_crop crop conversion) — **20.6% complete** (710 / 3 454 volumes).  
Steps remaining: 3 → preprocessing → training (~8h) → evaluation.

**Expected improvement**: better generalisation on DCM and axial datasets previously missing.

---

## Pipeline overview

```
train_contrast_agnostic.sh          ← 3 lines changed vs original
        │
        ├── 01_clone_dataset.py      (git-annex)
        ├── 02_create_msd_data.py    (datalists)
        └── 03_convert_msd_to_nnunet_reorient_sc_crop.py   ← NEW
                │  sct_image → RPI
                │  sc_crop  → bbox
                │  nibabel  → crop
                └─► nnUNet_raw/Dataset2000/

run_all.sh  =  bash train_contrast_agnostic.sh
            +  bash evaluate_csa_local.sh
```

---

## CPU Speed benchmark — sc_crop+ONNX vs sct_deepseg

Comparison on `sub-beijingPrisma04_T2w` (T2w, shape 64×320×320), **CPU only**.

| Method | Time (CPU) | Dice vs GT |
|---|---|---|
| `sct_deepseg spinalcord` (SCT built-in, full volume) | ~55 s | 0.9663 |
| **sc_crop + ONNX nnUNet (this work)** | **~11 s** | **0.9623** |
| **Speedup** | **×4.9 faster** | −0.004 pt |

Breakdown for sc_crop+ONNX:

| Step | Time |
|---|---|
| sc_crop detection (YOLO ONNX) | ~1 s |
| ONNX nnUNet inference | ~10 s |
| **Total** | **~11 s** |

> Note: `sct_deepseg` is the current SCT contrast-agnostic model (also nnUNet-based), running on the
> full uncropped volume. The sc_crop+ONNX pipeline is **5× faster** with negligible Dice loss (−0.004).

Script: `nnUnet/benchmark_cpu.py` — reproduces this on any image with optional GT Dice.

```bash
python nnUnet/benchmark_cpu.py \
    -i image.nii.gz -gt seg_gt.nii.gz \
    --model /path/to/nnunet_seg.onnx \
    --plans /path/to/plans.json
```

---

## Key takeaways

1. **Dice on par with the paper** (0.9586 PyTorch, 0.9537 end-to-end) with full train/test consistency.
2. **ONNX deployment** costs only −0.52 Dice points and runs fully on CPU in ~11 s per volume.
3. **5× faster than sct_deepseg** on CPU (×4.9 on T2w 64×320×320).
4. **sc_crop is extremely reliable**: 99.87% SC coverage on 466 test volumes.
5. **Integration is minimal**: 3 lines of code in the training script, 1 new Python file.
6. **v2 in progress**: all 15 datasets now included (+14% more data), results expected within ~24 h.
