# Towards Contrast-agnostic Soft Segmentation of the Spinal Cord

[![MedIA](https://img.shields.io/badge/MedIA-10.106/media.2025.103473-darkgreen.svg)](https://doi.org/10.1016/j.media.2025.103473)
[![ImagingNeuroscience](https://img.shields.io/badge/ImagingNeuroscience-10.1162/IMAG.a.1105-darkgreen.svg)](https://doi.org/10.1162/IMAG.a.1105)

Official repository for contrast-agnostic segmentation of the spinal cord. This `sc-crop-v4` branch trains the model on **detection-cropped** images (YOLO crop via [`sc-crop`](https://github.com/ivadomed/sc-crop) + [nnUNetv2](https://github.com/MIC-DKFZ/nnUNet)), and ships a standalone cropping + inference pipeline (`sc-segment-pt` / `sc-segment-onnx`) that does not require the full Spinal Cord Toolbox at runtime.

## Table of contents

1. [Lifelong-learning monitoring](#1-lifelong-learning-monitoring)
2. [Retraining the model](#2-retraining-the-model)
   - [2.1. Environment and data](#21-environment-and-data)
   - [2.2. Train](#22-train)
   - [2.3. Test](#23-test)
   - [2.4. Automated drift monitoring — the GitHub Action](#24-automated-drift-monitoring--the-github-action)
3. [Using the trained model](#3-using-the-trained-model)
4. [Citation](#4-citation)

---

## 1. Lifelong-learning monitoring

The spinal cord segmentation model is improved over time as new contrasts and pathologies are added. To make sure each new version does not silently regress, we **automatically monitor morphometric drift** across model versions: every time a new model is released, a GitHub Actions workflow recomputes the spinal cord cross-sectional area (CSA) and compares it against all previously released versions.

![Lifelong learning figure](imag.a.1105_fig2.png)

For a fair, reproducible comparison, every model version is evaluated on the **frozen test split** of the public spine-generic [`data-multi-subject`](https://github.com/spine-generic/data-multi-subject) dataset:

- **Metric:** C2–C3 CSA, computed per subject for all 6 contrasts.
- **Test set:** *n = 49* subjects, fixed in [`scripts/spine_generic_test_split_for_csa_drift_monitoring.yaml`](scripts/spine_generic_test_split_for_csa_drift_monitoring.yaml).
- **Output:** violin plots of CSA per contrast and of the CSA standard deviation across contrasts, comparing the new model against every previous release.

The end-to-end pipeline that produces these plots is described in [section 2.4](#24-automated-drift-monitoring--the-github-action).

---

## 2. Retraining the model

Three steps: set up the environment, train, test. The automated monitoring that runs on every release is detailed last, in [2.4](#24-automated-drift-monitoring--the-github-action).

### 2.1. Environment and data

1. Create and activate a conda environment:

   ```bash
   conda create -n contrast_agnostic_310 python=3.10 -y
   conda activate contrast_agnostic_310
   ```

2. Clone this branch:

   ```bash
   git clone https://github.com/quentinRevillon/contrast-agnostic-softseg-spinalcord.git --branch sc-crop-v4
   cd contrast-agnostic-softseg-spinalcord
   ```

3. Install the dependencies (nnUNet + PyTorch):

   ```bash
   pip install -r nnUnet/requirements.txt
   ```

   > **Note**
   > `requirements.txt` pins nnUNet to a specific GitHub commit (v2.6.0) compatible with PyTorch 2.8+cu128. PyTorch 2.8 is required for Blackwell GPUs (sm_120, e.g. RTX PRO 6000). The PyPI `nnunetv2==2.5.2` is broken with PyTorch>=2.4 (removed `verbose` parameter in `_LRScheduler`).

You do **not** need to download the datasets manually: `scripts/train_contrast_agnostic.sh` clones them from NeuroPoly's git-annex server as its first step.

### 2.2. Train

`scripts/train_contrast_agnostic.sh` runs the full pipeline in 5 steps: **(1)** clone datasets from git-annex, **(2)** create MSD-style datalists from the predefined splits, **(3)** convert to the nnUNet format with `sc-crop` detection-based cropping applied to each image/label pair, **(4)** nnUNet preprocessing, **(5)** nnUNet training.

```bash
bash scripts/train_contrast_agnostic.sh
```

> [!IMPORTANT]
> The script does **not** run out-of-the-box. You must set the user-specific variables at the top of the script (repo path, dataset download path, nnUNet `raw`/`preprocessed`/`results` paths, dataset number, GPU id, …). All variables are documented inline in the script.

> [!TIP]
> The script **auto-resumes**: it detects the furthest completed step and restarts from there. You can also force a range with `START_STEP` / `END_STEP`, e.g. `START_STEP=5 bash scripts/train_contrast_agnostic.sh` to (re)train only. Run it inside `tmux` or `screen` so training survives disconnections.

### 2.3. Test

Once training is complete, run inference on the held-out test set and compute Dice scores. The test set (`imagesTs` / `labelsTs`) is already in `nnUNet_raw`, populated by step 3 of the training script.

```bash
bash scripts/test_contrast_agnostic.sh
```

By default it uses `checkpoint_best.pth`. Results are written to:

```
nnUNet_results/Dataset7000_ContrastAgnosticScCrop/nnUNetTrainer__nnUNetPlans__3d_fullres/predictions_test_fold0_checkpoint_best/summary.json
```

To evaluate the final checkpoint instead:

```bash
CHECKPOINT=checkpoint_final.pth bash scripts/test_contrast_agnostic.sh
```

> Dice results for this branch (v4.1 vs v3.0) are documented in [issue #5](https://github.com/quentinRevillon/contrast-agnostic-softseg-spinalcord/issues/5).

### 2.4. Automated drift monitoring — the GitHub Action

Publishing a release triggers the workflow [`.github/workflows/run_morphometric_analysis.yml`](.github/workflows/run_morphometric_analysis.yml), which produces the CSA drift plots shown in [section 1](#1-lifelong-learning-monitoring).

**Creating the release**

Create a GitHub release with these conventions:

- **Tag:** `vX.Y` (e.g. `v3.0`, `v4.1`), where `X` is a major change (architecture / training strategy) and `Y` a minor one (new contrasts and/or pathologies). The tag name is reused to name the output CSV, so it matters.
- **Title:** anything (the workflow does not depend on it).
- **Description:** a drop-down summary of the dataset characteristics, generated from `nnUnet/utils.py`.
- **Assets:** the entire nnUNet output folder (containing `plans.json`, `dataset.json`, and `fold_0/checkpoint_best.pth`), zipped as `model_contrast_agnostic_<training-date>.zip`.

**The workflow (4 jobs)**

1. **`download_dataset`** — clones `data-multi-subject` via git-annex, downloading only the test-split subjects, and caches it.
2. **`compute_csa`** — splits the *n = 49* test subjects into 16 parallel batches. Each runner installs SCT + the `sc-crop` pipeline, downloads the model from the release, and computes the C2–C3 CSA for all 6 contrasts. **Inference uses the cropping pipeline (`sc-segment-pt`: YOLO crop + nnUNet, on CPU), not `sct_deepseg`.**
3. **`download_results`** — aggregates the per-batch CSVs into a single `csa_c2c3__model_<tag>.csv` and uploads it to the release.
4. **`generate_plots`** — downloads the `csa_c2c3__model_<tag>.csv` of the current and all previous releases, generates the comparison violin plots, and uploads them as `morphometric_plots.zip` to the release.

> [!TIP]
> **Test it locally before publishing a release.** The same scripts run without GitHub Actions — point the model argument at a local checkpoint and `SC_SEGMENT_BIN` at the `sc-crop` env binary:
> ```bash
> SC_SEGMENT_BIN=/path/to/sc_crop/bin/sc-segment-pt \
>     bash scripts/compute_morphometrics_spine_generic.sh "sub-barcelona06" "/path/to/fold_0/checkpoint_best.pth"
> ```
> Once the C2–C3 CSV is produced correctly, publish the release to trigger the workflow.

---

## 3. Using the trained model

How to segment a new image with a trained model. All steps are copy-paste ready and tested end-to-end.

### 3.1. Install the inference pipeline

```bash
conda create -n sc_crop python=3.10 -y
conda activate sc_crop
pip install \
    "sc-crop @ git+https://github.com/ivadomed/sc-crop.git@v0.4.1" \
    "nnunet-onnx @ git+https://github.com/quentinRevillon/nnunet-onnx.git" \
    torch nnunetv2 onnxscript onnx
```

### 3.2. Download the model and an example image

```bash
mkdir ~/ca-inference-test && cd ~/ca-inference-test

# Example T2w image (from sc-crop test data)
curl -L https://github.com/ivadomed/sc-crop/releases/download/test-data/t2.nii.gz -o t2.nii.gz

# v4.1 model
curl -L https://github.com/quentinRevillon/contrast-agnostic-softseg-spinalcord/releases/download/v4.1/model_contrast_agnostic_20260602.zip -o model_v4.zip
unzip model_v4.zip
```

### 3.3. Segment — PyTorch (`sc-segment-pt`)

Runs the full pipeline: YOLO detection crop (`sc-crop`) followed by nnUNet segmentation.

```bash
sc-segment-pt \
    -i t2.nii.gz -o seg_v4_pt.nii.gz \
    --checkpoint nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth
```

### 3.4. Segment — ONNX (`sc-segment-onnx`)

ONNX needs no nnUNet at runtime and is ~10× faster than `sct_deepseg spinalcord` on CPU (see benchmark in [issue #2](https://github.com/quentinRevillon/contrast-agnostic-softseg-spinalcord/issues/2)). First export the checkpoint once:

```bash
python -m nnunet_onnx.export \
    --checkpoint nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth \
    --output model_contrast_agnostic_v4.1.onnx
```

Then segment:

```bash
sc-segment-onnx \
    -i t2.nii.gz -o seg_v4_onnx.nii.gz \
    --model model_contrast_agnostic_v4.1.onnx
```

### 3.5. Segment — v3.0 model via SCT (baseline)

`sct_deepseg spinalcord` uses the v3.0 contrast-agnostic model bundled in [SCT](https://spinalcordtoolbox.com/stable/user_section/command-line/deepseg/spinalcord.html) — no separate download needed. Useful as a baseline for comparison.

```bash
sct_deepseg spinalcord -i t2.nii.gz -o seg_sct.nii.gz
```

### 3.6. Visualise

```bash
fsleyes t2.nii.gz seg_sct.nii.gz -cm red seg_v4_pt.nii.gz -cm blue seg_v4_onnx.nii.gz -cm green &
```

---

## 4. Citation

If you find this work and/or code useful for your research, please cite our papers:

```bibtex
@article{BEDARD2025103473,
title = {Towards contrast-agnostic soft segmentation of the spinal cord},
journal = {Medical Image Analysis},
volume = {101},
pages = {103473},
year = {2025},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2025.103473},
url = {https://www.sciencedirect.com/science/article/pii/S1361841525000210},
author = {Sandrine Bédard* and Enamundram Naga Karthik* and Charidimos Tsagkas and Emanuele Pravatà and Cristina Granziera and Andrew Smith and Kenneth Arnold {Weber II} and Julien Cohen-Adad},
note = {Shared authorship -- authors contributed equally}
}
```

```bibtex
@article{Karthik2026,
title = {Monitoring morphometric drift in lifelong learning segmentation of the spinal cord},
journal = {Imaging Neuroscience},
volume = {4},
pages = {IMAG.a.1105},
year = {2026},
doi = {https://doi.org/10.1162/IMAG.a.1105},
author = {Enamundram Naga Karthik and Sandrine Bédard and Jan Valošek and Christoph S Aigner and Elise Bannier and Josef Bednařík and Virginie Callot and Anna Combes and Armin Curt and Gergely David and Falk Eippert and Lynn Farner and Michael G Fehlings and Patrick Freund and Tobias Granberg and Cristina Granziera and Ulrike Horn and Tomáš Horák and Suzanne Humphreys and Markus Hupp and Anne Kerbrat and Nawal Kinany and Shannon Kolind and Petr Kudlička and Anna Lebret and Lisa Eunyoung Lee and Caterina Mainero and Allan R Martin and Megan McGrath and Govind Nair and Kristin P O'Grady and Jiwon Oh and Russell Ouellette and Nikolai Pfender and Dario Pfyffer and Pierre-François Pradat and Alexandre Prat and Emanuele Pravatà and Daniel S Reich and Ilaria Ricchi and Naama Rotem-Kohavi and Simon Schading-Sassenhausen and Maryam Seif and Andrew Smith and Seth A Smith and Grace Sweeney and Roger Tam and Anthony Traboulsee and Constantina Andrada Treaba and Charidimos Tsagkas and Zachary Vavasour and Dimitri Van De Ville and Kenneth Arnold Weber II and Sarath Chandar and Julien Cohen-Adad}
}
```
