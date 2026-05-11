# Issues found in `train_contrast_agnostic.sh`

These are bugs and missing setup steps encountered when running the script on a server via a
non-interactive shell (e.g. `set_slot`, `nohup`, SSH without login shell).

---

## 1. Script not changing to repo root before running

**Error:**
```
FileNotFoundError: datasplits/datasplit_lumbar-epfl_seed50.yaml
```

**Cause:** The script calls Python scripts with relative paths (e.g. `datasplits/...`) but never
`cd` to the repo root. If the script is launched from another directory, all relative paths break.

**Fix:** Add `cd ${PATH_REPO}` at the top of the script, right after `PATH_REPO` is defined.

---

## 2. `python` not resolved in non-interactive shells

**Error:**
```
ModuleNotFoundError: No module named 'loguru'
# or
command not found: python
```

**Cause:** The script calls `python` without specifying the conda environment. In a non-interactive
shell, `conda activate` is not run and the system Python (or no Python) is used instead.

**Fix:** Define an explicit Python path variable and use it throughout:
```bash
PYTHON=/path/to/conda/envs/contrast_agnostic/bin/python
```
Then replace all `python` calls with `${PYTHON}`.

---

## 3. `nnUNetv2_plan_and_preprocess` and `nnUNetv2_train` not found

**Error:**
```
command not found: nnUNetv2_plan_and_preprocess
command not found: nnUNetv2_train
```

**Cause:** nnUNet CLI commands are installed in the conda environment's `bin/`. Without exporting
`PATH`, they are not found in a non-interactive shell.

**Fix:**
```bash
export PATH=/path/to/conda/envs/contrast_agnostic/bin:$PATH
```

---

## 4. `sct_image` / SCT commands not found during dataset conversion

**Error:**
```
command not found: sct_image
```

**Cause:** `03_convert_msd_to_nnunet_reorient.py` calls SCT commands (e.g. `sct_image`,
`sct_register_multimodal`) via subprocess. SCT's `bin/` directory is not in `PATH` in a
non-interactive shell.

**Fix:**
```bash
export PATH=/path/to/spinalcordtoolbox/bin:$PATH
```

---

## 5. nnUNet environment variables not set

**Error:**
```
AssertionError  # or silent failure during nnUNet import
```

**Cause:** nnUNet requires three environment variables to be set at import time:
`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`. The script defines local variables
`PATH_NNUNET_RAW` / `PATH_NNUNET_RESULTS` but never exports them as the nnUNet-expected names.

**Fix:** Add before any nnUNet call:
```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

Note: these directories do **not** come with the nnUNet package — they must be created manually.

---

## 6. `cuda_visible_devices` hardcoded to GPU index 2 which may not exist

**Error:**
```
RuntimeError: No CUDA GPUs are available
```

**Cause:** `cuda_visible_devices=2` is hardcoded but the server may only have GPUs 0 and 1.
When `CUDA_VISIBLE_DEVICES` points to a non-existent index, PyTorch sees no GPU at all.

**Fix:** Check available GPUs with `nvidia-smi` and set `cuda_visible_devices` accordingly (e.g. `0` or `1`).

---

## 7. nnUNetv2 incompatible with PyTorch ≥ 2.8 (`PolyLRScheduler` verbose argument removed)

**Error:**
```
TypeError: __init__() takes from 2 to 3 positional arguments but 4 were given
  File "polylr.py", line 11: super().__init__(optimizer, ..., False)
```

**Cause:** PyTorch ≥ 2.8 removed the `verbose` positional argument from `LRScheduler.__init__()`.
nnUNetv2 ≤ 2.5.2 still passes it as `False`. Upgrading nnUNetv2 (`pip install --upgrade nnunetv2`)
does not fix this — the bug is still present in the latest release as of 2026-04-20.

The GPU (NVIDIA Blackwell sm_120) requires PyTorch ≥ 2.7 (cu128), so there is no PyTorch version
that satisfies both constraints without patching.

**Fix:** Patch `polylr.py` in the installed package:
```bash
sed -i \
  's/from torch.optim.lr_scheduler import _LRScheduler/from torch.optim.lr_scheduler import LRScheduler as _LRScheduler/' \
  /home/quentinr/.conda/envs/contrast_agnostic/lib/python3.9/site-packages/nnunetv2/training/lr_scheduler/polylr.py
sed -i \
  's/super().__init__(optimizer, current_step if current_step is not None else -1, False)/super().__init__(optimizer, current_step if current_step is not None else -1)/' \
  /home/quentinr/.conda/envs/contrast_agnostic/lib/python3.9/site-packages/nnunetv2/training/lr_scheduler/polylr.py
```

---

## 8. `torch.compile` crashes training via `BrokenProcessPool`

**Error:**
```
Using torch.compile...
concurrent.futures.process.BrokenProcessPool: A process in the process pool was terminated abruptly
```

**Cause:** nnUNetv2 ≥ 2.4 enables `torch.compile` by default. The inductor backend spawns
subprocesses for JIT compilation — on some systems/configurations these workers crash silently,
killing the training.

**Fix:** Disable torch.compile via environment variable before launching training:
```bash
export TORCHDYNAMO_DISABLE=1
```

---

## 9. PyTorch does not support NVIDIA Blackwell GPU (sm_120)

**Error:**
```
NVIDIA RTX PRO 6000 Blackwell Max-Q with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Cause:** The Blackwell architecture (sm_120) requires PyTorch ≥ 2.6. PyTorch 2.4 only supports
up to sm_90 (Hopper). This was masked by earlier errors and only visible after adding
`CUDA_LAUNCH_BLOCKING=1`.

**Fix:** PyTorch 2.7.0 with cu128 is the first version with (experimental) sm_120 support:
```bash
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```
Note: PyTorch 2.8.0 also supports sm_120 but breaks nnUNetv2 2.5.2 (`PolyLRScheduler` TypeError — see issue #7).
Support is prototype/experimental in 2.7.0 — known gaps in nvfuser and cuSPARSELt.
See: https://pytorch.org/blog/pytorch-2-7/

---

## 11. Custom trainer `nnUNetTrainer_CropAugmentation`

**Not a bug** — documentation of the custom trainer created for YOLO crop robustness.

**File:**
```
/home/quentinr/.conda/envs/contrast_agnostic/lib/python3.9/site-packages/nnunetv2/training/nnUNetTrainer/variants/data_augmentation/nnUNetTrainer_CropAugmentation.py
```

**What it does:**
- Prepends a `YOLOCropSimulationTransform` (p=0.5 per sample) to the standard nnUNet augmentation pipeline
- Computes the tight GT bounding box in each patch
- Adds random padding per face drawn from a 50/50 mixture: U[0, 10mm] or U[10mm, max_available]
- Zeros all voxels outside the resulting crop region; tensor stays at fixed patch_size

**Usage:**
```bash
nnUNetv2_train <DATASET_ID> 3d_fullres <FOLD> -tr nnUNetTrainer_CropAugmentation
```

---

## 10. `CUDA_VISIBLE_DEVICES` not set for preprocessing step

**Error:**
```
RuntimeError: No CUDA GPUs are available
```

**Cause:** The training command already had `CUDA_VISIBLE_DEVICES` set inline:
```bash
CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_train ...
```
But `nnUNetv2_plan_and_preprocess` (called before training) had no such inline assignment,
so it ran without knowing which GPU to use.

**Fix:** Export the variable after it is defined so it is available for all subsequent commands:
```bash
cuda_visible_devices=2
export CUDA_VISIBLE_DEVICES=${cuda_visible_devices}
```
