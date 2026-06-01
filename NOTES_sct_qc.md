# Notes on sct_qc column limitations

## Columns in the QC HTML report

| Column | Source | Controllable |
|--------|--------|-------------|
| **Dataset** | `-qc-dataset` flag | ✓ — used for method filter (`gt`, `seg_v4`, `seg_v3`) |
| **Subject** | `-qc-subject` flag | ✓ — set to `dataset/subject` (e.g. `basel-mp2rage/sub-C069`) |
| **Contrast** | `Path(input_image).resolve().parent.name` | ✗ — always `anat` for BIDS images |
| **Command** | `-p` flag (standalone) or internal command | ✗ — `sct_deepseg_sc` for gt/v4, `sct_deepseg` for v3 |

## Why Contrast = "anat" is unavoidable

In `spinalcordtoolbox/reports/qc2.py`, the contrast is derived as:

```python
path_input = Path(fname_in1).resolve()  # follows symlinks
contrast = path_input.parent.name       # always the real parent folder
```

`resolve()` follows symlinks, so creating symlinks in contrast-named folders
(e.g. `/tmp/UNIT1/image.nii.gz → /path/to/anat/image.nii.gz`) does not work:
the resolved path points to the actual `anat/` folder.

The only way to fix contrast would be to **copy** images into contrast-named
folders — too expensive for 510 large NIfTI files.

## Why Command = "sct_deepseg_sc" for gt and v4

`sct_qc` standalone (used for gt and v4, which have no built-in `-qc` flag)
only accepts old command names as `-p`:
```
{sct_propseg, sct_deepseg_sc, sct_deepseg_gm, sct_deepseg_lesion, ...}
```

The modern `sct_deepseg` is not a valid `-p` option. It is only used internally
when calling `sct_deepseg spinalcord -qc` (v3 inference), which goes through
`qc2.sct_deepseg()` and sets `command = 'sct_deepseg'`.

This is a SCT limitation, not a pipeline design issue. The rendering (axial SC
slices) is identical between `sct_deepseg_sc` and `sct_deepseg`.

## Effective filters

Given these constraints, the useful filters in the QC HTML interface are:
- **Dataset** → switch between `gt`, `seg_v4`, `seg_v3`
- **Subject** → identifies which image/dataset (e.g. `basel-mp2rage/sub-C069_UNIT1`)
