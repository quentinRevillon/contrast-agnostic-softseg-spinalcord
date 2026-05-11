"""
Custom nnUNetv2 trainer adding a random foreground-crop simulation augmentation
on top of the default nnUNetTrainer pipeline (used to train contrast-agnostic v3.0).

The augmentation crops each patch around the segmentation foreground bbox
with a random padding in [0, MAX_PAD_VOX] voxels, then zero-pads back to the
original patch size (centred). This teaches the model to segment correctly
when the cord is close to or touching the image border.

Reuses acvl_utils bbox utilities already present in the nnUNetv2 dependency tree:
  - get_bbox_from_mask : mask → [[lo,hi], ...]
  - pad_bbox           : expand bbox by N voxels, clipped to array_shape
  - bounding_box_to_slice : bbox → tuple of slices

Install (so nnUNetv2_train can discover it via -tr):
    cp nnUnet/nnUNetTrainer_RandomForegroundCrop.py \
       $(python -c "import nnunetv2; print(nnunetv2.__path__[0])")/training/nnUNetTrainer/variants/data_augmentation/

Usage (drop-in replacement for the v3.0 training command):
    CUDA_VISIBLE_DEVICES=X nnUNetv2_train DATASET_ID 3d_fullres FOLD \
        -tr nnUNetTrainer_RandomForegroundCrop -p nnUNetPlans
"""

import numpy as np
import torch

from acvl_utils.cropping_and_padding.bounding_boxes import (
    bounding_box_to_slice,
    get_bbox_from_mask,
    pad_bbox,
)
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# Maximum padding around the foreground bbox, in voxels.
# At the model's preprocessed spacing (~0.7–1.0 mm/vox) this is roughly 28–40 mm.
MAX_PAD_VOX = 40
CROP_PROB = 0.8


class ForegroundCropSimTransform(BasicTransform):
    """
    Randomly crop a patch around the segmentation foreground + random padding,
    then zero-pad back to the original patch size (centred).
    """

    def __init__(self, max_pad_vox: int = MAX_PAD_VOX, apply_probability: float = CROP_PROB):
        super().__init__()
        self.max_pad_vox = max_pad_vox
        self.apply_probability = apply_probability

    def get_parameters(self, **data_dict) -> dict:
        if np.random.random() > self.apply_probability:
            return {"skip": True}

        seg = data_dict["segmentation"]          # (C, X, Y, Z) torch.Tensor
        mask = (seg[0] > 0).numpy()
        if not mask.any():
            return {"skip": True}

        bbox = get_bbox_from_mask(mask)
        pad = int(np.random.randint(0, self.max_pad_vox + 1))
        bbox = pad_bbox(bbox, pad, array_shape=mask.shape)
        return {"skip": False, "slicer": bounding_box_to_slice(bbox), "orig_shape": seg.shape[1:]}

    def apply(self, data_dict: dict, **params) -> dict:
        if params["skip"]:
            return data_dict

        slicer = (slice(None),) + params["slicer"]   # keep channel dim
        orig_shape = np.array(params["orig_shape"])

        cropped_img = data_dict["image"][slicer]
        cropped_seg = data_dict["segmentation"][slicer]

        crop_shape = np.array(cropped_img.shape[1:])
        offset = (orig_shape - crop_shape) // 2
        dst = tuple(slice(int(offset[i]), int(offset[i]) + int(crop_shape[i])) for i in range(3))

        padded_img = torch.zeros_like(data_dict["image"])
        padded_seg = torch.zeros_like(data_dict["segmentation"])
        padded_img[(slice(None),) + dst] = cropped_img
        padded_seg[(slice(None),) + dst] = cropped_seg

        data_dict["image"] = padded_img
        data_dict["segmentation"] = padded_seg
        return data_dict


class nnUNetTrainer_RandomForegroundCrop(nnUNetTrainer):
    """
    Identical to nnUNetTrainer (used for contrast-agnostic v3.0) with a single
    addition: ForegroundCropSimTransform prepended to the augmentation pipeline.
    """

    @staticmethod
    def get_training_transforms(
        patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
        do_dummy_2d_data_aug, use_mask_for_norm=None, is_cascaded=False,
        foreground_labels=None, regions=None, ignore_label=None,
    ):
        base = nnUNetTrainer.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug, use_mask_for_norm=use_mask_for_norm,
            is_cascaded=is_cascaded, foreground_labels=foreground_labels,
            regions=regions, ignore_label=ignore_label,
        )
        return ComposeTransforms(
            [ForegroundCropSimTransform(MAX_PAD_VOX, CROP_PROB)] + base.transforms
        )
