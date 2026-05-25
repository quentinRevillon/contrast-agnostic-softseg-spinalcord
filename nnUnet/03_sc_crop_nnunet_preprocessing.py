"""
sc_crop preprocessing for nnUNet datasets — thin wrapper.

The full implementation lives in sc_crop.nnunet. Use it directly:

    from sc_crop.nnunet import preprocess_dataset

Or via CLI (after pip install sc-crop[yolo]):

    sc_crop preprocess-nnunet --nnunet-dir ... --output ...

This script is kept for compatibility with the contrast-agnostic training
pipeline (scripts/train_contrast_agnostic.sh step 3).
"""

from sc_crop.nnunet import main

if __name__ == "__main__":
    main()
