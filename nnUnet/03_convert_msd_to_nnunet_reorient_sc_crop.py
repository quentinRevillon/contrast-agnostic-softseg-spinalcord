"""
Convert MSD datalists to nnU-Net format with RPI reorientation and sc_crop detection-based cropping.

Identical to 03_convert_msd_to_nnunet_reorient.py except that after RPI reorientation each image
is cropped around the spinal cord using sc_crop (YOLO-based detection). No GT segmentation mask
is required — the crop coordinates come from the detector.

This matches the inference pipeline: at test time sc_crop detects the SC bbox before segmentation,
so training on cropped volumes ensures train/test consistency.

Usage:
    python 03_convert_msd_to_nnunet_reorient_sc_crop.py \
        -i /path/to/MSD/datalists/folder \
        -o /path/to/nnUNet_raw/folder \
        --taskname contrastAgnosticCropped \
        --tasknumber 1000

Author: Quentin Revillon (adapted from 03_convert_msd_to_nnunet_reorient.py by Pierre-Louis Benveniste / Naga Karthik)
"""

import os
import argparse
import json
import tempfile
from pathlib import Path
from collections import OrderedDict

import nibabel as nib
import numpy as np

from sc_crop import detect, crop


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MSD dataset to nnU-Net format with sc_crop detection-based cropping')
    parser.add_argument('-i', '--input',      type=str, required=True, help='Folder containing MSD json datalist files')
    parser.add_argument('-o', '--output',     type=str, required=True, help='Output directory (nnUNet_raw)')
    parser.add_argument('--taskname',         type=str, default='contrastAgnosticCropped', help='Dataset name')
    parser.add_argument('--tasknumber',       type=int, required=True, help='Dataset number (nnUNet -d argument)')
    parser.add_argument('--pad-rl',  type=float, default=10.0, help='Right-Left padding in mm (default: 10)')
    parser.add_argument('--pad-ap',  type=float, default=15.0, help='Anterior-Posterior padding in mm (default: 15)')
    parser.add_argument('--pad-si',  type=float, default=30.0, help='Superior-Inferior padding in mm (default: 30)')
    return parser.parse_args()


def force_orthonormal_affine(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Force direction cosines to be exactly orthonormal (SVD).

    SimpleITK/ITK rejects NIfTI files whose direction cosines are not orthonormal
    (e.g. oblique acquisitions where floating-point rounding survives sct_image -setorient RPI).
    SVD gives the nearest orthonormal matrix while preserving voxel spacing and origin.
    """
    affine = img.affine.copy()
    R      = affine[:3, :3]
    spacing = np.linalg.norm(R, axis=0)
    U, _, Vt = np.linalg.svd(R)
    affine[:3, :3] = (U @ Vt) * spacing
    return nib.Nifti1Image(np.asarray(img.dataobj), affine, img.header)


def process_single_image(img_dict, counter, path_out_images, path_out_labels, taskname,
                          pad_rl, pad_ap, pad_si):
    """Reorient to RPI, sc_crop detection-based crop, then save in nnUNet format."""
    image_file_nnunet = os.path.join(path_out_images, f'{taskname}_{counter:03d}_0000.nii.gz')
    label_file_nnunet = os.path.join(path_out_labels, f'{taskname}_{counter:03d}.nii.gz')

    with tempfile.TemporaryDirectory() as tmpdir:
        img_rpi = os.path.join(tmpdir, 'img_rpi.nii.gz')
        lbl_rpi = os.path.join(tmpdir, 'lbl_rpi.nii.gz')

        # Reorient image and label to RPI
        assert os.system(f"sct_image -i {img_dict['image']} -setorient RPI -o {img_rpi}") == 0
        assert os.system(f"sct_image -i {img_dict['label']} -setorient RPI -o {lbl_rpi}") == 0

        # sc_crop: detect spinal cord bbox (no GT mask required), crop image and label with same bbox
        bbox = detect(img_rpi, pad_left=pad_rl, pad_right=pad_rl,
                      pad_anterior=pad_ap, pad_posterior=pad_ap,
                      pad_superior=pad_si, pad_inferior=pad_si)

        nib.save(crop(nib.load(img_rpi), bbox), image_file_nnunet)
        nib.save(crop(nib.load(lbl_rpi), bbox), label_file_nnunet)

    # Align label to image (match dimensions/resolution exactly — same as original pipeline)
    assert os.system(
        f"sct_register_multimodal -i {label_file_nnunet} -d {image_file_nnunet} "
        f"-identity 1 -o {label_file_nnunet} "
        f"-owarp file_to_delete_{counter}.nii.gz -owarpinv file_to_delete_2_{counter}.nii.gz"
    ) == 0
    os.system(f"rm -f file_to_delete_{counter}.nii.gz file_to_delete_2_{counter}.nii.gz")
    os.system(f"rm -f {label_file_nnunet.replace('.nii.gz', '_inv.nii.gz')}")

    # Binarize label
    assert os.system(f"sct_maths -i {label_file_nnunet} -bin 0.5 -o {label_file_nnunet}") == 0

    # Force orthonormal direction cosines — SimpleITK/ITK rejects oblique affines
    for path in [image_file_nnunet, label_file_nnunet]:
        nib.save(force_orthonormal_affine(nib.load(path)), path)

    return {
        'image':        str(os.path.abspath(img_dict['image'])),
        'label':        str(os.path.abspath(img_dict['label'])),
        'image_nnunet': image_file_nnunet,
        'label_nnunet': label_file_nnunet,
    }


def load_json_datalist(datalist_path, key):
    with open(datalist_path) as f:
        return json.load(f)[key]


def process_dataset(data_list, path_out_images, path_out_labels, taskname,
                    start_counter, pad_rl, pad_ap, pad_si):
    results = []
    for i, item in enumerate(data_list):
        counter = start_counter + i
        print(f"  [{counter}] {os.path.basename(item['image'])}")
        results.append(process_single_image(
            item, counter, path_out_images, path_out_labels, taskname,
            pad_rl, pad_ap, pad_si,
        ))
    return results


def main():
    args = parse_args()

    path_out            = Path(args.output) / f'Dataset{args.tasknumber}_{args.taskname}'
    path_out_imagesTr   = path_out / 'imagesTr'
    path_out_imagesTs   = path_out / 'imagesTs'
    path_out_labelsTr   = path_out / 'labelsTr'
    path_out_labelsTs   = path_out / 'labelsTs'

    for p in [path_out, path_out_imagesTr, path_out_imagesTs, path_out_labelsTr, path_out_labelsTs]:
        p.mkdir(parents=True, exist_ok=True)

    datalists = sorted(f for f in os.listdir(args.input) if f.endswith('_seed50.json'))
    train_data, val_data, test_data = [], [], []
    for datalist in datalists:
        print(f"Loading dataset: {datalist}")
        train_data += load_json_datalist(os.path.join(args.input, datalist), 'train')
        val_data   += load_json_datalist(os.path.join(args.input, datalist), 'validation')
        test_data  += load_json_datalist(os.path.join(args.input, datalist), 'test')

    print(f"\nTraining+validation: {len(train_data) + len(val_data)} subjects")
    print(f"Test:                {len(test_data)} subjects")

    print("\nProcessing training data...")
    train_results = process_dataset(
        train_data + val_data,
        str(path_out_imagesTr), str(path_out_labelsTr),
        args.taskname, 1, args.pad_rl, args.pad_ap, args.pad_si,
    )

    print("\nProcessing test data...")
    test_results = process_dataset(
        test_data,
        str(path_out_imagesTs), str(path_out_labelsTs),
        args.taskname, 1, args.pad_rl, args.pad_ap, args.pad_si,
    )

    conversion_dict = {}
    for r in train_results + test_results:
        conversion_dict[r['image']] = r['image_nnunet']
        conversion_dict[r['label']] = r['label_nnunet']

    with open(path_out / 'conversion_dict.json', 'w') as f:
        json.dump(conversion_dict, f, indent=4)

    json_dict = OrderedDict({
        'name':             args.taskname,
        'description':      args.taskname,
        'tensorImageSize':  '3D',
        'reference':        'TBD',
        'licence':          'TBD',
        'release':          '0.0',
        'channel_names':    {'0': 'MRI'},
        'labels':           {'background': 0, 'sc': 1},
        'numTraining':      len(train_results),
        'numTest':          len(test_results),
        'file_ending':      '.nii.gz',
        'image_orientation':'RPI',
        'training': [{'image': r['image_nnunet'], 'label': r['label_nnunet']} for r in train_results],
        'test':     [{'image': r['image_nnunet'], 'label': r['label_nnunet']} for r in test_results],
    })

    with open(path_out / 'dataset.json', 'w') as f:
        json.dump(json_dict, f, indent=4)

    print(f"\nDone. Dataset saved to {path_out}")


if __name__ == '__main__':
    main()
