"""
03_convert_msd_to_nnunet_reorient_cropped.py — MSD → nnUNet conversion with RPI reorientation
and SC bounding box crop.

Identical to 03_convert_msd_to_nnunet_reorient.py with one addition: after reorientation and
label registration, both image and label are cropped to the spinal cord GT bounding box +
padding_mm per face. This gives nnUNet a realistic view of the SC field of view, leading to
a more relevant patch size during plan_and_preprocess.

Example:
    python 03_convert_msd_to_nnunet_reorient_cropped.py
        -i /path/to/MSD/datalists/folder
        -o /path/to/nnUNet_raw/folder
        --taskname TempContrastAgnosticCropped
        --tasknumber 998
        --padding-mm 10.0
        --workers 8

Author: Pierre-Louis Benveniste (adapted by Naga Karthik); SC crop added by Quentin Revillon
"""

import argparse
import json
import os
from collections import OrderedDict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import nibabel as nib
import numpy as np
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MSD dataset to nnU-Net format with SC bbox crop')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to the folder containing MSD json files')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--taskname', type=str, default='msLesionAgnostic',
                        help='Name of the task')
    parser.add_argument('--tasknumber', type=int, required=True,
                        help='Number of the task')
    parser.add_argument('--padding-mm', type=float, default=10.0,
                        help='Padding added per face of the SC bbox in mm (default: 10)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: number of CPU cores)')
    return parser.parse_args()


def crop_nifti_to_sc_bbox(image_path: str, label_path: str, padding_mm: float) -> None:
    """Crop image and label in place to the SC GT bounding box + padding_mm per face."""
    label_nib = nib.load(label_path)
    image_nib = nib.load(image_path)

    label_data = np.asarray(label_nib.dataobj)
    image_data = np.asarray(image_nib.dataobj)

    spacing = np.array(label_nib.header.get_zooms()[:3])

    fg = label_data > 0
    if not fg.any():
        return  # no SC voxels: leave files untouched

    coords = np.array(np.where(fg))
    min_coords = coords.min(axis=1)
    max_coords = coords.max(axis=1)
    shape = np.array(label_data.shape[:3])

    pad_voxels = np.ceil(padding_mm / spacing).astype(int)
    lo = np.maximum(0, min_coords - pad_voxels)
    hi = np.minimum(shape - 1, max_coords + pad_voxels)

    slices = tuple(slice(int(lo[i]), int(hi[i]) + 1) for i in range(3))

    cropped_label = label_data[slices]
    cropped_image = image_data[slices]

    # Shift affine origin to world coordinates of the new voxel [0,0,0]
    old_affine = label_nib.affine
    new_affine = old_affine.copy()
    new_affine[:3, 3] = (old_affine @ np.array([lo[0], lo[1], lo[2], 1.0]))[:3]

    nib.save(nib.Nifti1Image(cropped_label.astype(label_data.dtype), new_affine), label_path)
    nib.save(nib.Nifti1Image(cropped_image.astype(image_data.dtype), new_affine), image_path)


def process_single_image(args):
    """Reorient to RPI, align label to image, binarize, then crop to SC bbox."""
    img_dict, counter, path_out_images, path_out_labels, taskname, padding_mm = args

    image_file_nnunet = os.path.join(path_out_images, f'{taskname}_{counter:03d}_0000.nii.gz')
    label_file_nnunet = os.path.join(path_out_labels, f'{taskname}_{counter:03d}.nii.gz')

    # Reorient image to RPI
    assert os.system(f"sct_image -i {img_dict['image']} -setorient RPI -o {image_file_nnunet}") == 0

    # Reorient label to RPI
    assert os.system(f"sct_image -i {img_dict['label']} -setorient RPI -o {label_file_nnunet}") == 0

    # Put label to image to match dimension, resolution and orientation
    assert os.system(f"sct_register_multimodal -i {str(label_file_nnunet)} -d {str(image_file_nnunet)} "
                     f"-identity 1 -o {str(label_file_nnunet)} -owarp file_to_delete_{counter}.nii.gz "
                     f"-owarpinv file_to_delete_2_{counter}.nii.gz") == 0

    # Clean up temporary files
    os.system(f"rm file_to_delete_{counter}.nii.gz file_to_delete_2_{counter}.nii.gz")
    other_file_to_remove = str(label_file_nnunet).replace('.nii.gz', '_inv.nii.gz')
    os.system(f"rm {other_file_to_remove}")

    # Binarize label
    assert os.system(f"sct_maths -i {str(label_file_nnunet)} -bin 0.5 -o {str(label_file_nnunet)}") == 0

    # Crop both image and label to SC bbox + padding
    crop_nifti_to_sc_bbox(image_file_nnunet, label_file_nnunet, padding_mm)

    return {
        'image': str(os.path.abspath(img_dict['image'])),
        'label': str(os.path.abspath(img_dict['label'])),
        'image_nnunet': image_file_nnunet,
        'label_nnunet': label_file_nnunet
    }


def process_dataset_parallel(data_list, path_out_images, path_out_labels, taskname,
                              start_counter, num_workers, padding_mm):
    with Pool(processes=num_workers) as pool:
        work_items = [
            (item, start_counter + i, path_out_images, path_out_labels, taskname, padding_mm)
            for i, item in enumerate(data_list)
        ]
        results = list(tqdm.tqdm(
            pool.imap(process_single_image, work_items),
            total=len(work_items),
            desc="Processing images"
        ))
    return results


def load_json_datalist(datalist_path, key_to_extract):
    with open(datalist_path, 'r') as f:
        datalist = json.load(f)
    return datalist[key_to_extract]


def main():
    args = parse_args()
    if args.workers is None:
        args.workers = cpu_count()

    path_out = Path(os.path.join(args.output, f'Dataset{args.tasknumber}_{args.taskname}'))
    path_out_imagesTr = path_out / 'imagesTr'
    path_out_imagesTs = path_out / 'imagesTs'
    path_out_labelsTr = path_out / 'labelsTr'
    path_out_labelsTs = path_out / 'labelsTs'

    for path in [path_out, path_out_imagesTr, path_out_imagesTs, path_out_labelsTr, path_out_labelsTs]:
        path.mkdir(parents=True, exist_ok=True)

    datalists_list = [f for f in os.listdir(args.input) if f.endswith("_seed50.json")]
    train_data, val_data, test_data = [], [], []
    for datalist in sorted(datalists_list):
        print(f"Loading dataset: {datalist}")
        train_data += load_json_datalist(os.path.join(args.input, datalist), key_to_extract="train")
        val_data += load_json_datalist(os.path.join(args.input, datalist), key_to_extract="validation")
        test_data += load_json_datalist(os.path.join(args.input, datalist), key_to_extract="test")

    print(f"Processing {len(datalists_list)} datasets with {args.workers} workers...")
    print(f"Training samples: {len(train_data + val_data)}  |  Test samples: {len(test_data)}")
    print(f"SC bbox padding: {args.padding_mm} mm")

    print("Processing training data...")
    train_results = process_dataset_parallel(
        train_data + val_data, path_out_imagesTr, path_out_labelsTr,
        args.taskname, 1, args.workers, args.padding_mm
    )

    print("Processing test data...")
    test_results = process_dataset_parallel(
        test_data, path_out_imagesTs, path_out_labelsTs,
        args.taskname, 1, args.workers, args.padding_mm
    )

    conversion_dict = {}
    for result in train_results + test_results:
        conversion_dict[result['image']] = result['image_nnunet']
        conversion_dict[result['label']] = result['label_nnunet']

    with open(path_out / "conversion_dict.json", "w") as f:
        json.dump(conversion_dict, f, indent=4)

    json_dict = OrderedDict({
        'name': args.taskname,
        'description': f'{args.taskname} — SC bbox crop {args.padding_mm}mm padding',
        'tensorImageSize': "3D",
        'reference': "TBD",
        'licence': "TBD",
        'release': "0.0",
        'channel_names': {"0": "MRI"},
        'labels': {"background": 0, "sc": 1},
        'numTraining': len(train_results),
        'numTest': len(test_results),
        'file_ending': ".nii.gz",
        'image_orientation': "RPI",
        'training': [{'image': r['image_nnunet'], 'label': r['label_nnunet']} for r in train_results],
        'test': [{'image': r['image_nnunet'], 'label': r['label_nnunet']} for r in test_results]
    })

    with open(path_out / "dataset.json", "w") as f:
        json.dump(json_dict, f, indent=4)

    print("Conversion completed successfully!")


if __name__ == '__main__':
    main()
