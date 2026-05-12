"""
Identical to 03_convert_msd_to_nnunet_reorient.py with one addition: after binarization,
both image and label are cropped to the spinal cord GT bounding box with per-face padding in mm.

Images are in RPI orientation after reorientation, so:
  axis 0 (R→): lo = Left face,     hi = Right face
  axis 1 (P→): lo = Anterior face, hi = Posterior face
  axis 2 (I→): lo = Superior face, hi = Inferior face

Example:
    python 03_convert_msd_to_nnunet_reorient_cropped.py
        -i /path/to/MSD/datalists/folder
        -o /path/to/nnUNet_raw/folder
        --taskname ContrastAgnosticCropped
        --tasknumber 997
        --pad-left 20 --pad-right 20
        --pad-anterior 30 --pad-posterior 30
        --pad-superior 40 --pad-inferior 40
        --workers 8

Author: Pierre-Louis Benveniste (adapted for multiprocessing by Naga Karthik);
SC bbox crop added by Quentin Revillon
"""

import os
import argparse
import json
from pathlib import Path
import tqdm
from collections import OrderedDict
from multiprocessing import Pool, cpu_count

import nibabel as nib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MSD dataset to nnU-Net format with SC bbox crop')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the folder containing MSD json files')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output directory')
    parser.add_argument('--taskname', type=str, help='Name of the task', default='msLesionAgnostic')
    parser.add_argument('--tasknumber', type=int, required=True, help='Number of the task')
    parser.add_argument('--pad-left',      type=float, default=20.0, help='Padding on the Left face in mm (RPI axis 0 lo)')
    parser.add_argument('--pad-right',     type=float, default=20.0, help='Padding on the Right face in mm (RPI axis 0 hi)')
    parser.add_argument('--pad-anterior',  type=float, default=30.0, help='Padding on the Anterior face in mm (RPI axis 1 lo)')
    parser.add_argument('--pad-posterior', type=float, default=30.0, help='Padding on the Posterior face in mm (RPI axis 1 hi)')
    parser.add_argument('--pad-superior',  type=float, default=40.0, help='Padding on the Superior face in mm (RPI axis 2 lo)')
    parser.add_argument('--pad-inferior',  type=float, default=40.0, help='Padding on the Inferior face in mm (RPI axis 2 hi)')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: number of CPU cores)')
    return parser.parse_args()


def crop_to_sc_bbox(image_path, label_path, pad_lo, pad_hi):
    """Crop image and label in-place to the SC GT bounding box + per-face padding.

    pad_lo: [left, anterior, superior] in mm (faces at low indices in RPI)
    pad_hi: [right, posterior, inferior] in mm (faces at high indices in RPI)
    """
    lbl_nib = nib.load(label_path)
    img_nib = nib.load(image_path)
    lbl = np.asarray(lbl_nib.dataobj)
    img = np.asarray(img_nib.dataobj)
    spacing = np.array(lbl_nib.header.get_zooms()[:3])

    coords = np.array(np.where(lbl > 0))
    mn, mx = coords.min(axis=1), coords.max(axis=1)
    shape = np.array(lbl.shape[:3])

    pad_lo_vox = np.ceil(np.array(pad_lo) / spacing).astype(int)
    pad_hi_vox = np.ceil(np.array(pad_hi) / spacing).astype(int)
    lo = np.maximum(0, mn - pad_lo_vox)
    hi = np.minimum(shape - 1, mx + pad_hi_vox)

    slices = tuple(slice(lo[i], hi[i] + 1) for i in range(3))

    affine = lbl_nib.affine.copy()
    affine[:3, 3] = (lbl_nib.affine @ np.array([lo[0], lo[1], lo[2], 1.0]))[:3]
    # Re-orthonormalize rotation part (ITK requires exact orthonormality)
    R = affine[:3, :3]
    vox_spacing = np.linalg.norm(R, axis=0)
    U, _, Vt = np.linalg.svd(R / vox_spacing)
    affine[:3, :3] = (U @ Vt) * vox_spacing

    nib.save(nib.Nifti1Image(lbl[slices].astype(lbl.dtype), affine), label_path)
    nib.save(nib.Nifti1Image(img[slices].astype(img.dtype), affine), image_path)


def process_single_image(args):
    """Process a single image and its corresponding label"""
    img_dict, counter, path_out_images, path_out_labels, taskname, pad_lo, pad_hi = args

    image_file_nnunet = os.path.join(path_out_images, f'{taskname}_{counter:03d}_0000.nii.gz')
    label_file_nnunet = os.path.join(path_out_labels, f'{taskname}_{counter:03d}.nii.gz')

    # Sentinel file written only after all steps complete — safe for resumption
    done_file = image_file_nnunet + '.done'
    if os.path.exists(done_file):
        return {
            'image': str(os.path.abspath(img_dict['image'])),
            'label': str(os.path.abspath(img_dict['label'])),
            'image_nnunet': image_file_nnunet,
            'label_nnunet': label_file_nnunet
        }

    # Remove any partial output from a previous interrupted run
    os.system(f"rm -f {image_file_nnunet} {label_file_nnunet}")

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

    # Crop image and label to SC bbox with per-face padding
    crop_to_sc_bbox(image_file_nnunet, label_file_nnunet, pad_lo, pad_hi)

    # Mark as fully processed
    open(done_file, 'w').close()

    return {
        'image': str(os.path.abspath(img_dict['image'])),
        'label': str(os.path.abspath(img_dict['label'])),
        'image_nnunet': image_file_nnunet,
        'label_nnunet': label_file_nnunet
    }


def process_dataset_parallel(data_list, path_out_images, path_out_labels, taskname, start_counter, num_workers, pad_lo, pad_hi):
    """Process a dataset in parallel using multiple workers"""
    with Pool(processes=num_workers) as pool:
        work_items = [
            (item, start_counter + i, path_out_images, path_out_labels, taskname, pad_lo, pad_hi)
            for i, item in enumerate(data_list)
        ]
        results = list(tqdm.tqdm(
            pool.imap(process_single_image, work_items),
            total=len(work_items),
            desc="Processing images"
        ))
    return results


def load_json_datalist(datalist_path, key_to_extract):
    """Load a json datalist file and extract the specified key"""
    with open(datalist_path, 'r') as f:
        datalist = json.load(f)
    return datalist[key_to_extract]


def main():
    args = parse_args()
    if args.workers is None:
        args.workers = cpu_count()

    pad_lo = [args.pad_left, args.pad_anterior, args.pad_superior]
    pad_hi = [args.pad_right, args.pad_posterior, args.pad_inferior]

    path_out = Path(os.path.join(args.output, f'Dataset{args.tasknumber}_{args.taskname}'))
    path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
    path_out_imagesTs = Path(os.path.join(path_out, 'imagesTs'))
    path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
    path_out_labelsTs = Path(os.path.join(path_out, 'labelsTs'))

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
    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of validation samples: {len(val_data)}")
    print(f"Number of testing samples: {len(test_data)}")
    print(f"SC bbox padding (mm) — L:{args.pad_left} R:{args.pad_right} A:{args.pad_anterior} P:{args.pad_posterior} S:{args.pad_superior} I:{args.pad_inferior}")

    print("Processing training data...")
    train_results = process_dataset_parallel(
        train_data + val_data, path_out_imagesTr, path_out_labelsTr,
        args.taskname, 1, args.workers, pad_lo, pad_hi
    )

    print("Processing test data...")
    test_results = process_dataset_parallel(
        test_data, path_out_imagesTs, path_out_labelsTs,
        args.taskname, 1, args.workers, pad_lo, pad_hi
    )

    conversion_dict = {}
    for result in train_results + test_results:
        conversion_dict[result['image']] = result['image_nnunet']
        conversion_dict[result['label']] = result['label_nnunet']

    with open(os.path.join(path_out, "conversion_dict.json"), "w") as f:
        json.dump(conversion_dict, f, indent=4)

    json_dict = OrderedDict({
        'name': args.taskname,
        'description': args.taskname,
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
        'training': [{'image': str(r['image_nnunet']), 'label': str(r['label_nnunet'])} for r in train_results],
        'test': [{'image': str(r['image_nnunet']), 'label': str(r['label_nnunet'])} for r in test_results]
    })

    with open(os.path.join(path_out, "dataset.json"), "w") as f:
        json.dump(json_dict, f, indent=4)

    print("Conversion completed successfully!")


if __name__ == '__main__':
    main()
