"""
Create train/val/test datasplit YAML files for all datasets, assuming files are already on disk.

Identical to nnUnet/02_create_msd_data.py but skips git-annex get calls.
Use this when all NIfTI files are already downloaded locally.

Output: datasplits/datasplit_{dataset}_seed{seed}.yaml written relative to CWD.
Run from the repo root or a directory containing a datasplits/ subfolder.

Usage:
    cd /home/quentinr/contrast-agnostic-softseg-spinalcord
    python scripts_for_training_with_cropping/02_create_datasplits.py \
        --path-data /home/quentinr/data/<dataset> \
        --path-out /tmp/out \
        --seed 50 \
        [--include subjects_to_include.yml]
"""

import os
import re
import json
import glob
import yaml
import numpy as np
from tqdm import tqdm
import argparse
from loguru import logger
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from datetime import datetime

import sys
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', 'nnUnet')))
from utils import get_git_branch_and_commit

import pandas as pd
pd.set_option('display.max_colwidth', None)


FILESEG_SUFFIXES = {
    "basel-mp2rage": ["labels", "label-SC_seg"],
    "canproco": ["labels", "seg-manual"],
    "data-multi-subject": ["labels_softseg_bin", "desc-softseg_label-SC_seg"],
    "dcm-brno": ["labels", "seg"],
    "dcm-zurich": ["labels", "label-SC_seg"],
    "dcm-zurich-lesions": ["labels", "label-SC_mask-manual"],
    "dcm-zurich-lesions-20231115": ["labels", "label-SC_mask-manual"],
    "lumbar-epfl": ["labels", "seg-manual"],
    "lumbar-vanderbilt": ["labels", "label-SC_seg"],
    "sci-colorado": ["labels", "seg-manual"],
    "sci-paris": ["labels", "seg-manual"],
    "sci-zurich": ["labels", "seg-manual"],
    "sct-testing-large": ["labels", "seg-manual"],
    "site_006": ["labels", "label-SC_seg"],
    "site_007": ["labels", "label-SC_seg"],
}

PATHOLOGIES = ["ALS", "DCM", "NMO", "MS", "SYR", "SCI"]


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--path-data', required=True, type=str)
    parser.add_argument('--path-out', required=True, type=str)
    parser.add_argument('--include', type=str)
    parser.add_argument('--seed', default=42, type=int)
    return parser


def fetch_subject_nifti_details(filename_path):
    subject = re.search('sub-(.*?)[_/]', filename_path)
    subjectID = subject.group(0)[:-1] if subject else ""

    session = re.search('ses-(.*?)[_/]', filename_path)
    sessionID = session.group(0)[:-1] if session else ""

    orientation = re.search('acq-(.*?)[_/]', filename_path)
    orientationID = orientation.group(0)[:-1] if orientation else ""

    if 'data-multi-subject' in filename_path:
        contrast_pattern = r'.*_(space-other_T1w|space-other_T2w|space-other_T2star|flip-1_mt-on_space-other_MTS|flip-2_mt-off_space-other_MTS|rec-average_dwi).*'
    else:
        contrast_pattern = r'.*_(T1w|T2w|acq-sagthor_T2w|acq-sagcerv_T2w|acq-sagstir_T2w|acq-ax_T2w|T2star|PSIR|STIR|UNIT1|acq-MTon_MTR|acq-dwiMean_dwi|acq-T1w_MTR).*'
    contrast = re.search(contrast_pattern, filename_path)
    contrastID = contrast.group(1) if contrast else ""

    return subjectID, sessionID, orientationID, contrastID


def create_df(args, dataset_path):
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    labels_folder = FILESEG_SUFFIXES[dataset_name][0]
    labels_suffix = FILESEG_SUFFIXES[dataset_name][1]

    if dataset_name == 'sct-testing-large':
        path_files = os.path.join(dataset_path, 'derivatives', labels_folder, 'sub-*', '**', f'*_{labels_suffix}.nii.gz')
        df_participants = pd.read_csv(os.path.join(dataset_path, 'participants.tsv'), sep='\t')
        sct_testing_large_patho_subjects = []
        for pathology in PATHOLOGIES:
            subs = df_participants[df_participants['pathology'] == pathology]['participant_id'].tolist()
            sct_testing_large_patho_subjects.extend(subs)
        derivatives_subs = os.listdir(os.path.join(dataset_path, 'derivatives', labels_folder))
        sct_testing_large_patho_subjects = [sub for sub in sct_testing_large_patho_subjects if sub in derivatives_subs]
    elif dataset_name == 'canproco':
        path_files = os.path.join(dataset_path, 'derivatives', labels_folder, 'sub-*', 'ses-M0', '**', f'*_{labels_suffix}.nii.gz')
    else:
        path_files = os.path.join(dataset_path, 'derivatives', labels_folder, 'sub-*', '**', f'*_{labels_suffix}.nii.gz')

    fname_files = glob.glob(path_files, recursive=True)
    if len(fname_files) == 0:
        logger.info(f"No image/label files found in {dataset_path}")
        return None

    df = pd.DataFrame({'filename': fname_files})
    df['datasetName'] = dataset_name
    df['subjectID'], df['sessionID'], df['orientationID'], df['contrastID'] = zip(*df['filename'].map(fetch_subject_nifti_details))

    if dataset_name == 'basel-mp2rage':
        df['pathologyID'] = 'n/a'
        for subject in df['subjectID'].unique():
            df.loc[df['subjectID'] == subject, 'pathologyID'] = 'HC' if subject.startswith('sub-C') else 'MS'

    elif dataset_name == 'sct-testing-large':
        df_participants = pd.read_csv(os.path.join(dataset_path, 'participants.tsv'), sep='\t')
        df = df[~df['contrastID'].str.len().eq(0)]
        df = df[~df['contrastID'].str.contains('acq-dwiMean_dwi')]
        df = df[df['subjectID'].isin(sct_testing_large_patho_subjects)]
        df = df[~df['subjectID'].str.contains('sub-xuanwuChenxi002')]
        with open(args.include, 'r') as file:
            files_to_include = yaml.safe_load(file)[dataset_name]
            files_to_include = [os.path.basename(f) for f in files_to_include]
        df['fname_temp'] = df['filename'].apply(os.path.basename)
        df = df[df['fname_temp'].isin(files_to_include)]
        df.drop(columns=['fname_temp'], inplace=True)
        df = pd.merge(df, df_participants[['participant_id', 'pathology']], left_on='subjectID', right_on='participant_id', how='left')
        df.rename(columns={'pathology': 'pathologyID'}, inplace=True)

    elif dataset_name == 'canproco':
        exclude_subs_canproco = ['sub-cal088', 'sub-cal209', 'sub-cal161', 'sub-mon006', 'sub-mon009', 'sub-mon032', 'sub-mon097',
                                 'sub-mon113', 'sub-mon118', 'sub-mon148', 'sub-mon152', 'sub-mon168', 'sub-mon191', 'sub-van134',
                                 'sub-van135', 'sub-van171', 'sub-van176', 'sub-van181', 'sub-van201', 'sub-van206', 'sub-van207',
                                 'sub-tor014', 'sub-tor133', 'sub-cal149']
        df = df[~df['subjectID'].isin(exclude_subs_canproco)]
        df_participants = pd.read_csv(os.path.join(dataset_path, 'participants.tsv'), sep='\t')
        df = pd.merge(df, df_participants[['participant_id', 'phenotype_M0']], left_on='subjectID', right_on='participant_id', how='left')
        df['phenotype_M0'].fillna('HC', inplace=True)
        df.rename(columns={'phenotype_M0': 'pathologyID'}, inplace=True)

    elif dataset_name in ['site_006', 'site_007']:
        with open(args.include, 'r') as file:
            files_to_include = yaml.safe_load(file)[dataset_name]
            files_to_include = [os.path.basename(f) for f in files_to_include]
            files_to_include = [f.replace('.nii.gz', f'_{labels_suffix}.nii.gz') for f in files_to_include]
        df['fname_temp'] = df['filename'].apply(os.path.basename)
        df_filtered = df[df['fname_temp'].isin(files_to_include)].drop(columns=['fname_temp'])
        if len(df_filtered) == 0:
            logger.warning(f"{dataset_name}: subjects_to_include.yml IDs don't match files on disk — using all {len(df)} subjects found.")
            df.drop(columns=['fname_temp'], inplace=True)
        else:
            df = df_filtered
        df['pathologyID'] = 'AcuteSCI'

    else:
        df_participants = pd.read_csv(os.path.join(dataset_path, 'participants.tsv'), sep='\t')
        if 'pathology' in df_participants.columns:
            df = pd.merge(df, df_participants[['participant_id', 'pathology']], left_on='subjectID', right_on='participant_id', how='left')
            df.rename(columns={'pathology': 'pathologyID'}, inplace=True)
        elif 'sci' in dataset_name:
            df['pathologyID'] = 'SCI'
        elif dataset_name == 'lumbar-epfl':
            df['pathologyID'] = 'HC'
        else:
            df['pathologyID'] = 'n/a'

    df = df[['datasetName', 'subjectID', 'sessionID', 'orientationID', 'contrastID', 'pathologyID', 'filename']]
    return df


def get_boilerplate_json(dataset, dataset_commits):
    params = OrderedDict()
    params["description"] = "Datasets for contrast-agnostic spinal cord segmentation"
    params["labels"] = {"0": "background", "1": "sc-seg"}
    params["license"] = "MIT"
    params["modality"] = {"0": "MRI"}
    params["dataset"] = dataset
    params["reference"] = "BIDS: Brain Imaging Data Structure"
    params["tensorImageSize"] = "3D"
    params["datasetVersions"] = dataset_commits
    if dataset == 'data-multi-subject':
        params["subjectType"] = "HC"
    elif dataset == 'sct-testing-large':
        params["subjectType"] = PATHOLOGIES
    return params


def main():
    args = get_parser().parse_args()
    np.random.seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    os.makedirs(os.path.join(args.path_out, "logs"), exist_ok=True)
    logger.add(os.path.join(args.path_out, "logs", f"log_{os.path.basename(args.path_data)}_seed{args.seed}_{timestamp}.txt"))

    dataset_name = os.path.basename(os.path.normpath(args.path_data))
    logger.info(f"Processing {dataset_name} ...")

    df = create_df(args, args.path_data)

    branch, commit = get_git_branch_and_commit(args.path_data)
    dataset_commits = {dataset_name: f"git-{branch}-{commit}"}

    train_ratio, val_ratio, test_ratio = (0.65, 0.15, 0.2) if dataset_name == 'data-multi-subject' else (0.8, 0.1, 0.1)

    all_subjects = df['subjectID'].unique()
    train_subjects, test_subjects = train_test_split(all_subjects, test_size=test_ratio)
    train_subjects, val_subjects = train_test_split(train_subjects, test_size=val_ratio / (train_ratio + val_ratio))
    train_subjects, val_subjects, test_subjects = sorted(train_subjects), sorted(val_subjects), sorted(test_subjects)

    df['split'] = 'none'
    df.loc[df['subjectID'].isin(train_subjects), 'split'] = 'train'
    df.loc[df['subjectID'].isin(val_subjects), 'split'] = 'validation'
    df.loc[df['subjectID'].isin(test_subjects), 'split'] = 'test'

    params = get_boilerplate_json(dataset_name, dataset_commits)
    labels_folder = FILESEG_SUFFIXES[dataset_name][0]
    labels_suffix = FILESEG_SUFFIXES[dataset_name][1]

    subjects_to_remove = []
    all_subjects_list = [{"train": train_subjects}, {"validation": val_subjects}, {"test": test_subjects}]

    for subjects_dict in tqdm(all_subjects_list, desc="Building datalist"):
        for name, subs_list in subjects_dict.items():
            temp_list = []
            for subject in subs_list:
                num_files = len(df[df['subjectID'] == subject])
                for idx in range(num_files):
                    fname_label = df[df['subjectID'] == subject]['filename'].values[idx]
                    fname_image = fname_label.replace(f'/derivatives/{labels_folder}', '').replace(f'_{labels_suffix}.nii.gz', '.nii.gz')

                    if os.path.exists(fname_image) and os.path.exists(fname_label):
                        temp_list.append({"image": fname_image, "label": fname_label})
                    else:
                        subjects_to_remove.append(subject)
            params[name] = temp_list

    params["contrasts"] = df['contrastID'].unique().tolist()
    params["numTrainingImagesTotal"] = len(params["train"])
    params["numValidationImagesTotal"] = len(params["validation"])
    params["numTestImagesTotal"] = len(params["test"])
    params["seed"] = args.seed

    train_subs_all = sorted(set(train_subjects) - set(subjects_to_remove))
    val_subs_all = sorted(set(val_subjects) - set(subjects_to_remove))
    test_subs_all = sorted(set(test_subjects) - set(subjects_to_remove))
    params["numTrainingSubjects"] = len(train_subs_all)
    params["numValidationSubjects"] = len(val_subs_all)
    params["numTestSubjects"] = len(test_subs_all)

    logger.info(f"train={params['numTrainingImagesTotal']} val={params['numValidationImagesTotal']} test={params['numTestImagesTotal']} images")

    df = df[~df['subjectID'].isin(subjects_to_remove)]

    params["numImagesPerContrast"] = {"train": {}, "validation": {}, "test": {}}
    for contrast in params["contrasts"]:
        params["numImagesPerContrast"]["train"][contrast] = len(df[df['subjectID'].isin(train_subs_all) & (df['contrastID'] == contrast)])
        params["numImagesPerContrast"]["validation"][contrast] = len(df[df['subjectID'].isin(val_subs_all) & (df['contrastID'] == contrast)])
        params["numImagesPerContrast"]["test"][contrast] = len(df[df['subjectID'].isin(test_subs_all) & (df['contrastID'] == contrast)])

    with open(f"datasplits/datasplit_{dataset_name}_seed{args.seed}.yaml", 'w') as file:
        yaml.dump({'train': train_subs_all, 'val': val_subs_all, 'test': test_subs_all}, file, indent=2, sort_keys=True)

    df = df[['datasetName', 'subjectID', 'sessionID', 'orientationID', 'contrastID', 'pathologyID', 'split', 'filename']]
    df = df.sort_values(by=['subjectID'])
    for f in df['filename']:
        df['filename'] = df['filename'].replace(f, os.path.basename(f))
    df.to_csv(os.path.join(args.path_out, f"df_{dataset_name}_seed{args.seed}.csv"), index=False)

    os.makedirs(args.path_out, exist_ok=True)
    with open(os.path.join(args.path_out, f"datasplit_{dataset_name}_seed{args.seed}.json"), 'w') as f:
        f.write(json.dumps(params, indent=4, sort_keys=True))


if __name__ == "__main__":
    main()
