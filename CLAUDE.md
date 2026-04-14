ARCHITECTURE DU PROJET — RÈGLES POUR CLAUDE CODE
=================================================

STYLE DE CODE
- Pas de gestion d'exceptions, pas de try/catch — le code plante sur inputs invalides
- Une seule responsabilité par fonction
- Code court : favoriser les primitives et librairies existantes (glue coding)
- Fichiers courts : un script = une responsabilité, pas d'abstractions prématurées
- nibabel pour NIfTI (SCT non requis sauf usage explicite de sct_deepseg)
- Pas de résolution cible codée en dur — le code plante si les données ne sont pas au format attendu
- Toujours mettre à jour le fichier CLAUDE.md du projet pour refléter exactement l'état du projet

PRINCIPE DE SIMPLIFICATION
- Pas de helpers inutiles : si une fonction n'est utilisée qu'une fois, l'inliner
- Pas de fallback : dict explicite par dataset plutôt que logique de découverte générique
- Pas de classes : fonctions + dicts suffisent pour ce projet
- La reorientation est virtuelle (permutation d'axes via ornt_transform, aucun voxel rechargé)

ENVIRONNEMENT
- conda environment : contrast_agnostic
- toujours activer avant d'exécuter du code : conda activate contrast_agnostic
- SCT installé dans /home/quentinr/spinalcordtoolbox/bin/ (sct_deepseg, etc.)
- modèle de segmentation : /home/quentinr/spinalcordtoolbox/data/deepseg_models/model_seg_sc_contrast_agnostic_nnunet/

DONNÉES
- ~/data/ est le répertoire centralisé de toutes les données brutes BIDS, partagé entre projets
- test set gelé spine-generic (49 sujets) : scripts/spine_generic_test_split_for_csa_drift_monitoring.yaml
- images      : ~/data/data-multi-subject/<subject>/anat/<subject>_<contrast>.nii.gz
- GT masks    : ~/data/data-multi-subject/derivatives/labels_softseg_bin/<subject>/anat/
                  <subject>_<contrast>_desc-softseg_label-SC_seg.nii.gz
- contrasts disponibles par sujet : T1w, T2w, T2star, MTS (flip-1_mt-on, flip-2_mt-off)

INFÉRENCE NNUNET
- nnunetv2 doit être installé dans l'env : pip install nnunetv2
- numpy doit être <2 (numpy 2.x casse blosc2/acvl_utils) : pip install "numpy<2"
- modèle : MODEL_DIR = ~/spinalcordtoolbox/data/deepseg_models/model_seg_sc_contrast_agnostic_nnunet/nnUNetTrainer__nnUNetPlans__3d_fullres
- variables d'env nnUNet_raw/preprocessed/results requises à l'import (valeur dummy suffit)

SCRIPTS — cropping_YOLO/
  eval_crop_robustness.py        ← expérience oracle via sct_deepseg (subprocess, CPU)
  eval_crop_robustness_nnunet.py ← même expérience via nnUNetPredictor direct (GPU/CPU)
                                    predictor chargé une seule fois, --device cuda|cpu
                                    → results_crop_robustness.csv (subject, contrast, padding_mm, dice)
                                    padding -1 = pas de crop (baseline pleine image)
