# État de l'entraînement — Résumé des problèmes et nouvelle approche

## Contexte

Objectif : reproduire l'entraînement du modèle contrast-agnostic v3.0 (papier NeuroPoly) en ajoutant
un trainer custom `nnUNetTrainer_CropAugmentation` qui simule des crops YOLO imprécis pendant
l'entraînement, pour rendre le modèle robuste à la variabilité de la détection YOLO.

---

## Problèmes rencontrés

### 1. Problèmes de setup du script (shell non-interactif)

Le script `scripts/train_contrast_agnostic.sh` est conçu pour être lancé à la main en session
interactive. Il échoue dans un shell non-interactif (e.g. `set_slot`, `nohup`, SSH sans login) car :

- `cd` vers le repo absent → chemins relatifs cassés
- `python` non résolu → conda non activé
- `nnUNetv2_train`, `nnUNetv2_plan_and_preprocess` non trouvés → PATH conda absent
- `sct_image` non trouvé → PATH SCT absent
- Variables nnUNet (`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`) non exportées
- `CUDA_VISIBLE_DEVICES=2` hardcodé → GPU 2 inexistant (server a GPU 0 et 1 seulement)
- `export CUDA_VISIBLE_DEVICES` avant la définition de la variable → valeur vide

**Fix :** `scripts_modified/train_contrast_agnostic.sh` — toutes les variables exportées
explicitement avec chemins absolus, `CUDA_VISIBLE_DEVICES=0`.

---

### 2. Incompatibilités PyTorch / nnUNetv2 / GPU Blackwell

Le GPU du serveur (NVIDIA RTX PRO 6000 Blackwell, sm_120) requiert PyTorch ≥ 2.7 cu128.
PyTorch 2.8.0 casse nnUNetv2 2.5.2 (`PolyLRScheduler` : `_LRScheduler` renommé, arg `verbose`
supprimé). `torch.compile` crash via `BrokenProcessPool`.

**Fix :**
- PyTorch 2.8.0 + cu128 (seule version supportant sm_120)
- Patch de `polylr.py` dans le package installé
- `export TORCHDYNAMO_DISABLE=1`

Voir `ISSUES.md` pour les détails et commandes exactes.

---

### 3. Fingerprint généré sur un run de debug (18 cas seulement)

**Cause :** Le script a d'abord été lancé en mode debug avec `DATASETS=("data-multi-subject")`
et un petit split → `dataset_fingerprint.json` généré sur 18 cas spine-generic uniquement
(le 20 avril 22h34). Lors du vrai run (21 avril), nnUNet a réutilisé ce fichier sans le
recalculer (`if not isfile(properties_file)`).

**Conséquence :** Le fingerprint ne reflète qu'une seule résolution ([0.39, 0.39, 0.8] mm,
shape [640, 639, 64]) au lieu de la diversité des 15 datasets. Le plan généré est donc
complètement différent du papier :

| | Plan actuel (biaisé) | Papier v3.0 |
|---|---|---|
| Spacing cible | [0.8, 0.39, 0.39] mm | [0.9, 0.7, 1.0] mm |
| Median shape | [64, 640, 639] | [96, 320, 318] |
| Patch size | [28, 256, 256] | [64, 224, 160] |

---

### 4. Split train/val généré sur 18 cas seulement

**Cause :** `splits_final.json` créé lors du même debug run (20 avril 22h35), avant que les
2563 cas complets soient présents. nnUNet réutilise ce fichier s'il existe.

**Conséquence :** L'entraînement qui a tourné (epoch 647, trainer `nnUNetTrainer`) utilisait
**14 cas en train et 4 en validation** sur les 2563 disponibles. Les 2545 autres cas ont été
ignorés.

---

### 5. Preprocessing réalisé à la mauvaise résolution

Les 2563 cas ont été preprocessés (636 GB de `.npz`/`.npy`/`.pkl`) selon le plan biaisé
(target spacing [0.8, 0.39, 0.39] mm). Ces fichiers sont invalides pour reproduire le papier.

---

### 6. Entraînement erroné (à jeter)

L'entraînement `nnUNetTrainer__nnUNetPlans__3d_fullres` fold 0 a tourné jusqu'à l'epoch 647
avec le mauvais plan, le mauvais split, et le mauvais trainer. Le checkpoint `checkpoint_latest.pth`
et `checkpoint_best.pth` sont à rejeter.

---

## Nouvelle approche

### Étape 1 — Nettoyage

Supprimer les fichiers erronés (à confirmer avant exécution) :

```bash
# Fichiers de plan et split
rm /home/quentinr/nnunet-v2/nnUNet_preprocessed/Dataset999_TempContrastAgnostic/dataset_fingerprint.json
rm /home/quentinr/nnunet-v2/nnUNet_preprocessed/Dataset999_TempContrastAgnostic/nnUNetPlans.json
rm /home/quentinr/nnunet-v2/nnUNet_preprocessed/Dataset999_TempContrastAgnostic/splits_final.json

# Preprocessing à la mauvaise résolution (636 GB)
rm -rf /home/quentinr/nnunet-v2/nnUNet_preprocessed/Dataset999_TempContrastAgnostic/nnUNetPlans_3d_fullres/

# Résultats d'entraînement erronés
rm -rf /home/quentinr/nnunet-v2/nnUNet_results/Dataset999_TempContrastAgnostic/nnUNetTrainer__nnUNetPlans__3d_fullres/
```

### Étape 2 — Re-fingerprint + re-plan + re-preprocessing

Via `set_slot` (job long, plusieurs heures) :

```bash
set_slot 0 /bin/bash -c "
export nnUNet_raw=/home/quentinr/nnunet-v2/nnUNet_raw
export nnUNet_preprocessed=/home/quentinr/nnunet-v2/nnUNet_preprocessed
export nnUNet_results=/home/quentinr/nnunet-v2/nnUNet_results
export PATH=/home/quentinr/.conda/envs/contrast_agnostic/bin:$PATH
export TORCHDYNAMO_DISABLE=1
nnUNetv2_plan_and_preprocess -d 999 -c 3d_fullres --verify_dataset_integrity
"
```

Cette commande va :
1. Recalculer le fingerprint sur **tous les 2563 cas** (raw, déjà corrects)
2. Générer un nouveau `nnUNetPlans.json` basé sur la vraie distribution des datasets
3. Re-preprocesser tous les cas au nouvel espacement cible

Le plan résultant devrait se rapprocher du papier ([64, 224, 160], [0.9, 0.7, 1.0] mm).
Si le plan diffère encore, on pourra forcer le spacing via `--overwrite_target_spacing`.

### Étape 3 — Entraînement avec le bon trainer

Une fois le preprocessing terminé, lancer via `set_slot` :

```bash
set_slot 1 /bin/bash -c "
export nnUNet_raw=/home/quentinr/nnunet-v2/nnUNet_raw
export nnUNet_preprocessed=/home/quentinr/nnunet-v2/nnUNet_preprocessed
export nnUNet_results=/home/quentinr/nnunet-v2/nnUNet_results
export PATH=/home/quentinr/.conda/envs/contrast_agnostic/bin:$PATH
export TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
nnUNetv2_train 999 3d_fullres 0 -tr nnUNetTrainer_CropAugmentation -p nnUNetPlans
"
```

Ou utiliser `scripts_modified/train_contrast_agnostic.sh` (déjà configuré avec `NNUNET_TRAINER="nnUNetTrainer_CropAugmentation"`).

---

## Ce qui est correct et ne change pas

- `nnUNet_raw/` (2563 cas convertis depuis BIDS) — **intact, ne pas toucher**
- `nnUNetTrainer_CropAugmentation.py` — trainer custom, **prêt à l'emploi**
- `scripts_modified/train_contrast_agnostic.sh` — script corrigé pour shell non-interactif
- Patch `polylr.py` — **déjà appliqué**

---

## Fichiers de référence

| Fichier | Contenu |
|---|---|
| `ISSUES.md` | Tous les bugs du script + fixes (PyTorch, PolyLR, CUDA, etc.) |
| `NNUNET_PATCHES.md` | Comment nnUNet détermine le patch size (anisotropie, RPI, transpose) |
| `NNUNET_PADDING_INFERENCE.md` | Pipeline de padding à l'inférence (`pad_nd_image`, sliding window) |
| `NNUNET_CROP_AUGMENTATION_TRAINER.md` | Fonctionnement du trainer custom |
