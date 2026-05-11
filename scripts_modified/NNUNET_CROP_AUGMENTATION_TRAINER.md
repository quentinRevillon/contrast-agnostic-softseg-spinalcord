# nnUNetTrainer_CropAugmentation

## Localisation du fichier

```
/home/quentinr/.conda/envs/contrast_agnostic/lib/python3.9/site-packages/nnunetv2/
  training/nnUNetTrainer/variants/data_augmentation/
    nnUNetTrainer_CropAugmentation.py
```

## Motivation

La pipeline de segmentation de la moelle est composée de deux modèles en série :

```
Image IRM → YOLO (détection) → crop → nnUNet (segmentation)
```

Si le crop YOLO est imprécis (trop serré, trop large, décalé), le modèle de segmentation voit une image différente de ce qu'il a vu à l'entraînement (images complètes). Ce trainer simule ces crops imprécis pendant l'entraînement pour rendre le modèle robuste.

## Utilisation

```bash
nnUNetv2_train <DATASET_ID> 3d_fullres <FOLD> -tr nnUNetTrainer_CropAugmentation
```

Dans `scripts_modified/train_contrast_agnostic.sh` :
```bash
NNUNET_TRAINER="nnUNetTrainer_CropAugmentation"
```

## Architecture

Deux classes dans le même fichier :

### `YOLOCropSimulationTransform(BasicTransform)`

Transform batchgeneratorsv2 qui opère sur un patch de taille fixe `(C, D, H, W)`.

**Étapes :**

1. **Calcul de la bbox GT** — trouve les coordonnées min/max du foreground (`seg[0] > 0`) sur les 3 axes

2. **Tirage du padding par face** — 6 faces indépendantes (D_lo, D_hi, H_lo, H_hi, W_lo, W_hi), chacune tirée selon une mixture 50/50 :
   - 50 % : `U[0, 10mm]` → petit offset, crop serré
   - 50 % : `U[10mm, max_available]` → crop lâche

   Le max disponible par face est la distance en voxels entre le bord de la bbox GT et le bord du patch.

   Conversion mm → voxels via `self.spacing` (lu depuis `configuration_manager.spacing`).

3. **Masque de crop** — tensor booléen `(D, H, W)` : `True` dans la région bbox+padding, `False` ailleurs

4. **Application** — les voxels à `False` sont mis à **0** dans l'image et dans la segmentation

**Si le patch ne contient pas de foreground** (cas rare) : le transform est un no-op (retourne l'image intacte).

### `nnUNetTrainer_CropAugmentation(nnUNetTrainer)`

Override de `get_training_transforms` en méthode d'instance (pas `@staticmethod`) pour accéder à `self.configuration_manager.spacing`.

**Pipeline résultant :**
```
RandomTransform(YOLOCropSimulationTransform, p=0.5)   ← ajouté en tête
→ SpatialTransform (rotation, scaling)
→ GaussianNoise
→ GaussianBlur
→ MultiplicativeBrightness
→ Contrast
→ SimulateLowResolution
→ Gamma (×2)
→ MirrorTransform
→ DownsampleSegForDSTransform
```

Le crop est appliqué **avant** les augmentations spatiales, ce qui est cohérent : on simule d'abord ce que YOLO donne (image avec contexte variable autour de la moelle), puis les augmentations classiques sont appliquées par-dessus.

## Paramètres clés

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `p_per_sample` | 0.5 | Probabilité d'appliquer le crop (via `RandomTransform`) |
| Seuil petit/grand crop | 10 mm | Frontière de la mixture bimodale |
| Spacing (dataset moelle) | `[0.8, 0.39, 0.39]` mm | Lu automatiquement depuis `configuration_manager` |
| Patch size | `[28, 256, 256]` | Taille fixe après transpose `[2, 0, 1]` (axe S-I en premier) |

## Correspondance avec le seuil 10mm

Pour le dataset moelle avec patch `[28, 256, 256]` et spacing `[0.8, 0.39, 0.39]mm` :

| Axe | Seuil 10mm en voxels | Max disponible (bord → bbox) |
|-----|---------------------|------------------------------|
| D (S-I, 0.8mm) | ~12 voxels | jusqu'à ~14 (moitié du patch) |
| H (axial, 0.39mm) | ~25 voxels | jusqu'à ~128 (moitié du patch) |
| W (axial, 0.39mm) | ~25 voxels | jusqu'à ~128 (moitié du patch) |

## Cohérence avec l'inférence réelle

À l'inférence, l'image YOLO-croppée est **centrée** dans le patch avec des zéros (`pad_nd_image` de acvl_utils). Dans l'augmentation, le patch est tiré aléatoirement dans l'image complète et les zéros sont placés selon la bbox et le padding tiré. La position du contenu dans le patch diffère légèrement, mais l'effet pour le réseau est identique : une région de signal entourée de zéros à étendue variable.

Voir `NNUNET_PADDING_INFERENCE.md` pour l'analyse complète du pipeline de padding.
