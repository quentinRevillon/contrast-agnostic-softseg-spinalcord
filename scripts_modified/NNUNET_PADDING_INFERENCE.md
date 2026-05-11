# nnUNet — Padding et pipeline d'inférence sur image croppée

## Contexte

La pipeline de segmentation de la moelle est :
1. YOLO détecte la moelle et crop l'image originale
2. L'image croppée est donnée à nnUNet pour segmentation
3. nnUNet gère le fait que cette image peut être plus petite que son patch_size

Question : quand l'image croppée est plus petite que `patch_size`, où est-elle placée dans le patch ? Est-elle centrée ? Collée en haut à gauche ?

---

## À l'entraînement — le data loader

**Source :** `nnunetv2/training/dataloading/data_loader_3d.py` + `base_data_loader.py`

Le dataloader charge l'image préprocessée complète (toujours plus grande que patch_size en général), puis tire **aléatoirement** une position de patch via `get_bbox()`.

Si l'image est plus petite que `patch_size` selon un axe (cas d'une image croppée très tight) :

```python
# base_data_loader.py
if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
    need_to_pad[d] = self.patch_size[d] - data_shape[d]

lbs = [- need_to_pad[i] // 2 for i in range(dim)]
ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]
```

Exemple concret avec `shape=20`, `patch_size=28` :
- `need_to_pad = 8`, `lbs = -4`, `ubs = -4`
- `bbox_lbs` est contraint à `-4` (une seule valeur possible)
- Le padding résultant est `(4, 4)` → **image centrée**

Le padding lui-même est calculé comme :
```python
padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
data_all[j] = np.pad(data, padding, 'constant', constant_values=0)  # zéros pour l'image
seg_all[j]  = np.pad(seg,  padding, 'constant', constant_values=-1) # -1 (ignore) pour la GT
```

---

## À l'inférence — le sliding window predictor

**Source :** `nnunetv2/inference/predict_from_raw_data.py` + `acvl_utils.cropping_and_padding.padding.pad_nd_image`

```python
data, slicer_revert_padding = pad_nd_image(
    input_image,
    self.configuration_manager.patch_size,
    'constant', {'value': 0},
    return_slicer=True,
    shape_must_be_divisible_by=None
)
```

Le docstring de `pad_nd_image` est explicite :

> *"Padding is done such that the original content will be at the center of the padded image.
> If the amount of padding needed is odd, the padding 'above' the content is larger."*

```python
pad_below = difference // 2
pad_above = difference // 2 + difference % 2
```

→ **L'image croppée par YOLO est centrée dans le patch, avec des zéros symétriques de chaque côté.**

Exemple : image de `[67, 192, 192]` voxels, patch_size `[28, 256, 256]` :
- Axe D : `67 > 28` → pas de padding, sliding window possible
- Axe H : `256 - 192 = 64` → `pad_below=32`, `pad_above=32`
- Axe W : idem

Après padding et prédiction sur la fenêtre, le padding est retiré via `slicer_revert_padding` pour retrouver la taille originale.

---

## Pipeline d'inférence complète

```
Image originale (e.g. [400, 800, 800] à 1mm isotrope)
    │
    ▼ YOLO crop
Image croppée autour de la moelle (e.g. [120, 80, 80] voxels à 1mm)
    │
    ▼ resample nnUNet → espacement cible [0.8, 0.39, 0.39] mm
Image resampleée (e.g. [150, 205, 205] voxels)
    │
    ▼ pad_nd_image centré avec zéros si < patch_size
Image paddée à [150, 256, 256] (zéros sur les côtés H et W)
    │
    ▼ sliding window (patch_size = [28, 256, 256])
    → plusieurs positions sur l'axe D, une seule sur H et W
    │
    ▼ retrait du padding (slicer_revert_padding)
    ▼ resample vers l'espacement original
Masque de segmentation à la taille originale
```

---

## Implication pour l'augmentation `YOLOCropSimulationTransform`

L'augmentation créée simule ce comportement en mettant à zéro tout ce qui est en dehors d'une bbox GT + padding aléatoire. La différence avec l'inférence réelle :

| Inférence | Augmentation (entraînement) |
|-----------|----------------------------|
| Image YOLO-croppée **centrée** dans le patch | Patch tiré aléatoirement dans l'image complète, contenu **non nécessairement centré** |
| Zéros autour du contenu utile | Zéros autour de la bbox GT + padding |

Cette différence est acceptable : du point de vue du réseau, les deux cas produisent la même situation — une région de signal utile entourée de zéros, à position variable dans le patch. La simulation est valide pour entraîner la robustesse à des crops YOLO imprécis.

---

## Fichiers sources

- `nnunetv2/training/dataloading/base_data_loader.py` — `get_bbox()`, `need_to_pad`, padding constant
- `nnunetv2/training/dataloading/data_loader_3d.py` — `generate_train_batch()`, extraction + padding du patch
- `nnunetv2/inference/predict_from_raw_data.py` — appel à `pad_nd_image` avant sliding window
- `acvl_utils/cropping_and_padding/padding.py` — `pad_nd_image()`, centrage symétrique
