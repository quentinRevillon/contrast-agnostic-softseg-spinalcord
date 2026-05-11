# nnUNet — Comprendre les patches

## Comment nnUNet détermine le patch size

Le patch size est calculé automatiquement lors de `nnUNetv2_plan_and_preprocess` par l'`ExperimentPlanner`. Il prend en compte :

1. **La forme médiane des images** après resampling à l'espacement cible
2. **La mémoire GPU disponible** — le patch doit tenir en mémoire avec le batch size
3. **Les contraintes du réseau** — nombre d'opérations de pooling possibles

Le résultat est stocké dans :
```
/home/quentinr/nnunet-v2/nnUNet_preprocessed/Dataset999_TempContrastAgnostic/nnUNetPlans.json
```

## Patch size pour le dataset de la moelle (RPI, anisotrope)

Les images de moelle épinière sont orientées en **RPI** (Right-Posterior-Inferior) :
- La moelle court le long de l'axe **S-I (Supérieur-Inférieur)** = axe Z en RPI
- Le plan axial = axes X-Y

nnUNet détecte l'anisotropie et **transpose les axes** pour mettre le plus grossier en premier :

```json
"transpose_forward": [2, 0, 1],
"transpose_backward": [1, 2, 0],
"spacing": [0.8mm, 0.39mm, 0.39mm]
"patch_size": [28, 256, 256]
```

Ce qui donne :
| Axe | Voxels | Espacement | Étendue réelle |
|-----|--------|------------|----------------|
| S-I (long de la moelle) | 28 | 0.8 mm | ~22.4 mm |
| Axial H | 256 | 0.39 mm | ~100 mm |
| Axial W | 256 | 0.39 mm | ~100 mm |

Le patch est "plat" dans la direction S-I car la résolution y est plus basse → nnUNet alloue moins de voxels pour tenir dans la mémoire GPU.

## Format des patches dans le pipeline d'augmentation

Les patches arrivent dans le pipeline d'augmentation avec une **taille fixe** `[batch, channels, D, H, W]`. Le preprocessing (resampling, padding) se fait en amont — **l'augmentation n'a pas à gérer le redimensionnement**.

Si on veut simuler un crop YOLO dans un transform custom :
- Le tensor reste toujours à la taille `patch_size`
- On met à zéro la région extérieure au crop simulé
- Les valeurs réelles de l'image sont conservées dans la région bbox GT + padding

## Références

- Trainer de base : `/home/quentinr/.conda/envs/contrast_agnostic/lib/python3.9/site-packages/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`
- Exemple trainer custom : `/home/quentinr/.conda/envs/contrast_agnostic/lib/python3.9/site-packages/nnunetv2/training/nnUNetTrainer/variants/data_augmentation/nnUNetTrainerNoDA.py`
- Plans du dataset : `/home/quentinr/nnunet-v2/nnUNet_preprocessed/Dataset999_TempContrastAgnostic/nnUNetPlans.json`
