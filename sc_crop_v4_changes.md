# Notes — branche `sc-crop-v4` vs `main`

## Contexte

`main` = pipeline original de Naga Karthik (segmentation contrast-agnostique v3, inférence via `sct_deepseg`).  
`sc-crop-v4` = adaptation de Quentin Revillon introduisant un **pré-traitement par détection/recadrage YOLO** (`sc-crop`) avant l'inférence nnUNet.

---

## Comment fonctionne le GitHub Action

### Qu'est-ce qu'un GitHub Action ?

Un GitHub Action = **un fichier YAML dans `.github/workflows/`**. C'est tout. GitHub lit automatiquement ce dossier au moment du push et enregistre le workflow. Il n'y a rien à installer ou activer côté GitHub — la présence du fichier suffit.

Il n'y a qu'un seul workflow dans ce repo :

```
.github/workflows/run_morphometric_analysis.yml
```

### Ce qui déclenche le workflow

```yaml
on:
  release:
    types: [published]
```

Le workflow se déclenche **uniquement quand tu publies une release** sur GitHub. Pas un push, pas un tag seul — une release *publiée* (pas draft). C'est l'action manuelle : GitHub → Releases → New Release → choisir un tag → uploader les fichiers → **Publish release**.

**Important** : le model `.zip` doit être attaché à la release **avant** de la publier, sinon le workflow échoue (guard `MODEL_URL` ajouté pour donner une erreur claire dans ce cas).

### Pourquoi un `.zip` pour le modèle ?

Un modèle nnUNet n'est pas un fichier unique. C'est un **dossier entier** dont `sc-segment-pt` a besoin des trois composantes simultanément :

```
nnUNet_results/
  Dataset7000_ContrastAgnosticScCrop/
    nnUNetTrainer__nnUNetPlans__3d_fullres/
      fold_0/
        checkpoint_best.pth   ← poids du modèle entraîné
      dataset.json            ← description des classes et canaux d'entrée
      plans.json              ← paramètres de preprocessing (spacing, patch size...)
```

GitHub ne permet pas d'uploader un dossier directement sur une release — donc on zippe tout, on uploade le `.zip`, et le workflow le dézippe au moment de l'évaluation.

Le `main` (Naga) uploadait un `.pt` seul parce qu'il utilisait `sct_deepseg` qui a son propre format monofichier. L'approche nnUNet impose le `.zip`.

### Flux complet — fichiers appelés dans l'ordre

```
[DÉCLENCHEUR] Publication d'une release avec model_*.zip attaché
       │
       ▼
[JOB 1] download_dataset                         (1 runner)
  scripts/download_spine_generic_test_data.sh
    └─ git clone spine-generic/data-multi-subject (tag r20250310)
    └─ git annex get  ×49 sujets                  (images + derivatives)
    └─ résultat mis en cache GitHub Actions
       │
       ▼  (16 runners démarrent en parallèle, chacun restaure le cache)
[JOB 2 ×16] compute_csa (batch 1 à 16, ~3 sujets chacun)
  ① pip install sc-crop v0.4.1 + nnunet-onnx  (dans venv SCT)
  ② curl → API GitHub → trouve l'URL du model_*.zip dans la release
  ③ curl -L → télécharge le model_*.zip (~220 MB)
  ④ unzip  → extrait fold_0/checkpoint_best.pth + dataset.json + plans.json

  scripts/compute_morphometrics_spine_generic.sh
    └─ ensure_model() + ensure_cls_model()     ← télécharge les poids YOLO
    └─ sct_run_batch → scripts/compute_csa.sh  (1 appel par sujet)
         └─ copy_gt_softseg_bin()              ← copie GT segmentation binarisée
         └─ copy_gt_disc_labels()              ← copie GT labels disques
         └─ label_vertebrae()
         │     └─ sct_label_vertebrae -discfile ← identifie C2-C3
         └─ sct_process_segmentation -vert 2:3  ← calcule CSA GT (mm²)
         └─ segment_sc()
         │     └─ sc-segment-pt --checkpoint    ← YOLO crop + nnUNet → segmentation prédite
         │     └─ sct_qc                        ← rapport QC HTML
         └─ sct_process_segmentation -vert 2:3  ← calcule CSA prédite (mm²)
         └─ résultat → logs_results/results/csa_c2c3.csv
  ⑤ artifact upload : csa-results-batch-N (contient logs + CSV)
       │
       ▼  (attend que les 16 batches soient finis)
[JOB 3] download_results                         (1 runner)
  ① télécharge les 16 artifacts csa-results-batch-*
  scripts/merge_run_batch_results.sh
    └─ scripts/merge_csvs.py                   ← fusionne 16 × csa_c2c3.csv → 1 fichier
  ② renomme → csa_c2c3__model_<tag>.csv
  ③ upload sur la release GitHub              ← visible dans Release assets
       │
       ▼
[JOB 4] generate_plots                           (1 runner)
  ① curl → API GitHub → liste toutes les releases du repo
  ② pour chaque release : télécharge le CSV si présent
  scripts/generate_morphometrics_plots.sh
    └─ csa_generate_figures/analyse_csa_across_releases.py
         └─ violin plots PNG (CSA par contraste, STD inter-contrastes)
         └─ comparaison entre toutes les versions de modèle
  ③ zip → morphometric_plots.zip
  ④ upload sur la release GitHub              ← visible dans Release assets
```

### Ce que tu retrouves dans la release à la fin

| Fichier | Mis là par | Contenu |
|---|---|---|
| `model_contrast_agnostic_YYYYMMDD.zip` | **toi** (upload manuel avant publish) | Poids nnUNet (fold_0, dataset.json, plans.json) |
| `csa_c2c3__model_<tag>.csv` | job `download_results` (automatique) | CSA mm² — 49 sujets × 6 contrastes × GT + prédiction |
| `morphometric_plots.zip` | job `generate_plots` (automatique) | PNG violin plots comparant toutes les releases |

### Comment voir les plots

```bash
# Depuis la release GitHub → télécharger morphometric_plots.zip, puis :
unzip morphometric_plots.zip -d plots/
ls plots/csvs_model_releases/   # fichiers PNG

# Ou regénérer manuellement depuis les CSV :
python csa_generate_figures/analyse_csa_across_releases.py -i csvs_model_releases/
```

---

## Modifications par rapport à `main`

### 1. Moteur d'inférence — `scripts/compute_csa.sh`

| | `main` | `sc-crop-v4` |
|---|---|---|
| Segmentation | `sct_deepseg spinalcord -i ... -qc ...` | `sc-segment-pt -i ... --checkpoint ... --device cpu` + `sct_qc` séparé |
| Arguments reçus de sct_run_batch | `MODEL_VERSION`, `PATH_NNUNET_SCRIPT`, `PATH_NNUNET_MODEL` | `MODEL_VERSION`, `PATH_NNUNET_MODEL` (chemin absolu vers `fold_0/checkpoint_best.pth`) |
| Labellisation vertébrale | `sct_label_utils -disc` (supprimé dans SCT récent) | `sct_label_vertebrae -discfile` (équivalent, compatible SCT actuel) |

### 2. Script de lancement batch — `scripts/compute_morphometrics_spine_generic.sh`

| | `main` | `sc-crop-v4` |
|---|---|---|
| Résolution du modèle | `sct_deepseg spinalcord -install -custom-url <url>` | `curl -L` + `unzip` → `find checkpoint_best.pth` → `realpath` |
| Format modèle attendu | `.pt` monofichier via sct_deepseg | `.zip` contenant dossier nnUNet (`fold_0/`, `dataset.json`, `plans.json`) |
| Pré-chauffe du cache sc-crop | absent | `ensure_model()` + `ensure_cls_model()` en série avant le batch parallèle (évite race conditions SHA256 lors du téléchargement concurrent des poids YOLO) |
| Passage checkpoint à sct_run_batch | `-script-args "${MODEL_VERSION}"` | `-script-args "${MODEL_VERSION} ${PATH_CHECKPOINT}"` |

### 3. Workflow CI — `.github/workflows/run_morphometric_analysis.yml`

| Modification | Raison |
|---|---|
| **Step "Free disk space"** en premier dans `compute_csa` : `sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /opt/hostedtoolcache/CodeQL` | ubuntu-latest ≈ 14 GB dispos. SCT (~5 GB) + cache dataset spine-generic (~8 GB) dépassaient la limite → crash du runner pendant "Restore cached dataset" |
| **`fail-fast: false`** dans la matrix strategy | Sans ça, si 1 batch sur 16 échoue, les 15 autres sont annulés — oblige à tout re-télécharger au prochain essai |
| **Guard `MODEL_URL`** : `if [[ -z "$MODEL_URL" ]]; then exit 1; fi` | Si la release est publiée sans model.zip, l'erreur était cryptique (curl silencieux sur URL vide) |
| **Installation sc-crop + nnunet-onnx dans le venv SCT** (`venv_sct`) | SCT fournit déjà torch + nnunetv2 ; installer dans le venv système 3.9 échouait avec "No module named torch" |
| **Step debug "Print per-subject error logs"** + tmate borné à 5 min | Affiche les logs d'erreur par sujet en cas d'échec ; tmate permet un accès SSH au runner pendant 5 min |

### 4. Pré-traitement dataset — `nnUnet/03_convert_msd_to_nnunet_reorient.py`

Ajouts par rapport à main :

- **`keep_largest_component()`** : nettoie les GT en ne gardant que la plus grande composante connexe — supprime les voxels isolés (bruit d'annotation) qui faussaient la bbox de crop
- **`force_orthonormal_affine()`** : corrige les affines non-orthonormales (acquisitions obliques, ex. sci-zurich T2w sagittal) que ITK/nnUNet rejette
- **Cropping sc-crop à la conversion** : chaque image est détectée + recadrée par YOLO avant d'être sauvegardée en format nnUNet → cohérence train/test (l'inférence fait aussi le crop)
- **Rapport QC de crop** : génère `crop_qc_report.csv` + `crop_qc_summary.json`
- Dataset ID : `999 / TempContrastAgnostic` (main) → `7000 / ContrastAgnosticScCrop` (sc-crop-v4)

### 5. Script d'entraînement — `scripts/train_contrast_agnostic.sh`

- Chemins adaptés pour la machine de Quentin (`/home/quentinr/`)
- **Contrôle des étapes** (`START_STEP` / `END_STEP`) : auto-détection de la reprise + possibilité de forcer une étape spécifique (`START_STEP=4 bash train_contrast_agnostic.sh`)
- Trainer : `nnUNetTrainer_5epochs` (debug) → `nnUNetTrainer` (production)
- `TORCHDYNAMO_DISABLE=1` ajouté pour compatibilité GPU sm_120+ (architecture Blackwell)
- `set -e` ajouté

### 6. Requirements — `nnUnet/requirements.txt`

Ajouts par rapport à main :

```
torch==2.8.0+cu128           # GPU Blackwell (sm_120+), absent de main
nnunetv2 @ git+...@503b1d2   # fix bug _LRScheduler absent de PyPI v2.5.2
sc-crop @ git+...@v0.4.1     # YOLO crop + inférence nnUNet, absent de main
```

---

## Résumé en une phrase

`sc-crop-v4` remplace `sct_deepseg` par un pipeline en deux étapes — détection YOLO (`sc-crop`) puis nnUNet sur le crop — appliqué de manière cohérente à l'entraînement ET à l'évaluation, avec un CI robuste (disk cleanup, fail-fast false, guard MODEL_URL).
