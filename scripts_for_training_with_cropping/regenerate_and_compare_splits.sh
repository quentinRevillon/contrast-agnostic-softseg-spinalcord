#!/bin/bash
# Regenerate all datasplits with seed 50 in datasplits_cropping/ and compare with originals in datasplits/.
# Usage: bash scripts_for_training_with_cropping/regenerate_and_compare_splits.sh
#
# Outputs a diff summary: IDENTICAL or diff per dataset.

set -o pipefail

REPO=/home/quentinr/contrast-agnostic-softseg-spinalcord
DATA_DIR=/home/quentinr/data
PYTHON=/home/quentinr/.conda/envs/contrast_agnostic/bin/python
INCLUDE=$REPO/subjects_to_include.yml
ORIGINAL_DIR=$REPO/datasplits
OUT_DIR=$REPO/datasplits_cropping

mkdir -p "$OUT_DIR/datasplits" "$OUT_DIR/out"

DATASETS=(
    "basel-mp2rage"
    "canproco"
    "data-multi-subject"
    "dcm-brno"
    "dcm-zurich"
    "dcm-zurich-lesions"
    "dcm-zurich-lesions-20231115"
    "lumbar-epfl"
    "lumbar-vanderbilt"
    "sci-colorado"
    "sci-paris"
    "sci-zurich"
    "sct-testing-large"
    "site_006"
    "site_007"
)

for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Processing: $dataset"
    echo "=========================================="
    if [ ! -d "$DATA_DIR/$dataset" ]; then
        echo "  -> SKIPPED (not on disk)"
        continue
    fi

    cd "$OUT_DIR"
    $PYTHON $REPO/scripts_for_training_with_cropping/02_create_datasplits.py \
        --seed 50 \
        --path-data "$DATA_DIR/$dataset" \
        --path-out "$OUT_DIR/out" \
        --include "$INCLUDE" || { echo "  -> FAILED (script error)"; continue; }

    GENERATED="$OUT_DIR/datasplits/datasplit_${dataset}_seed50.yaml"
    ORIGINAL="$ORIGINAL_DIR/datasplit_${dataset}_seed50.yaml"

    if [ ! -f "$ORIGINAL" ]; then
        echo "  -> No original found at $ORIGINAL"
    elif diff -q "$GENERATED" "$ORIGINAL" > /dev/null 2>&1; then
        echo "  -> IDENTICAL"
    else
        echo "  -> DIFFERS:"
        diff "$GENERATED" "$ORIGINAL" || true
    fi
    echo ""
done

echo "Done. Results in: $OUT_DIR/datasplits/"
