"""
Check how many subjects in each datasplit YAML are still present on disk.

For each datasplit file, and for each split (train/val/test), computes the
percentage of listed subjects that cannot be found in the corresponding
dataset directory. Plots a grouped bar chart per dataset.

Usage:
    python scripts_for_training_with_cropping/check_datasplit_coverage.py
    python scripts_for_training_with_cropping/check_datasplit_coverage.py \
        --datasplits-dir datasplits \
        --data-root /home/quentinr/data \
        --output datasplit_coverage.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

SPLITS = ["train", "val", "test"]
SPLIT_COLORS = {"train": "steelblue", "val": "darkorange", "test": "seagreen"}

DATASPLITS_DIR = Path(__file__).parent.parent / "datasplits"
DATA_ROOT = Path("/home/quentinr/data")


def dataset_name_from_yaml(yaml_path: Path) -> str:
    """Extract dataset name from filename: datasplit_{name}_seed50.yaml → {name}."""
    name = yaml_path.stem  # datasplit_{name}_seed50
    name = name.removeprefix("datasplit_")
    name = name.removesuffix("_seed50")
    return name


def subjects_on_disk(dataset_dir: Path) -> set:
    """Return set of subject folder names present in dataset_dir.
    For site_007, translates sub-van* → sub-007* to match YAML naming convention."""
    if not dataset_dir.exists():
        return set()
    subjects = {p.name for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("sub-")}
    if dataset_dir.name == "site_007":
        subjects = {s.replace("sub-van", "sub-007") for s in subjects}
    return subjects


def missing_pct(subjects_in_yaml: list, subjects_on_disk: set) -> float:
    if not subjects_in_yaml:
        return 0.0
    missing = sum(1 for s in subjects_in_yaml if s not in subjects_on_disk)
    return 100.0 * missing / len(subjects_in_yaml)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasplits-dir", type=Path, default=DATASPLITS_DIR)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    yaml_files = sorted(args.datasplits_dir.glob("datasplit_*_seed50.yaml"))
    assert yaml_files, f"No datasplit YAML files found in {args.datasplits_dir}"

    results = {}
    for yaml_path in yaml_files:
        dataset = dataset_name_from_yaml(yaml_path)
        split_data = yaml.safe_load(yaml_path.read_text())
        on_disk = subjects_on_disk(args.data_root / dataset)

        pcts = {}
        for split in SPLITS:
            subjects = split_data.get(split, [])
            pcts[split] = missing_pct(subjects, on_disk)
            n_total = len(subjects)
            n_missing = round(pcts[split] * n_total / 100)
            print(f"{dataset:<35} {split:<5}  {n_missing:>3}/{n_total:<3}  missing  ({pcts[split]:.1f}%)")

        results[dataset] = pcts

    # --- plot ---
    datasets = list(results.keys())
    n = len(datasets)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, n * 0.9), 6))
    for i, split in enumerate(SPLITS):
        vals = [results[d][split] for d in datasets]
        ax.bar(x + i * width, vals, width, label=split, color=SPLIT_COLORS[split], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("% subjects missing from disk")
    ax.set_ylim(0, 105)
    ax.set_title("Datasplit coverage — % of subjects in YAML not found on disk")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output = args.output or args.datasplits_dir.parent / "scripts_for_training_with_cropping" / "datasplit_coverage.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {output}")


if __name__ == "__main__":
    main()
