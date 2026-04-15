"""
Plot Dice vs padding results from eval_crop_robustness_nnunet.py (frozen test split).

One subplot per contrast. Each point = one padding level (mean ± std across subjects).
Padding -1 (no crop / baseline) is the rightmost point, labelled "no crop" on the x-axis.

Usage:
    python cropping_YOLO/plot_crop_robustness.py results_crop_robustness.csv
    python cropping_YOLO/plot_crop_robustness.py results_crop_robustness.csv --output fig.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONTRAST_ORDER = ["T1w", "T2w", "T2star", "flip-1_mt-on_MTS", "flip-2_mt-off_MTS", "rec-average_dwi"]
CONTRAST_LABELS = {
    "T1w": "T1w",
    "T2w": "T2w",
    "T2star": "T2*",
    "flip-1_mt-on_MTS": "MTon",
    "flip-2_mt-off_MTS": "MToff",
    "rec-average_dwi": "DWI",
}


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV file from eval_crop_robustness_nnunet.py")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output figure path (.png or .pdf). Default: same stem as input.")
    return parser


def main():
    args = get_parser().parse_args()
    df = pd.read_csv(args.input)

    # Compute statistics for the suptitle
    n_subjects = df["subject"].nunique()
    n_volumes = len(df[df["padding_mm"] == -1])  # count baseline rows (one per volume)
    split_info = f"{n_subjects} subjects, {n_volumes} volumes (frozen test split)"

    contrasts = [c for c in CONTRAST_ORDER if c in df["contrast"].unique()]
    contrasts += [c for c in df["contrast"].unique() if c not in CONTRAST_ORDER]

    # Build ordered x-axis: sorted crop paddings, then baseline (-1) at the end
    crop_paddings = sorted(df[df["padding_mm"] != -1]["padding_mm"].unique())
    x_values = crop_paddings + [-1]
    x_positions = list(range(len(x_values)))
    x_labels = [f"{int(p)}mm" for p in crop_paddings] + ["no crop"]

    n_cols = 3
    n_rows = int(np.ceil(len(contrasts) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
    axes = np.array(axes).flatten()

    for ax, contrast in zip(axes, contrasts):
        sub = df[df["contrast"] == contrast]
        stats = (
            sub.groupby("padding_mm")["dice"]
            .agg(["mean", "std"])
            .reindex(x_values)
        )
        means = stats["mean"].values
        stds = stats["std"].fillna(0).values

        ax.plot(x_positions, means, marker="o", color="steelblue")
        ax.fill_between(x_positions, means - stds, means + stds, alpha=0.2, color="steelblue")

        # vertical separator before baseline
        ax.axvline(len(x_positions) - 1.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_title(CONTRAST_LABELS.get(contrast, contrast), fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_ylabel("Dice")
        ax.grid(axis="y", alpha=0.3)

    # hide unused subplots
    for ax in axes[len(contrasts):]:
        ax.set_visible(False)

    fig.text(0.5, 0.02, "Additional padding to the minimal enclosing box of GT (mm)",
             ha="center", fontsize=11)
    fig.suptitle(f"SC segmentation Dice vs crop padding (mean ± std across subjects)\n{split_info}", y=0.995)
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])

    output = args.output or args.input.with_suffix(".png")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved → {output}")


if __name__ == "__main__":
    main()
