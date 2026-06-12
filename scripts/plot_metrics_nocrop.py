"""
Generate Dice violin plots and statistical summary from the NO-CROP metrics.json.

Twin of plot_metrics.py for the no-crop ablation report (generate_qc_report_nocrop.py),
whose metrics.json has keys dice_nocrop + dice_v3 (and no crop QC).

Produces:
  - dice_by_contrast.png  — Dice per contrast, v3 vs no-crop
  - dice_by_dataset.png   — Dice per dataset, v3 vs no-crop
  - stats.json            — Wilcoxon paired test + per-contrast/dataset mean±std

Usage:
    python scripts/plot_metrics_nocrop.py \
        --metrics /path/to/qc_results_dataset7001/metrics.json \
        --output-dir /path/to/qc_results_dataset7001/plots/

Author: Quentin Revillon
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics",    required=True, help="Path to no-crop metrics.json")
    parser.add_argument("--output-dir", required=True, help="Where to write plots and stats.json")
    return parser.parse_args()


def violin_plot(df: pd.DataFrame, group_col: str, title: str, out_path: Path) -> None:
    melted = df.melt(
        id_vars=[group_col],
        value_vars=["dice_nocrop", "dice_v3"],
        var_name="model",
        value_name="Dice",
    )
    melted["model"] = melted["model"].map({"dice_nocrop": "no-crop (nnUNet)", "dice_v3": "v3 (sct_deepseg)"})

    order = sorted(df[group_col].unique())
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 1.4), 5))
    sns.violinplot(data=melted, x=group_col, y="Dice", hue="model",
                   order=order, split=False, inner="box", ax=ax, palette=["#4CAF50", "#FF9800"])
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def group_stats(df: pd.DataFrame, group_col: str) -> dict:
    out = {}
    for group, gdf in df.groupby(group_col):
        out[group] = {
            "n": len(gdf),
            "dice_nocrop": {"mean": round(float(gdf["dice_nocrop"].mean()), 4),
                            "std":  round(float(gdf["dice_nocrop"].std()),  4)},
            "dice_v3":     {"mean": round(float(gdf["dice_v3"].mean()), 4),
                            "std":  round(float(gdf["dice_v3"].std()),  4)},
        }
    return out


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads(Path(args.metrics).read_text())
    df      = pd.DataFrame(metrics["subjects"])

    # Wilcoxon paired test (no-crop vs v3)
    stat, p = wilcoxon(df["dice_nocrop"], df["dice_v3"])

    stats = {
        "wilcoxon": {"statistic": float(stat), "p_value": float(p)},
        "by_contrast": group_stats(df, "contrast"),
        "by_dataset":  group_stats(df, "dataset"),
    }

    (output_dir / "stats.json").write_text(json.dumps(stats, indent=4))
    print(f"Wilcoxon no-crop vs v3: stat={stat:.1f}, p={p:.3e}")

    violin_plot(df, "contrast", "Dice per contrast — no-crop vs v3", output_dir / "dice_by_contrast.png")
    violin_plot(df, "dataset",  "Dice per dataset — no-crop vs v3",  output_dir / "dice_by_dataset.png")

    print(f"Done. Results in {output_dir}")


if __name__ == "__main__":
    main()
