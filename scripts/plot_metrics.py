"""
Generate Dice violin plots and statistical summary from metrics.json.

Produces:
  - dice_by_contrast.png  — Dice per contrast, v3 vs v4
  - dice_by_dataset.png   — Dice per dataset, v3 vs v4
  - stats.json            — Wilcoxon paired test + per-contrast/dataset mean±std

Usage:
    python scripts/plot_metrics.py \
        --metrics /path/to/results/metrics.json \
        --output-dir /path/to/results/plots/

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
    parser.add_argument("--metrics",    required=True, help="Path to metrics.json")
    parser.add_argument("--output-dir", required=True, help="Where to write plots and stats.json")
    return parser.parse_args()


def violin_plot(df: pd.DataFrame, group_col: str, title: str, out_path: Path) -> None:
    melted = df.melt(
        id_vars=[group_col],
        value_vars=["dice_v4", "dice_v3"],
        var_name="model",
        value_name="Dice",
    )
    melted["model"] = melted["model"].map({"dice_v4": "v4 (ours)", "dice_v3": "v3 (sct_deepseg)"})

    order = sorted(df[group_col].unique())
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 1.4), 5))
    sns.violinplot(data=melted, x=group_col, y="Dice", hue="model",
                   order=order, split=False, inner="box", ax=ax, palette=["#2196F3", "#FF9800"])
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
            "dice_v4": {"mean": round(float(gdf["dice_v4"].mean()), 4),
                        "std":  round(float(gdf["dice_v4"].std()),  4)},
            "dice_v3": {"mean": round(float(gdf["dice_v3"].mean()), 4),
                        "std":  round(float(gdf["dice_v3"].std()),  4)},
        }
    return out


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads(Path(args.metrics).read_text())
    df      = pd.DataFrame(metrics["subjects"])

    # Wilcoxon paired test
    stat, p = wilcoxon(df["dice_v4"], df["dice_v3"])

    stats = {
        "wilcoxon": {"statistic": float(stat), "p_value": float(p)},
        "by_contrast": group_stats(df, "contrast"),
        "by_dataset":  group_stats(df, "dataset"),
        "crop_ok_split": {
            "crop_ok": {
                "n": int(df["crop_ok"].sum()),
                "dice_v4": {"mean": round(float(df[df["crop_ok"]]["dice_v4"].mean()), 4),
                            "std":  round(float(df[df["crop_ok"]]["dice_v4"].std()),  4)},
                "dice_v3": {"mean": round(float(df[df["crop_ok"]]["dice_v3"].mean()), 4),
                            "std":  round(float(df[df["crop_ok"]]["dice_v3"].std()),  4)},
            },
            "crop_failed": {
                "n": int((~df["crop_ok"]).sum()),
                "dice_v4": {"mean": round(float(df[~df["crop_ok"]]["dice_v4"].mean()), 4) if (~df["crop_ok"]).any() else None,
                            "std":  round(float(df[~df["crop_ok"]]["dice_v4"].std()),  4) if (~df["crop_ok"]).any() else None},
            },
        },
    }

    (output_dir / "stats.json").write_text(json.dumps(stats, indent=4))
    print(f"Wilcoxon v4 vs v3: stat={stat:.1f}, p={p:.3e}")

    violin_plot(df, "contrast", "Dice per contrast — v4 vs v3", output_dir / "dice_by_contrast.png")
    violin_plot(df, "dataset",  "Dice per dataset — v4 vs v3",  output_dir / "dice_by_dataset.png")

    print(f"Done. Results in {output_dir}")


if __name__ == "__main__":
    main()
