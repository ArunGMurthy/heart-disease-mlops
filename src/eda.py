"""
Exploratory Data Analysis for the Heart Disease UCI dataset.

Generates and saves professional visualizations to reports/figures/:
  - target class balance
  - numeric feature histograms
  - correlation heatmap

Usage:
    python src/eda.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.data_preprocessing import NUMERIC_FEATURES, clean, load_raw  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
sns.set_theme(style="whitegrid")


def plot_class_balance(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["target"].value_counts().sort_index()
    labels = ["No Disease (0)", "Disease (1)"]
    sns.barplot(x=labels, y=counts.values, hue=labels, ax=ax, palette="viridis", legend=False)
    ax.set_ylabel("Count")
    ax.set_title("Target Class Balance")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(out_dir / "class_balance.png", dpi=150)
    plt.close(fig)


def plot_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    n = len(NUMERIC_FEATURES)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()
    for i, feat in enumerate(NUMERIC_FEATURES):
        sns.histplot(df[feat].dropna(), kde=True, ax=axes[i], color="#4C72B0")
        axes[i].set_title(feat)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Numeric Feature Distributions", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def run_eda(out_dir: Path = FIG_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = clean(load_raw())
    plot_class_balance(df, out_dir)
    plot_histograms(df, out_dir)
    plot_correlation_heatmap(df, out_dir)
    print(f"EDA figures saved to {out_dir}")
    print(df.describe(include="all").T)


if __name__ == "__main__":
    run_eda()
