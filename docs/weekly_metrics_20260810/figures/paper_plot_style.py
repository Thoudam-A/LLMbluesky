"""Shared report plotting style."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


FIG_DIR = Path(__file__).resolve().parent
DATA_PATH = FIG_DIR.parent / "metrics_summary.json"
COLORS = {
    "blue": "#2563EB",
    "cyan": "#0891B2",
    "green": "#059669",
    "amber": "#D97706",
    "red": "#DC2626",
    "gray": "#64748B",
}

matplotlib.rcParams.update(
    {
        "font.size": 11,
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def label_bars(ax: plt.Axes, bars: object) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png")
    plt.close(fig)
