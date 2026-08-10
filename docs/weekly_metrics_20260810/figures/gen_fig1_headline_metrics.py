#!/usr/bin/env python3
"""Generate the headline weekly metrics figure."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt

from paper_plot_style import COLORS, DATA_PATH, label_bars, save


data = json.loads(DATA_PATH.read_text(encoding="utf-8"))["headline"]
labels = ["模仿宏平均召回", "意图类型F1", "完整意图框架准确率"]
values = [
    data["controller_imitation_macro_recall"] * 100,
    data["intent_type_f1"] * 100,
    data["full_intent_frame_accuracy"] * 100,
]

fig, ax = plt.subplots(figsize=(7.2, 3.8))
bars = ax.bar(labels, values, color=[COLORS["blue"], COLORS["green"], COLORS["amber"]], width=0.6)
ax.set_ylabel("指标值（%）")
ax.set_ylim(0, 105)
ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
ax.set_axisbelow(True)
label_bars(ax, bars)
save(fig, "fig1_headline_metrics")
