#!/usr/bin/env python3
"""Generate the controller-imitation metric breakdown."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt

from paper_plot_style import COLORS, DATA_PATH, label_bars, save


section = json.loads(DATA_PATH.read_text(encoding="utf-8"))["controller_imitation"]
metrics = section["metrics"]
labels = ["宏平均召回", "微平均召回", "系统指令精度", "参数准确率", "严格参数宏召回", "高度召回", "速度召回"]
values = [
    metrics["controller_imitation_macro_recall"] * 100,
    metrics["controller_imitation_micro_recall"] * 100,
    metrics["system_command_precision"] * 100,
    metrics["parameter_accuracy_on_comparable_matches"] * 100,
    metrics["strict_parameter_macro_recall"] * 100,
    section["by_family"]["altitude"]["recall"] * 100,
    section["by_family"]["speed"]["recall"] * 100,
]
colors = [COLORS["blue"], COLORS["cyan"], COLORS["green"], COLORS["amber"], COLORS["gray"], COLORS["blue"], COLORS["red"]]

fig, ax = plt.subplots(figsize=(10.5, 4.2))
bars = ax.bar(labels, values, color=colors, width=0.68)
ax.set_ylabel("指标值（%）")
ax.set_ylim(0, 100)
ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(axis="x", rotation=18)
label_bars(ax, bars)
save(fig, "fig2_imitation_breakdown")
