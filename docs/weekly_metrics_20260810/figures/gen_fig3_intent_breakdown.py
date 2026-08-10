#!/usr/bin/env python3
"""Generate the controller-intent metric breakdown."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt

from paper_plot_style import COLORS, DATA_PATH, label_bars, save


section = json.loads(DATA_PATH.read_text(encoding="utf-8"))["intent_understanding"]
metrics = section["metrics"]
slots = section["slot_accuracy"]
labels = ["类型精确率", "类型召回率", "类型F1", "意图集合完全正确", "参考意图命中", "动作命中", "完整框架", "目标值槽位"]
values = [
    metrics["intent_type_precision"] * 100,
    metrics["intent_type_recall"] * 100,
    metrics["intent_type_f1"] * 100,
    metrics["utterance_exact_type_set_accuracy"] * 100,
    metrics["gold_intent_hit_rate"] * 100,
    metrics["action_hit_rate"] * 100,
    metrics["full_intent_frame_accuracy"] * 100,
    slots["target_value"]["accuracy"] * 100,
]
colors = [COLORS["blue"], COLORS["cyan"], COLORS["green"], COLORS["gray"], COLORS["amber"], COLORS["cyan"], COLORS["amber"], COLORS["red"]]

fig, ax = plt.subplots(figsize=(12.2, 4.5))
bars = ax.bar(labels, values, color=colors, width=0.66)
ax.set_ylabel("准确率（%）")
ax.set_ylim(0, 105)
ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(axis="x", rotation=16)
label_bars(ax, bars)
save(fig, "fig3_intent_breakdown")
