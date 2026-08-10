#!/usr/bin/env python3
"""Build a redacted, reproducible weekly ATC metrics bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percent(value: Any) -> str:
    return "NA" if value is None else f"{float(value) * 100:.2f}%"


def build_summary(
    imitation: dict[str, Any],
    replay: dict[str, Any],
    intent: dict[str, Any],
    intent_type_set: dict[str, Any],
    merge: dict[str, Any],
    inputs: dict[str, Path],
) -> dict[str, Any]:
    imitation_metrics = imitation["metrics"]
    return {
        "bundle_version": "weekly_atc_metrics_v1.0",
        "report_date": "2026-08-10",
        "headline": {
            "controller_imitation_macro_recall": imitation_metrics[
                "controller_imitation_macro_recall"
            ],
            "intent_type_f1": intent_type_set["intent_type_f1"],
            "full_intent_frame_accuracy": intent["frame_accuracy"],
        },
        "controller_imitation": {
            "system": imitation.get("system_name"),
            "scope": {
                "replay_date": replay.get("local_date"),
                "processed_ticks": replay.get("processed_ticks"),
                "historical_states_frozen": replay.get("historical_states_frozen"),
                "reference_hidden_during_decision": replay.get("reference_hidden"),
                "reference_scope": imitation.get("reference_scope"),
                "families": imitation.get("scope_families"),
                "state_altitude_band_m": imitation.get("reference_state_altitude_band_m"),
                "match_window_sec": 60,
            },
            "counts": imitation["counts"],
            "metrics": imitation_metrics,
            "by_family": imitation["by_family"],
            "runtime_sec": replay.get("runtime_sec"),
            "qwen_enabled": bool(replay.get("qwen", {}).get("enabled")),
        },
        "intent_understanding": {
            "scope": {
                "gold_rows": intent["gold_rows"],
                "prediction_rows": intent["prediction_rows"],
                "skipped_gold_rows": intent["skipped_gold_rows"],
                "evaluated_gold_intents": intent["evaluated_gold_intents"],
                "asr_included": False,
                "reference_note": (
                    "固化的500条结构化参考标注；当前证据不能等同于独立人工双盲金标准。"
                ),
            },
            "metrics": {
                "intent_type_precision": intent_type_set["intent_type_precision"],
                "intent_type_recall": intent_type_set["intent_type_recall"],
                "intent_type_f1": intent_type_set["intent_type_f1"],
                "utterance_exact_type_set_accuracy": intent_type_set[
                    "utterance_exact_type_set_accuracy"
                ],
                "gold_intent_hit_rate": intent["intent_type_accuracy"],
                "action_hit_rate": intent["action_accuracy"],
                "full_intent_frame_accuracy": intent["frame_accuracy"],
                "false_negative_count": intent["false_negative_count"],
                "false_positive_count": intent["false_positive_count"],
            },
            "slot_accuracy": intent["slot_accuracy"],
            "per_intent_type": intent["per_intent_type"],
            "type_set_per_intent_type": intent_type_set["per_intent_type"],
            "prediction_merge": merge,
        },
        "evidence": {
            name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for name, path in sorted(inputs.items())
        },
        "claim_boundaries": [
            "模仿精度是冻结历史态势离线回放的一致性，不是真实同场景复现或运行安全证明。",
            "意图类型准确率衡量大类识别；完整意图框架准确率还要求动作与必需参数槽位正确。",
            "意图类型F1同时惩罚漏检和多报；90.73%的参考意图命中率不惩罚额外预测，不能单独作为正式准确率。",
            "意图评估不包含语音识别误差，参考标注尚不能宣称为独立人工双盲金标准。",
        ],
    }


def markdown(summary: dict[str, Any]) -> str:
    imitation = summary["controller_imitation"]
    intent = summary["intent_understanding"]
    im = imitation["metrics"]
    it = intent["metrics"]
    lines = [
        "# 本周指标评估结果（2026-08-10）",
        "",
        "本周完成管制员指令模仿精度全日冻结回放，以及500条管制话语意图识别结果重算。",
        "",
        "## 汇报结论",
        "",
        f"- 管制员指令模仿宏平均召回：**{percent(im['controller_imitation_macro_recall'])}**；",
        f"- 意图类型识别F1：**{percent(it['intent_type_f1'])}**；",
        f"- 意图类型精确率 / 召回率：**{percent(it['intent_type_precision'])} / {percent(it['intent_type_recall'])}**；",
        f"- 参考意图命中率（旧口径）：**{percent(it['gold_intent_hit_rate'])}**；",
        f"- 完整意图框架准确率：**{percent(it['full_intent_frame_accuracy'])}**。",
        "",
        "> 本周建议把“识别准确率”汇报为意图类型F1 80.34%。90.73%只表示参考意图命中率，没有惩罚多报；当前综合平台的严格完整框架口径为35.13%。",
        "",
        "## 可视化",
        "",
        "### 指标总览",
        "",
        "![指标总览](figures/fig1_headline_metrics.png)",
        "",
        "### 管制员指令模仿精度分项",
        "",
        "![管制员指令模仿精度分项](figures/fig2_imitation_breakdown.png)",
        "",
        "### 管制意图识别分项",
        "",
        "![管制意图识别分项](figures/fig3_intent_breakdown.png)",
        "",
        "## 评估范围",
        "",
        "| 项目 | 管制员指令模仿精度 | 管制意图识别 |",
        "|---|---:|---:|",
        f"| 测试规模 | {imitation['scope']['processed_ticks']:,} tick | {intent['scope']['gold_rows']}条话语 / {intent['scope']['evaluated_gold_intents']}个意图 |",
        f"| 参考数量 | {imitation['counts']['reference_events']}条历史指令 | {intent['scope']['gold_rows']}条固化参考标注 |",
        f"| 系统输出 | {imitation['counts']['system_outputs']}条指令 | {intent['scope']['prediction_rows']}条预测 |",
        "| 数据范围 | 2025-07-02上海进近冻结历史态势 | 管制指令文本，不含ASR |",
        "",
        "## 模仿精度分项",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 宏平均召回 | {percent(im['controller_imitation_macro_recall'])} |",
        f"| 微平均召回 | {percent(im['controller_imitation_micro_recall'])} |",
        f"| 系统指令精度 | {percent(im['system_command_precision'])} |",
        f"| 匹配后参数准确率 | {percent(im['parameter_accuracy_on_comparable_matches'])} |",
        f"| 严格参数宏平均召回 | {percent(im['strict_parameter_macro_recall'])} |",
        f"| 高度指令召回 | {percent(imitation['by_family']['altitude']['recall'])} |",
        f"| 速度指令召回 | {percent(imitation['by_family']['speed']['recall'])} |",
        "",
        "当前系统输出全部为高度指令，因此速度指令召回仍为0%，这是下一阶段最明确的改进点。",
        "",
        "## 意图识别分项",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 意图类型精确率 | {percent(it['intent_type_precision'])} |",
        f"| 意图类型召回率 | {percent(it['intent_type_recall'])} |",
        f"| 意图类型F1 | {percent(it['intent_type_f1'])} |",
        f"| 话语级意图集合完全正确率 | {percent(it['utterance_exact_type_set_accuracy'])} |",
        f"| 参考意图命中率（不惩罚多报） | {percent(it['gold_intent_hit_rate'])} |",
        f"| 动作命中率 | {percent(it['action_hit_rate'])} |",
        f"| 完整意图框架准确率 | {percent(it['full_intent_frame_accuracy'])} |",
        f"| 呼号槽位准确率 | {percent(intent['slot_accuracy']['callsign']['accuracy'])} |",
        f"| 目标值槽位准确率 | {percent(intent['slot_accuracy']['target_value']['accuracy'])} |",
        f"| 单位槽位准确率 | {percent(intent['slot_accuracy']['unit']['accuracy'])} |",
        "",
        "意图类型召回较高，但额外预测导致精确率只有71.71%；目标值、单位、频率、跑道及进近类型等参数槽位进一步限制了完整框架准确率。",
        "",
        "## 结果边界",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["claim_boundaries"])
    lines.extend(
        [
            "",
            "## 复现",
            "",
            "图表从同目录 `metrics_summary.json` 自动生成，原始数据、模型权重和逐条预测不进入Git仓库。",
            "",
            "```powershell",
            "python .\\figures\\gen_fig1_headline_metrics.py",
            "python .\\figures\\gen_fig2_imitation_breakdown.py",
            "python .\\figures\\gen_fig3_intent_breakdown.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imitation", required=True, type=Path)
    parser.add_argument("--replay-summary", required=True, type=Path)
    parser.add_argument("--intent", required=True, type=Path)
    parser.add_argument("--intent-type-set", required=True, type=Path)
    parser.add_argument("--intent-merge-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    inputs = {
        "imitation_metric": args.imitation,
        "replay_summary": args.replay_summary,
        "intent_metric": args.intent,
        "intent_type_set_metric": args.intent_type_set,
        "intent_merge_manifest": args.intent_merge_manifest,
    }
    summary = build_summary(
        load_json(args.imitation),
        load_json(args.replay_summary),
        load_json(args.intent),
        load_json(args.intent_type_set),
        load_json(args.intent_merge_manifest),
        inputs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "WEEKLY_METRICS_REPORT.md").write_text(
        markdown(summary), encoding="utf-8"
    )
    print(json.dumps(summary["headline"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
