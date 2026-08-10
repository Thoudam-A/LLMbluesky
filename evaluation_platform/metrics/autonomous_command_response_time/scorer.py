"""Score post-detection H-PPO decision and command-generation response time."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "episode",
    "decision_count",
    "command_executions",
    "decision_response_samples",
    "mean_decision_response_ms",
    "max_decision_response_ms",
}


def _number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _integer(row: dict[str, str], key: str) -> int:
    return max(0, int(_number(row, key)))


def read_diagnostics(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return list(reader)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = sum(row["samples"] for row in rows)
    decisions = sum(row["decision_count"] for row in rows)
    commands = sum(row["command_executions"] for row in rows)
    total_ms = sum(row["samples"] * row["mean_ms"] for row in rows)
    return {
        "episodes": len(rows),
        "response_samples": samples,
        "decision_count": decisions,
        "command_executions": commands,
        "weighted_mean_ms": None if samples == 0 else total_ms / samples,
        "max_ms": None if samples == 0 else max((row["max_ms"] for row in rows if row["samples"]), default=0.0),
        "sample_coverage": None if decisions == 0 else samples / decisions,
        "amortized_ms_per_applied_command": None if commands == 0 else total_ms / commands,
    }


def score(diagnostics_path: Path) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for index, row in enumerate(read_diagnostics(diagnostics_path), 1):
        samples = _integer(row, "decision_response_samples")
        details.append({
            "episode": row.get("episode", index),
            "scenario": row.get("scenario") or "未标注场景",
            "mode": row.get("mode") or "unknown",
            "decision_count": _integer(row, "decision_count"),
            "command_executions": _integer(row, "command_executions"),
            "samples": samples,
            "mean_ms": max(0.0, _number(row, "mean_decision_response_ms")),
            "max_ms": max(0.0, _number(row, "max_decision_response_ms")),
        })

    aggregate = _aggregate(details)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[row["scenario"]].append(row)
    by_scenario = {name: _aggregate(rows) for name, rows in sorted(grouped.items())}
    value = aggregate["weighted_mean_ms"]
    return {
        "metric_id": "autonomous_command_response_time",
        "status": "complete" if value is not None else "not_evaluable",
        "primary": {
            "name": "加权平均自主决策与指令生成响应时间",
            "value": None if value is None else round(value, 6),
            "unit": "ms",
            "direction": "lower_is_better",
        },
        "metrics": {
            "weighted_mean_decision_response_ms": None if value is None else round(value, 6),
            "max_decision_response_ms": None if aggregate["max_ms"] is None else round(aggregate["max_ms"], 6),
            "response_sample_coverage": None if aggregate["sample_coverage"] is None else round(aggregate["sample_coverage"], 6),
            "amortized_ms_per_applied_command": (
                None if aggregate["amortized_ms_per_applied_command"] is None
                else round(aggregate["amortized_ms_per_applied_command"], 6)
            ),
        },
        "counts": {
            "episodes": aggregate["episodes"],
            "evaluable_episodes": sum(row["samples"] > 0 for row in details),
            "response_samples": aggregate["response_samples"],
            "decision_count": aggregate["decision_count"],
            "command_executions": aggregate["command_executions"],
        },
        "by_scenario": {
            name: {
                **values,
                "weighted_mean_ms": None if values["weighted_mean_ms"] is None else round(values["weighted_mean_ms"], 6),
                "max_ms": None if values["max_ms"] is None else round(values["max_ms"], 6),
                "sample_coverage": None if values["sample_coverage"] is None else round(values["sample_coverage"], 6),
                "amortized_ms_per_applied_command": (
                    None if values["amortized_ms_per_applied_command"] is None
                    else round(values["amortized_ms_per_applied_command"], 6)
                ),
            }
            for name, values in by_scenario.items()
        },
        "details": details,
        "definition": {
            "aggregation": "sum(samples * episode_mean_ms) / sum(samples)",
            "coverage": "sum(response_samples) / sum(decision_count)",
        },
        "claim_boundary": (
            "计时从已形成的冲突状态进入 H-PPO 决策步骤开始，到策略推理、参数化及 BlueSky 指令调用完成为止；"
            "不包含冲突检测、离线 LLM 生成、调度等待和航空器执行过程。"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = score(args.diagnostics)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
