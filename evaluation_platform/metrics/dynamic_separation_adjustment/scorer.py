"""Formal scorer for user-triggered dynamic separation adjustments.

Only runtime ``separation_adjustment_outcome`` events are admissible.  Older
episode-level logs may still be inspected elsewhere, but must never be used as
a substitute for a user changing a separation standard during a simulation.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            result.append(row)
    return result


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _event_detail(row: dict[str, Any]) -> dict[str, Any]:
    affected = int(_number(row.get("affected_pair_count", row.get("initial_violation_count", 0))))
    config_applied = bool(row.get("config_applied", False))
    recomputed = bool(row.get("detection_recomputed", False))
    stable = str(row.get("state", "")).upper() == "SUCCESS" and bool(row.get("success", False))
    secondary = int(_number(row.get("secondary_conflict_count", row.get("secondary_conflicts", 0))))
    eligible = config_applied and recomputed and affected > 0
    success = eligible and stable and secondary == 0
    return {
        "adjustment_id": row.get("adjustment_id"),
        "episode": row.get("episode"),
        "sim_time_s": _number(row.get("sim_time_s")),
        "old_horizontal_nm": _number(row.get("old_horizontal_nm", row.get("previous_horizontal_nm"))),
        "new_horizontal_nm": _number(row.get("new_horizontal_nm", row.get("horizontal_nm"))),
        "old_vertical_ft": _number(row.get("old_vertical_ft", row.get("previous_vertical_ft"))),
        "new_vertical_ft": _number(row.get("new_vertical_ft", row.get("vertical_ft"))),
        "old_time_s": _number(row.get("old_time_s", row.get("previous_time_s"))),
        "new_time_s": _number(row.get("new_time_s", row.get("time_s"))),
        "change_type": row.get("change_type", "user_adjustment"),
        "affected_pair_count": affected,
        "config_applied": config_applied,
        "detection_recomputed": recomputed,
        "stable": stable,
        "secondary_conflict_count": secondary,
        "response_time_s": row.get("response_time_s"),
        "eligible": eligible,
        "success": success,
        "reason": row.get("reason") or ("stable_compliance" if success else "not_stably_resolved"),
    }


def score(inputs: dict[str, str], output_dir: str | Path | None = None, **_: Any) -> dict[str, Any]:
    events_path = Path(inputs.get("events") or inputs.get("events_jsonl") or "")
    outcomes = [
        _event_detail(row)
        for row in _rows(events_path)
        if row.get("event") == "separation_adjustment_outcome" and row.get("adjustment_id")
    ]
    eligible = [row for row in outcomes if row["eligible"]]
    successful = [row for row in eligible if row["success"]]
    by_type: dict[str, dict[str, Any]] = defaultdict(lambda: {"events": 0, "successes": 0})
    for row in eligible:
        bucket = by_type[str(row["change_type"])]
        bucket["events"] += 1
        bucket["successes"] += int(row["success"])
    for bucket in by_type.values():
        bucket["success_rate"] = bucket["successes"] / bucket["events"] if bucket["events"] else None

    response_times = [_number(row["response_time_s"]) for row in successful if row.get("response_time_s") is not None]
    reasons = Counter(row["reason"] for row in outcomes if not row["success"])
    value = len(successful) / len(eligible) if eligible else None
    result = {
        "metric_id": "dynamic_separation_adjustment",
        "status": "complete" if eligible else "not_evaluable",
        "primary": {"name": "dynamic_separation_adjustment_success_rate", "value": value, "unit": "ratio"},
        "metrics": {
            "dynamic_separation_adjustment_success_rate": value,
            "formal_dynamic_separation_adjustment_success_rate": value,
            "stable_compliance_rate": value,
            "mean_response_time_s": sum(response_times) / len(response_times) if response_times else None,
            "secondary_conflict_rate": sum(row["secondary_conflict_count"] > 0 for row in eligible) / len(eligible) if eligible else None,
        },
        "counts": {
            "outcome_events": len(outcomes),
            "change_events": len(outcomes),
            "eligible_events": len(eligible),
            "successful_events": len(successful),
            "formal_eligible_events": len(eligible),
            "formal_successful_events": len(successful),
        },
        "details": outcomes,
        "by_change_type": dict(by_type),
        "failure_reasons": [{"reason": key, "count": count} for key, count in reasons.most_common()],
        "claim_boundary": (
            "正式动态间隔调整成功率仅统计用户修改间隔标准后产生的运行时审计事件："
            "配置生效、受影响冲突对已重新检测、在截止时间内稳定满足新标准，并且没有次生冲突。"
        ),
    }
    if output_dir:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
