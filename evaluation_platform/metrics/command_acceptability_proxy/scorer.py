"""Score offline command acceptability without pretending to have human labels.

The scorer is deliberately a proxy metric.  It evaluates commands with hard
operational guards and short-horizon evidence from the immutable H-PPO event
log.  It does not claim to measure controller or pilot human acceptance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MACRO_NAMES = {
    1: "RESUME_ROUTE",
    2: "HEADING",
    3: "ALTITUDE",
    4: "SPEED",
}
ACTIVE_MACROS = set(MACRO_NAMES)
DEFAULTS = {
    "heading_min_deg": 0.0,
    "heading_max_deg": 360.0,
    "altitude_min_ft": 29000.0,
    "altitude_max_ft": 41000.0,
    "speed_min_kt": 420.0,
    "speed_max_kt": 500.0,
    "duplicate_window_s": 30.0,
    "oscillation_window_s": 60.0,
    "accept_threshold": 0.75,
    "conditional_threshold": 0.60,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def read_diagnostics(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {
            int(row["episode"]): row
            for row in csv.DictReader(stream)
            if row.get("episode") not in (None, "")
        }


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _sim_time(row: dict[str, Any]) -> float:
    return _number(row.get("sim_time_s")) or 0.0


def _episode(row: dict[str, Any]) -> int:
    return _int(row.get("episode"), -1)


def _command_text(row: dict[str, Any]) -> str:
    return str(row.get("command") or "").strip()


def _target_value(row: dict[str, Any], macro: int) -> float | None:
    target = row.get("parameter_target")
    if isinstance(target, dict):
        keys = {
            2: ("heading_deg", "heading"),
            3: ("altitude_ft", "altitude"),
            4: ("speed_kt", "speed_kts", "speed"),
        }.get(macro, ())
        for key in keys:
            value = _number(target.get(key))
            if value is not None:
                return value

    command = _command_text(row)
    if macro == 2:
        pattern = r"\b(?:HDG|HEADING)\s+\S+\s+([-+]?\d+(?:\.\d+)?)"
    elif macro == 3:
        pattern = r"\b(?:ALT|ALTITUDE)\s+\S+\s+([-+]?\d+(?:\.\d+)?)"
    elif macro == 4:
        pattern = r"\b(?:SPD|SPEED|CAS)\s+\S+\s+([-+]?\d+(?:\.\d+)?)"
    else:
        return None
    match = re.search(pattern, command, flags=re.IGNORECASE)
    return _number(match.group(1)) if match else None


def _target_delta(macro: int, left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    delta = right - left
    if macro == 2:
        delta = (delta + 180.0) % 360.0 - 180.0
    return delta


def _same_target(left: dict[str, Any], right: dict[str, Any]) -> bool:
    macro = _int(left.get("macro_action"))
    if macro != _int(right.get("macro_action")):
        return False
    if macro == 1:
        return True
    first = _target_value(left, macro)
    second = _target_value(right, macro)
    delta = _target_delta(macro, first, second)
    if delta is None:
        return False
    tolerance = {2: 5.0, 3: 200.0, 4: 3.0}[macro]
    return abs(delta) <= tolerance


def _risk_snapshot(row: dict[str, Any]) -> dict[str, float | bool | None]:
    severity = _number(row.get("risk_severity_at_issue"))
    if severity is None:
        severity = _number(row.get("conflict_severity"))
    return {
        "severity": severity,
        "tcpa_s": _number(row.get("risk_tcpa_s_at_issue"))
        or _number(row.get("min_tcpa_s")),
        "horizontal_km": _number(row.get("min_horizontal_km")),
        "vertical_ft": _number(row.get("min_vertical_ft")),
        "loss_of_separation": _bool(row.get("loss_of_separation")),
    }


def _diagnostic_safety(row: dict[str, str] | None) -> tuple[float | None, str]:
    if not row:
        return None, "no_episode_diagnostics"
    collisions = _number(row.get("collision_events")) or 0.0
    violations = _number(row.get("safety_violations")) or 0.0
    secondary = _number(row.get("secondary_conflicts")) or 0.0
    if collisions > 0:
        return 0.0, "collision"
    if violations > 0 or secondary > 0:
        return 0.2, "safety_or_secondary_conflict"
    if _bool(row.get("safe_success")):
        return 1.0, "safe_success"
    if str(row.get("reason", "")).strip().lower() == "all_arrived":
        return 0.85, "all_arrived_without_recorded_violation"
    return 0.45, "episode_not_confirmed_safe"


def _score_effect(before: dict[str, Any], after: dict[str, Any] | None) -> tuple[float, str]:
    before_severity = before.get("severity")
    after_severity = after.get("severity") if after else None
    if before_severity is not None and after_severity is not None:
        delta = float(before_severity) - float(after_severity)
        if delta >= 0.10:
            return 1.0, "risk_reduced"
        if delta <= -0.10:
            return 0.0, "risk_increased"
        return max(0.0, min(1.0, 0.5 + delta)) , "risk_change_small"
    return 0.5, "risk_change_not_observed"


def _hard_reason(
    row: dict[str, Any],
    macro: int,
    applied: bool,
    target: float | None,
    previous: dict[str, Any] | None,
    cfg: dict[str, float],
) -> str:
    command = _command_text(row)
    if str(row.get("event", "")) == "command_failed":
        return "command_failed"
    if not applied:
        return "command_not_applied"
    if not command or command.upper() == "NOOP":
        return "empty_or_noop_command"
    if macro == 2 and (target is None or not cfg["heading_min_deg"] <= target <= cfg["heading_max_deg"]):
        return "heading_out_of_range_or_missing"
    if macro == 3 and (target is None or not cfg["altitude_min_ft"] <= target <= cfg["altitude_max_ft"]):
        return "altitude_out_of_range_or_missing"
    if macro == 4 and (target is None or not cfg["speed_min_kt"] <= target <= cfg["speed_max_kt"]):
        return "speed_out_of_range_or_missing"
    if previous is not None and _same_target(previous, row):
        previous_time = _sim_time(previous)
        if _sim_time(row) - previous_time <= cfg["duplicate_window_s"]:
            if not _bool(row.get("replan_triggered")) and not _bool(row.get("loss_of_separation")):
                return "duplicate_target_within_hold_window"
    if macro in {2, 3, 4}:
        risk = _risk_snapshot(row)
        target_ids = row.get("target_ids")
        if (risk.get("severity") or 0.0) <= 0.0 and not target_ids:
            return "unnecessary_intervention_without_conflict"
    return ""


def score(
    events_path: Path,
    diagnostics_path: Path | None = None,
    config: dict[str, float] | None = None,
) -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    if config:
        cfg.update({key: float(value) for key, value in config.items() if key in cfg})
    events = read_jsonl(events_path)
    diagnostics = read_diagnostics(diagnostics_path)

    timelines: defaultdict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    candidates: list[dict[str, Any]] = []
    ignored = Counter()
    for row in events:
        if row.get("event") != "action_executed" or not row.get("acid"):
            if row.get("event") == "command_failed" and row.get("acid"):
                pass
            else:
                continue
        macro = _int(row.get("macro_action"))
        if macro not in ACTIVE_MACROS:
            ignored["non_control_macro"] += 1
            continue
        timelines[(_episode(row), str(row.get("acid")))].append(row)
        candidates.append(row)

    for rows in timelines.values():
        rows.sort(key=_sim_time)
    candidates.sort(key=lambda row: (_episode(row), _sim_time(row), str(row.get("acid"))))

    details: list[dict[str, Any]] = []
    previous_by_aircraft: dict[tuple[int, str], dict[str, Any]] = {}
    for row in candidates:
        macro = _int(row.get("macro_action"))
        acid = str(row.get("acid"))
        key = (_episode(row), acid)
        applied = bool(row.get("command_applied", row.get("event") == "action_executed"))
        target = _target_value(row, macro)
        previous = previous_by_aircraft.get(key)
        reason = _hard_reason(row, macro, applied, target, previous, cfg)
        if row.get("event") == "action_executed" and applied:
            previous_by_aircraft[key] = row

        timeline = timelines[key]
        index = next((i for i, item in enumerate(timeline) if item is row), -1)
        future = timeline[index + 1:] if index >= 0 else []
        after_row = next((item for item in future if _sim_time(item) > _sim_time(row)), None)
        before_risk = _risk_snapshot(row)
        after_risk = _risk_snapshot(after_row) if after_row else None
        effect_score, effect_reason = _score_effect(before_risk, after_risk)
        safety_score, safety_reason = _diagnostic_safety(diagnostics.get(_episode(row)))
        if safety_score is None:
            safety_score = 0.0 if after_risk and after_risk["loss_of_separation"] else 0.7 if after_row else 0.45
        oscillation = False
        if after_row and _int(after_row.get("macro_action")) == macro and macro != 1:
            delta = _target_delta(macro, target, _target_value(after_row, macro))
            oscillation = delta is not None and abs(delta) > {2: 5.0, 3: 200.0, 4: 3.0}[macro]
        stability_score = 0.35 if oscillation and _sim_time(after_row) - _sim_time(row) <= cfg["oscillation_window_s"] else 1.0
        evidence_complete = bool(after_row or diagnostics.get(_episode(row)))
        feasibility_score = 0.0 if reason else 1.0
        efficiency_score = 0.5
        score_value = (
            0.45 * safety_score
            + 0.30 * effect_score
            + 0.15 * stability_score
            + 0.05 * feasibility_score
            + 0.05 * efficiency_score
        )
        if reason:
            grade = "rejected"
        elif not evidence_complete:
            grade = "conditional"
            reason = "insufficient_post_command_evidence"
        elif score_value >= cfg["accept_threshold"]:
            grade = "accepted"
            reason = ""
        elif score_value >= cfg["conditional_threshold"]:
            grade = "conditional"
            # A safe episode may still be conditionally accepted when the local
            # post-command evidence is weak. Do not report ``safe_success`` as
            # a rejection reason in the UI.
            reason = (
                effect_reason
                if effect_reason != "risk_reduced"
                else "score_below_accept_threshold"
            )
        else:
            grade = "rejected"
            reason = safety_reason if safety_score < effect_score else effect_reason
        details.append({
            "episode": _episode(row),
            "sim_time_s": _sim_time(row),
            "aircraft_id": acid,
            "macro_action": macro,
            "macro_name": MACRO_NAMES[macro],
            "command": _command_text(row),
            "command_applied": applied,
            "target_value": target,
            "grade": grade,
            "accepted": grade == "accepted",
            "conditional": grade == "conditional",
            "score": round(score_value, 6),
            "hard_reject_reason": reason if reason in {
                "command_failed",
                "command_not_applied",
                "empty_or_noop_command",
                "heading_out_of_range_or_missing",
                "altitude_out_of_range_or_missing",
                "speed_out_of_range_or_missing",
                "duplicate_target_within_hold_window",
                "unnecessary_intervention_without_conflict",
            } else "",
            "reason": reason,
            "safety_score": round(safety_score, 6),
            "effect_score": round(effect_score, 6),
            "stability_score": round(stability_score, 6),
            "feasibility_score": round(feasibility_score, 6),
            "efficiency_score": round(efficiency_score, 6),
            "evidence_complete": evidence_complete,
            "effect_evidence": effect_reason,
            "safety_evidence": safety_reason,
            "oscillation_detected": oscillation,
        })

    eligible = len(details)
    accepted = sum(row["accepted"] for row in details)
    conditional = sum(row["conditional"] for row in details)
    rejected = eligible - accepted - conditional
    hard_rejected = sum(bool(row["hard_reject_reason"]) for row in details)
    by_macro: dict[str, dict[str, Any]] = {}
    for macro, name in MACRO_NAMES.items():
        rows = [row for row in details if row["macro_action"] == macro]
        count = len(rows)
        accepted_count = sum(row["accepted"] for row in rows)
        conditional_count = sum(row["conditional"] for row in rows)
        by_macro[str(macro)] = {
            "name": name,
            "eligible": count,
            "accepted": accepted_count,
            "conditional": conditional_count,
            "rejected": count - accepted_count - conditional_count,
            "acceptance_rate": None if not count else accepted_count / count,
            "practical_rate": None if not count else (accepted_count + conditional_count) / count,
        }
    value = None if not eligible else accepted / eligible
    practical = None if not eligible else (accepted + conditional) / eligible
    return {
        "metric_id": "command_acceptability_proxy",
        "status": "complete" if eligible else "not_evaluable",
        # Metric values are stored as ratios in [0, 1]. The UI is responsible
        # for percentage formatting, as it is for the other platform metrics.
        "primary": {"name": "offline_command_acceptability_rate", "value": value, "unit": "ratio"},
        "metrics": {
            "offline_command_acceptability_rate": value,
            "offline_command_practical_rate": practical,
            "offline_command_conditional_rate": None if not eligible else conditional / eligible,
            "offline_command_hard_reject_rate": None if not eligible else hard_rejected / eligible,
            "evidence_complete_rate": None if not eligible else sum(row["evidence_complete"] for row in details) / eligible,
            "oscillation_rate": None if not eligible else sum(row["oscillation_detected"] for row in details) / eligible,
        },
        "counts": {
            "eligible_commands": eligible,
            "accepted_commands": accepted,
            "conditional_commands": conditional,
            "rejected_commands": rejected,
            "hard_rejected_commands": hard_rejected,
            "ignored_non_control_events": sum(ignored.values()),
        },
        "by_macro": by_macro,
        "failure_reasons": [
            {"reason": key, "count": count}
            for key, count in Counter(row["reason"] for row in details if row["reason"]).most_common()
        ],
        "details": details,
        "configuration": cfg,
        "claim_boundary": (
            "该指标是基于规则、短时风险变化和回合结果的自动化离线可接受性代理评分；"
            "不等同于真实管制员、飞行员或机组的人因接受度。"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = score(args.events, args.diagnostics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
