#!/usr/bin/env python3
"""Score ATC predictions on a safety-critical automation scope."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SPACE_RE = re.compile(r"\s+")
FILLERS = {"啊", "呀", "呢", "吧", "的"}
UNIT_ALIASES = {
    "knots": "kt",
    "kts": "kt",
    "knot": "kt",
    "meters": "m",
    "meter": "m",
    "degrees": "degree",
    "deg": "degree",
}
DEFAULT_SCOPE = [
    "altitude_change",
    "altitude_maintain",
    "speed_adjust",
    "speed_procedure",
    "heading_change",
    "direct_to_fix",
    "approach_clearance",
    "qnh_setting",
    "restriction_cancel",
    "frequency_transfer",
    "radar_service",
    "landing_clearance",
    "takeoff_or_departure_instruction",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = SPACE_RE.sub("", str(value).strip()).lower()
    for filler in FILLERS:
        text = text.replace(filler, "")
    return text


def norm_unit(value: Any) -> str:
    text = norm_text(value)
    return UNIT_ALIASES.get(text, text)


def norm_number(value: Any, slot: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if slot == "frequency":
        try:
            return f"{float(text):g}"
        except ValueError:
            return norm_text(text)
    if slot == "target_value":
        try:
            if "." in text:
                return f"{float(text):g}"
            return str(int(text))
        except ValueError:
            return norm_text(text)
    return norm_text(text)


def norm_slot(intent: dict[str, Any], slot: str) -> str:
    value = intent.get(slot)
    if slot == "unit":
        return norm_unit(value)
    if slot in {"target_value", "frequency"}:
        return norm_number(value, slot)
    if slot == "runway":
        return norm_text(value).upper()
    return norm_text(value)


def slot_match(gold: dict[str, Any], pred: dict[str, Any], slot: str) -> bool:
    return norm_slot(gold, slot) == norm_slot(pred, slot)


def scope_set(scope_arg: str) -> set[str]:
    return {item.strip() for item in scope_arg.split(",") if item.strip()}


def in_scope(intent: dict[str, Any], scope: set[str]) -> bool:
    return str(intent.get("intent_type") or "") in scope


def gate_slots(intent: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    intent_type = str(intent.get("intent_type") or "")
    spec = schema.get("intent_types", {}).get(intent_type, {})
    slots = [slot for slot in spec.get("required_slots", []) if slot not in {"callsign", "intent_type", "action"}]
    conditional = spec.get("conditional_required_slots", {})
    if intent_type == "heading_change" and intent.get("target_value") not in {None, ""}:
        for slot in conditional.get("explicit_heading_or_angle", []):
            if slot not in slots:
                slots.append(slot)
    return slots


def candidate_score(gold: dict[str, Any], pred: dict[str, Any], schema: dict[str, Any]) -> tuple[int, int]:
    type_ok = int(slot_match(gold, pred, "intent_type"))
    action_ok = int(type_ok and slot_match(gold, pred, "action"))
    slot_hits = sum(int(slot_match(gold, pred, slot)) for slot in gate_slots(gold, schema))
    return (type_ok + action_ok, slot_hits)


def safety_success(gold: dict[str, Any], pred: dict[str, Any], schema: dict[str, Any]) -> bool:
    if not slot_match(gold, pred, "intent_type"):
        return False
    if not slot_match(gold, pred, "action"):
        return False
    return all(slot_match(gold, pred, slot) for slot in gate_slots(gold, schema))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def score(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]], schema: dict[str, Any], scope: set[str]) -> dict[str, Any]:
    pred_by_utt = {row.get("utterance_id"): row for row in pred_rows}
    evaluated_gold = 0
    success_count = 0
    type_correct = 0
    action_correct = 0
    false_negative = 0
    false_positive = 0
    skipped_gold_rows = 0
    pending_gold_rows = 0
    invalid_pred_rows = 0
    utterance_total = 0
    utterance_success = 0
    control_detection_total = 0
    control_detection_correct = 0
    errors: list[dict[str, Any]] = []
    per_type: dict[str, dict[str, int]] = {}
    slot_stats: dict[str, dict[str, int]] = {}
    callsign_stats = {"total": 0, "correct": 0}

    for gold_row in gold_rows:
        status = gold_row.get("annotation_status")
        gold_all = [item for item in gold_row.get("intents", []) if isinstance(item, dict)]
        if status == "pending":
            pending_gold_rows += 1
            continue
        if status == "skip":
            skipped_gold_rows += 1
            gold_scoped: list[dict[str, Any]] = []
        elif status == "done":
            gold_scoped = [item for item in gold_all if in_scope(item, scope)]
        else:
            pending_gold_rows += 1
            continue

        pred_row = pred_by_utt.get(gold_row.get("utterance_id"), {})
        pred_all = pred_row.get("intents", [])
        if not isinstance(pred_all, list):
            pred_all = []
            invalid_pred_rows += 1
        pred_scoped = [item for item in pred_all if isinstance(item, dict) and in_scope(item, scope)]

        control_detection_total += 1
        control_detection_correct += int(bool(gold_scoped) == bool(pred_scoped))

        unused = set(range(len(pred_scoped)))
        row_success = True
        row_has_work = bool(gold_scoped or pred_scoped)

        if not gold_scoped and not pred_scoped:
            if row_has_work:
                utterance_total += 1
                utterance_success += 1
            continue

        utterance_total += 1

        for gold in gold_scoped:
            evaluated_gold += 1
            intent_type = str(gold.get("intent_type") or "<missing>")
            per_type.setdefault(intent_type, {"gold": 0, "intent_type_correct": 0, "action_correct": 0, "success": 0})
            per_type[intent_type]["gold"] += 1

            best_idx = None
            best_score = (-1, -1)
            for idx in unused:
                pred = pred_scoped[idx]
                current = candidate_score(gold, pred, schema)
                if current > best_score:
                    best_score = current
                    best_idx = idx

            if best_idx is None:
                false_negative += 1
                row_success = False
                errors.append(
                    {
                        "utterance_id": gold_row.get("utterance_id"),
                        "annotation_id": gold_row.get("annotation_id"),
                        "error_type": "missing_intent",
                        "text": gold_row.get("text"),
                        "gold": gold,
                    }
                )
                continue

            pred = pred_scoped[best_idx]
            unused.remove(best_idx)
            type_ok = slot_match(gold, pred, "intent_type")
            action_ok = type_ok and slot_match(gold, pred, "action")
            ok = action_ok and safety_success(gold, pred, schema)
            callsign_stats["total"] += 1
            callsign_stats["correct"] += int(slot_match(gold, pred, "callsign"))

            type_correct += int(type_ok)
            action_correct += int(action_ok)
            success_count += int(ok)
            per_type[intent_type]["intent_type_correct"] += int(type_ok)
            per_type[intent_type]["action_correct"] += int(action_ok)
            per_type[intent_type]["success"] += int(ok)

            for slot in gate_slots(gold, schema):
                slot_stats.setdefault(slot, {"total": 0, "correct": 0})
                slot_stats[slot]["total"] += 1
                slot_stats[slot]["correct"] += int(slot_match(gold, pred, slot))

            if not ok:
                row_success = False
                if not type_ok:
                    error_type = "wrong_intent_type"
                elif not slot_match(gold, pred, "action"):
                    error_type = "wrong_action"
                else:
                    error_type = "wrong_critical_slot"
                errors.append(
                    {
                        "utterance_id": gold_row.get("utterance_id"),
                        "annotation_id": gold_row.get("annotation_id"),
                        "error_type": error_type,
                        "text": gold_row.get("text"),
                        "gold": gold,
                        "prediction": pred,
                    }
                )

        false_positive += len(unused)
        if unused:
            row_success = False
        for idx in unused:
            errors.append(
                {
                    "utterance_id": gold_row.get("utterance_id"),
                    "annotation_id": gold_row.get("annotation_id"),
                    "error_type": "extra_critical_intent",
                    "text": gold_row.get("text"),
                    "prediction": pred_scoped[idx],
                }
            )

        utterance_success += int(row_success)

    def ratio(num: int, den: int) -> float | None:
        return None if den == 0 else num / den

    return {
        "scope_intent_types": sorted(scope),
        "gold_rows": len(gold_rows),
        "prediction_rows": len(pred_rows),
        "pending_gold_rows": pending_gold_rows,
        "skipped_gold_rows": skipped_gold_rows,
        "evaluated_safety_gold_intents": evaluated_gold,
        "safety_critical_intent_success": ratio(success_count, evaluated_gold),
        "intent_type_accuracy": ratio(type_correct, evaluated_gold),
        "action_accuracy": ratio(action_correct, evaluated_gold),
        "critical_false_negative_count": false_negative,
        "critical_false_positive_count": false_positive,
        "control_detection_accuracy": ratio(control_detection_correct, control_detection_total),
        "utterance_success_rate": ratio(utterance_success, utterance_total),
        "callsign_accuracy_diagnostic": ratio(callsign_stats["correct"], callsign_stats["total"]),
        "invalid_prediction_rows": invalid_pred_rows,
        "per_intent_type": {
            key: {
                **value,
                "success_rate": ratio(value["success"], value["gold"]),
            }
            for key, value in sorted(per_type.items())
        },
        "slot_accuracy": {
            key: {
                **value,
                "accuracy": ratio(value["correct"], value["total"]),
            }
            for key, value in sorted(slot_stats.items())
        },
        "errors": errors,
    }


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# ATC Safety-Critical Intent Report",
        "",
        "## Task Definition",
        "",
        "This report evaluates only high-frequency, automation-ready control intents.",
        "Callsign remains a diagnostic field, but it is not a hard success gate for intent understanding.",
        "",
        f"- Scope intent types: {', '.join(metrics['scope_intent_types'])}",
        "",
        "## Summary",
        "",
        f"- Gold rows: {metrics['gold_rows']}",
        f"- Prediction rows: {metrics['prediction_rows']}",
        f"- Evaluated safety-critical gold intents: {metrics['evaluated_safety_gold_intents']}",
        f"- Safety-critical intent success: {fmt(metrics['safety_critical_intent_success'])}",
        f"- Intent type accuracy: {fmt(metrics['intent_type_accuracy'])}",
        f"- Action accuracy: {fmt(metrics['action_accuracy'])}",
        f"- Control detection accuracy: {fmt(metrics['control_detection_accuracy'])}",
        f"- Utterance success rate: {fmt(metrics['utterance_success_rate'])}",
        f"- Callsign accuracy (diagnostic only): {fmt(metrics['callsign_accuracy_diagnostic'])}",
        f"- Critical false negatives: {metrics['critical_false_negative_count']}",
        f"- Critical false positives: {metrics['critical_false_positive_count']}",
        "",
        "## Slot Accuracy",
        "",
    ]
    for slot, item in metrics["slot_accuracy"].items():
        lines.append(f"- {slot}: {fmt(item['accuracy'])} ({item['correct']}/{item['total']})")
    lines.extend(["", "## Intent Type Success", ""])
    for intent_type, item in metrics["per_intent_type"].items():
        lines.append(f"- {intent_type}: {fmt(item['success_rate'])} ({item['success']}/{item['gold']})")
    lines.extend(["", "## First Errors", ""])
    for err in metrics["errors"][:30]:
        lines.append(f"- {err.get('utterance_id')} {err.get('error_type')}: {err.get('text', '')}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--scope", default=",".join(DEFAULT_SCOPE))
    args = parser.parse_args()

    schema = json.load(open(args.schema, encoding="utf-8"))
    metrics = score(read_jsonl(Path(args.gold)), read_jsonl(Path(args.pred)), schema, scope_set(args.scope))
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(Path(args.report), metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "errors"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
