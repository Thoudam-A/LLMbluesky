#!/usr/bin/env python3
"""Score ATC intent predictions against reviewed gold annotations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SPACE_RE = re.compile(r"\s+")
FILLERS = {"啊", "呢", "嗯", "呃", "的"}
UNIT_ALIASES = {
    "knots": "kt",
    "kts": "kt",
    "knot": "kt",
    "meters": "m",
    "meter": "m",
    "degrees": "degree",
    "deg": "degree",
}


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
    if slot in {"frequency"}:
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
    if slot == "callsign":
        return norm_text(value)
    if slot == "runway":
        return norm_text(value).upper()
    return norm_text(value)


def slot_match(gold: dict[str, Any], pred: dict[str, Any], slot: str) -> bool:
    return norm_slot(gold, slot) == norm_slot(pred, slot)


def required_slots(intent: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    intent_type = intent.get("intent_type")
    spec = schema.get("intent_types", {}).get(intent_type, {})
    slots = list(spec.get("required_slots", []))
    conditional = spec.get("conditional_required_slots", {})
    if intent_type == "heading_change":
        if intent.get("target_value") not in {None, ""}:
            for slot in conditional.get("explicit_heading_or_angle", []):
                if slot not in slots:
                    slots.append(slot)
    return slots


def candidate_score(gold: dict[str, Any], pred: dict[str, Any], schema: dict[str, Any]) -> tuple[int, int, int, int]:
    slots = required_slots(gold, schema)
    callsign = int(slot_match(gold, pred, "callsign"))
    intent_type = int(slot_match(gold, pred, "intent_type"))
    action = int(slot_match(gold, pred, "action"))
    slot_hits = sum(1 for slot in slots if slot_match(gold, pred, slot))
    return (intent_type + action + callsign, slot_hits, intent_type, action)


def frame_correct(gold: dict[str, Any], pred: dict[str, Any], schema: dict[str, Any]) -> bool:
    for slot in required_slots(gold, schema):
        if not slot_match(gold, pred, slot):
            return False
    return True


def score(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
    pred_by_utt = {row.get("utterance_id"): row for row in pred_rows}
    evaluated_gold = 0
    intent_type_correct = 0
    action_correct = 0
    frame_correct_count = 0
    false_negative = 0
    false_positive = 0
    skipped_gold_rows = 0
    pending_gold_rows = 0
    invalid_pred_rows = 0
    errors: list[dict[str, Any]] = []
    per_type: dict[str, dict[str, int]] = {}
    slot_stats: dict[str, dict[str, int]] = {}

    for gold_row in gold_rows:
        status = gold_row.get("annotation_status")
        gold_intents = gold_row.get("intents", [])
        if status == "pending":
            pending_gold_rows += 1
            continue
        if status == "skip" or not gold_intents:
            skipped_gold_rows += 1
            continue
        if status != "done":
            pending_gold_rows += 1
            continue

        pred_row = pred_by_utt.get(gold_row.get("utterance_id"), {})
        pred_intents = pred_row.get("intents", [])
        if not isinstance(pred_intents, list):
            pred_intents = []
            invalid_pred_rows += 1
        unused = set(range(len(pred_intents)))

        for gold in gold_intents:
            if not isinstance(gold, dict):
                continue
            evaluated_gold += 1
            intent_type = str(gold.get("intent_type") or "<missing>")
            per_type.setdefault(intent_type, {"gold": 0, "intent_type_correct": 0, "action_correct": 0, "frame_correct": 0})
            per_type[intent_type]["gold"] += 1

            best_idx = None
            best_score = (-1, -1, -1, -1)
            for idx in unused:
                pred = pred_intents[idx]
                if not isinstance(pred, dict):
                    continue
                current = candidate_score(gold, pred, schema)
                if current > best_score:
                    best_score = current
                    best_idx = idx

            if best_idx is None:
                false_negative += 1
                errors.append({"utterance_id": gold_row.get("utterance_id"), "error_type": "missing_intent", "gold": gold})
                continue

            pred = pred_intents[best_idx]
            unused.remove(best_idx)
            type_ok = slot_match(gold, pred, "intent_type")
            action_ok = type_ok and slot_match(gold, pred, "action")
            frame_ok = action_ok and frame_correct(gold, pred, schema)

            intent_type_correct += int(type_ok)
            action_correct += int(action_ok)
            frame_correct_count += int(frame_ok)
            per_type[intent_type]["intent_type_correct"] += int(type_ok)
            per_type[intent_type]["action_correct"] += int(action_ok)
            per_type[intent_type]["frame_correct"] += int(frame_ok)

            for slot in required_slots(gold, schema):
                slot_stats.setdefault(slot, {"total": 0, "correct": 0})
                slot_stats[slot]["total"] += 1
                slot_stats[slot]["correct"] += int(slot_match(gold, pred, slot))

            if not frame_ok:
                if not type_ok:
                    error_type = "wrong_intent_type"
                elif not slot_match(gold, pred, "action"):
                    error_type = "wrong_action"
                elif not slot_match(gold, pred, "callsign"):
                    error_type = "wrong_callsign"
                else:
                    error_type = "wrong_slot_value"
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
        for idx in unused:
            errors.append(
                {
                    "utterance_id": gold_row.get("utterance_id"),
                    "annotation_id": gold_row.get("annotation_id"),
                    "error_type": "extra_intent",
                    "text": gold_row.get("text"),
                    "prediction": pred_intents[idx],
                }
            )

    def ratio(num: int, den: int) -> float | None:
        return None if den == 0 else num / den

    return {
        "gold_rows": len(gold_rows),
        "prediction_rows": len(pred_rows),
        "pending_gold_rows": pending_gold_rows,
        "skipped_gold_rows": skipped_gold_rows,
        "evaluated_gold_intents": evaluated_gold,
        "intent_type_accuracy": ratio(intent_type_correct, evaluated_gold),
        "action_accuracy": ratio(action_correct, evaluated_gold),
        "frame_accuracy": ratio(frame_correct_count, evaluated_gold),
        "false_negative_count": false_negative,
        "false_positive_count": false_positive,
        "invalid_prediction_rows": invalid_pred_rows,
        "per_intent_type": {
            key: {
                **value,
                "frame_accuracy": ratio(value["frame_correct"], value["gold"]),
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


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# ATC Intent Scoring Report",
        "",
        "## Summary",
        "",
        f"- Gold rows: {metrics['gold_rows']}",
        f"- Prediction rows: {metrics['prediction_rows']}",
        f"- Pending gold rows ignored: {metrics['pending_gold_rows']}",
        f"- Skipped gold rows ignored: {metrics['skipped_gold_rows']}",
        f"- Evaluated gold intents: {metrics['evaluated_gold_intents']}",
        f"- Intent type accuracy: {fmt(metrics['intent_type_accuracy'])}",
        f"- Action accuracy: {fmt(metrics['action_accuracy'])}",
        f"- Frame accuracy: {fmt(metrics['frame_accuracy'])}",
        f"- False negatives: {metrics['false_negative_count']}",
        f"- False positives: {metrics['false_positive_count']}",
        "",
    ]
    if metrics["evaluated_gold_intents"] == 0:
        lines.extend(
            [
                "No official score was computed because the gold file has no reviewed `annotation_status=done` intents.",
                "",
            ]
        )
    lines.extend(["## Slot Accuracy", ""])
    for slot, item in metrics["slot_accuracy"].items():
        lines.append(f"- {slot}: {fmt(item['accuracy'])} ({item['correct']}/{item['total']})")
    lines.extend(["", "## Intent Type Frame Accuracy", ""])
    for intent_type, item in metrics["per_intent_type"].items():
        lines.append(f"- {intent_type}: {fmt(item['frame_accuracy'])} ({item['frame_correct']}/{item['gold']})")
    lines.extend(["", "## First Errors", ""])
    for err in metrics["errors"][:30]:
        lines.append(f"- {err.get('utterance_id')} {err.get('error_type')}: {err.get('text', '')}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="/home/cjj/atc_intent_eval/data/gold/annotation_sample_500.jsonl")
    parser.add_argument("--pred", default="/home/cjj/atc_intent_eval/outputs/qwen3_4b/predictions.jsonl")
    parser.add_argument("--schema", default="/home/cjj/atc_intent_eval/schemas/atc_intent_schema.json")
    parser.add_argument("--out-json", default="/home/cjj/atc_intent_eval/outputs/qwen3_4b/score.json")
    parser.add_argument("--report", default="/home/cjj/atc_intent_eval/reports/qwen3_4b_score_report.md")
    args = parser.parse_args()

    schema = json.load(open(args.schema, encoding="utf-8"))
    metrics = score(read_jsonl(Path(args.gold)), read_jsonl(Path(args.pred)), schema)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(Path(args.report), metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "errors"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
