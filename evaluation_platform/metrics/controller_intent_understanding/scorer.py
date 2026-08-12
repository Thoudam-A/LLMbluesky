"""Score direct controller-instruction intent classification by event ID."""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any, Iterable


METRIC_ID = "controller_intent_understanding_accuracy"
METRIC_VERSION = "controller_intent_frame_v1.0"
TYPE_FAMILY = {
    "altitude_change": "altitude",
    "altitude_maintain": "altitude",
    "speed_adjust": "speed",
    "speed_procedure": "speed",
    "heading_change": "heading",
}
TOLERANCES = {
    "altitude": {"m": 100.0, "ft": 300.0, "fl": 3.0},
    "speed": {"kt": 10.0, "mps": 5.0},
    "heading": {"deg": 10.0},
}
UNIT_ALIASES = {
    "degree": "deg",
    "degrees": "deg",
    "knot": "kt",
    "knots": "kt",
    "kts": "kt",
    "meter": "m",
    "meters": "m",
    "feet": "ft",
}


def jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def keyed(rows: Iterable[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        event_id = str(row.get("event_id") or row.get("reference_event_id") or "").strip()
        if not event_id:
            raise ValueError(f"{source}:{index}: event_id is required")
        if source == "controller_instructions" and not str(row.get("instruction_text") or "").strip():
            raise ValueError(f"{source}:{index}: instruction_text is required")
        if event_id in result:
            raise ValueError(f"{source}:{index}: duplicate event_id {event_id!r}")
        result[event_id] = row
    return result


def normalized_callsign(value: Any) -> str | None:
    text = "".join(char for char in str(value or "").upper() if char.isalnum())
    if not text or not any(char.isalpha() for char in text) or not any(char.isdigit() for char in text):
        return None
    return text


def normalized_label(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return text or None


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalized_unit(value: Any) -> str | None:
    text = normalized_label(value)
    return UNIT_ALIASES.get(text, text) if text else None


def parameter_match(reference: dict[str, Any], prediction: dict[str, Any]) -> bool | None:
    ref_value = finite(reference.get("target_value"))
    if ref_value is None:
        return None
    pred_value = finite(prediction.get("target_value"))
    if pred_value is None:
        return False
    family = normalized_label(reference.get("intent_family"))
    ref_unit = normalized_unit(reference.get("unit"))
    pred_unit = normalized_unit(prediction.get("unit"))
    if not family or not ref_unit or ref_unit != pred_unit:
        return False
    tolerance = TOLERANCES.get(family, {}).get(ref_unit)
    if tolerance is None:
        return False
    difference = (
        abs((ref_value - pred_value + 180.0) % 360.0 - 180.0)
        if family == "heading"
        else abs(ref_value - pred_value)
    )
    return difference <= tolerance


def reference_valid(row: dict[str, Any]) -> bool:
    intent_type = normalized_label(row.get("intent_type"))
    family = normalized_label(row.get("intent_family"))
    return bool(
        normalized_callsign(row.get("callsign"))
        and intent_type in TYPE_FAMILY
        and family == TYPE_FAMILY.get(intent_type)
        and normalized_label(row.get("action"))
    )


def prediction_valid(row: dict[str, Any]) -> bool:
    return reference_valid(row)


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score(instructions_path: Path, references_path: Path, predictions_path: Path) -> dict[str, Any]:
    instructions = keyed(jsonl_rows(instructions_path), "controller_instructions")
    references = keyed(jsonl_rows(references_path), "intent_reference_events")
    predictions = keyed(jsonl_rows(predictions_path), "intent_predictions")

    instruction_ids = set(instructions)
    reference_ids = set(references)
    prediction_ids = set(predictions)
    paired_ids = sorted(instruction_ids & reference_ids)
    invalid_reference_ids = [event_id for event_id in paired_ids if not reference_valid(references[event_id])]
    evaluable_ids = [event_id for event_id in paired_ids if event_id not in set(invalid_reference_ids)]

    counts = collections.Counter()
    details: list[dict[str, Any]] = []
    by_type: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    prediction_type_counts = collections.Counter(
        normalized_label(row.get("intent_type")) or "invalid"
        for row in predictions.values()
    )

    for event_id in evaluable_ids:
        reference = references[event_id]
        prediction = predictions.get(event_id)
        ref_type = normalized_label(reference.get("intent_type")) or "invalid"
        summary = by_type[ref_type]
        summary["reference_count"] += 1
        if prediction is None:
            counts["missing_prediction"] += 1
            if finite(reference.get("target_value")) is not None:
                counts["parameter_comparable"] += 1
            details.append({
                "event_id": event_id,
                "instruction_text": instructions[event_id].get("instruction_text"),
                "reference": reference,
                "prediction": None,
                "callsign_match": False,
                "intent_family_match": False,
                "intent_type_match": False,
                "action_match": False,
                "parameter_match": False if finite(reference.get("target_value")) is not None else None,
                "full_match": False,
                "error": "missing_prediction",
            })
            continue

        valid_prediction = prediction_valid(prediction)
        if not valid_prediction:
            counts["invalid_prediction_total"] += 1
        callsign_hit = normalized_callsign(reference.get("callsign")) == normalized_callsign(prediction.get("callsign"))
        family_hit = normalized_label(reference.get("intent_family")) == normalized_label(prediction.get("intent_family"))
        type_hit = normalized_label(reference.get("intent_type")) == normalized_label(prediction.get("intent_type"))
        action_hit = normalized_label(reference.get("action")) == normalized_label(prediction.get("action"))
        target_hit = parameter_match(reference, prediction)
        full_hit = bool(valid_prediction and callsign_hit and family_hit and type_hit and action_hit and target_hit is not False)

        counts["callsign_hit"] += int(callsign_hit)
        counts["family_hit"] += int(family_hit)
        counts["type_hit"] += int(type_hit)
        counts["action_hit"] += int(action_hit)
        if target_hit is not None:
            counts["parameter_comparable"] += 1
            counts["parameter_hit"] += int(target_hit)
        counts["full_hit"] += int(full_hit)
        summary["prediction_count"] += 1
        summary["type_hit"] += int(type_hit)
        summary["full_hit"] += int(full_hit)

        if full_hit:
            error = "full_match"
        elif not callsign_hit:
            error = "callsign_error"
        elif not family_hit or not type_hit:
            error = "intent_type_error"
        elif not action_hit:
            error = "action_error"
        elif target_hit is False:
            error = "parameter_error"
        else:
            error = "invalid_prediction"
        counts[error] += 1
        details.append({
            "event_id": event_id,
            "instruction_text": instructions[event_id].get("instruction_text"),
            "reference": reference,
            "prediction": prediction,
            "callsign_match": callsign_hit,
            "intent_family_match": family_hit,
            "intent_type_match": type_hit,
            "action_match": action_hit,
            "parameter_match": target_hit,
            "full_match": full_hit,
            "error": error,
        })

    denominator = len(evaluable_ids)
    raw_prediction_count = len(predictions)
    exact_accuracy = safe_ratio(counts["full_hit"], denominator)
    semantic_precision = safe_ratio(counts["full_hit"], raw_prediction_count)
    semantic_f1 = (
        2 * exact_accuracy * semantic_precision / (exact_accuracy + semantic_precision)
        if exact_accuracy is not None and semantic_precision is not None and exact_accuracy + semantic_precision > 0
        else 0.0 if denominator and raw_prediction_count else None
    )
    return {
        "metric_id": METRIC_ID,
        "metric_version": METRIC_VERSION,
        "status": "complete" if denominator else "not_evaluable",
        "primary": {
            "name": "完整意图框架准确率",
            "value": None if exact_accuracy is None else round(exact_accuracy, 6),
            "unit": "%",
            "direction": "higher_is_better",
        },
        "metrics": {
            "full_intent_frame_accuracy": None if exact_accuracy is None else round(exact_accuracy, 6),
            "callsign_accuracy": None if not denominator else round(counts["callsign_hit"] / denominator, 6),
            "intent_family_accuracy": None if not denominator else round(counts["family_hit"] / denominator, 6),
            "intent_type_accuracy": None if not denominator else round(counts["type_hit"] / denominator, 6),
            "action_accuracy": None if not denominator else round(counts["action_hit"] / denominator, 6),
            "parameter_accuracy": (
                None if not counts["parameter_comparable"]
                else round(counts["parameter_hit"] / counts["parameter_comparable"], 6)
            ),
            "semantic_precision": None if semantic_precision is None else round(semantic_precision, 6),
            "semantic_f1": None if semantic_f1 is None else round(semantic_f1, 6),
            "paired_input_coverage": None if not instruction_ids else round(len(paired_ids) / len(instruction_ids), 6),
        },
        "counts": {
            "controller_instructions": len(instructions),
            "reference_events": len(references),
            "classifier_outputs": len(predictions),
            "paired_instruction_references": len(paired_ids),
            "evaluable_events": denominator,
            "full_intent_hits": counts["full_hit"],
            "missing_predictions": counts["missing_prediction"],
            "extra_predictions": len(prediction_ids - instruction_ids),
            "unpaired_instructions": len(instruction_ids - reference_ids),
            "unpaired_references": len(reference_ids - instruction_ids),
            "invalid_references": len(invalid_reference_ids),
            "invalid_predictions": counts["invalid_prediction_total"],
            "parameter_comparable_events": counts["parameter_comparable"],
        },
        "by_intent_type": {
            intent_type: {
                "reference_count": values["reference_count"],
                "prediction_count": prediction_type_counts[intent_type],
                "type_hit": values["type_hit"],
                "full_hit": values["full_hit"],
                "full_accuracy": (
                    round(values["full_hit"] / values["reference_count"], 6)
                    if values["reference_count"] else None
                ),
            }
            for intent_type, values in sorted(by_type.items())
        },
        "error_counts": {
            key: counts[key]
            for key in (
                "full_match", "missing_prediction", "invalid_prediction", "callsign_error",
                "intent_type_error", "action_error", "parameter_error"
            )
        },
        "details": details,
        "definition": {
            "pairing": "controller instruction, reference intent and prediction share the same event_id",
            "full_match": "callsign + intent_family + intent_type + action + comparable target tolerance",
            "denominator": "valid instruction-reference pairs; missing classifier output counts as incorrect",
            "parameter_tolerances": TOLERANCES,
        },
        "claim_boundary": (
            "衡量同一事件ID下管制指令文本的结构化意图分类准确率；"
            "不包含语音识别误差，也不评价指令执行效果或运行安全。"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instructions", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = score(args.instructions, args.references, args.predictions)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
