#!/usr/bin/env python3
"""Score controller-command imitation with order-preserving one-to-one matching."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from seu_imitation_common import (
    event_epoch,
    family_for_intent,
    finite_or_none,
    jsonl_rows,
    normalize_callsign,
    target_matches,
)


METRIC_VERSION = "controller_imitation_sequence_v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--system-outputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--match-window-sec", type=float, default=60.0)
    parser.add_argument("--system-name", default="current_decision_system")
    parser.add_argument(
        "--families",
        default="altitude,speed,heading",
        help="Comma-separated evaluation scope: altitude,speed,heading.",
    )
    parser.add_argument(
        "--reference-scope",
        choices=("all", "inside_sector", "outside_sector"),
        default="all",
        help="Optionally filter references by target_state.inside_sector.",
    )
    parser.add_argument("--min-state-altitude-m", type=float)
    parser.add_argument("--max-state-altitude-m", type=float)
    return parser.parse_args()


def canonicalize(row: dict[str, Any], index: int, source: str) -> dict[str, Any] | None:
    callsign = normalize_callsign(row.get("callsign") or row.get("acid") or row.get("flight_id"))
    family = family_for_intent(
        row.get("intent_family")
        or row.get("intent_type")
        or row.get("command_type")
        or row.get("type")
    )
    if not callsign or not family:
        return None
    result = dict(row)
    result["_index"] = index
    result["_source"] = source
    result["_callsign"] = callsign
    result["_family"] = family
    result["_epoch"] = event_epoch(row)
    return result


def better(candidate: tuple[int, float], current: tuple[int, float]) -> bool:
    return candidate[0] > current[0] or (
        candidate[0] == current[0] and candidate[1] < current[1] - 1e-12
    )


def align_sequence(
    references: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    window_sec: float,
    require_parameter_match: bool = False,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """LCS-style alignment: maximize matches, then minimize total time error."""

    n, m = len(references), len(predictions)
    dp = [[(0, 0.0) for _ in range(m + 1)] for _ in range(n + 1)]
    back = [["" for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        back[i][0] = "skip_ref"
    for j in range(1, m + 1):
        back[0][j] = "skip_pred"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = dp[i - 1][j]
            action = "skip_ref"
            if better(dp[i][j - 1], best):
                best = dp[i][j - 1]
                action = "skip_pred"
            ref = references[i - 1]
            pred = predictions[j - 1]
            delta = abs(ref["_epoch"] - pred["_epoch"])
            parameter_ok = target_matches(ref, pred) is True
            reference_action = str(ref.get("action") or "").lower()
            prediction_action = str(pred.get("action") or "").lower()
            action_ok = True
            if ref["_family"] == "altitude":
                if reference_action in {"climb", "descend"} and prediction_action in {"climb", "descend"}:
                    action_ok = reference_action == prediction_action
            if (
                ref["_family"] == pred["_family"]
                and delta <= window_sec
                and (not require_parameter_match or (parameter_ok and action_ok))
            ):
                prior = dp[i - 1][j - 1]
                matched = (prior[0] + 1, prior[1] + delta)
                if better(matched, best):
                    best = matched
                    action = "match"
            dp[i][j] = best
            back[i][j] = action

    pairs: list[tuple[int, int]] = []
    matched_ref: set[int] = set()
    matched_pred: set[int] = set()
    i, j = n, m
    while i > 0 or j > 0:
        action = back[i][j]
        if action == "match":
            pairs.append((i - 1, j - 1))
            matched_ref.add(i - 1)
            matched_pred.add(j - 1)
            i -= 1
            j -= 1
        elif action == "skip_pred":
            j -= 1
        else:
            i -= 1
    pairs.reverse()
    return (
        pairs,
        [idx for idx in range(n) if idx not in matched_ref],
        [idx for idx in range(m) if idx not in matched_pred],
    )


def score(
    references_raw: list[dict[str, Any]],
    predictions_raw: list[dict[str, Any]],
    window_sec: float,
    system_name: str,
) -> dict[str, Any]:
    invalid_references = 0
    invalid_predictions = 0
    references: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(references_raw):
        canonical = canonicalize(row, index, "reference")
        if canonical is None:
            invalid_references += 1
        else:
            references.append(canonical)
    for index, row in enumerate(predictions_raw):
        try:
            canonical = canonicalize(row, index, "prediction")
        except (TypeError, ValueError):
            canonical = None
        if canonical is None:
            invalid_predictions += 1
        else:
            predictions.append(canonical)

    ref_by_call: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    pred_by_call: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in references:
        ref_by_call[row["_callsign"]].append(row)
    for row in predictions:
        pred_by_call[row["_callsign"]].append(row)
    for rows in list(ref_by_call.values()) + list(pred_by_call.values()):
        rows.sort(key=lambda row: (row["_epoch"], row["_index"]))

    matches: list[dict[str, Any]] = []
    unmatched_references: list[dict[str, Any]] = []
    unmatched_predictions: list[dict[str, Any]] = []
    per_aircraft: dict[str, dict[str, Any]] = {}
    all_callsigns = sorted(set(ref_by_call) | set(pred_by_call))
    for callsign in all_callsigns:
        refs = ref_by_call.get(callsign, [])
        preds = pred_by_call.get(callsign, [])
        pairs, ref_misses, pred_misses = align_sequence(refs, preds, window_sec)
        for ref_index, pred_index in pairs:
            ref = refs[ref_index]
            pred = preds[pred_index]
            parameter_match = target_matches(ref, pred)
            matches.append(
                {
                    "callsign": callsign,
                    "reference_event_id": ref.get("reference_event_id") or ref.get("event_id"),
                    "reference_time_utc": ref.get("event_time_utc"),
                    "prediction_event_id": pred.get("event_id") or pred.get("command_id"),
                    "prediction_time_utc": pred.get("event_time_utc") or pred.get("timestamp"),
                    "intent_family": ref["_family"],
                    "time_delta_sec": round(abs(ref["_epoch"] - pred["_epoch"]), 6),
                    "reference_action": ref.get("action"),
                    "prediction_action": pred.get("action"),
                    "reference_target": ref.get("target_value"),
                    "prediction_target": pred.get("target_value"),
                    "reference_unit": ref.get("unit"),
                    "prediction_unit": pred.get("unit"),
                    "parameter_match": parameter_match,
                }
            )
        unmatched_references.extend(
            {
                "callsign": callsign,
                "reference_event_id": refs[idx].get("reference_event_id") or refs[idx].get("event_id"),
                "event_time_utc": refs[idx].get("event_time_utc"),
                "intent_family": refs[idx]["_family"],
                "action": refs[idx].get("action"),
                "target_value": refs[idx].get("target_value"),
                "unit": refs[idx].get("unit"),
            }
            for idx in ref_misses
        )
        unmatched_predictions.extend(
            {
                "callsign": callsign,
                "prediction_event_id": preds[idx].get("event_id") or preds[idx].get("command_id"),
                "event_time_utc": preds[idx].get("event_time_utc") or preds[idx].get("timestamp"),
                "intent_family": preds[idx]["_family"],
                "action": preds[idx].get("action"),
                "target_value": preds[idx].get("target_value"),
                "unit": preds[idx].get("unit"),
            }
            for idx in pred_misses
        )
        if refs:
            per_aircraft[callsign] = {
                "reference_count": len(refs),
                "system_output_count": len(preds),
                "matched_count": len(pairs),
                "recall": round(len(pairs) / len(refs), 6),
                "precision": round(len(pairs) / len(preds), 6) if preds else None,
            }

    ref_count = len(references)
    pred_count = len(predictions)
    match_count = len(matches)
    family_reference = collections.Counter(row["_family"] for row in references)
    family_prediction = collections.Counter(row["_family"] for row in predictions)
    family_match = collections.Counter(row["intent_family"] for row in matches)
    by_family = {}
    for family in sorted(set(family_reference) | set(family_prediction)):
        by_family[family] = {
            "reference_count": family_reference[family],
            "system_output_count": family_prediction[family],
            "matched_count": family_match[family],
            "recall": (
                round(family_match[family] / family_reference[family], 6)
                if family_reference[family]
                else None
            ),
            "precision": (
                round(family_match[family] / family_prediction[family], 6)
                if family_prediction[family]
                else None
            ),
        }

    comparable = [row for row in matches if row["parameter_match"] is not None]
    parameter_hits = sum(row["parameter_match"] is True for row in comparable)
    macro_recall = (
        sum(row["recall"] for row in per_aircraft.values()) / len(per_aircraft)
        if per_aircraft
        else None
    )

    strict_ref_by_call: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in references:
        if finite_or_none(row.get("target_value")) is not None and row.get("unit"):
            strict_ref_by_call[row["_callsign"]].append(row)
    strict_match_count = 0
    strict_per_aircraft: dict[str, float] = {}
    strict_family_ref = collections.Counter()
    strict_family_match = collections.Counter()
    for callsign, refs in strict_ref_by_call.items():
        refs.sort(key=lambda row: (row["_epoch"], row["_index"]))
        preds = pred_by_call.get(callsign, [])
        pairs, _ref_misses, _pred_misses = align_sequence(
            refs,
            preds,
            window_sec,
            require_parameter_match=True,
        )
        strict_match_count += len(pairs)
        strict_per_aircraft[callsign] = len(pairs) / len(refs)
        strict_family_ref.update(row["_family"] for row in refs)
        strict_family_match.update(refs[ref_index]["_family"] for ref_index, _ in pairs)
    strict_reference_count = sum(len(rows) for rows in strict_ref_by_call.values())
    strict_macro_recall = (
        sum(strict_per_aircraft.values()) / len(strict_per_aircraft)
        if strict_per_aircraft
        else None
    )
    return {
        "metric_version": METRIC_VERSION,
        "system_name": system_name,
        "definition": {
            "primary": (
                "macro average across reference aircraft of matched historical commands / "
                "historical reference commands"
            ),
            "matching": (
                "same standardized callsign, same intent family, absolute time difference within "
                f"{window_sec:g}s, order-preserving one-to-one alignment"
            ),
            "parameter_match": (
                "supplementary numeric target agreement using family/unit tolerances; "
                "not required for the primary action match"
            ),
        },
        "counts": {
            "reference_events": ref_count,
            "system_outputs": pred_count,
            "matched_events": match_count,
            "unmatched_references": len(unmatched_references),
            "unmatched_system_outputs": len(unmatched_predictions),
            "reference_aircraft": len(ref_by_call),
            "invalid_reference_rows": invalid_references,
            "invalid_system_output_rows": invalid_predictions,
        },
        "metrics": {
            "controller_imitation_macro_recall": round(macro_recall, 6) if macro_recall is not None else None,
            "controller_imitation_micro_recall": round(match_count / ref_count, 6) if ref_count else None,
            "system_command_precision": round(match_count / pred_count, 6) if pred_count else None,
            "command_inflation_ratio": round(pred_count / ref_count, 6) if ref_count else None,
            "parameter_accuracy_on_comparable_matches": (
                round(parameter_hits / len(comparable), 6) if comparable else None
            ),
            "parameter_comparable_match_count": len(comparable),
            "strict_parameter_macro_recall": (
                round(strict_macro_recall, 6) if strict_macro_recall is not None else None
            ),
            "strict_parameter_micro_recall": (
                round(strict_match_count / strict_reference_count, 6)
                if strict_reference_count
                else None
            ),
            "strict_parameter_reference_count": strict_reference_count,
            "strict_parameter_matched_count": strict_match_count,
        },
        "by_family": by_family,
        "strict_parameter_by_family": {
            family: {
                "reference_count": strict_family_ref[family],
                "matched_count": strict_family_match[family],
                "recall": (
                    round(strict_family_match[family] / strict_family_ref[family], 6)
                    if strict_family_ref[family]
                    else None
                ),
            }
            for family in sorted(strict_family_ref)
        },
        "per_aircraft": per_aircraft,
        "matches": matches,
        "unmatched_references": unmatched_references,
        "unmatched_system_outputs": unmatched_predictions,
        "warnings": [
            "A score is meaningful only when system outputs are produced independently in historical shadow replay.",
            "Do not call the model only at historical command times; that would leak reference timing.",
            "Report full-scope and conflict-resolution-only subsets separately when the system supports only conflict resolution.",
        ],
    }


def main() -> int:
    args = parse_args()
    families = {item.strip().lower() for item in args.families.split(",") if item.strip()}
    invalid_families = families - {"altitude", "speed", "heading"}
    if not families or invalid_families:
        raise SystemExit(f"invalid --families value: {args.families!r}")
    references = []
    for row in jsonl_rows(args.references):
        if family_for_intent(row.get("intent_family") or row.get("intent_type")) not in families:
            continue
        inside = bool((row.get("target_state") or {}).get("inside_sector"))
        if args.reference_scope == "inside_sector" and not inside:
            continue
        if args.reference_scope == "outside_sector" and inside:
            continue
        state_altitude = (row.get("target_state") or {}).get("altitude_m")
        if args.min_state_altitude_m is not None:
            if state_altitude is None or float(state_altitude) < args.min_state_altitude_m:
                continue
        if args.max_state_altitude_m is not None:
            if state_altitude is None or float(state_altitude) > args.max_state_altitude_m:
                continue
        references.append(row)
    predictions = [
        row
        for row in jsonl_rows(args.system_outputs)
        if family_for_intent(
            row.get("intent_family")
            or row.get("intent_type")
            or row.get("command_type")
            or row.get("type")
        )
        in families
    ]
    result = score(references, predictions, args.match_window_sec, args.system_name)
    result["scope_families"] = sorted(families)
    result["reference_scope"] = args.reference_scope
    result["reference_state_altitude_band_m"] = [
        args.min_state_altitude_m,
        args.max_state_altitude_m,
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: result[key] for key in ("system_name", "counts", "metrics", "by_family")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
