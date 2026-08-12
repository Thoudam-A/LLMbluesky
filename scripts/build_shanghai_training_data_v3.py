#!/usr/bin/env python3
"""Build semantically corrected, time-aware Shanghai training records v3.

The builder consumes the independently validated v2 lossless dataset. It does
not overwrite v2. Current-instruction text remains in a separate label audit;
the assembled model input contains only causal state, prior instructions,
causally available plans, time-aware route context, and causal traffic.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow

from build_historical_habit_events import ZH_NUMBER_CHARS, chinese_integer
from build_shanghai_lossless_training_data_v2 import (
    distance_nm,
    segment_projection_nm,
)


VERSION = "shanghai_training_data_builder_v3.0"
ROUTE_COLUMNS_V2 = {
    "route_point_count", "route_resolved_point_count", "route_resolved_fraction",
    "active_leg_method", "active_leg_previous_fix", "active_leg_next_fix",
    "active_leg", "active_leg_cross_track_nm", "active_leg_fraction",
    "distance_to_next_fix_nm", "estimated_seconds_to_next_fix",
    "raw_eto_delta_to_next_fix_sec", "next_planned_level_m",
    "next_planned_level_source_unit", "next_planned_level_parse_status",
    "ispass_first_unpassed_index", "ispass_self_leg", "ispass_time_stale",
}
STATE_INPUT_COLUMNS = [
    "inside_sector", "sector_hit", "latitude", "longitude", "altitude_m",
    "ground_speed_kt", "track_heading_deg", "vertical_rate_mps",
    "time_from_sector_entry_sec", "aircraft_type", "wake_turbulence_category",
]
PLAN_INPUT_COLUMNS = [
    "plan_features_causally_available", "plan_adep", "plan_ades", "sid", "star",
    "departure_runway", "arrival_runway", "requested_level_m",
    "requested_level_source_unit", "requested_level_parse_status", "planned_speed_kt",
]
OPERATIONAL_CUE_PATTERNS = {
    "direct_to": r"直飞",
    "route_offset": r"偏置|航路左侧|航路右侧",
    "heading_or_turn": r"航向|左转|右转",
    "crossing_condition": r"交叉|通过.+以后|过.+以后",
    "expedite_or_rate": r"上升率|下降率|尽快|加速爬升|加速下降",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_iso_epoch(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def strict_bool(value: Any) -> bool:
    """Treat null/NaN as false instead of relying on bool(np.nan) == True."""
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_parquet(path: Path, rows: list[dict[str, Any]], empty_columns: list[str]) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame({column: pd.Series(dtype="string") for column in empty_columns})
    frame.to_parquet(path, index=False)


def boundary_suffix(transcript: str, raw_span: str) -> str | None:
    if not raw_span:
        return None
    if raw_span.rstrip().endswith("以上"):
        return "above"
    if raw_span.rstrip().endswith("以下"):
        return "below"
    start = 0
    while True:
        start = transcript.find(raw_span, start)
        if start < 0:
            break
        suffix = transcript[start + len(raw_span): start + len(raw_span) + 4]
        if suffix.startswith("以上"):
            return "above"
        if suffix.startswith("以下"):
            return "below"
        start += len(raw_span)
    return None


def extract_vertical_rate_constraints(transcript: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    number = rf"[0-9{ZH_NUMBER_CHARS}]{{1,8}}"
    pattern = re.compile(
        rf"(?P<kind>上升率|下降率)\s*(?P<value>{number})\s*(?P<unit>英尺|米)?\s*(?P<bound>以上|以下)?"
    )
    constraints: list[dict[str, Any]] = []
    for match in pattern.finditer(transcript):
        parsed = chinese_integer(match.group("value"))
        if parsed is None:
            continue
        unit_raw = match.group("unit")
        unit = (
            "m/min" if unit_raw == "米"
            else "ft/min" if unit_raw == "英尺"
            else config["label_semantics"]["vertical_rate_default_unit"]
        )
        bound = match.group("bound")
        constraints.append({
            "kind": "climb_rate" if match.group("kind") == "上升率" else "descent_rate",
            "operator": ">=" if bound == "以上" else "<=" if bound == "以下" else "=",
            "target_value": parsed.value,
            "unit": unit,
            "unit_inferred": unit_raw is None,
            "raw_span": match.group(0),
        })
    return constraints


def operational_cues(transcript: str) -> list[str]:
    return [name for name, pattern in OPERATIONAL_CUE_PATTERNS.items() if re.search(pattern, transcript)]


def semantic_label(instruction: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    transcript = str(instruction.get("transcript") or "")
    raw_span = str(instruction.get("raw_span") or "")
    family = instruction.get("intent_family")
    original_action = instruction.get("action")
    action = original_action
    target = instruction.get("target_value")
    unit = instruction.get("unit")
    operator = instruction.get("constraint_operator") or ("=" if target is not None else None)
    target_required = target is not None
    target_semantics = instruction.get("target_semantics") or (
        "exact_value" if target is not None else "categorical_no_numeric_target"
    )
    correction_reasons: list[str] = []
    expanded_raw_span = raw_span
    speed_basis = None

    if family == "speed":
        speed_basis = (
            "indicated_airspeed_explicit"
            if config["label_semantics"]["explicit_ias_marker"] in transcript
            else config["label_semantics"]["speed_default_basis"]
        )
        if "速度正常" in transcript or "表速正常" in transcript:
            action = config["label_semantics"]["speed_normal_action"]
            target = None
            unit = None
            operator = None
            target_required = False
            target_semantics = "cancel_active_speed_constraint"
            expanded_raw_span = "表速正常" if "表速正常" in transcript else "速度正常"
            correction_reasons.append("speed_normal_is_valid_non_numeric_command")
        else:
            bound = boundary_suffix(transcript, raw_span)
            if bound == "above":
                operator = config["label_semantics"]["speed_above_operator"]
                action = (
                    "increase_to_min_speed"
                    if original_action == "increase_speed" or "增速" in raw_span or "加速" in raw_span
                    else "maintain_min_speed"
                )
                target_semantics = "lower_bound"
                expanded_raw_span = raw_span if raw_span.rstrip().endswith("以上") else raw_span + "以上"
                correction_reasons.append("preserved_speed_lower_bound_modifier")
            elif bound == "below":
                operator = config["label_semantics"]["speed_below_operator"]
                action = (
                    "reduce_to_max_speed"
                    if original_action == "reduce_speed" or "减速" in raw_span
                    else "maintain_max_speed"
                )
                target_semantics = "upper_bound"
                expanded_raw_span = raw_span if raw_span.rstrip().endswith("以下") else raw_span + "以下"
                correction_reasons.append("preserved_speed_upper_bound_modifier")
    elif family == "altitude":
        target_required = True

    vertical_constraints = extract_vertical_rate_constraints(transcript, config)
    cues = operational_cues(transcript)
    return {
        "intent_family": family,
        "action": action,
        "constraint_operator": operator,
        "target_value": target,
        "target_unit": unit,
        "target_required": target_required,
        "target_semantics": target_semantics,
        "speed_command_basis": speed_basis,
        "vertical_rate_constraints": vertical_constraints,
        "operational_cues": cues,
        "expanded_raw_span": expanded_raw_span,
        "semantic_correction_applied": bool(correction_reasons),
        "semantic_correction_reasons": correction_reasons,
        "original_action": original_action,
        "original_target_value": instruction.get("target_value"),
        "original_unit": instruction.get("unit"),
        "original_raw_span": raw_span,
    }


def route_context_v3(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    state = record.get("pre_command_state_raw")
    plan_safe = bool((record.get("quality") or {}).get("plan_features_causally_available"))
    result = {
        "route_feature_mask": False,
        "route_context_confidence": "none",
        "active_leg_method": "unavailable_plan" if not plan_safe else "unresolved",
        "active_leg_previous_fix": None,
        "active_leg_next_fix": None,
        "active_leg": None,
        "active_leg_cross_track_nm": None,
        "active_leg_fraction": None,
        "distance_to_next_fix_nm": None,
        "time_to_next_fix_sec": None,
        "raw_eto_delta_to_next_fix_sec": None,
        "next_planned_level_m": None,
        "next_planned_level_source_unit": None,
        "next_planned_level_parse_status": None,
        "plan_snapshot_age_sec": None,
        "plan_snapshot_fresh": False,
        "ispass_eto_conflict": False,
        "route_point_count": 0,
        "route_resolved_point_count": 0,
        "route_resolved_fraction": 0.0,
    }
    if not plan_safe or not state:
        return result
    points = record.get("route_points_raw_and_resolved") or []
    result["route_point_count"] = len(points)
    resolved_count = sum(point.get("resolved_latitude") is not None for point in points)
    result["route_resolved_point_count"] = resolved_count
    result["route_resolved_fraction"] = 0.0 if not points else resolved_count / len(points)
    state_epoch = float(state["event_time_epoch"])
    plan_epoch = parse_iso_epoch(state.get("plan_filtim_utc"))
    if plan_epoch is not None:
        age = state_epoch - plan_epoch
        result["plan_snapshot_age_sec"] = age
        result["plan_snapshot_fresh"] = 0 <= age <= float(config["route"]["maximum_plan_snapshot_age_sec"])
    if plan_epoch is None:
        result["active_leg_method"] = "unknown_plan_snapshot_time"
        return result
    if not result["plan_snapshot_fresh"]:
        result["active_leg_method"] = "stale_plan_snapshot"
        return result

    tolerance = float(config["route"]["eto_stale_tolerance_sec"])
    first_unpassed = next(
        (i for i, point in enumerate(points) if str(point.get("ispass_raw") or "").upper() != "Y"),
        None,
    )
    if first_unpassed is not None:
        first_delta = points[first_unpassed].get("eto_delta_sec")
        result["ispass_eto_conflict"] = first_delta is not None and float(first_delta) < -tolerance

    selected_index: int | None = None
    method = None
    confidence = "none"
    cross_track = None
    fraction = None
    distance_to_next = None
    geometric_candidates: list[tuple[float, int, float, float]] = []
    latitude = float(state["latitude"])
    longitude = float(state["longitude"])
    for index in range(len(points) - 1):
        a, b = points[index], points[index + 1]
        coords = (
            a.get("resolved_latitude"), a.get("resolved_longitude"),
            b.get("resolved_latitude"), b.get("resolved_longitude"),
        )
        if any(value is None for value in coords):
            continue
        leg_length = distance_nm(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]))
        if leg_length > float(config["route"]["maximum_geometric_leg_length_nm"]):
            continue
        b_delta = b.get("eto_delta_sec")
        a_delta = a.get("eto_delta_sec")
        if b_delta is not None and float(b_delta) < -tolerance:
            continue
        if a_delta is not None and float(a_delta) > tolerance:
            continue
        projection = segment_projection_nm(
            latitude, longitude, float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
        )
        if projection[0] <= float(config["route"]["maximum_geometric_cross_track_nm"]):
            geometric_candidates.append((projection[0], index, projection[1], projection[2]))
    if geometric_candidates:
        cross_track, selected_index, fraction, distance_to_next = min(geometric_candidates)
        method = "geometric_temporal_consistent"
        confidence = "high"
    else:
        bracket_candidates = []
        for index in range(len(points) - 1):
            a_delta = points[index].get("eto_delta_sec")
            b_delta = points[index + 1].get("eto_delta_sec")
            if a_delta is None or b_delta is None:
                continue
            if float(a_delta) <= 0 < float(b_delta):
                bracket_candidates.append((float(b_delta), index))
        if bracket_candidates:
            _, selected_index = min(bracket_candidates)
            method = "eto_time_bracket"
            confidence = "medium"
        elif first_unpassed is not None:
            first_delta = points[first_unpassed].get("eto_delta_sec")
            if first_delta is None or float(first_delta) >= -tolerance:
                selected_index = max(0, first_unpassed - 1)
                method = "ispass_recent_fallback"
                confidence = "low"

    if selected_index is None or not points:
        return result
    next_index = min(selected_index + 1, len(points) - 1)
    previous = points[selected_index]
    next_point = points[next_index]
    previous_fix = previous.get("ptid_raw")
    next_fix = next_point.get("ptid_raw")
    if distance_to_next is None and next_point.get("resolved_latitude") is not None:
        distance_to_next = distance_nm(
            latitude, longitude,
            float(next_point["resolved_latitude"]), float(next_point["resolved_longitude"]),
        )
    raw_delta = next_point.get("eto_delta_sec")
    result.update({
        "route_feature_mask": True,
        "route_context_confidence": confidence,
        "active_leg_method": method,
        "active_leg_previous_fix": previous_fix,
        "active_leg_next_fix": next_fix,
        "active_leg": None if previous_fix is None or next_fix is None else f"{previous_fix}->{next_fix}",
        "active_leg_cross_track_nm": cross_track,
        "active_leg_fraction": fraction,
        "distance_to_next_fix_nm": distance_to_next,
        "time_to_next_fix_sec": None if raw_delta is None else max(0.0, float(raw_delta)),
        "raw_eto_delta_to_next_fix_sec": raw_delta,
        "next_planned_level_m": next_point.get("planned_level_m"),
        "next_planned_level_source_unit": next_point.get("planned_level_source_unit"),
        "next_planned_level_parse_status": next_point.get("planned_level_parse_status"),
    })
    return result


def causal_other_plan(audit_row: pd.Series) -> tuple[bool, dict[str, Any]]:
    available = strict_bool(audit_row.get("other_raw_plan_available_at_point"))
    plan_epoch = parse_iso_epoch(audit_row.get("other_raw_plan_filtim_utc"))
    other_epoch = audit_row.get("other_event_time_epoch")
    safe = available and plan_epoch is not None and other_epoch is not None and plan_epoch <= float(other_epoch) + 1e-9
    values = {
        "other_plan_adep": audit_row.get("other_raw_plan_adep") if safe else None,
        "other_plan_ades": audit_row.get("other_raw_plan_ades") if safe else None,
        "other_sid": audit_row.get("other_raw_sid") if safe else None,
        "other_star": audit_row.get("other_raw_star") if safe else None,
        "other_departure_runway": audit_row.get("other_raw_departure_runway") if safe else None,
        "other_arrival_runway": audit_row.get("other_raw_arrival_runway") if safe else None,
    }
    return safe, values


def state_tier_v3(main_row: pd.Series, label: dict[str, Any]) -> tuple[str, list[str]]:
    issues = json.loads(main_row.get("quality_issues_json") or "[]")
    if label["target_semantics"] == "cancel_active_speed_constraint":
        issues = [issue for issue in issues if issue != "missing_numeric_target"]
    old_tier = str(main_row.get("structural_quality_tier"))
    if old_tier == "C" and not issues:
        return "A", issues
    return old_tier, issues


def model_history_item(previous: dict[str, Any], current_epoch: float, config: dict[str, Any]) -> dict[str, Any]:
    label = semantic_label(previous, config)
    transcript = str(previous.get("transcript") or "")
    previous_rate_constraints = []
    for constraint in label["vertical_rate_constraints"]:
        previous_constraint = dict(constraint)
        previous_constraint["previous_raw_span"] = previous_constraint.pop("raw_span", None)
        previous_rate_constraints.append(previous_constraint)
    return {
        "previous_reference_event_id": previous.get("reference_event_id"),
        "previous_event_time_epoch": float(previous["event_time_epoch"]),
        "seconds_since_previous_event": current_epoch - float(previous["event_time_epoch"]),
        "intent_family": label["intent_family"],
        "action": label["action"],
        "constraint_operator": label["constraint_operator"],
        "target_value": label["target_value"],
        "target_unit": label["target_unit"],
        "target_semantics": label["target_semantics"],
        "full_previous_transcript": transcript,
        "previous_operational_cues": operational_cues(transcript),
        "previous_vertical_rate_constraints": previous_rate_constraints,
    }


def active_clearance_context(history: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "active_altitude_constraint": None,
        "active_speed_constraint": None,
        "normal_speed_resumed": False,
        "most_recent_speed_event": None,
        "recent_operational_cues": [],
        "recent_compound_transcripts": [],
    }
    speed_state_resolved = False
    # History is newest-first. The first altitude/speed item is therefore the
    # most recent observable clearance, which is the safest available proxy
    # for the active constraint in this transcript-only dataset.
    for item in history:
        if result["active_altitude_constraint"] is None and item["intent_family"] == "altitude":
            result["active_altitude_constraint"] = {
                key: item[key] for key in [
                    "action", "constraint_operator", "target_value", "target_unit",
                    "target_semantics", "seconds_since_previous_event",
                    "previous_reference_event_id",
                ]
            }
        if not speed_state_resolved and item["intent_family"] == "speed":
            event = {
                key: item[key] for key in [
                    "action", "constraint_operator", "target_value", "target_unit",
                    "target_semantics", "seconds_since_previous_event",
                    "previous_reference_event_id",
                ]
            }
            result["most_recent_speed_event"] = event
            if item["target_semantics"] == "cancel_active_speed_constraint":
                result["normal_speed_resumed"] = True
            else:
                result["active_speed_constraint"] = event
            speed_state_resolved = True
        if item["previous_operational_cues"]:
            result["recent_operational_cues"].extend(item["previous_operational_cues"])
            result["recent_compound_transcripts"].append(item["full_previous_transcript"])
    result["recent_operational_cues"] = sorted(set(result["recent_operational_cues"]))
    return result


def input_coverage(plan_safe: bool, route: dict[str, Any], history: list[dict[str, Any]], traffic_count: int) -> dict[str, Any]:
    missing = []
    if not plan_safe:
        missing.append("causal_flight_plan")
    if not route["route_feature_mask"]:
        missing.append("usable_route_progress")
    if not history:
        missing.append("prior_clearance_history")
    if traffic_count == 0:
        missing.append("causal_traffic_context")
    available = 4 - len(missing)
    tier = "high" if available == 4 else "medium" if available >= 2 else "low"
    return {
        "input_coverage_tier": tier,
        "available_context_groups": available,
        "total_context_groups": 4,
        "missing_context_groups": missing,
        "not_a_decision_sufficiency_label": True,
    }


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    minimum_arrow = int(config["runtime"]["minimum_pyarrow_major"])
    if int(pyarrow.__version__.split(".", 1)[0]) < minimum_arrow:
        raise SystemExit(f"pyarrow>={minimum_arrow} required; found {pyarrow.__version__}")
    v2_manifest = json.loads((args.v2_dir / "manifest.json").read_text(encoding="utf-8"))
    if v2_manifest["dataset_version"] != config["source_dataset_version"]:
        raise SystemExit("v2 source dataset version does not match v3 config")

    nested = read_jsonl(args.v2_dir / "event_state_records.jsonl")
    nested.sort(key=lambda row: (float(row["instruction"]["event_time_epoch"]), row["reference_event_id"]))
    if args.limit is not None:
        nested = nested[: args.limit]
    selected_ids = {row["reference_event_id"] for row in nested}
    main_v2 = pd.read_parquet(args.v2_dir / "model_safe_supervised_records_v2.parquet")
    main_v2 = main_v2[main_v2["reference_event_id"].isin(selected_ids)].set_index("reference_event_id")
    traffic_safe = pd.read_parquet(args.v2_dir / "model_safe_traffic_relations_v2.parquet")
    traffic_safe = traffic_safe[traffic_safe["reference_event_id"].isin(selected_ids)]
    traffic_audit = pd.read_parquet(args.v2_dir / "traffic_relations_audit_v2.parquet")
    traffic_audit = traffic_audit[
        traffic_audit["reference_event_id"].isin(selected_ids)
        & traffic_audit["relation_model_eligible"].fillna(False)
    ]
    audit_index = traffic_audit.set_index(["reference_event_id", "relation_source_row_index"], drop=False)
    traffic_groups = {ref: frame.sort_values("relation_risk_rank") for ref, frame in traffic_safe.groupby("reference_event_id")}

    all_instructions = [row["instruction"] for row in read_jsonl(args.v2_dir / "event_state_records.jsonl")]
    by_callsign: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for instruction in all_instructions:
        by_callsign[str(instruction.get("callsign") or "").upper()].append(instruction)
    for values in by_callsign.values():
        values.sort(key=lambda item: float(item["event_time_epoch"]))

    bundle_members: dict[tuple[float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in nested:
        item = record["instruction"]
        bundle_key = (
            float(item["event_time_epoch"]),
            str(item.get("callsign") or "").upper(),
            str(item.get("transcript") or ""),
        )
        bundle_members[bundle_key].append(item)
    bundle_by_reference: dict[str, dict[str, Any]] = {}
    for key, members in bundle_members.items():
        members.sort(key=lambda item: item["reference_event_id"])
        bundle_id = "bundle-" + hashlib.sha256(
            json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        member_labels = [semantic_label(item, config) for item in members]
        for item in members:
            bundle_by_reference[item["reference_event_id"]] = {
                "instruction_bundle_id": bundle_id,
                "bundle_size": len(members),
                "member_reference_event_ids": [member["reference_event_id"] for member in members],
                "member_labels": member_labels,
            }

    assembled: list[dict[str, Any]] = []
    main_rows: list[dict[str, Any]] = []
    label_audit_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    traffic_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for record in nested:
        ref = record["reference_event_id"]
        instruction = record["instruction"]
        current_epoch = float(instruction["event_time_epoch"])
        main_row = main_v2.loc[ref]
        label = semantic_label(instruction, config)
        bundle = bundle_by_reference[ref]
        route = route_context_v3(record, config)
        plan_safe = strict_bool(main_row.get("plan_features_causally_available"))

        prior = [
            item for item in by_callsign[str(instruction.get("callsign") or "").upper()]
            if 0 < current_epoch - float(item["event_time_epoch"]) <= float(config["history"]["lookback_sec"])
        ][-int(config["history"]["maximum_previous_events"]):]
        history = [model_history_item(item, current_epoch, config) for item in reversed(prior)]
        for rank, item in enumerate(history, start=1):
            item["history_rank"] = rank
        clearance = active_clearance_context(history)
        for rank, item in enumerate(history, start=1):
            history_rows.append({"reference_event_id": ref, **item})

        event_traffic: list[dict[str, Any]] = []
        group = traffic_groups.get(ref, pd.DataFrame())
        for _, traffic in group.iterrows():
            key = (ref, traffic["relation_source_row_index"])
            audit = audit_index.loc[key]
            if isinstance(audit, pd.DataFrame):
                audit = audit.iloc[-1]
            other_plan_safe, other_plan = causal_other_plan(audit)
            item = {key: json_safe(value) for key, value in traffic.to_dict().items()}
            item.update({"other_plan_causally_available": other_plan_safe, **json_safe(other_plan)})
            item["same_sid"] = bool(
                plan_safe and other_plan_safe and main_row.get("sid") is not None
                and str(main_row.get("sid")) == str(item.get("other_sid"))
            )
            item["same_star"] = bool(
                plan_safe and other_plan_safe and main_row.get("star") is not None
                and str(main_row.get("star")) == str(item.get("other_star"))
            )
            item["same_departure_runway"] = bool(
                plan_safe and other_plan_safe and main_row.get("departure_runway") is not None
                and str(main_row.get("departure_runway")) == str(item.get("other_departure_runway"))
            )
            item["same_arrival_runway"] = bool(
                plan_safe and other_plan_safe and main_row.get("arrival_runway") is not None
                and str(main_row.get("arrival_runway")) == str(item.get("other_arrival_runway"))
            )
            event_traffic.append(item)
            traffic_rows.append(item)

        traffic_summary = {
            "traffic_count": len(event_traffic),
            "inside_sector_traffic_count": sum(bool(item.get("other_inside_sector")) for item in event_traffic),
            "predicted_conflict_count": sum(bool(item.get("predicted_pair_conflict")) for item in event_traffic),
            "same_sid_traffic_count": sum(bool(item.get("same_sid")) for item in event_traffic),
            "same_star_traffic_count": sum(bool(item.get("same_star")) for item in event_traffic),
            "same_departure_runway_traffic_count": sum(bool(item.get("same_departure_runway")) for item in event_traffic),
            "nearest_horizontal_nm": min((item["current_horizontal_nm"] for item in event_traffic if item.get("current_horizontal_nm") is not None), default=None),
            "minimum_cpa_horizontal_nm": min((item["cpa_horizontal_nm"] for item in event_traffic if item.get("cpa_horizontal_nm") is not None), default=None),
            "minimum_vertical_at_cpa_m": min((item["vertical_at_cpa_m"] for item in event_traffic if item.get("vertical_at_cpa_m") is not None), default=None),
        }
        coverage = input_coverage(plan_safe, route, history, len(event_traffic))
        tier, issues = state_tier_v3(main_row, label)
        base_weight = float(config["training_weights"][f"state_tier_{tier.lower()}"])
        if label["semantic_correction_applied"]:
            base_weight *= float(config["training_weights"]["known_modifier_correction_multiplier"])
        if label["target_semantics"] == "cancel_active_speed_constraint":
            base_weight *= float(config["training_weights"]["non_numeric_categorical_multiplier"])
        use = "weak_train_core" if tier == "A" else "weak_train_weighted" if tier == "B" else "audit_only"
        if base_weight <= 0:
            use = "audit_only"

        target_state = {column: json_safe(main_row.get(column)) for column in STATE_INPUT_COLUMNS}
        target_state.update({
            "speed_observation_basis": "ground_speed",
            "speed_command_observation_comparable": False,
        })
        plan_context = {column: json_safe(main_row.get(column)) for column in PLAN_INPUT_COLUMNS}
        model_input = {
            "target_aircraft_state": target_state,
            "flight_plan_context": plan_context,
            "route_context": json_safe(route),
            "active_clearance_context": json_safe(clearance),
            "instruction_history": json_safe(history),
            "traffic_summary": json_safe(traffic_summary),
            "traffic_context": json_safe(event_traffic),
            "input_coverage": coverage,
        }
        training_control = {
            "state_quality_tier": tier,
            "state_quality_issues": issues,
            "label_quality_status": (
                "rule_corrected_known_semantics"
                if label["semantic_correction_applied"]
                else "rule_extracted_weak_label"
            ),
            # A verified controller speaker role does not verify the parsed
            # action/parameter label. These labels remain weak until reviewed.
            "label_human_verified": False,
            "speaker_role_human_verified": bool(instruction.get("speaker_role_human_verified")),
            "recommended_use": use,
            "recommended_training_weight": base_weight,
            "current_instruction_text_is_model_input": False,
        }
        assembled.append({
            "schema_version": config["dataset_version"],
            "reference_event_id": ref,
            "model_input": model_input,
            "label": {
                **{key: value for key, value in label.items() if not key.startswith("original_") and key != "expanded_raw_span"},
                "instruction_bundle": {
                    "instruction_bundle_id": bundle["instruction_bundle_id"],
                    "bundle_size": bundle["bundle_size"],
                    "member_reference_event_ids": bundle["member_reference_event_ids"],
                    "actions": [
                        {key: value for key, value in member_label.items() if not key.startswith("original_") and key != "expanded_raw_span"}
                        for member_label in bundle["member_labels"]
                    ],
                },
            },
            "training_control": training_control,
            "provenance": {
                "event_time_epoch": current_epoch,
                "callsign": instruction.get("callsign"),
                "trajectory_id": main_row.get("trajectory_id"),
                "state_event_time_epoch": main_row.get("state_event_time_epoch"),
                "source_v2_reference_event_id": ref,
                "instruction_bundle_id": bundle["instruction_bundle_id"],
            },
        })
        flat = {
            "schema_version": config["dataset_version"],
            "reference_event_id": ref,
            "instruction_bundle_id": bundle["instruction_bundle_id"],
            "instruction_bundle_size": bundle["bundle_size"],
            "event_time_epoch": current_epoch,
            "callsign": instruction.get("callsign"),
            **target_state,
            **plan_context,
            **route,
            **traffic_summary,
            "history_event_count_15min": len(history),
            "active_altitude_action": None if clearance["active_altitude_constraint"] is None else clearance["active_altitude_constraint"]["action"],
            "active_altitude_target": None if clearance["active_altitude_constraint"] is None else clearance["active_altitude_constraint"]["target_value"],
            "active_speed_action": None if clearance["active_speed_constraint"] is None else clearance["active_speed_constraint"]["action"],
            "active_speed_target": None if clearance["active_speed_constraint"] is None else clearance["active_speed_constraint"]["target_value"],
            "normal_speed_resumed": clearance["normal_speed_resumed"],
            "recent_operational_cues_json": json.dumps(clearance["recent_operational_cues"], ensure_ascii=False),
            "input_coverage_tier": coverage["input_coverage_tier"],
            "missing_context_groups_json": json.dumps(coverage["missing_context_groups"], ensure_ascii=False),
            "intent_family": label["intent_family"],
            "action": label["action"],
            "constraint_operator": label["constraint_operator"],
            "target_value": label["target_value"],
            "target_unit": label["target_unit"],
            "target_semantics": label["target_semantics"],
            "speed_command_basis": label["speed_command_basis"],
            "vertical_rate_constraints_json": json.dumps(label["vertical_rate_constraints"], ensure_ascii=False),
            "state_quality_tier": tier,
            "label_quality_status": training_control["label_quality_status"],
            "recommended_use": use,
            "recommended_training_weight": base_weight,
        }
        main_rows.append(flat)
        label_audit_rows.append({
            "reference_event_id": ref,
            "transcript": instruction.get("transcript"),
            "original_raw_span": label["original_raw_span"],
            "expanded_raw_span": label["expanded_raw_span"],
            "original_action": label["original_action"],
            "corrected_action": label["action"],
            "original_target_value": label["original_target_value"],
            "corrected_target_value": label["target_value"],
            "constraint_operator": label["constraint_operator"],
            "target_semantics": label["target_semantics"],
            "speed_command_basis": label["speed_command_basis"],
            "semantic_correction_applied": label["semantic_correction_applied"],
            "semantic_correction_reasons_json": json.dumps(label["semantic_correction_reasons"], ensure_ascii=False),
            "vertical_rate_constraints_json": json.dumps(label["vertical_rate_constraints"], ensure_ascii=False),
            "operational_cues_json": json.dumps(label["operational_cues"], ensure_ascii=False),
            "parse_review_required": instruction.get("parse_review_required"),
            "parse_rule_confidence": instruction.get("parse_rule_confidence"),
        })
        for point in record.get("route_points_raw_and_resolved") or []:
            route_rows.append({"reference_event_id": ref, **point})
        counters[f"state_tier_{tier}"] += 1
        counters[f"route_method_{route['active_leg_method']}"] += 1
        counters[f"coverage_{coverage['input_coverage_tier']}"] += 1
        counters["semantic_corrections"] += int(label["semantic_correction_applied"])
        counters["valid_non_numeric_labels"] += int(label["target_semantics"] == "cancel_active_speed_constraint")

    assembled_by_bundle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in assembled:
        assembled_by_bundle[record["provenance"]["instruction_bundle_id"]].append(record)
    bundle_training_records: list[dict[str, Any]] = []
    for bundle_id, members in assembled_by_bundle.items():
        members.sort(key=lambda item: item["provenance"]["source_v2_reference_event_id"])
        first = members[0]
        actions = first["label"]["instruction_bundle"]["actions"]
        semantic_keys = [
            (action.get("intent_family"), action.get("action"), action.get("constraint_operator"),
             action.get("target_value"), action.get("target_unit"), action.get("target_semantics"))
            for action in actions
        ]
        conflicting_alternatives = len(semantic_keys) != len(set(semantic_keys)) or any(
            semantic_keys[i][0] == semantic_keys[j][0]
            and semantic_keys[i][3] == semantic_keys[j][3]
            and semantic_keys[i] != semantic_keys[j]
            for i in range(len(semantic_keys)) for j in range(i + 1, len(semantic_keys))
        )
        weights = [float(member["training_control"]["recommended_training_weight"]) for member in members]
        bundle_weight = 0.0 if conflicting_alternatives else min(weights)
        bundle_training_records.append({
            "schema_version": config["dataset_version"],
            "instruction_bundle_id": bundle_id,
            "member_reference_event_ids": [member["provenance"]["source_v2_reference_event_id"] for member in members],
            "model_input": first["model_input"],
            "label": {"actions": actions, "action_count": len(actions)},
            "training_control": {
                "recommended_use": "audit_only" if bundle_weight <= 0 else "weak_train_multiaction",
                "recommended_training_weight": bundle_weight,
                "label_alternative_conflict": conflicting_alternatives,
                "current_instruction_text_is_model_input": False,
                "label_human_verified": False,
            },
            "provenance": {
                "event_time_epoch": first["provenance"]["event_time_epoch"],
                "callsign": first["provenance"]["callsign"],
                "trajectory_id": first["provenance"]["trajectory_id"],
            },
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "assembled_training_records_v3.jsonl": args.output_dir / "assembled_training_records_v3.jsonl",
        "instruction_bundle_training_records_v3.jsonl": args.output_dir / "instruction_bundle_training_records_v3.jsonl",
        "supervised_main_v3.parquet": args.output_dir / "supervised_main_v3.parquet",
        "traffic_context_v3.parquet": args.output_dir / "traffic_context_v3.parquet",
        "instruction_history_v3.parquet": args.output_dir / "instruction_history_v3.parquet",
        "semantic_label_audit_v3.parquet": args.output_dir / "semantic_label_audit_v3.parquet",
        "route_points_audit_v3.parquet": args.output_dir / "route_points_audit_v3.parquet",
    }
    write_jsonl(paths["assembled_training_records_v3.jsonl"], assembled)
    write_jsonl(paths["instruction_bundle_training_records_v3.jsonl"], bundle_training_records)
    write_parquet(paths["supervised_main_v3.parquet"], main_rows, ["schema_version", "reference_event_id"])
    write_parquet(paths["traffic_context_v3.parquet"], traffic_rows, ["reference_event_id", "relation_risk_rank"])
    write_parquet(paths["instruction_history_v3.parquet"], history_rows, ["reference_event_id", "history_rank"])
    write_parquet(paths["semantic_label_audit_v3.parquet"], label_audit_rows, ["reference_event_id", "transcript"])
    write_parquet(paths["route_points_audit_v3.parquet"], route_rows, ["reference_event_id", "route_point_index"])

    output_hashes = {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in paths.items()}
    summary = {
        "builder_version": VERSION,
        "dataset_version": config["dataset_version"],
        "source_dataset_version": config["source_dataset_version"],
        "scope": {
            "development_local_date": config["development_local_date"],
            "reserved_test_local_date": config["reserved_test_local_date"],
            "reserved_test_labels_loaded": False,
            "limited_run": args.limit is not None,
            "limit": args.limit,
        },
        "counts": {
            "reference_records_preserved": len(assembled),
            "instruction_bundle_training_units": len(bundle_training_records),
            "multi_action_bundles": sum(len(record["label"]["actions"]) > 1 for record in bundle_training_records),
            "conflicting_bundle_labels_audit_only": sum(record["training_control"]["label_alternative_conflict"] for record in bundle_training_records),
            "main_rows": len(main_rows),
            "traffic_rows": len(traffic_rows),
            "history_rows": len(history_rows),
            "label_audit_rows": len(label_audit_rows),
            "route_point_rows": len(route_rows),
            **dict(sorted(counters.items())),
        },
        "semantic_improvements": {
            "speed_boundary_operators_preserved": True,
            "speed_normal_is_valid_non_numeric_command": True,
            "vertical_rate_constraints_preserved": True,
            "current_instruction_text_separated_from_model_input": True,
            "prior_full_transcript_preserved_as_causal_history": True,
        },
        "route_improvements": {
            "geometric_requires_temporal_consistency": True,
            "eto_time_bracket_fallback": True,
            "stale_ispass_rejected": True,
            "plan_snapshot_age_recorded": True,
            "route_confidence_and_mask_recorded": True,
        },
        "input_improvements": {
            "active_altitude_and_speed_clearance_context": True,
            "compound_prior_instruction_cues_preserved": True,
            "other_aircraft_plan_is_causally_gated": True,
            "same_sid_star_runway_relations": True,
            "ground_speed_not_treated_as_commanded_ias": True,
            "all_causal_traffic_relations_preserved": True,
        },
        "warnings": [
            "All current labels remain rule-derived weak labels; none are human gold.",
            "Input coverage tier is not a decision sufficiency or correctness label.",
            "ETO time-bracket route context is a medium-confidence fallback, not observed waypoint passage.",
            "Commanded speed basis is unavailable unless explicitly stated; ground speed is not IAS.",
            "Compound operational cues preserve text but are not fully normalized into executable clearances.",
            "History is limited to previously extracted altitude/speed reference events; unextracted heading or lateral clearances may be absent.",
        ],
        "schema_roles": {
            "assembled_training_records_v3.jsonl": {
                "model_input": "causal inference input only",
                "label": "current weak supervision target; never feed as input",
                "training_control": "filtering and sample weighting metadata",
                "provenance": "audit metadata; exclude identifiers from training input",
            },
            "instruction_bundle_training_records_v3.jsonl": {
                "model_input": "recommended causal training input",
                "label": "all actions from one controller utterance; recommended multi-action target",
                "training_control": "conflicting alternative parses are audit_only with zero weight",
                "provenance": "audit metadata; exclude identifiers from training input",
            },
            "supervised_main_v3.parquet": "flattened supervised table containing both inputs and labels; split columns explicitly before training",
            "semantic_label_audit_v3.parquet": "contains current transcript and must never be used as model input",
            "traffic_context_v3.parquet": "one-to-many causal traffic input table",
            "instruction_history_v3.parquet": "causal prior-instruction input table",
            "route_points_audit_v3.parquet": "route audit only; not a default model input",
        },
        "input": {
            "v2_dir": str(args.v2_dir.resolve()),
            "v2_manifest_sha256": sha256_file(args.v2_dir / "manifest.json"),
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
            "builder_script": str(Path(__file__).resolve()),
            "builder_script_sha256": sha256_file(Path(__file__).resolve()),
            "runtime": {"python": sys.version, "pyarrow": pyarrow.__version__, "pandas": pd.__version__},
        },
        "outputs": output_hashes,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
