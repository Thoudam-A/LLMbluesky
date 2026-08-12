#!/usr/bin/env python3
"""Build lossless, quality-tiered Shanghai approach training-state records.

The v2 format deliberately separates raw source fields, parsed fields, quality
flags, relational traffic state and label eligibility.  No development-date
reference is silently dropped; unusable examples remain in the audit ledger.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.dataset as ds

from seu_imitation_common import jsonl_rows, write_jsonl
from shanghai_program_policy import (
    MPS_TO_KT,
    clean_category,
    parse_planned_speed_kt,
    velocity_nm_sec,
    xy_nm,
)


VERSION = "shanghai_lossless_training_builder_v2.0"
MINIMUM_PYARROW_MAJOR = 20
FEET_TO_METRES = 0.3048


def parse_flight_level(value: Any) -> dict[str, Any]:
    """Parse ICAO metric/feet level codes without discarding source semantics.

    S/M codes are tens of metres and F/A codes are hundreds of feet. Bare
    numbers retain the legacy tens-of-metres interpretation, but are explicitly
    marked as assumed rather than silently treated as an S level.
    """

    raw = json_safe(value)
    text = clean_category(value).upper().replace(" ", "")
    result = {
        "raw": raw,
        "normalized_code": None if text == "MISSING" else text,
        "value_m": None,
        "source_unit": None,
        "parse_status": "missing" if text == "MISSING" else "unparsed",
    }
    if text == "MISSING":
        return result
    prefix = text[0] if text and text[0].isalpha() else ""
    payload = text[1:] if prefix else text
    try:
        numeric = float(payload)
    except ValueError:
        result["parse_status"] = "invalid_numeric_payload"
        return result
    if not math.isfinite(numeric):
        result["parse_status"] = "non_finite_numeric_payload"
        return result
    if prefix in {"S", "M"}:
        result.update(
            value_m=numeric * 10.0,
            source_unit="tens_of_metres",
            parse_status="parsed_metric_code",
        )
    elif prefix in {"F", "A"}:
        result.update(
            value_m=numeric * 100.0 * FEET_TO_METRES,
            source_unit="hundreds_of_feet",
            parse_status="parsed_feet_code",
        )
    elif not prefix:
        result.update(
            value_m=numeric * 10.0,
            source_unit="assumed_tens_of_metres",
            parse_status="parsed_with_legacy_unit_assumption",
        )
    else:
        result["parse_status"] = f"unsupported_prefix_{prefix}"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-parquet", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    if pd.isna(value):
        return None
    return value


def local_day_bounds(local_date: str, timezone: str) -> tuple[int, int]:
    zone = ZoneInfo(timezone)
    start = dt.datetime.fromisoformat(local_date).replace(tzinfo=zone)
    return int(start.timestamp()), int((start + dt.timedelta(days=1)).timestamp())


def reference_local_date(reference: dict[str, Any], timezone: str) -> str:
    epoch = float(reference["event_time_epoch"])
    return dt.datetime.fromtimestamp(epoch, ZoneInfo(timezone)).date().isoformat()


def load_nav_index(waypoint_file: Path, airport_file: Path) -> dict[str, list[tuple[float, float, str]]]:
    index: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    with waypoint_file.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 4 or str(row[0]).lstrip().startswith("#"):
                continue
            try:
                index[str(row[0]).strip().upper()].append(
                    (float(row[2]), float(row[3]), "waypoints-old.dat")
                )
            except ValueError:
                continue
    with airport_file.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 4 or str(row[0]).lstrip().startswith("#"):
                continue
            try:
                index[str(row[0]).strip().upper()].append(
                    (float(row[2]), float(row[3]), "airports.dat")
                )
            except ValueError:
                continue
    return index


def distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    mean_lat = (lat1 + lat2) / 2.0
    x = (lon2 - lon1) * 60.0 * math.cos(math.radians(mean_lat))
    y = (lat2 - lat1) * 60.0
    return math.hypot(x, y)


def resolve_navpoint(
    point_id: Any,
    nav_index: dict[str, list[tuple[float, float, str]]],
    target_latitude: float,
    target_longitude: float,
) -> tuple[float | None, float | None, str | None]:
    values = nav_index.get(clean_category(point_id).upper(), [])
    if not values:
        return None, None, None
    selected = min(
        values,
        key=lambda item: distance_nm(
            target_latitude, target_longitude, item[0], item[1]
        ),
    )
    return selected


def segment_projection_nm(
    target_latitude: float,
    target_longitude: float,
    start_latitude: float,
    start_longitude: float,
    end_latitude: float,
    end_longitude: float,
) -> tuple[float, float, float]:
    center_lat = target_latitude
    scale_x = 60.0 * math.cos(math.radians(center_lat))
    ax = (start_longitude - target_longitude) * scale_x
    ay = (start_latitude - target_latitude) * 60.0
    bx = (end_longitude - target_longitude) * scale_x
    by = (end_latitude - target_latitude) * 60.0
    vx, vy = bx - ax, by - ay
    vv = vx * vx + vy * vy
    fraction = 0.0 if vv <= 1e-12 else max(0.0, min(1.0, -((ax * vx + ay * vy) / vv)))
    px, py = ax + fraction * vx, ay + fraction * vy
    cross_track = math.hypot(px, py)
    distance_to_end = math.hypot(bx, by)
    return cross_track, fraction, distance_to_end


def parsed_route_points(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    try:
        points = json.loads(value) if isinstance(value, str) else list(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    return [dict(point) for point in points if isinstance(point, dict)]


def parse_iso_epoch(value: Any) -> float | None:
    text = clean_category(value)
    if text == "MISSING":
        return None
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def plan_availability_at_state(state: dict[str, Any] | None) -> tuple[bool, list[str]]:
    """Return whether plan-derived features were causally available."""

    if state is None:
        return False, ["no_pre_command_state"]
    issues: list[str] = []
    if not bool(state.get("plan_available_at_point")):
        issues.append("plan_not_available_at_pre_state")
    plan_epoch = parse_iso_epoch(state.get("plan_filtim_utc"))
    state_epoch = float(state["event_time_epoch"])
    if plan_epoch is None:
        issues.append("plan_filter_time_missing")
    elif plan_epoch > state_epoch + 1e-6:
        issues.append("plan_update_after_pre_state")
    return not issues, issues


def route_context_v2(
    row: dict[str, Any],
    nav_index: dict[str, list[tuple[float, float, str]]],
    maximum_leg_distance_nm: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    points = parsed_route_points(row.get("plan_rtepts_json"))
    epoch = float(row["event_time_epoch"])
    latitude = float(row["latitude"])
    longitude = float(row["longitude"])
    speed_mps = max(0.0, float(row.get("ground_speed_mps") or 0.0))
    resolved: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        level = parse_flight_level(point.get("flight_level"))
        lat, lon, source = resolve_navpoint(
            point.get("ptid"), nav_index, latitude, longitude
        )
        eto_epoch = None
        try:
            eto_epoch = dt.datetime.fromisoformat(
                clean_category(point.get("eto_utc")).replace("Z", "+00:00")
            ).timestamp()
        except ValueError:
            pass
        resolved.append(
            {
                "route_point_index": index,
                "ptid_raw": point.get("ptid"),
                "sector_raw": point.get("sector"),
                "flight_level_raw": point.get("flight_level"),
                "eto_utc_raw": point.get("eto_utc"),
                "ispass_raw": point.get("ispass"),
                "planned_level_m": level["value_m"],
                "planned_level_source_unit": level["source_unit"],
                "planned_level_parse_status": level["parse_status"],
                "eto_epoch": eto_epoch,
                "eto_delta_sec": None if eto_epoch is None else eto_epoch - epoch,
                "resolved_latitude": lat,
                "resolved_longitude": lon,
                "coordinate_source": source,
            }
        )

    best: tuple[float, int, float, float] | None = None
    for index in range(len(resolved) - 1):
        a, b = resolved[index], resolved[index + 1]
        if any(
            value is None
            for value in (
                a["resolved_latitude"],
                a["resolved_longitude"],
                b["resolved_latitude"],
                b["resolved_longitude"],
            )
        ):
            continue
        cross_track, fraction, distance_to_end = segment_projection_nm(
            latitude,
            longitude,
            float(a["resolved_latitude"]),
            float(a["resolved_longitude"]),
            float(b["resolved_latitude"]),
            float(b["resolved_longitude"]),
        )
        candidate = (cross_track, index, fraction, distance_to_end)
        if best is None or candidate < best:
            best = candidate

    first_unpassed = next(
        (
            index
            for index, point in enumerate(resolved)
            if clean_category(point.get("ispass_raw")).upper() != "Y"
        ),
        max(0, len(resolved) - 1),
    ) if resolved else None
    ispass_previous = None if first_unpassed is None else max(0, first_unpassed - 1)
    ispass_self_leg = bool(
        resolved
        and first_unpassed is not None
        and clean_category(resolved[ispass_previous]["ptid_raw"])
        == clean_category(resolved[first_unpassed]["ptid_raw"])
    )

    method = "unresolved"
    selected_index = None
    cross_track_nm = None
    segment_fraction = None
    distance_to_next_nm = None
    if best is not None and best[0] <= maximum_leg_distance_nm:
        method = "geometric_projection"
        cross_track_nm, selected_index, segment_fraction, distance_to_next_nm = best
    elif first_unpassed is not None:
        method = "ispass_fallback"
        selected_index = max(0, first_unpassed - 1)

    previous_fix = None
    next_fix = None
    next_level = None
    next_level_unit = None
    next_level_parse_status = None
    next_eto_delta = None
    if selected_index is not None and resolved:
        previous_fix = clean_category(resolved[selected_index]["ptid_raw"])
        next_point_index = min(selected_index + 1, len(resolved) - 1)
        next_point = resolved[next_point_index]
        next_fix = clean_category(next_point["ptid_raw"])
        next_level = next_point["planned_level_m"]
        next_level_unit = next_point["planned_level_source_unit"]
        next_level_parse_status = next_point["planned_level_parse_status"]
        next_eto_delta = next_point["eto_delta_sec"]
        if distance_to_next_nm is None and next_point["resolved_latitude"] is not None:
            distance_to_next_nm = distance_nm(
                latitude,
                longitude,
                float(next_point["resolved_latitude"]),
                float(next_point["resolved_longitude"]),
            )
    estimated_seconds = None
    if distance_to_next_nm is not None and speed_mps > 1.0:
        estimated_seconds = distance_to_next_nm * 1852.0 / speed_mps

    context = {
        "route_point_count": len(resolved),
        "route_resolved_point_count": sum(
            point["resolved_latitude"] is not None for point in resolved
        ),
        "route_resolved_fraction": (
            0.0
            if not resolved
            else sum(point["resolved_latitude"] is not None for point in resolved)
            / len(resolved)
        ),
        "active_leg_method": method,
        "active_leg_previous_fix": previous_fix,
        "active_leg_next_fix": next_fix,
        "active_leg": (
            None if previous_fix is None or next_fix is None else f"{previous_fix}->{next_fix}"
        ),
        "active_leg_cross_track_nm": cross_track_nm,
        "active_leg_fraction": segment_fraction,
        "distance_to_next_fix_nm": distance_to_next_nm,
        "estimated_seconds_to_next_fix": estimated_seconds,
        "raw_eto_delta_to_next_fix_sec": next_eto_delta,
        "next_planned_level_m": next_level,
        "next_planned_level_source_unit": next_level_unit,
        "next_planned_level_parse_status": next_level_parse_status,
        "ispass_first_unpassed_index": first_unpassed,
        "ispass_self_leg": ispass_self_leg,
        "ispass_time_stale": bool(
            first_unpassed is not None
            and resolved[first_unpassed]["eto_delta_sec"] is not None
            and float(resolved[first_unpassed]["eto_delta_sec"]) < 0
        ),
    }
    return context, resolved


def select_pre_state(
    times: list[float],
    rows: list[dict[str, Any]],
    event_epoch: float,
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    desired_sec = float(config["state_link"]["desired_pre_command_sec"])
    fallback = float(config["state_link"]["fallback_max_target_delta_sec"])
    desired_epoch = event_epoch - desired_sec
    # Strictly causal: a point timestamped exactly at command time may already
    # contain a clearance-driven update and must not be selected as pre-state.
    end_index = bisect.bisect_left(times, event_epoch)
    start_index = bisect.bisect_left(times, desired_epoch - fallback)
    candidates = rows[start_index:end_index]
    if not candidates:
        return None, {
            "desired_pre_state_epoch": desired_epoch,
            "pre_state_linked": False,
            "pre_state_selection_method": "no_pre_command_point_within_fallback",
        }
    selected = min(
        candidates,
        key=lambda row: (
            abs(float(row["event_time_epoch"]) - desired_epoch),
            -float(row["event_time_epoch"]),
        ),
    )
    selected_epoch = float(selected["event_time_epoch"])
    return selected, {
        "desired_pre_state_epoch": desired_epoch,
        "pre_state_linked": True,
        "pre_state_selection_method": "nearest_pre_command_point_to_t_minus_4",
        "pre_state_event_epoch": selected_epoch,
        "pre_state_age_before_command_sec": event_epoch - selected_epoch,
        "pre_state_target_delta_sec": abs(selected_epoch - desired_epoch),
    }


def direction_consistent(reference: dict[str, Any], row: dict[str, Any] | None) -> bool | None:
    if row is None or reference.get("target_value") is None:
        return None
    try:
        target = float(reference["target_value"])
    except (TypeError, ValueError):
        return None
    family = clean_category(reference.get("intent_family"))
    action = clean_category(reference.get("action")).lower()
    if family == "altitude":
        delta = target - float(row["altitude_m"])
    elif family == "speed":
        delta = target - float(row["ground_speed_mps"]) * MPS_TO_KT
    else:
        return None
    if action in {"climb", "increase", "accelerate", "increase_speed"}:
        return delta >= -1e-6
    if action in {
        "descend", "decrease", "decelerate", "reduce", "reduce_speed"
    }:
        return delta <= 1e-6
    return True


def structural_quality(
    reference: dict[str, Any],
    state: dict[str, Any] | None,
    link: dict[str, Any],
    direction_ok: bool | None,
    config: dict[str, Any],
) -> tuple[str, list[str], str, float]:
    issues: list[str] = []
    target = reference.get("target_value")
    try:
        numeric_target = target is not None and math.isfinite(float(target))
    except (TypeError, ValueError):
        numeric_target = False
    if not numeric_target:
        issues.append("missing_numeric_target")
    if state is None:
        issues.append("no_pre_command_state")
        return "C", issues, "audit_only", 0.0
    target_delta = float(link.get("pre_state_target_delta_sec", float("inf")))
    if target_delta > float(config["state_link"]["tier_a_max_target_delta_sec"]):
        issues.append("pre_state_target_delta_gt_8s")
    if target_delta > float(config["state_link"]["tier_b_max_target_delta_sec"]):
        issues.append("pre_state_target_delta_gt_16s")
    inside = bool(state.get("inside_sector"))
    altitude = float(state.get("altitude_m") or 0.0)
    core = config["core_scope"]
    if not inside:
        issues.append("outside_sector_boundary_context")
    if altitude < float(core["minimum_altitude_m"]):
        issues.append("below_core_altitude_band")
    if altitude > float(core["maximum_altitude_m"]):
        issues.append("above_core_altitude_band")
    if direction_ok is False:
        issues.append("historical_direction_mismatch")
    if not numeric_target:
        return "C", issues, "audit_only", 0.0
    tier_a = not any(
        issue in issues
        for issue in {
            "pre_state_target_delta_gt_8s",
            "outside_sector_boundary_context",
            "below_core_altitude_band",
            "above_core_altitude_band",
            "historical_direction_mismatch",
        }
    )
    if tier_a:
        return "A", issues, "weak_train_core", 1.0
    if target_delta <= float(config["state_link"]["tier_b_max_target_delta_sec"]):
        return "B", issues, "weak_train_weighted", 0.5
    return "C", issues, "audit_only", 0.0


def pair_relation(
    target: dict[str, Any], other: dict[str, Any], horizon_sec: float
) -> dict[str, Any] | None:
    try:
        required = {
            "target_latitude": float(target["latitude"]),
            "target_longitude": float(target["longitude"]),
            "target_heading": float(target["track_heading_deg"]),
            "target_speed": float(target["ground_speed_mps"]),
            "target_altitude": float(target["altitude_m"]),
            "other_latitude": float(other["latitude"]),
            "other_longitude": float(other["longitude"]),
            "other_heading": float(other["track_heading_deg"]),
            "other_speed": float(other["ground_speed_mps"]),
            "other_altitude": float(other["altitude_m"]),
        }
        if not all(math.isfinite(value) for value in required.values()):
            return None
        tx, ty = xy_nm(required["target_latitude"], required["target_longitude"])
        ox, oy = xy_nm(required["other_latitude"], required["other_longitude"])
        tvx, tvy = velocity_nm_sec(
            required["target_heading"], required["target_speed"]
        )
        ovx, ovy = velocity_nm_sec(
            required["other_heading"], required["other_speed"]
        )
        target_alt = required["target_altitude"]
        other_alt = required["other_altitude"]
    except (KeyError, TypeError, ValueError):
        return None
    target_vr_raw = target.get("vertical_rate_mps")
    other_vr_raw = other.get("vertical_rate_mps")
    try:
        target_vr_value = float(target_vr_raw)
    except (TypeError, ValueError):
        target_vr_value = float("nan")
    try:
        other_vr_value = float(other_vr_raw)
    except (TypeError, ValueError):
        other_vr_value = float("nan")
    target_vr_assumed_zero = not math.isfinite(target_vr_value)
    other_vr_assumed_zero = not math.isfinite(other_vr_value)
    target_vr = 0.0 if target_vr_assumed_zero else target_vr_value
    other_vr = 0.0 if other_vr_assumed_zero else other_vr_value
    rx, ry = ox - tx, oy - ty
    rvx, rvy = ovx - tvx, ovy - tvy
    horizontal = math.hypot(rx, ry)
    vv = rvx * rvx + rvy * rvy
    tcpa = 0.0 if vv <= 1e-12 else max(
        0.0, min(horizon_sec, -((rx * rvx + ry * rvy) / vv))
    )
    cpa_x, cpa_y = rx + rvx * tcpa, ry + rvy * tcpa
    cpa_horizontal = math.hypot(cpa_x, cpa_y)
    vertical_now = abs(other_alt - target_alt)
    vertical_at_cpa = abs(
        (other_alt + other_vr * tcpa) - (target_alt + target_vr * tcpa)
    )
    closing_kt = 0.0
    if horizontal > 1e-9:
        closing_kt = -((rx * rvx + ry * rvy) / horizontal) * 3600.0
    bearing = (math.degrees(math.atan2(rx, ry)) + 360.0) % 360.0
    relative_bearing = (
        bearing - float(target["track_heading_deg"]) + 540.0
    ) % 360.0 - 180.0
    heading_difference = abs(
        (float(other["track_heading_deg"]) - float(target["track_heading_deg"]) + 180.0)
        % 360.0
        - 180.0
    )
    return {
        "current_horizontal_nm": horizontal,
        "current_vertical_m": vertical_now,
        "relative_bearing_deg": relative_bearing,
        "heading_difference_deg": heading_difference,
        "closing_speed_kt": closing_kt,
        "tcpa_sec": tcpa,
        "cpa_horizontal_nm": cpa_horizontal,
        "vertical_at_cpa_m": vertical_at_cpa,
        "target_vertical_rate_assumed_zero": target_vr_assumed_zero,
        "other_vertical_rate_assumed_zero": other_vr_assumed_zero,
        "vertical_cpa_quality": (
            "assumed_zero_missing_vertical_rate"
            if target_vr_assumed_zero or other_vr_assumed_zero
            else "observed_vertical_rates"
        ),
    }


def flatten(prefix: str, row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    return {f"{prefix}{key}": json_safe(value) for key, value in row.items()}


def select_causal_other_row_indices(
    rows: list[dict[str, Any]], target_state_epoch: float
) -> set[int]:
    """Select one latest non-future observation per other trajectory."""

    selected: dict[str, tuple[float, int]] = {}
    for index, row in enumerate(rows):
        try:
            row_epoch = float(row["event_time_epoch"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(row_epoch) or row_epoch > target_state_epoch + 1e-9:
            continue
        trajectory_id = str(row.get("trajectory_id"))
        candidate = (row_epoch, index)
        if trajectory_id not in selected or candidate > selected[trajectory_id]:
            selected[trajectory_id] = candidate
    return {index for _, index in selected.values()}


def rank_relations(rows: list[dict[str, Any]]) -> None:
    """Assign a complete, stable 1..N risk ranking in place."""

    rows.sort(
        key=lambda item: (
            not item["relation_valid"],
            not item["predicted_pair_conflict"],
            float("inf")
            if item["cpa_horizontal_nm"] is None
            else item["cpa_horizontal_nm"],
            float("inf")
            if item["vertical_at_cpa_m"] is None
            else item["vertical_at_cpa_m"],
            item["relation_source_row_index"],
        )
    )
    for rank, relation in enumerate(rows, start=1):
        relation["relation_risk_rank"] = rank


def load_rows(
    dataset: ds.Dataset,
    columns: list[str],
    expression: ds.Expression,
) -> list[dict[str, Any]]:
    frame = dataset.scanner(
        columns=columns, filter=expression, batch_size=65536
    ).to_table().to_pandas()
    return frame.to_dict("records")


def write_parquet_records(
    rows: list[dict[str, Any]], path: Path, empty_columns: list[str]
) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            {column: pd.Series(dtype="string") for column in empty_columns}
        )
    frame.to_parquet(path, index=False)


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    minimum_pyarrow_major = int(
        config.get("runtime", {}).get("minimum_pyarrow_major", MINIMUM_PYARROW_MAJOR)
    )
    pyarrow_major = int(pyarrow.__version__.split(".", 1)[0])
    if pyarrow_major < minimum_pyarrow_major:
        raise SystemExit(
            f"pyarrow>={minimum_pyarrow_major} is required for this "
            "Arrow-25-created Parquet; "
            f"found {pyarrow.__version__}. Install the data-pipeline requirements."
        )
    local_date = config["development_local_date"]
    timezone = config["local_timezone"]
    start, end = local_day_bounds(local_date, timezone)
    references = [
        reference
        for reference in jsonl_rows(args.references)
        if reference_local_date(reference, timezone) == local_date
        and reference.get("intent_family") in set(config["instruction_families"])
    ]
    references.sort(key=lambda row: (float(row["event_time_epoch"]), str(row.get("reference_event_id"))))
    if args.limit is not None:
        references = references[: args.limit]
    dataset = ds.dataset(str(args.trajectory_parquet), format="parquet")
    source_columns = list(dataset.schema.names)
    trajectory_ids = sorted(
        {
            str((reference.get("target_state") or {}).get("trajectory_id"))
            for reference in references
            if (reference.get("target_state") or {}).get("trajectory_id")
        }
    )
    scan_buffer_sec = float(
        config["state_link"]["fallback_max_target_delta_sec"]
    ) + float(config["state_link"]["desired_pre_command_sec"])
    if trajectory_ids:
        expression = (
            (ds.field("time_bucket_epoch") >= int(start - scan_buffer_sec))
            & (ds.field("time_bucket_epoch") < end)
            & ds.field("trajectory_id").isin(trajectory_ids)
        )
        target_rows = load_rows(dataset, source_columns, expression)
    else:
        target_rows = []
    by_trajectory: dict[str, tuple[list[float], list[dict[str, Any]]]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        grouped[str(row["trajectory_id"])].append(row)
    for trajectory_id, values in grouped.items():
        ordered = sorted(values, key=lambda row: float(row["event_time_epoch"]))
        by_trajectory[trajectory_id] = (
            [float(row["event_time_epoch"]) for row in ordered],
            ordered,
        )

    nav_root = args.config.resolve().parents[1]
    waypoint_path = nav_root / config["route"]["waypoint_file"]
    airport_path = nav_root / config["route"]["airport_file"]
    nav_index = load_nav_index(waypoint_path, airport_path)

    prior_by_callsign: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_dev_references = [
        reference
        for reference in jsonl_rows(args.references)
        if reference_local_date(reference, timezone) == local_date
    ]
    all_dev_references.sort(key=lambda row: float(row["event_time_epoch"]))
    for reference in all_dev_references:
        prior_by_callsign[clean_category(reference.get("callsign")).upper()].append(reference)

    nested_records: list[dict[str, Any]] = []
    audit_states: list[dict[str, Any]] = []
    model_safe_states: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    selected_states: list[tuple[dict[str, Any], dict[str, Any]]] = []
    grade_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    for reference in references:
        trajectory_id = str((reference.get("target_state") or {}).get("trajectory_id") or "")
        event_epoch = float(reference["event_time_epoch"])
        trajectory = by_trajectory.get(trajectory_id)
        if trajectory is None:
            state, link = None, {
                "desired_pre_state_epoch": event_epoch - float(config["state_link"]["desired_pre_command_sec"]),
                "pre_state_linked": False,
                "pre_state_selection_method": "trajectory_not_found",
            }
        else:
            state, link = select_pre_state(trajectory[0], trajectory[1], event_epoch, config)
        direction_ok = direction_consistent(reference, state)
        grade, issues, recommended_use, weight = structural_quality(
            reference, state, link, direction_ok, config
        )
        grade_counts[grade] += 1
        issue_counts.update(issues)
        route_context: dict[str, Any] = {}
        resolved_points: list[dict[str, Any]] = []
        if state is not None:
            route_context, resolved_points = route_context_v2(
                state,
                nav_index,
                float(config["route"]["maximum_geometric_leg_distance_nm"]),
            )
            selected_states.append((reference, state))
        plan_safe, plan_issues = plan_availability_at_state(state)
        issues.extend(issue for issue in plan_issues if issue not in issues)
        issue_counts.update(plan_issues)
        callsign = clean_category(reference.get("callsign")).upper()
        prior_values = [
            item
            for item in prior_by_callsign.get(callsign, [])
            if 0 < event_epoch - float(item["event_time_epoch"]) <= float(config["history"]["lookback_sec"])
        ][-int(config["history"]["maximum_previous_events"]):]
        history = []
        for position, item in enumerate(reversed(prior_values), start=1):
            history_item = {
                "history_rank": position,
                "previous_reference_event_id": item.get("reference_event_id"),
                "previous_event_time_epoch": item.get("event_time_epoch"),
                "seconds_since_previous_event": event_epoch - float(item["event_time_epoch"]),
                "previous_intent_family": item.get("intent_family"),
                "previous_action": item.get("action"),
                "previous_target_value": item.get("target_value"),
                "previous_unit": item.get("unit"),
                "previous_raw_span": item.get("raw_span"),
            }
            history.append(history_item)
            history_rows.append({"reference_event_id": reference.get("reference_event_id"), **history_item})
        quality = {
            "structural_quality_tier": grade,
            "quality_issues": issues,
            "recommended_use": recommended_use,
            "recommended_training_weight": weight,
            "direction_consistent": direction_ok,
            "label_source": "rule_extracted_reference_candidate",
            "label_human_verified": bool(reference.get("speaker_role_human_verified")),
            "parse_review_required": bool(reference.get("parse_review_required")),
            "potential_post_command_leakage_fields": [
                "state_raw_cleared_flight_level",
                "instruction.target_state",
            ],
            "plan_features_causally_available": plan_safe,
        }
        cleared_level = parse_flight_level(None if state is None else state.get("cleared_flight_level"))
        requested_level = parse_flight_level(None if state is None else state.get("requested_flight_level"))
        parsed = {
            "cleared_level_m": cleared_level["value_m"],
            "cleared_level_source_unit": cleared_level["source_unit"],
            "cleared_level_parse_status": cleared_level["parse_status"],
            "requested_level_m": requested_level["value_m"],
            "requested_level_source_unit": requested_level["source_unit"],
            "requested_level_parse_status": requested_level["parse_status"],
            "planned_speed_kt": None if state is None else json_safe(parse_planned_speed_kt(state.get("planned_speed"))),
            "current_ground_speed_kt": None if state is None else float(state["ground_speed_mps"]) * MPS_TO_KT,
        }
        record = {
            "schema_version": config["dataset_version"],
            "reference_event_id": reference.get("reference_event_id"),
            "instruction": json_safe(reference),
            "state_link": json_safe(link),
            "pre_command_state_raw": json_safe(state),
            "parsed_state": parsed,
            "route_context": json_safe(route_context),
            "route_points_raw_and_resolved": json_safe(resolved_points),
            "previous_instruction_context": json_safe(history),
            "quality": quality,
        }
        nested_records.append(record)
        flat = {
            "schema_version": config["dataset_version"],
            "reference_event_id": reference.get("reference_event_id"),
            "event_time_epoch": event_epoch,
            "callsign": reference.get("callsign"),
            "intent_family": reference.get("intent_family"),
            "action": reference.get("action"),
            "target_value": reference.get("target_value"),
            "target_unit": reference.get("unit"),
            "raw_span": reference.get("raw_span"),
            "transcript": reference.get("transcript"),
            "structural_quality_tier": grade,
            "recommended_use": recommended_use,
            "recommended_training_weight": weight,
            "quality_issues_json": json.dumps(issues, ensure_ascii=False),
            "direction_consistent": direction_ok,
            "history_event_count_15min": len(history),
            **link,
            **parsed,
            **route_context,
            **flatten("state_raw_", state),
        }
        audit_states.append(flat)
        safe_plan_values = {
            "plan_adep": state.get("plan_adep") if plan_safe and state is not None else None,
            "plan_ades": state.get("plan_ades") if plan_safe and state is not None else None,
            "sid": state.get("sid") if plan_safe and state is not None else None,
            "star": state.get("star") if plan_safe and state is not None else None,
            "departure_runway": state.get("departure_runway") if plan_safe and state is not None else None,
            "arrival_runway": state.get("arrival_runway") if plan_safe and state is not None else None,
            "requested_level_m": parsed["requested_level_m"] if plan_safe else None,
            "requested_level_source_unit": parsed["requested_level_source_unit"] if plan_safe else None,
            "requested_level_parse_status": parsed["requested_level_parse_status"] if plan_safe else None,
            "planned_speed_kt": parsed["planned_speed_kt"] if plan_safe else None,
        }
        safe_route_values = {
            key: (value if plan_safe else None)
            for key, value in route_context.items()
        }
        model_safe_states.append({
            "schema_version": config["dataset_version"],
            "reference_event_id": reference.get("reference_event_id"),
            "event_time_epoch": event_epoch,
            "callsign": reference.get("callsign"),
            "intent_family": reference.get("intent_family"),
            "action": reference.get("action"),
            "target_value": reference.get("target_value"),
            "target_unit": reference.get("unit"),
            "structural_quality_tier": grade,
            "recommended_use": recommended_use,
            "recommended_training_weight": weight,
            "quality_issues_json": json.dumps(issues, ensure_ascii=False),
            "direction_consistent": direction_ok,
            "history_event_count_15min": len(history),
            "plan_features_causally_available": plan_safe,
            **link,
            "trajectory_id": None if state is None else state.get("trajectory_id"),
            "state_event_time_epoch": None if state is None else state.get("event_time_epoch"),
            "time_bucket_epoch": None if state is None else state.get("time_bucket_epoch"),
            "inside_sector": None if state is None else state.get("inside_sector"),
            "sector_hit": None if state is None else state.get("sector_hit"),
            "latitude": None if state is None else state.get("latitude"),
            "longitude": None if state is None else state.get("longitude"),
            "altitude_m": None if state is None else state.get("altitude_m"),
            "ground_speed_kt": parsed["current_ground_speed_kt"],
            "track_heading_deg": None if state is None else state.get("track_heading_deg"),
            "vertical_rate_mps": None if state is None else state.get("vertical_rate_mps"),
            "time_from_sector_entry_sec": None if state is None else state.get("time_from_sector_entry_sec"),
            "aircraft_type": state.get("aircraft_type") if plan_safe and state is not None else None,
            "wake_turbulence_category": state.get("wake_turbulence_category") if plan_safe and state is not None else None,
            **safe_plan_values,
            **safe_route_values,
        })
        for point in resolved_points:
            route_rows.append({
                "reference_event_id": reference.get("reference_event_id"),
                "trajectory_id": trajectory_id,
                **point,
            })
        quality_rows.append({
            "reference_event_id": reference.get("reference_event_id"),
            "event_time_epoch": event_epoch,
            "callsign": reference.get("callsign"),
            "intent_family": reference.get("intent_family"),
            "target_value": reference.get("target_value"),
            **link,
            **quality,
        })

    buckets = sorted({int(state["time_bucket_epoch"]) for _, state in selected_states})
    if buckets:
        traffic_expression = (
            (ds.field("time_bucket_epoch") >= int(start - scan_buffer_sec))
            & (ds.field("time_bucket_epoch") < end)
            & ds.field("time_bucket_epoch").isin(buckets)
        )
        traffic_rows = load_rows(dataset, source_columns, traffic_expression)
    else:
        traffic_rows = []
    traffic_by_bucket: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in traffic_rows:
        traffic_by_bucket[int(row["time_bucket_epoch"])].append(row)
    relation_rows: list[dict[str, Any]] = []
    model_safe_relation_rows: list[dict[str, Any]] = []
    relation_counts: Counter[str] = Counter()
    expected_relation_rows = 0
    for reference, state in selected_states:
        event_id = reference.get("reference_event_id")
        command_epoch = float(reference["event_time_epoch"])
        bucket = int(state["time_bucket_epoch"])
        event_relations: list[dict[str, Any]] = []
        safe_event_relations: list[dict[str, Any]] = []
        other_rows = [
            other
            for other in traffic_by_bucket.get(bucket, [])
            if str(other.get("trajectory_id")) != str(state.get("trajectory_id"))
        ]
        causal_indices = select_causal_other_row_indices(
            other_rows, float(state["event_time_epoch"])
        )
        for relation_source_row_index, other in enumerate(other_rows):
            other_epoch = float(other["event_time_epoch"])
            if other_epoch >= command_epoch:
                temporal_status = "at_or_after_command_audit_only"
            elif other_epoch > float(state["event_time_epoch"]):
                temporal_status = "after_target_pre_state_audit_only"
            elif relation_source_row_index not in causal_indices:
                temporal_status = "superseded_duplicate_trajectory_audit_only"
            else:
                temporal_status = "causal_model_snapshot"
            model_eligible = relation_source_row_index in causal_indices
            relation = pair_relation(
                state, other, float(config["traffic"]["prediction_horizon_sec"])
            )
            relation_valid = relation is not None
            if not relation_valid:
                relation_counts["invalid_relation"] += 1
                relation = {
                    "current_horizontal_nm": None,
                    "current_vertical_m": None,
                    "relative_bearing_deg": None,
                    "heading_difference_deg": None,
                    "closing_speed_kt": None,
                    "tcpa_sec": None,
                    "cpa_horizontal_nm": None,
                    "vertical_at_cpa_m": None,
                    "target_vertical_rate_assumed_zero": None,
                    "other_vertical_rate_assumed_zero": None,
                    "vertical_cpa_quality": "unavailable_invalid_required_kinematics",
                }
            conflict = bool(
                relation_valid
                and relation["cpa_horizontal_nm"] < float(config["traffic"]["horizontal_conflict_nm"])
                and relation["vertical_at_cpa_m"] < float(config["traffic"]["vertical_conflict_m"])
            )
            base_relation = {
                "reference_event_id": event_id,
                "command_event_time_epoch": command_epoch,
                "target_trajectory_id": state.get("trajectory_id"),
                "target_callsign": state.get("callsign"),
                "target_state_event_time_epoch": state.get("event_time_epoch"),
                "state_time_bucket_epoch": bucket,
                "relation_source_row_index": relation_source_row_index,
                "other_trajectory_id": other.get("trajectory_id"),
                "other_callsign": other.get("callsign"),
                "other_event_time_epoch": other_epoch,
                "other_event_time_delta_sec": float(other["event_time_epoch"]) - float(state["event_time_epoch"]),
                "relation_temporal_status": temporal_status,
                "relation_model_eligible": model_eligible,
                "relation_valid": relation_valid,
                "relation_invalid_reason": None if relation_valid else "missing_or_invalid_required_kinematics",
                **relation,
                "predicted_pair_conflict": conflict,
            }
            event_relations.append({
                **base_relation,
                **flatten("other_raw_", other),
            })
            if model_eligible:
                safe_event_relations.append({
                    **base_relation,
                    "other_inside_sector": json_safe(other.get("inside_sector")),
                    "other_sector_hit": json_safe(other.get("sector_hit")),
                    "other_latitude": json_safe(other.get("latitude")),
                    "other_longitude": json_safe(other.get("longitude")),
                    "other_altitude_m": json_safe(other.get("altitude_m")),
                    "other_ground_speed_kt": (
                        None
                        if other.get("ground_speed_mps") is None
                        else json_safe(float(other["ground_speed_mps"]) * MPS_TO_KT)
                    ),
                    "other_track_heading_deg": json_safe(other.get("track_heading_deg")),
                    "other_vertical_rate_mps": json_safe(other.get("vertical_rate_mps")),
                })
        expected_relation_rows += len(event_relations)
        rank_relations(event_relations)
        rank_relations(safe_event_relations)
        relation_rows.extend(event_relations)
        model_safe_relation_rows.extend(safe_event_relations)
        relation_counts.update(
            relation["relation_temporal_status"] for relation in event_relations
        )
        relation_counts["events_with_audit_relations"] += int(bool(event_relations))
        relation_counts["events_with_model_safe_relations"] += int(bool(safe_event_relations))
        relation_counts["events_with_predicted_conflict"] += int(
            any(item["predicted_pair_conflict"] for item in safe_event_relations)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_path = args.output_dir / "event_state_records.jsonl"
    audit_state_path = args.output_dir / "state_audit_v2.parquet"
    model_state_path = args.output_dir / "model_safe_supervised_records_v2.parquet"
    relation_path = args.output_dir / "traffic_relations_audit_v2.parquet"
    model_relation_path = args.output_dir / "model_safe_traffic_relations_v2.parquet"
    route_path = args.output_dir / "route_points_audit_v2.parquet"
    history_path = args.output_dir / "instruction_history_v2.parquet"
    quality_path = args.output_dir / "quality_ledger.jsonl"
    write_jsonl(event_path, nested_records)
    write_parquet_records(
        audit_states,
        audit_state_path,
        ["schema_version", "reference_event_id"],
    )
    write_parquet_records(
        model_safe_states,
        model_state_path,
        ["schema_version", "reference_event_id"],
    )
    write_parquet_records(
        relation_rows,
        relation_path,
        ["reference_event_id", "relation_valid"],
    )
    write_parquet_records(
        model_safe_relation_rows,
        model_relation_path,
        ["reference_event_id", "relation_valid"],
    )
    write_parquet_records(
        route_rows,
        route_path,
        ["reference_event_id", "route_point_index"],
    )
    write_parquet_records(
        history_rows,
        history_path,
        ["reference_event_id", "previous_reference_event_id"],
    )
    write_jsonl(quality_path, quality_rows)

    flat_frame = pd.DataFrame(audit_states)
    route_method = (
        flat_frame["active_leg_method"]
        if "active_leg_method" in flat_frame
        else pd.Series(dtype="string")
    )
    self_leg = (
        flat_frame["ispass_self_leg"]
        if "ispass_self_leg" in flat_frame
        else pd.Series(dtype=bool)
    )
    stale_ispass = (
        flat_frame["ispass_time_stale"]
        if "ispass_time_stale" in flat_frame
        else pd.Series(dtype=bool)
    )
    estimated_seconds = (
        flat_frame["estimated_seconds_to_next_fix"]
        if "estimated_seconds_to_next_fix" in flat_frame
        else pd.Series(dtype=float)
    )
    summary = {
        "builder_version": VERSION,
        "dataset_version": config["dataset_version"],
        "scope": {
            "development_local_date": local_date,
            "reserved_test_local_date": config["reserved_test_local_date"],
            "reserved_test_labels_loaded": False,
            "instruction_families": config["instruction_families"],
            "limited_run": args.limit is not None,
            "limit": args.limit,
        },
        "counts": {
            "reference_events_preserved": len(references),
            "state_linked": sum(row["pre_state_linked"] for row in quality_rows),
            "quality_tiers": dict(sorted(grade_counts.items())),
            "quality_issues": dict(sorted(issue_counts.items())),
            "state_audit_rows": len(audit_states),
            "model_safe_supervised_rows": len(model_safe_states),
            "traffic_relation_audit_rows": len(relation_rows),
            "model_safe_traffic_relation_rows": len(model_safe_relation_rows),
            "route_point_rows": len(route_rows),
            "instruction_history_rows": len(history_rows),
            **dict(sorted(relation_counts.items())),
        },
        "information_preservation": {
            "source_parquet_columns": len(source_columns),
            "source_parquet_columns_preserved_with_state_raw_prefix": source_columns,
            "raw_reference_event_nested": True,
            "raw_route_json_preserved": True,
            "resolved_route_points_are_additive": True,
            "all_same_bucket_traffic_aircraft_preserved": True,
            "all_same_bucket_rows_are_audit_only_until_causal_selection": True,
            "model_safe_relation_selection": "one latest row per other trajectory at or before target pre-state",
            "expected_same_bucket_relation_rows": expected_relation_rows,
            "actual_same_bucket_relation_rows": len(relation_rows),
            "same_bucket_relation_row_count_matches": len(relation_rows) == expected_relation_rows,
            "model_safe_relations_strictly_not_after_target_pre_state": all(
                float(row["other_event_time_epoch"])
                <= float(row["target_state_event_time_epoch"]) + 1e-9
                for row in model_safe_relation_rows
            ),
            "excluded_events_are_in_quality_ledger": True,
            "potential_post_command_leakage_fields_flagged": [
                "state_raw_cleared_flight_level",
                "instruction.target_state",
            ],
        },
        "model_safe_schema": {
            "model_input_columns": [
                "inside_sector", "sector_hit", "latitude", "longitude",
                "altitude_m", "ground_speed_kt", "track_heading_deg",
                "vertical_rate_mps", "time_from_sector_entry_sec",
                "plan_features_causally_available", "aircraft_type",
                "wake_turbulence_category", "plan_adep", "plan_ades",
                "sid", "star", "departure_runway", "arrival_runway",
                "requested_level_m", "requested_level_source_unit",
                "requested_level_parse_status", "planned_speed_kt",
                "route_point_count", "route_resolved_point_count",
                "route_resolved_fraction", "active_leg_method",
                "active_leg_previous_fix", "active_leg_next_fix",
                "active_leg", "active_leg_cross_track_nm",
                "active_leg_fraction", "distance_to_next_fix_nm",
                "estimated_seconds_to_next_fix",
                "raw_eto_delta_to_next_fix_sec", "next_planned_level_m",
                "next_planned_level_source_unit", "next_planned_level_parse_status",
                "ispass_first_unpassed_index", "ispass_self_leg",
                "ispass_time_stale", "history_event_count_15min"
            ],
            "label_columns": [
                "intent_family", "action", "target_value", "target_unit"
            ],
            "metadata_columns": [
                "schema_version", "reference_event_id", "event_time_epoch",
                "callsign", "structural_quality_tier", "recommended_use",
                "recommended_training_weight", "quality_issues_json",
                "direction_consistent", "desired_pre_state_epoch",
                "pre_state_linked", "pre_state_selection_method",
                "pre_state_event_epoch", "pre_state_age_before_command_sec",
                "pre_state_target_delta_sec", "trajectory_id",
                "state_event_time_epoch", "time_bucket_epoch"
            ],
            "forbidden_feature_patterns": [
                "state_raw_*", "instruction.*", "transcript", "raw_span",
                "cleared_flight_level", "reference_event_id", "target_*",
                "action", "intent_family"
            ],
        },
        "route_quality": {
            "geometric_projection": int((route_method == "geometric_projection").sum()),
            "ispass_fallback": int((route_method == "ispass_fallback").sum()),
            "unresolved": int((route_method == "unresolved").sum()),
            "self_leg_count": int(self_leg.fillna(False).sum()),
            "stale_ispass_time_count": int(stale_ispass.fillna(False).sum()),
            "negative_estimated_seconds_count": int((estimated_seconds < 0).sum()),
        },
        "input": {
            "builder_script": str(Path(__file__).resolve()),
            "builder_script_sha256": sha256_file(Path(__file__).resolve()),
            "trajectory_parquet": str(args.trajectory_parquet.resolve()),
            "trajectory_parquet_sha256": sha256_file(args.trajectory_parquet),
            "references": str(args.references.resolve()),
            "references_sha256": sha256_file(args.references),
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
            "waypoint_file": str(waypoint_path.resolve()),
            "waypoint_file_sha256": sha256_file(waypoint_path),
            "airport_file": str(airport_path.resolve()),
            "airport_file_sha256": sha256_file(airport_path),
            "runtime": {
                "python": __import__("sys").version,
                "pyarrow": pyarrow.__version__,
                "pandas": pd.__version__,
                "minimum_pyarrow_major": minimum_pyarrow_major,
            },
        },
        "outputs": {},
        "warnings": [
            "Reference labels remain rule-extracted weak labels, not human-reviewed gold.",
            "Boundary-context tier B is preserved but must not be mixed into the core metric without an explicit scope decision.",
            "Cleared flight level is preserved for audit but flagged as potential post-command leakage.",
            "Nested instruction.target_state is preserved for audit and must not be used as a pre-command model feature.",
            "Geometric route context depends on local navdata coverage; unresolved procedure points remain raw and flagged.",
        ],
    }
    for path in [
        event_path,
        audit_state_path,
        model_state_path,
        relation_path,
        model_relation_path,
        route_path,
        history_path,
        quality_path,
    ]:
        summary["outputs"][path.name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
