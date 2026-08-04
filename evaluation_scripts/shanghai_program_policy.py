#!/usr/bin/env python3
"""Shared feature, candidate and safety code for the Shanghai program-aware policy."""

from __future__ import annotations

import datetime as dt
import json
import math
from collections import Counter
from typing import Any, Iterable

import numpy as np
import pandas as pd


SHANGHAI_AIRPORTS = {"ZSSS", "ZSPD"}
MPS_TO_KT = 1.0 / 0.514444
EARTH_NM_PER_DEG = 60.0

NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "altitude_m",
    "ground_speed_kt",
    "track_heading_deg",
    "vertical_rate_mps",
    "time_from_sector_entry_sec",
    "requested_level_m",
    "planned_speed_kt",
    "next_planned_level_m",
    "seconds_to_next_fix",
    "route_progress",
    "traffic_count",
    "nearest_horizontal_nm",
    "nearest_vertical_m",
    "min_cpa_horizontal_nm",
    "min_tcpa_sec",
    "candidate_target",
    "candidate_delta_altitude_m",
    "candidate_delta_speed_kt",
    "candidate_matches_next_level",
    "candidate_matches_requested_level",
]

CATEGORICAL_FEATURES = [
    "phase",
    "vertical_phase",
    "plan_adep",
    "plan_ades",
    "sid",
    "star",
    "departure_runway",
    "arrival_runway",
    "active_leg",
    "next_fix",
    "candidate_kind",
    "candidate_direction",
]

PARQUET_COLUMNS = [
    "trajectory_id",
    "sector_code",
    "callsign",
    "event_time_epoch",
    "time_bucket_epoch",
    "inside_sector",
    "longitude",
    "latitude",
    "altitude_m",
    "ground_speed_mps",
    "track_heading_deg",
    "vertical_rate_mps",
    "time_from_sector_entry_sec",
    "flight_plan_found",
    "plan_available_at_point",
    "plan_adep",
    "plan_ades",
    "requested_flight_level",
    "planned_speed",
    "sid",
    "star",
    "departure_runway",
    "arrival_runway",
    "planned_route",
    "planned_route_point_count",
    "plan_rtepts_json",
]


def clean_category(value: Any) -> str:
    if value is None:
        return "MISSING"
    try:
        if pd.isna(value):
            return "MISSING"
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else "MISSING"


def parse_s_level(value: Any) -> float:
    text = clean_category(value).upper()
    if text == "MISSING":
        return float("nan")
    if text.startswith("S"):
        text = text[1:]
    try:
        # ICAO China flight-plan metric levels use tens of metres:
        # S0600 = 6000 m, S0150 = 1500 m.
        return float(text) * 10.0
    except ValueError:
        return float("nan")


def parse_planned_speed_kt(value: Any) -> float:
    text = clean_category(value).upper()
    if text == "MISSING":
        return float("nan")
    try:
        if text.startswith("K"):
            return float(text[1:]) / 1.852
        if text.startswith("N"):
            return float(text[1:])
        return float(text)
    except ValueError:
        return float("nan")


def derive_phase(adep: Any, ades: Any) -> str:
    dep = clean_category(adep)
    arr = clean_category(ades)
    if dep in SHANGHAI_AIRPORTS:
        return "departure"
    if arr in SHANGHAI_AIRPORTS:
        return "arrival"
    return "overflight"


def derive_vertical_phase(vertical_rate_mps: Any) -> str:
    try:
        rate = float(vertical_rate_mps)
    except (TypeError, ValueError):
        return "unknown"
    if rate > 0.5:
        return "climbing"
    if rate < -0.5:
        return "descending"
    return "level"


def parse_iso_epoch(value: Any) -> float:
    text = clean_category(value)
    if text == "MISSING":
        return float("nan")
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return float("nan")


def route_context(value: Any, epoch: float) -> dict[str, Any]:
    result = {
        "active_leg": "MISSING",
        "next_fix": "MISSING",
        "next_planned_level_m": float("nan"),
        "seconds_to_next_fix": float("nan"),
        "route_progress": float("nan"),
        "route_point_count": 0,
    }
    if value is None:
        return result
    try:
        points = json.loads(value) if isinstance(value, str) else list(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return result
    if not points:
        return result
    next_index = next(
        (index for index, point in enumerate(points) if clean_category(point.get("ispass")).upper() != "Y"),
        len(points) - 1,
    )
    previous_index = max(0, next_index - 1)
    previous_fix = clean_category(points[previous_index].get("ptid"))
    next_point = points[next_index]
    next_fix = clean_category(next_point.get("ptid"))
    next_epoch = parse_iso_epoch(next_point.get("eto_utc"))
    result.update(
        {
            "active_leg": f"{previous_fix}->{next_fix}",
            "next_fix": next_fix,
            "next_planned_level_m": parse_s_level(next_point.get("flight_level")),
            "seconds_to_next_fix": (
                float(next_epoch - epoch) if math.isfinite(next_epoch) else float("nan")
            ),
            "route_progress": float(
                sum(clean_category(point.get("ispass")).upper() == "Y" for point in points)
                / len(points)
            ),
            "route_point_count": len(points),
        }
    )
    return result


def xy_nm(latitude: float, longitude: float, center_latitude: float = 30.92095, center_longitude: float = 120.96985) -> tuple[float, float]:
    x = (longitude - center_longitude) * EARTH_NM_PER_DEG * math.cos(math.radians(center_latitude))
    y = (latitude - center_latitude) * EARTH_NM_PER_DEG
    return x, y


def velocity_nm_sec(track_heading_deg: float, ground_speed_mps: float) -> tuple[float, float]:
    radians = math.radians(track_heading_deg)
    speed_nm_sec = ground_speed_mps / 1852.0
    return speed_nm_sec * math.sin(radians), speed_nm_sec * math.cos(radians)


def pair_geometry(a: dict[str, Any], b: dict[str, Any], horizon_sec: float = 300.0) -> tuple[float, float, float]:
    ax, ay = xy_nm(float(a["latitude"]), float(a["longitude"]))
    bx, by = xy_nm(float(b["latitude"]), float(b["longitude"]))
    avx, avy = velocity_nm_sec(float(a["track_heading_deg"]), float(a["ground_speed_mps"]))
    bvx, bvy = velocity_nm_sec(float(b["track_heading_deg"]), float(b["ground_speed_mps"]))
    rx, ry = bx - ax, by - ay
    vx, vy = bvx - avx, bvy - avy
    vv = vx * vx + vy * vy
    tcpa = 0.0 if vv <= 1e-12 else max(0.0, min(horizon_sec, -((rx * vx + ry * vy) / vv)))
    dx, dy = rx + vx * tcpa, ry + vy * tcpa
    return tcpa, math.hypot(dx, dy), math.hypot(rx, ry)


def traffic_features(target: dict[str, Any], rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    nearest_h = float("inf")
    nearest_v = float("inf")
    min_cpa_h = float("inf")
    min_tcpa = float("inf")
    count = 0
    for other in rows:
        if (
            clean_category(other.get("trajectory_id"))
            == clean_category(target.get("trajectory_id"))
        ):
            continue
        try:
            vertical = abs(float(target["altitude_m"]) - float(other["altitude_m"]))
            tcpa, cpa_h, current_h = pair_geometry(target, other)
        except (KeyError, TypeError, ValueError):
            continue
        count += 1
        nearest_h = min(nearest_h, current_h)
        nearest_v = min(nearest_v, vertical)
        if cpa_h < min_cpa_h:
            min_cpa_h = cpa_h
            min_tcpa = tcpa
    return {
        "traffic_count": float(count),
        "nearest_horizontal_nm": nearest_h if math.isfinite(nearest_h) else 999.0,
        "nearest_vertical_m": nearest_v if math.isfinite(nearest_v) else 99999.0,
        "min_cpa_horizontal_nm": min_cpa_h if math.isfinite(min_cpa_h) else 999.0,
        "min_tcpa_sec": min_tcpa if math.isfinite(min_tcpa) else 9999.0,
    }


def state_features(row: dict[str, Any], tick_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    epoch = float(row["time_bucket_epoch"])
    route = route_context(row.get("plan_rtepts_json"), epoch)
    feature = {
        "trajectory_id": clean_category(row.get("trajectory_id")),
        "callsign": clean_category(row.get("callsign")).upper(),
        "time_bucket_epoch": int(epoch),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "altitude_m": float(row["altitude_m"]),
        "ground_speed_kt": float(row["ground_speed_mps"]) * MPS_TO_KT,
        "track_heading_deg": float(row["track_heading_deg"]) % 360.0,
        "vertical_rate_mps": float(row.get("vertical_rate_mps") or 0.0),
        "time_from_sector_entry_sec": float(row.get("time_from_sector_entry_sec") or 0.0),
        "phase": derive_phase(row.get("plan_adep"), row.get("plan_ades")),
        "vertical_phase": derive_vertical_phase(row.get("vertical_rate_mps")),
        "plan_adep": clean_category(row.get("plan_adep")),
        "plan_ades": clean_category(row.get("plan_ades")),
        "sid": clean_category(row.get("sid")),
        "star": clean_category(row.get("star")),
        "departure_runway": clean_category(row.get("departure_runway")),
        "arrival_runway": clean_category(row.get("arrival_runway")),
        "requested_level_m": parse_s_level(row.get("requested_flight_level")),
        "planned_speed_kt": parse_planned_speed_kt(row.get("planned_speed")),
        **route,
        **traffic_features(row, tick_rows),
    }
    return feature


def candidate_id(kind: str, target: float | int | None = None) -> str:
    if kind == "hold":
        return "HOLD"
    prefix = "ALT" if kind == "altitude" else "SPD"
    return f"{prefix}_{int(round(float(target)))}"


def historical_candidate_id(reference: dict[str, Any]) -> str | None:
    family = clean_category(reference.get("intent_family"))
    value = reference.get("target_value")
    if family not in {"altitude", "speed"} or value is None:
        return None
    try:
        return candidate_id(family, float(value))
    except (TypeError, ValueError):
        return None


def generate_candidates(state: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    settings = config["candidates"]
    altitude = float(state["altitude_m"])
    speed = float(state["ground_speed_kt"])
    phase = state["phase"]
    minimum_altitude_change = float(settings["minimum_altitude_change_m"])
    minimum_speed_change = float(settings["minimum_speed_change_kt"])
    candidates: list[dict[str, Any]] = [
        {
            "candidate_id": "HOLD",
            "candidate_kind": "hold",
            "candidate_target": 0.0,
            "candidate_direction": "hold",
            "candidate_delta_altitude_m": 0.0,
            "candidate_delta_speed_kt": 0.0,
        }
    ]
    for target in settings["altitude_levels_m"]:
        delta = float(target) - altitude
        if abs(delta) < minimum_altitude_change:
            direction = "maintain"
        else:
            direction = "climb" if delta > 0 else "descend"
        if bool(settings.get("phase_direction_gate", True)):
            if phase == "departure" and direction == "descend":
                continue
            if phase == "arrival" and direction == "climb":
                continue
        candidates.append(
            {
                "candidate_id": candidate_id("altitude", target),
                "candidate_kind": "altitude",
                "candidate_target": float(target),
                "candidate_direction": direction,
                "candidate_delta_altitude_m": delta,
                "candidate_delta_speed_kt": 0.0,
            }
        )
    for target in settings["speed_targets_kt"]:
        delta = float(target) - speed
        direction = (
            "maintain_speed"
            if abs(delta) < minimum_speed_change
            else ("accelerate" if delta > 0 else "decelerate")
        )
        candidates.append(
            {
                "candidate_id": candidate_id("speed", target),
                "candidate_kind": "speed",
                "candidate_target": float(target),
                "candidate_direction": direction,
                "candidate_delta_altitude_m": 0.0,
                "candidate_delta_speed_kt": delta,
            }
        )
    next_level = float(state.get("next_planned_level_m", float("nan")))
    requested_level = float(state.get("requested_level_m", float("nan")))
    for candidate in candidates:
        target = float(candidate["candidate_target"])
        candidate["candidate_matches_next_level"] = float(
            candidate["candidate_kind"] == "altitude"
            and math.isfinite(next_level)
            and abs(target - next_level) <= 150.0
        )
        candidate["candidate_matches_requested_level"] = float(
            candidate["candidate_kind"] == "altitude"
            and math.isfinite(requested_level)
            and abs(target - requested_level) <= 150.0
        )
    return candidates


def expand_candidate_rows(
    state: dict[str, Any],
    config: dict[str, Any],
    true_candidate_id: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in generate_candidates(state, config):
        item = {**state, **candidate}
        if true_candidate_id is not None:
            item["is_selected"] = int(candidate["candidate_id"] == true_candidate_id)
        rows.append(item)
    return rows


def command_for_candidate(callsign: str, candidate: dict[str, Any]) -> str | None:
    kind = candidate["candidate_kind"]
    if kind == "altitude":
        direction = {
            "climb": "CLIMB",
            "descend": "DESCEND",
        }.get(str(candidate["candidate_direction"]))
        suffix = f",{direction}" if direction else ""
        return f"ALT_M {callsign},{int(round(candidate['candidate_target']))}{suffix}"
    if kind == "speed":
        return f"SPD {callsign},{int(round(candidate['candidate_target']))}"
    return None


def runtime_candidate_rejection(
    candidate: dict[str, Any] | pd.Series,
    epoch: float,
    command_cooldown: dict[str, float],
    axis_cooldown: dict[tuple[str, str], float],
    last_target: dict[tuple[str, str], tuple[str, float]],
    runtime: dict[str, Any],
) -> str | None:
    """Return the shared pre-safety runtime rejection reason, if any."""

    callsign = str(candidate["callsign"])
    kind = str(candidate["candidate_kind"])
    candidate_id = str(candidate["candidate_id"])
    key = (callsign, kind)
    if epoch < command_cooldown.get(callsign, float("-inf")):
        return "aircraft_command_cooldown"
    if epoch < axis_cooldown.get(key, float("-inf")):
        return "axis_cooldown"
    previous = last_target.get(key)
    if (
        previous is not None
        and candidate_id == previous[0]
        and epoch - previous[1] < float(runtime["target_memory_sec"])
    ):
        return "repeat_target"
    current_target = float(candidate["candidate_target"])
    if previous is not None and kind == "altitude":
        previous_target = float(previous[0].split("_", 1)[1])
        phase = str(candidate["phase"])
        if (
            phase == "departure" and current_target < previous_target
        ) or (
            phase == "arrival" and current_target > previous_target
        ):
            return "regressive_altitude_target"
    if previous is not None and kind == "speed":
        previous_target = float(previous[0].split("_", 1)[1])
        current_speed = float(candidate["ground_speed_kt"])
        prior_direction = previous_target - current_speed
        new_direction = current_target - current_speed
        if prior_direction * new_direction < 0:
            return "speed_direction_reversal"
    return None


def record_runtime_candidate(
    candidate: dict[str, Any] | pd.Series,
    epoch: float,
    command_cooldown: dict[str, float],
    axis_cooldown: dict[tuple[str, str], float],
    last_target: dict[tuple[str, str], tuple[str, float]],
    runtime: dict[str, Any],
) -> None:
    """Update shared cooldown and target memory after issuing a command."""

    callsign = str(candidate["callsign"])
    kind = str(candidate["candidate_kind"])
    command_cooldown[callsign] = (
        epoch + float(runtime["aircraft_command_cooldown_sec"])
    )
    axis_cooldown[(callsign, kind)] = (
        epoch + float(runtime["same_axis_cooldown_sec"])
    )
    last_target[(callsign, kind)] = (str(candidate["candidate_id"]), epoch)


def _distance_after_speed_command(
    start_speed_kt: float,
    target_speed_kt: float,
    elapsed_after_delay: float,
    acceleration_ktps: float,
) -> float:
    if elapsed_after_delay <= 0:
        return 0.0
    delta = target_speed_kt - start_speed_kt
    if abs(delta) <= 1e-9:
        return start_speed_kt * elapsed_after_delay / 3600.0
    ramp = abs(delta) / acceleration_ktps
    used = min(ramp, elapsed_after_delay)
    direction = 1.0 if delta > 0 else -1.0
    end_speed = start_speed_kt + direction * acceleration_ktps * used
    distance = (start_speed_kt + end_speed) * 0.5 * used / 3600.0
    if elapsed_after_delay > ramp:
        distance += target_speed_kt * (elapsed_after_delay - ramp) / 3600.0
    return distance


def predicted_motion(
    row: dict[str, Any],
    candidate: dict[str, Any] | None,
    t_sec: float,
    safety: dict[str, Any],
) -> tuple[float, float, float]:
    x0, y0 = xy_nm(float(row["latitude"]), float(row["longitude"]))
    heading = math.radians(float(row["track_heading_deg"]))
    start_speed_kt = float(row["ground_speed_mps"]) * MPS_TO_KT
    delay = float(safety["command_delay_sec"])
    before = min(t_sec, delay)
    distance_nm = start_speed_kt * before / 3600.0
    remaining = max(0.0, t_sec - delay)
    if candidate and candidate["candidate_kind"] == "speed":
        distance_nm += _distance_after_speed_command(
            start_speed_kt,
            float(candidate["candidate_target"]),
            remaining,
            float(safety["speed_acceleration_ktps"]),
        )
    else:
        distance_nm += start_speed_kt * remaining / 3600.0
    x = x0 + distance_nm * math.sin(heading)
    y = y0 + distance_nm * math.cos(heading)

    altitude = float(row["altitude_m"])
    current_vertical_rate = float(row.get("vertical_rate_mps") or 0.0)
    altitude += current_vertical_rate * before
    if candidate and candidate["candidate_kind"] == "altitude" and remaining > 0:
        target = float(candidate["candidate_target"])
        direction = 1.0 if target > altitude else -1.0
        commanded = altitude + direction * float(safety["vertical_rate_mps"]) * remaining
        altitude = min(target, commanded) if direction > 0 else max(target, commanded)
    else:
        altitude += current_vertical_rate * remaining
    return x, y, altitude


def candidate_safety(
    target: dict[str, Any],
    candidate: dict[str, Any],
    traffic_rows: Iterable[dict[str, Any]],
    safety: dict[str, Any],
) -> dict[str, Any]:
    if candidate["candidate_kind"] == "hold":
        return {
            "safe": True,
            "checked_pairs": 0,
            "minimum_horizontal_nm": None,
            "minimum_vertical_m_when_horizontal_loss": None,
            "unsafe_pairs": [],
        }
    minimum_h = float("inf")
    minimum_v_when_hloss = float("inf")
    unsafe_pairs: list[dict[str, Any]] = []
    checked = 0
    horizon = int(safety["horizon_sec"])
    step = int(safety["step_sec"])
    hsep_limit = float(safety["minimum_horizontal_separation_nm"])
    vsep_limit = float(safety["minimum_vertical_separation_m"])
    target_id = clean_category(target.get("trajectory_id"))
    for other in traffic_rows:
        if clean_category(other.get("trajectory_id")) == target_id:
            continue
        try:
            float(other["latitude"])
            float(other["longitude"])
            float(other["altitude_m"])
            float(other["ground_speed_mps"])
            float(other["track_heading_deg"])
        except (KeyError, TypeError, ValueError):
            continue
        checked += 1
        pair_failed = False
        pair_min_h = float("inf")
        pair_min_v = float("inf")
        for t_sec in range(0, horizon + step, step):
            ax, ay, aalt = predicted_motion(target, candidate, float(t_sec), safety)
            bx, by, balt = predicted_motion(other, None, float(t_sec), safety)
            horizontal = math.hypot(bx - ax, by - ay)
            vertical = abs(balt - aalt)
            minimum_h = min(minimum_h, horizontal)
            pair_min_h = min(pair_min_h, horizontal)
            if horizontal < hsep_limit:
                minimum_v_when_hloss = min(minimum_v_when_hloss, vertical)
                pair_min_v = min(pair_min_v, vertical)
                if vertical < vsep_limit:
                    pair_failed = True
        if pair_failed:
            unsafe_pairs.append(
                {
                    "other_callsign": clean_category(other.get("callsign")).upper(),
                    "minimum_horizontal_nm": round(pair_min_h, 3),
                    "minimum_vertical_m_when_horizontal_loss": (
                        None if not math.isfinite(pair_min_v) else round(pair_min_v, 1)
                    ),
                }
            )
    return {
        "safe": not unsafe_pairs,
        "checked_pairs": checked,
        "minimum_horizontal_nm": None if not math.isfinite(minimum_h) else round(minimum_h, 3),
        "minimum_vertical_m_when_horizontal_loss": (
            None if not math.isfinite(minimum_v_when_hloss) else round(minimum_v_when_hloss, 1)
        ),
        "unsafe_pairs": unsafe_pairs[:20],
    }


def top_candidate_decision(scored: pd.DataFrame, threshold: float) -> dict[str, Any]:
    hold = scored[scored["candidate_id"] == "HOLD"].sort_values(
        "candidate_score", ascending=False
    )
    non_hold = scored[scored["candidate_id"] != "HOLD"].sort_values(
        ["candidate_score", "candidate_id"], ascending=[False, True]
    )
    hold_score = float(hold.iloc[0]["candidate_score"]) if not hold.empty else 0.0
    if non_hold.empty:
        return {
            "selected_candidate_id": "HOLD",
            "margin": float("-inf"),
            "hold_score": hold_score,
            "non_hold_score": None,
        }
    best = non_hold.iloc[0]
    margin = float(best["candidate_score"]) - hold_score
    return {
        "selected_candidate_id": (
            str(best["candidate_id"]) if margin >= threshold else "HOLD"
        ),
        "margin": margin,
        "hold_score": hold_score,
        "non_hold_score": float(best["candidate_score"]),
    }


def count_by_phase(frame: pd.DataFrame) -> dict[str, int]:
    return {str(key): int(value) for key, value in Counter(frame["phase"]).items()}
