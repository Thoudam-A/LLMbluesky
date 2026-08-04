from __future__ import annotations

import json

from .models import ConflictScene
from .schemas import CANDIDATE_OUTPUT_SCHEMA, COARSE_CANDIDATE_OUTPUT_SCHEMA


PROMPT_VERSION = "hppo-candidate-v2"
COARSE_PROMPT_VERSION = "hppo-coarse-candidate-v1"


def _relevant_scene_payload(scene: ConflictScene) -> dict:
    """Keep API input bounded to decision-relevant aircraft and conflict facts."""
    relevant_ids = set(scene.target_ids)
    for edge in scene.conflicts:
        relevant_ids.add(edge.ownship_id)
        relevant_ids.add(edge.intruder_id)
    aircraft = [item for item in scene.aircraft if item.aircraft_id in relevant_ids]
    return {
        "schema_version": scene.schema_version,
        "scene_id": scene.scene_id,
        "simulation_time_s": scene.simulation_time_s,
        "separation_standard": {
            "horizontal_km": scene.separation_horizontal_km,
            "vertical_ft": scene.separation_vertical_ft,
            "temporal_s": scene.separation_time_s,
        },
        "target_ids": scene.target_ids,
        "aircraft": [
            {
                "aircraft_id": item.aircraft_id,
                "latitude_deg": item.latitude_deg,
                "longitude_deg": item.longitude_deg,
                "altitude_ft": item.altitude_ft,
                "speed_kt": item.speed_kt,
                "heading_deg": item.heading_deg,
                "vertical_speed_fpm": item.vertical_speed_fpm,
                "current_macro": item.current_macro,
                "hold_remaining_s": item.hold_remaining_s,
            }
            for item in aircraft
        ],
        "conflicts": [
            {
                "ownship_id": edge.ownship_id,
                "intruder_id": edge.intruder_id,
                "horizontal_km": edge.horizontal_km,
                "vertical_ft": edge.vertical_ft,
                "tcpa_s": edge.tcpa_s,
                "time_gap_s": edge.time_gap_s,
                "relative_bearing_deg": edge.relative_bearing_deg,
                "severity": edge.severity,
                "loss_of_separation": edge.loss_of_separation,
            }
            for edge in scene.conflicts
        ],
    }


def _situation_summary(scene: ConflictScene) -> str:
    """Small natural-language layer for relational reasoning; JSON remains authoritative."""
    if not scene.conflicts:
        return "No active conflict edge is supplied; do not invent a maneuver."
    highest = max(scene.conflicts, key=lambda item: (item.severity, -item.tcpa_s))
    status = "current separation loss" if highest.loss_of_separation else "predicted separation risk"
    return (
        f"{status}: {highest.ownship_id} versus {highest.intruder_id}; "
        f"severity={highest.severity:.2f}, horizontal={highest.horizontal_km:.2f} km, "
        f"vertical={highest.vertical_ft:.0f} ft, TCPA={highest.tcpa_s:.0f} s. "
        "Produce candidates only for target_ids and preserve the configured separation standard."
    )


def _action_limits(config) -> dict:
    ranges = config.command_ranges
    hold = config.runtime.min_command_hold_s
    return {
        "heading_delta_deg": [-float(ranges.heading_offset_deg), float(ranges.heading_offset_deg)],
        "altitude_delta_ft": [-float(ranges.altitude_delta_ft), float(ranges.altitude_delta_ft)],
        "speed_delta_kt": [-float(ranges.speed_delta_kt), float(ranges.speed_delta_kt)],
        "minimum_duration_s": float(hold),
        "recovery_action": "RESUME_ROUTE",
    }


def build_candidate_prompt(scene: ConflictScene, config=None) -> str:
    limits = _action_limits(config) if config is not None else {
        "heading_delta_deg": [-45.0, 45.0],
        "altitude_delta_ft": [-3000.0, 3000.0],
        "speed_delta_kt": [-30.0, 30.0],
        "minimum_duration_s": 10.0,
        "recovery_action": "RESUME_ROUTE",
    }
    return "\n".join(
        [
            "You are an air-traffic conflict-resolution candidate generator.",
            "Return JSON only. Do not execute commands, invent aircraft, or invent action types.",
            "Generate at most three candidates per target aircraft. Prefer one primary maneuver and diverse alternatives.",
            "NO_NEW_COMMAND and RESUME_ROUTE must use an empty parameters object.",
            "All angles are degrees, altitude changes are feet, speed changes are knots, and duration is seconds.",
            "Natural-language situation summary (JSON below is authoritative):",
            _situation_summary(scene),
            "Executable action limits:",
            json.dumps(limits, separators=(",", ":"), sort_keys=True),
            'Example JSON: {"schema_version":"1.0","scene_id":"copy-from-input","candidates":[{"candidate_id":"C1","aircraft_id":"KL0","action_type":"TURN_RIGHT","parameters":{"heading_delta_deg":20,"duration_s":20},"prior_score":0.7,"expected_effect":"increase horizontal separation","intent":"LATERAL_SPLIT","minimum_duration_s":20,"release_stable_s":15,"recovery_action":"RESUME_ROUTE","fallback_candidate_id":"C2"}]}',
            "Output schema:",
            json.dumps(CANDIDATE_OUTPUT_SCHEMA, separators=(",", ":"), sort_keys=True),
            "Structured conflict scene:",
            json.dumps(_relevant_scene_payload(scene), separators=(",", ":"), sort_keys=True, allow_nan=False),
        ]
    )


def build_coarse_candidate_prompt(scene: ConflictScene, config=None) -> str:
    """Prompt for the v1.1 candidate language; RL owns exact action parameters."""
    limits = _action_limits(config) if config is not None else {
        "heading_delta_deg": [-45.0, 45.0],
        "altitude_delta_ft": [-3000.0, 3000.0],
        "speed_delta_kt": [-30.0, 30.0],
        "minimum_duration_s": 10.0,
        "recovery_action": "RESUME_ROUTE",
    }
    return "\n".join(
        [
            "You are an air-traffic conflict-resolution candidate generator.",
            "Return JSON only. Do not execute commands, invent aircraft, or invent action types.",
            "Generate at most three diverse candidates per target aircraft.",
            "Output coarse candidates only: choose macro action, magnitude level, duration level, timing, and a short reason.",
            "Do not output exact heading, altitude, speed, or duration values; parameters must be {}.",
            "Use PRIORITY_PASS only with distinct priority_aircraft_id and yield_aircraft_id; aircraft_id must be the yield aircraft.",
            "For an aircraft with hold_remaining_s greater than zero, do not override its active command immediately: use NO_NEW_COMMAND or execution_timing AFTER_CURRENT_HOLD.",
            "Valid macro_action values: NO_NEW_COMMAND, HOLD_HEADING, TURN_LEFT, TURN_RIGHT, CLIMB, DESCEND, ACCELERATE, DECELERATE.",
            "Natural-language situation summary (JSON below is authoritative):",
            _situation_summary(scene),
            "Configured executable limits; RL will choose exact values within them:",
            json.dumps(limits, separators=(",", ":"), sort_keys=True),
            "Output schema:",
            json.dumps(COARSE_CANDIDATE_OUTPUT_SCHEMA, separators=(",", ":"), sort_keys=True),
            "Structured conflict scene:",
            json.dumps(_relevant_scene_payload(scene), separators=(",", ":"), sort_keys=True, allow_nan=False),
        ]
    )
