"""Fixed-size, validated LLM candidate pools for the future candidate Actor.

The existing H-PPO policy remains a five-action policy.  This module provides
an intermediate, versioned candidate language and only adapts candidates to the
legacy action prior when that mapping is semantically safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .models import (
    CandidateAction,
    CoarseCandidateAction,
    DURATION_LEVELS,
    EXECUTION_TIMINGS,
    MAGNITUDE_LEVELS,
)


COARSE_FEATURE_DIM = 18  # action 8 + magnitude 3 + duration 3 + timing 2 + score + coordination flag


@dataclass(slots=True)
class CandidatePool:
    candidates: list[CoarseCandidateAction | None]
    candidate_mask: np.ndarray
    features: np.ndarray
    rejection_counts: dict[str, int]


def _one_hot(value: str, values: tuple[str, ...]) -> np.ndarray:
    encoded = np.zeros(len(values), dtype=np.float32)
    if value in values:
        encoded[values.index(value)] = 1.0
    return encoded


def candidate_features(candidate: CoarseCandidateAction) -> np.ndarray:
    action_order = (
        "NO_NEW_COMMAND", "HOLD_HEADING", "TURN_LEFT", "TURN_RIGHT",
        "CLIMB", "DESCEND", "ACCELERATE", "DECELERATE",
    )
    return np.concatenate(
        (
            _one_hot(candidate.macro_action, action_order),
            _one_hot(candidate.magnitude_level, MAGNITUDE_LEVELS),
            _one_hot(candidate.duration_level, DURATION_LEVELS),
            _one_hot(candidate.execution_timing, EXECUTION_TIMINGS),
            np.array([candidate.prior_score, float(candidate.coordination_intent == "PRIORITY_PASS")], dtype=np.float32),
        )
    )


def _family(candidate: CoarseCandidateAction) -> str:
    if candidate.macro_action in {"TURN_LEFT", "TURN_RIGHT"}:
        return "lateral"
    if candidate.macro_action in {"CLIMB", "DESCEND"}:
        return "vertical"
    if candidate.macro_action in {"ACCELERATE", "DECELERATE"}:
        return "speed"
    return candidate.macro_action


def build_candidate_pool(candidates: Iterable[CoarseCandidateAction], slots: int = 6) -> CandidatePool:
    """Deduplicate, retain cross-dimension diversity, then pad a fixed pool."""
    if slots <= 0:
        raise ValueError("candidate pool slots must be positive")
    rejection_counts: dict[str, int] = {}
    unique: dict[tuple[str, ...], CoarseCandidateAction] = {}
    for candidate in candidates:
        if not candidate.safe:
            reason = candidate.rejection_reason or "unsafe"
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        key = (
            candidate.aircraft_id,
            candidate.macro_action,
            candidate.magnitude_level,
            candidate.duration_level,
            candidate.execution_timing,
            candidate.priority_aircraft_id,
            candidate.yield_aircraft_id,
        )
        if key not in unique or candidate.prior_score > unique[key].prior_score:
            unique[key] = candidate
    ranked = sorted(unique.values(), key=lambda item: (-item.prior_score, item.candidate_id))
    selected: list[CoarseCandidateAction] = []
    families: set[str] = set()
    for candidate in ranked:
        family = _family(candidate)
        if family not in families:
            selected.append(candidate)
            families.add(family)
        if len(selected) == slots:
            break
    if len(selected) < slots:
        for candidate in ranked:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) == slots:
                break
    mask = np.zeros(slots, dtype=np.float32)
    features = np.zeros((slots, COARSE_FEATURE_DIM), dtype=np.float32)
    padded: list[CoarseCandidateAction | None] = [None] * slots
    for index, candidate in enumerate(selected):
        padded[index] = candidate
        mask[index] = 1.0
        features[index] = candidate_features(candidate)
    return CandidatePool(padded, mask, features, rejection_counts)


def level_value(level: str, maximum: float) -> float:
    """Representative value for safety screening; exact value remains RL-owned."""
    fractions = {"SMALL": 1.0 / 6.0, "MEDIUM": 0.5, "LARGE": 5.0 / 6.0}
    return float(maximum * fractions[level])


def duration_value(level: str, minimum_hold_s: float) -> float:
    nominal = {"SHORT": 10.0, "MEDIUM": 20.0, "LONG": 30.0}[level]
    return max(nominal, float(minimum_hold_s))


def to_legacy_candidate(candidate: CoarseCandidateAction, config) -> CandidateAction | None:
    """Map an executable coarse candidate to the present five-action baseline.

    HOLD_HEADING has no lossless equivalent in the baseline: NO_NEW_COMMAND
    continues an existing instruction and does not explicitly issue a heading
    hold, therefore this candidate is intentionally withheld from the adapter.
    """
    if candidate.macro_action == "HOLD_HEADING":
        return None
    ranges = config.command_ranges
    duration = duration_value(candidate.duration_level, config.runtime.min_command_hold_s)
    parameters: dict[str, float] = {}
    action_type = candidate.macro_action
    if action_type in {"TURN_LEFT", "TURN_RIGHT"}:
        delta = level_value(candidate.magnitude_level, ranges.heading_offset_deg)
        parameters = {"heading_delta_deg": -delta if action_type == "TURN_LEFT" else delta, "duration_s": duration}
    elif action_type in {"CLIMB", "DESCEND"}:
        delta = level_value(candidate.magnitude_level, ranges.altitude_delta_ft)
        parameters = {"altitude_delta_ft": delta if action_type == "CLIMB" else -delta, "duration_s": duration}
    elif action_type in {"ACCELERATE", "DECELERATE"}:
        delta = level_value(candidate.magnitude_level, ranges.speed_delta_kt)
        parameters = {"speed_delta_kt": delta if action_type == "ACCELERATE" else -delta, "duration_s": duration}
    elif action_type != "NO_NEW_COMMAND":
        return None
    return CandidateAction(
        candidate_id=candidate.candidate_id,
        aircraft_id=candidate.aircraft_id,
        action_type=action_type,
        parameters=parameters,
        prior_score=candidate.prior_score,
        expected_effect=candidate.expected_effect,
        intent=candidate.coordination_intent,
        minimum_duration_s=duration,
        release_stable_s=duration,
    )


def coarse_from_legacy(candidate: CandidateAction) -> CoarseCandidateAction:
    """Adapt cached v1.0 candidates into the v1.1 pool without losing safety data."""
    values = candidate.parameters
    action = candidate.action_type
    magnitude = "MEDIUM"
    duration = "MEDIUM"
    if "duration_s" in values:
        duration = "SHORT" if values["duration_s"] <= 12.5 else "LONG" if values["duration_s"] >= 25.0 else "MEDIUM"
    magnitude_value = abs(next((float(values[key]) for key in ("heading_delta_deg", "altitude_delta_ft", "speed_delta_kt") if key in values), 0.0))
    if action in {"TURN_LEFT", "TURN_RIGHT"}:
        magnitude = "SMALL" if magnitude_value <= 15.0 else "LARGE" if magnitude_value >= 30.0 else "MEDIUM"
    elif action in {"CLIMB", "DESCEND"}:
        magnitude = "SMALL" if magnitude_value <= 1000.0 else "LARGE" if magnitude_value >= 2000.0 else "MEDIUM"
    elif action in {"ACCELERATE", "DECELERATE"}:
        magnitude = "SMALL" if magnitude_value <= 10.0 else "LARGE" if magnitude_value >= 20.0 else "MEDIUM"
    supported = action in {
        "NO_NEW_COMMAND", "HOLD_HEADING", "TURN_LEFT", "TURN_RIGHT",
        "CLIMB", "DESCEND", "ACCELERATE", "DECELERATE",
    }
    return CoarseCandidateAction(
        candidate_id=candidate.candidate_id,
        aircraft_id=candidate.aircraft_id,
        # The present H-PPO baseline can cache RESUME_ROUTE, but v1.1 is a
        # conflict-resolution candidate language and intentionally excludes it.
        # Keep the object parseable while preventing it from entering the pool.
        macro_action=action if supported else "NO_NEW_COMMAND",
        magnitude_level=magnitude,
        duration_level=duration,
        execution_timing="IMMEDIATE",
        coordination_intent=candidate.intent if candidate.intent in {"NONE", "PRIORITY_PASS"} else "NONE",
        prior_score=candidate.prior_score,
        expected_effect=candidate.expected_effect,
        recovery_condition=candidate.recovery_action,
        safe=bool(candidate.safe and supported),
        rejection_reason=(
            candidate.rejection_reason
            if candidate.rejection_reason
            else ("legacy_macro_not_supported_by_coarse_language" if not supported else "")
        ),
        predicted_min_horizontal_km=candidate.predicted_min_horizontal_km,
        predicted_min_vertical_ft=candidate.predicted_min_vertical_ft,
    )
