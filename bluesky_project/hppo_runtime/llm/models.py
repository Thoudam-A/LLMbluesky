from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any


ACTION_TO_MACRO = {
    "NO_NEW_COMMAND": 0,
    "RESUME_ROUTE": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 2,
    "CLIMB": 3,
    "DESCEND": 3,
    "ACCELERATE": 4,
    "DECELERATE": 4,
}

COARSE_MACRO_ACTIONS = (
    "NO_NEW_COMMAND",
    "HOLD_HEADING",
    "TURN_LEFT",
    "TURN_RIGHT",
    "CLIMB",
    "DESCEND",
    "ACCELERATE",
    "DECELERATE",
)
MAGNITUDE_LEVELS = ("SMALL", "MEDIUM", "LARGE")
DURATION_LEVELS = ("SHORT", "MEDIUM", "LONG")
EXECUTION_TIMINGS = ("IMMEDIATE", "AFTER_CURRENT_HOLD")
COORDINATION_INTENTS = ("NONE", "PRIORITY_PASS")


@dataclass(slots=True)
class AircraftSnapshot:
    aircraft_id: str
    latitude_deg: float
    longitude_deg: float
    altitude_ft: float
    speed_kt: float
    heading_deg: float
    vertical_speed_fpm: float
    current_macro: int
    hold_remaining_s: float


@dataclass(slots=True)
class ConflictEdge:
    ownship_id: str
    intruder_id: str
    horizontal_km: float
    vertical_ft: float
    tcpa_s: float
    time_gap_s: float | None
    relative_bearing_deg: float
    severity: float
    loss_of_separation: bool


@dataclass(slots=True)
class ConflictScene:
    schema_version: str
    scene_id: str
    simulation_time_s: float
    separation_horizontal_km: float
    separation_vertical_ft: float
    separation_time_s: float
    aircraft: list[AircraftSnapshot]
    conflicts: list[ConflictEdge]
    target_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CandidateAction:
    candidate_id: str
    aircraft_id: str
    action_type: str
    parameters: dict[str, float] = field(default_factory=dict)
    prior_score: float = 0.0
    expected_effect: str = ""
    intent: str = ""
    minimum_duration_s: float = 10.0
    release_stable_s: float = 15.0
    recovery_action: str = "RESUME_ROUTE"
    fallback_candidate_id: str = ""
    safe: bool = True
    rejection_reason: str = ""
    predicted_min_horizontal_km: float | None = None
    predicted_min_vertical_ft: float | None = None

    @property
    def macro_action(self) -> int:
        return ACTION_TO_MACRO[self.action_type]


@dataclass(slots=True)
class CandidateProposal:
    schema_version: str
    scene_id: str
    provider: str
    prompt_version: str
    candidates: list[CandidateAction]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "provider": self.provider,
            "prompt_version": self.prompt_version,
            "candidates": [asdict(candidate) for candidate in self.candidates],
        }


@dataclass(slots=True)
class CoarseCandidateAction:
    """Versioned LLM candidate language used before H-PPO parameter selection.

    It deliberately contains levels rather than executable continuous values.
    ``HOLD_HEADING`` is retained for the future eight-action policy but cannot
    be losslessly executed by the present five-action H-PPO baseline.
    """

    candidate_id: str
    aircraft_id: str
    macro_action: str
    magnitude_level: str
    duration_level: str
    execution_timing: str
    coordination_intent: str = "NONE"
    priority_aircraft_id: str = ""
    yield_aircraft_id: str = ""
    parameters: dict[str, float] = field(default_factory=dict)
    prior_score: float = 0.0
    expected_effect: str = ""
    reason: str = ""
    recovery_condition: str = "stable_separation"
    safe: bool = True
    rejection_reason: str = ""
    predicted_min_horizontal_km: float | None = None
    predicted_min_vertical_ft: float | None = None


@dataclass(slots=True)
class CoarseCandidateProposal:
    schema_version: str
    scene_id: str
    provider: str
    prompt_version: str
    candidates: list[CoarseCandidateAction]


def parse_coarse_candidate_proposal(
    payload: str | dict[str, Any], provider: str, prompt_version: str
) -> CoarseCandidateProposal:
    """Parse the strict v1.1 LLM candidate language without executing actions."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(data, dict) or set(data) != {"schema_version", "scene_id", "candidates"}:
        raise ValueError("coarse candidate output must contain only schema_version, scene_id, and candidates")
    if data["schema_version"] != "1.1" or not isinstance(data["scene_id"], str):
        raise ValueError("unsupported coarse candidate schema version or invalid scene_id")
    if not isinstance(data["candidates"], list):
        raise ValueError("coarse candidates must be a list")
    required = {
        "candidate_id",
        "aircraft_id",
        "macro_action",
        "magnitude_level",
        "duration_level",
        "parameters",
        "prior_score",
        "expected_effect",
        "recovery_condition",
    }
    optional = {
        "execution_timing",
        "coordination_intent",
        "priority_aircraft_id",
        "yield_aircraft_id",
        "reason",
    }
    candidates: list[CoarseCandidateAction] = []
    for item in data["candidates"]:
        if not isinstance(item, dict) or not required.issubset(item) or not set(item).issubset(required | optional):
            raise ValueError("coarse candidate has missing or unknown fields")
        if item["macro_action"] not in COARSE_MACRO_ACTIONS:
            raise ValueError(f"unknown coarse macro action: {item['macro_action']}")
        if item["magnitude_level"] not in MAGNITUDE_LEVELS or item["duration_level"] not in DURATION_LEVELS:
            raise ValueError("unknown magnitude or duration level")
        if item.get("execution_timing", "IMMEDIATE") not in EXECUTION_TIMINGS:
            raise ValueError("unknown execution timing")
        if item.get("coordination_intent", "NONE") not in COORDINATION_INTENTS:
            raise ValueError("unknown coordination intent")
        if not isinstance(item["parameters"], dict) or item["parameters"]:
            raise ValueError("coarse candidate parameters must be an empty object")
        score = float(item["prior_score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError("coarse candidate prior_score must be in [0,1]")
        candidates.append(
            CoarseCandidateAction(
                candidate_id=str(item["candidate_id"]),
                aircraft_id=str(item["aircraft_id"]),
                macro_action=str(item["macro_action"]),
                magnitude_level=str(item["magnitude_level"]),
                duration_level=str(item["duration_level"]),
                execution_timing=str(item.get("execution_timing", "IMMEDIATE")),
                coordination_intent=str(item.get("coordination_intent", "NONE")),
                priority_aircraft_id=str(item.get("priority_aircraft_id", "")),
                yield_aircraft_id=str(item.get("yield_aircraft_id", "")),
                parameters={},
                prior_score=score,
                expected_effect=str(item["expected_effect"]),
                reason=str(item.get("reason", "")),
                recovery_condition=str(item["recovery_condition"]),
            )
        )
    return CoarseCandidateProposal("1.1", data["scene_id"], provider, prompt_version, candidates)


def parse_candidate_proposal(payload: str | dict[str, Any], provider: str, prompt_version: str) -> CandidateProposal:
    data = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(data, dict) or set(data) != {"schema_version", "scene_id", "candidates"}:
        raise ValueError("candidate output must contain only schema_version, scene_id, and candidates")
    if data["schema_version"] != "1.0" or not isinstance(data["scene_id"], str):
        raise ValueError("unsupported schema version or invalid scene_id")
    if not isinstance(data["candidates"], list):
        raise ValueError("candidates must be a list")
    candidates: list[CandidateAction] = []
    required = {"candidate_id", "aircraft_id", "action_type", "parameters", "prior_score"}
    optional = {
        "expected_effect",
        "intent",
        "minimum_duration_s",
        "release_stable_s",
        "recovery_action",
        "fallback_candidate_id",
    }
    for item in data["candidates"]:
        if not isinstance(item, dict) or not required.issubset(item) or not set(item).issubset(required | optional):
            raise ValueError("candidate has missing or unknown fields")
        if item["action_type"] not in ACTION_TO_MACRO:
            raise ValueError(f"unknown action type: {item['action_type']}")
        if not isinstance(item["parameters"], dict):
            raise ValueError("candidate parameters must be an object")
        score = float(item["prior_score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError("candidate prior_score must be in [0,1]")
        candidates.append(
            CandidateAction(
                candidate_id=str(item["candidate_id"]),
                aircraft_id=str(item["aircraft_id"]),
                action_type=str(item["action_type"]),
                parameters={str(key): float(value) for key, value in item["parameters"].items()},
                prior_score=score,
                expected_effect=str(item.get("expected_effect", "")),
                intent=str(item.get("intent", "")),
                minimum_duration_s=float(item.get("minimum_duration_s", 10.0)),
                release_stable_s=float(item.get("release_stable_s", 15.0)),
                recovery_action=str(item.get("recovery_action", "RESUME_ROUTE")),
                fallback_candidate_id=str(item.get("fallback_candidate_id", "")),
            )
        )
    return CandidateProposal(
        schema_version="1.0",
        scene_id=data["scene_id"],
        provider=provider,
        prompt_version=prompt_version,
        candidates=candidates,
    )


def conflict_scene_from_dict(data: dict[str, Any]) -> ConflictScene:
    return ConflictScene(
        schema_version=str(data["schema_version"]),
        scene_id=str(data["scene_id"]),
        simulation_time_s=float(data["simulation_time_s"]),
        separation_horizontal_km=float(data["separation_horizontal_km"]),
        separation_vertical_ft=float(data["separation_vertical_ft"]),
        separation_time_s=float(data["separation_time_s"]),
        aircraft=[AircraftSnapshot(**item) for item in data["aircraft"]],
        conflicts=[ConflictEdge(**item) for item in data["conflicts"]],
        target_ids=[str(item) for item in data["target_ids"]],
    )


def candidate_proposal_from_dict(data: dict[str, Any]) -> CandidateProposal:
    return CandidateProposal(
        schema_version=str(data["schema_version"]),
        scene_id=str(data["scene_id"]),
        provider=str(data["provider"]),
        prompt_version=str(data["prompt_version"]),
        candidates=[CandidateAction(**item) for item in data["candidates"]],
    )
