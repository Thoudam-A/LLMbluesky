"""Stable state contracts shared by the BlueSky adapter and research tooling.

The policy consumes normalized numeric tensors only.  This module documents the
meaning and ordering of every feature so that offline data generation, model
training and LLM scene descriptions do not silently drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeatureGroup:
    name: str
    size: int
    description: str


# `_own_features()` in hppo_environment.py.  Angles are represented as sin/cos
# pairs; distances, speeds and altitudes are normalized before entering policy.
OWN_AIRCRAFT_FEATURES = FeatureGroup(
    "own_aircraft",
    18,
    "position, speed, altitude, heading, route intent and goal-progress features",
)
COMMAND_CONTEXT_FEATURES = FeatureGroup(
    "command_context",
    21,
    "macro one-hot, age/hold, route and conflict context, command phase, target progress, effect status and branch cooldowns",
)
SEPARATION_STANDARD_FEATURES = FeatureGroup(
    "separation_standard",
    3,
    "configured horizontal, vertical and temporal separation standards",
)
INTRUDER_FEATURES = FeatureGroup(
    "intruder",
    10,
    "horizontal range, relative bearing, relative altitude/speed/track, TCPA, current loss flag and time gap",
)


LOCAL_OBSERVATION_GROUPS = (
    OWN_AIRCRAFT_FEATURES,
    COMMAND_CONTEXT_FEATURES,
    SEPARATION_STANDARD_FEATURES,
)


def expected_local_observation_dim(max_intruders: int) -> int:
    """Return the fixed Top-K local-observation dimension."""
    if max_intruders < 0:
        raise ValueError("max_intruders must be non-negative")
    return sum(group.size for group in LOCAL_OBSERVATION_GROUPS) + max_intruders * INTRUDER_FEATURES.size


def expected_global_slot_dim() -> int:
    """Own-aircraft features plus conflict, command and separation context."""
    # 18 own + active + severity + command-active + command-age + 3 standards + time-gap.
    return 26


def validate_network_dimensions(config) -> None:
    """Fail early when a config and the environment feature layout diverge."""
    local_dim = expected_local_observation_dim(int(config.max_intruders))
    if int(config.network.local_obs_dim) != local_dim:
        raise ValueError(
            "local_obs_dim does not match the fixed Top-K observation contract: "
            f"configured={config.network.local_obs_dim}, expected={local_dim}"
        )
    global_dim = expected_global_slot_dim()
    if int(config.network.global_slot_dim) != global_dim:
        raise ValueError(
            "global_slot_dim does not match the padded Critic slot contract: "
            f"configured={config.network.global_slot_dim}, expected={global_dim}"
        )


def observation_contract() -> dict[str, object]:
    """Machine-readable layout for experiment metadata and future dataset tools."""
    return {
        "version": "hppo-observation-v2",
        "local_groups": [
            {"name": group.name, "size": group.size, "description": group.description}
            for group in (*LOCAL_OBSERVATION_GROUPS, INTRUDER_FEATURES)
        ],
        "global_slot_dim": expected_global_slot_dim(),
        "global_pooling": "masked mean pooling over fixed padded aircraft slots",
    }
