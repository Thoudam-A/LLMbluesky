SCENE_SCHEMA_VERSION = "1.0"

CANDIDATE_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "scene_id", "candidates"],
    "properties": {
        "schema_version": {"const": SCENE_SCHEMA_VERSION},
        "scene_id": {"type": "string"},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["candidate_id", "aircraft_id", "action_type", "parameters", "prior_score"],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "aircraft_id": {"type": "string"},
                    "action_type": {
                        "enum": [
                            "NO_NEW_COMMAND",
                            "RESUME_ROUTE",
                            "TURN_LEFT",
                            "TURN_RIGHT",
                            "CLIMB",
                            "DESCEND",
                            "ACCELERATE",
                            "DECELERATE",
                        ]
                    },
                    "parameters": {"type": "object"},
                    "prior_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "expected_effect": {"type": "string"},
                    "intent": {"type": "string"},
                    "minimum_duration_s": {"type": "number", "minimum": 0.0},
                    "release_stable_s": {"type": "number", "minimum": 0.0},
                    "recovery_action": {"enum": ["RESUME_ROUTE"]},
                    "fallback_candidate_id": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

COARSE_CANDIDATE_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "scene_id", "candidates"],
    "properties": {
        "schema_version": {"const": "1.1"},
        "scene_id": {"type": "string"},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "candidate_id", "aircraft_id", "macro_action", "magnitude_level",
                    "duration_level", "parameters", "prior_score", "expected_effect", "recovery_condition",
                ],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "aircraft_id": {"type": "string"},
                    "coordination_intent": {"enum": ["NONE", "PRIORITY_PASS"]},
                    "priority_aircraft_id": {"type": "string"},
                    "yield_aircraft_id": {"type": "string"},
                    "macro_action": {"enum": ["NO_NEW_COMMAND", "HOLD_HEADING", "TURN_LEFT", "TURN_RIGHT", "CLIMB", "DESCEND", "ACCELERATE", "DECELERATE"]},
                    "magnitude_level": {"enum": ["SMALL", "MEDIUM", "LARGE"]},
                    "duration_level": {"enum": ["SHORT", "MEDIUM", "LONG"]},
                    "execution_timing": {"enum": ["IMMEDIATE", "AFTER_CURRENT_HOLD"]},
                    "parameters": {"type": "object", "maxProperties": 0},
                    "prior_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "expected_effect": {"type": "string"},
                    "reason": {"type": "string"},
                    "recovery_condition": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}
