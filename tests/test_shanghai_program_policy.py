from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from shanghai_program_policy import (  # noqa: E402
    candidate_safety,
    command_for_candidate,
    generate_candidates,
    route_context,
    runtime_candidate_rejection,
    top_candidate_decision,
)


CONFIG = {
    "candidates": {
        "altitude_levels_m": [2400, 3000, 3600, 4200, 6000],
        "speed_targets_kt": [210, 230, 250, 280, 300],
        "minimum_altitude_change_m": 150,
        "minimum_speed_change_kt": 8,
        "phase_direction_gate": True,
    }
}


def state(phase: str, altitude: float = 3600.0, speed: float = 250.0) -> dict:
    return {
        "phase": phase,
        "altitude_m": altitude,
        "ground_speed_kt": speed,
        "next_planned_level_m": 3000.0,
        "requested_level_m": 6000.0,
    }


class ProgramContextTests(unittest.TestCase):
    def test_route_context_finds_first_unpassed_fix(self) -> None:
        points = [
            {
                "ptid": "A",
                "flight_level": "S0300",
                "eto_utc": "2025-07-01T00:00:00Z",
                "ispass": "Y",
            },
            {
                "ptid": "B",
                "flight_level": "S0420",
                "eto_utc": "2025-07-01T00:05:00Z",
                "ispass": "N",
            },
        ]
        context = route_context(points, 1751328000.0)
        self.assertEqual(context["active_leg"], "A->B")
        self.assertEqual(context["next_fix"], "B")
        self.assertEqual(context["next_planned_level_m"], 4200.0)
        self.assertEqual(context["route_progress"], 0.5)

    def test_phase_gate_blocks_wrong_altitude_direction(self) -> None:
        departure = generate_candidates(state("departure"), CONFIG)
        arrival = generate_candidates(state("arrival"), CONFIG)
        self.assertFalse(
            any(
                item["candidate_kind"] == "altitude"
                and item["candidate_direction"] == "descend"
                for item in departure
            )
        )
        self.assertFalse(
            any(
                item["candidate_kind"] == "altitude"
                and item["candidate_direction"] == "climb"
                for item in arrival
            )
        )

    def test_near_current_targets_are_maintain_candidates(self) -> None:
        candidates = generate_candidates(state("arrival", altitude=3590.0), CONFIG)
        target = next(item for item in candidates if item["candidate_id"] == "ALT_3600")
        self.assertEqual(target["candidate_direction"], "maintain")


class DecisionTests(unittest.TestCase):
    def test_speed_memory_allows_same_direction_new_target(self) -> None:
        runtime = {
            "aircraft_command_cooldown_sec": 60,
            "same_axis_cooldown_sec": 12,
            "target_memory_sec": 900,
        }
        candidate = {
            "callsign": "CES100",
            "candidate_kind": "speed",
            "candidate_id": "SPD_280",
            "candidate_target": 280,
            "phase": "departure",
            "ground_speed_kt": 250,
        }
        reason = runtime_candidate_rejection(
            candidate,
            1061,
            {"CES100": 1060},
            {("CES100", "speed"): 1012},
            {("CES100", "speed"): ("SPD_270", 1000)},
            runtime,
        )
        self.assertIsNone(reason)

    def test_speed_memory_rejects_repeat_and_reversal(self) -> None:
        runtime = {
            "aircraft_command_cooldown_sec": 60,
            "same_axis_cooldown_sec": 12,
            "target_memory_sec": 900,
        }
        memory = {("CES100", "speed"): ("SPD_270", 1000)}
        base = {
            "callsign": "CES100",
            "candidate_kind": "speed",
            "phase": "departure",
            "ground_speed_kt": 250,
        }
        repeat = {**base, "candidate_id": "SPD_270", "candidate_target": 270}
        reversal = {**base, "candidate_id": "SPD_230", "candidate_target": 230}
        self.assertEqual(
            runtime_candidate_rejection(
                repeat, 1061, {}, {}, memory, runtime
            ),
            "repeat_target",
        )
        self.assertEqual(
            runtime_candidate_rejection(
                reversal, 1061, {}, {}, memory, runtime
            ),
            "speed_direction_reversal",
        )

    def test_hold_wins_below_margin_threshold(self) -> None:
        frame = pd.DataFrame(
            [
                {"candidate_id": "HOLD", "candidate_score": 0.6},
                {"candidate_id": "ALT_3000", "candidate_score": 0.55},
            ]
        )
        decision = top_candidate_decision(frame, threshold=0.0)
        self.assertEqual(decision["selected_candidate_id"], "HOLD")

    def test_maintain_altitude_command_uses_parseable_form(self) -> None:
        command = command_for_candidate(
            "CES100",
            {
                "candidate_kind": "altitude",
                "candidate_target": 3600,
                "candidate_direction": "maintain",
            },
        )
        self.assertEqual(command, "ALT_M CES100,3600")

    def test_pairwise_safety_rejects_same_track_coincident_pair(self) -> None:
        target = {
            "trajectory_id": "a",
            "callsign": "AAA1",
            "latitude": 31.0,
            "longitude": 121.0,
            "altitude_m": 3000.0,
            "ground_speed_mps": 120.0,
            "track_heading_deg": 90.0,
            "vertical_rate_mps": 0.0,
        }
        other = {
            **target,
            "trajectory_id": "b",
            "callsign": "BBB2",
            "longitude": 121.01,
        }
        candidate = {
            "candidate_kind": "speed",
            "candidate_target": 230.0,
        }
        safety = {
            "horizon_sec": 60,
            "step_sec": 4,
            "command_delay_sec": 4,
            "vertical_rate_mps": 10,
            "speed_acceleration_ktps": 1,
            "minimum_horizontal_separation_nm": 5,
            "minimum_vertical_separation_m": 300,
        }
        result = candidate_safety(target, candidate, [target, other], safety)
        self.assertFalse(result["safe"])
        self.assertEqual(result["checked_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
