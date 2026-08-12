from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_shanghai_lossless_training_data_v2 import (  # noqa: E402
    flatten,
    direction_consistent,
    pair_relation,
    parse_flight_level,
    rank_relations,
    route_context_v2,
    select_pre_state,
    select_causal_other_row_indices,
    structural_quality,
)


CONFIG = {
    "state_link": {
        "desired_pre_command_sec": 4,
        "tier_a_max_target_delta_sec": 8,
        "tier_b_max_target_delta_sec": 16,
        "fallback_max_target_delta_sec": 32,
    },
    "core_scope": {
        "minimum_altitude_m": 1400,
        "maximum_altitude_m": 6200,
    },
}


class StateLinkTests(unittest.TestCase):
    def test_pre_state_never_uses_post_command_point(self) -> None:
        rows = [
            {"event_time_epoch": 92.0},
            {"event_time_epoch": 96.0},
            {"event_time_epoch": 101.0},
        ]
        state, audit = select_pre_state(
            [92.0, 96.0, 101.0], rows, 100.0, CONFIG
        )
        self.assertEqual(state["event_time_epoch"], 96.0)
        self.assertEqual(audit["pre_state_age_before_command_sec"], 4.0)

    def test_state_exactly_at_command_time_is_not_selected(self) -> None:
        rows = [{"event_time_epoch": 100.0}]
        state, audit = select_pre_state([100.0], rows, 100.0, CONFIG)
        self.assertIsNone(state)
        self.assertFalse(audit["pre_state_linked"])

    def test_state_beyond_tier_b_window_is_audit_only(self) -> None:
        reference = {"target_value": 3000}
        state = {"inside_sector": True, "altitude_m": 3000}
        tier, issues, use, weight = structural_quality(
            reference, state, {"pre_state_target_delta_sec": 20.0}, True, CONFIG
        )
        self.assertEqual(tier, "C")
        self.assertIn("pre_state_target_delta_gt_16s", issues)
        self.assertEqual(use, "audit_only")
        self.assertEqual(weight, 0.0)

    def test_boundary_state_is_weighted_not_dropped(self) -> None:
        reference = {"target_value": 3000}
        state = {"inside_sector": False, "altitude_m": 3000}
        link = {"pre_state_target_delta_sec": 1.0}
        tier, issues, use, weight = structural_quality(
            reference, state, link, True, CONFIG
        )
        self.assertEqual(tier, "B")
        self.assertIn("outside_sector_boundary_context", issues)
        self.assertEqual(use, "weak_train_weighted")
        self.assertGreater(weight, 0.0)

    def test_missing_target_remains_audit_only(self) -> None:
        state = {"inside_sector": True, "altitude_m": 3000}
        tier, issues, use, weight = structural_quality(
            {"target_value": None},
            state,
            {"pre_state_target_delta_sec": 0.0},
            None,
            CONFIG,
        )
        self.assertEqual(tier, "C")
        self.assertIn("missing_numeric_target", issues)
        self.assertEqual(use, "audit_only")
        self.assertEqual(weight, 0.0)

    def test_speed_action_alias_is_direction_checked(self) -> None:
        reference = {
            "intent_family": "speed",
            "action": "increase_speed",
            "target_value": 300,
        }
        state = {"ground_speed_mps": 160.0}
        self.assertFalse(direction_consistent(reference, state))


class RouteAndRelationTests(unittest.TestCase):
    def test_causal_other_selection_deduplicates_and_rejects_future(self) -> None:
        rows = [
            {"trajectory_id": "DUP", "event_time_epoch": 98.0},
            {"trajectory_id": "DUP", "event_time_epoch": 99.0},
            {"trajectory_id": "DUP", "event_time_epoch": 101.0},
            {"trajectory_id": "ONE", "event_time_epoch": 97.0},
        ]
        self.assertEqual(select_causal_other_row_indices(rows, 100.0), {1, 3})

    def test_relation_ranks_remain_unique_for_duplicate_trajectory_ids(self) -> None:
        rows = [
            {
                "other_trajectory_id": "DUP",
                "relation_source_row_index": 0,
                "relation_valid": True,
                "predicted_pair_conflict": False,
                "cpa_horizontal_nm": 2.0,
                "vertical_at_cpa_m": 400.0,
            },
            {
                "other_trajectory_id": "DUP",
                "relation_source_row_index": 1,
                "relation_valid": True,
                "predicted_pair_conflict": False,
                "cpa_horizontal_nm": 1.0,
                "vertical_at_cpa_m": 400.0,
            },
        ]
        rank_relations(rows)
        self.assertEqual([row["relation_risk_rank"] for row in rows], [1, 2])
        self.assertEqual([row["relation_source_row_index"] for row in rows], [1, 0])
    def test_flight_level_parser_preserves_metric_and_feet_semantics(self) -> None:
        metric = parse_flight_level("S0600")
        feet = parse_flight_level("F240")
        self.assertEqual(metric["value_m"], 6000.0)
        self.assertEqual(metric["source_unit"], "tens_of_metres")
        self.assertAlmostEqual(feet["value_m"], 7315.2)
        self.assertEqual(feet["source_unit"], "hundreds_of_feet")
        self.assertEqual(feet["parse_status"], "parsed_feet_code")

    def test_geometric_route_context_uses_resolved_leg(self) -> None:
        row = {
            "event_time_epoch": 1000.0,
            "latitude": 31.0,
            "longitude": 121.05,
            "ground_speed_mps": 150.0,
            "plan_rtepts_json": (
                '[{"ptid":"A","flight_level":"S0300","eto_utc":"2025-01-01T00:00:00Z","ispass":"N"},'
                '{"ptid":"B","flight_level":"S0420","eto_utc":"2025-01-01T00:10:00Z","ispass":"N"}]'
            ),
        }
        nav = {
            "A": [(31.0, 121.0, "test")],
            "B": [(31.0, 121.1, "test")],
        }
        context, points = route_context_v2(row, nav, 50.0)
        self.assertEqual(context["active_leg_method"], "geometric_projection")
        self.assertEqual(context["active_leg"], "A->B")
        self.assertGreater(context["estimated_seconds_to_next_fix"], 0.0)
        self.assertEqual(len(points), 2)

    def test_pair_relation_keeps_vertical_at_same_cpa(self) -> None:
        target = {
            "latitude": 31.0,
            "longitude": 121.0,
            "track_heading_deg": 90.0,
            "ground_speed_mps": 100.0,
            "altitude_m": 3000.0,
            "vertical_rate_mps": 5.0,
        }
        other = {
            "latitude": 31.0,
            "longitude": 121.1,
            "track_heading_deg": 270.0,
            "ground_speed_mps": 100.0,
            "altitude_m": 3600.0,
            "vertical_rate_mps": -5.0,
        }
        relation = pair_relation(target, other, 300.0)
        self.assertIsNotNone(relation)
        self.assertLess(relation["cpa_horizontal_nm"], relation["current_horizontal_nm"])
        self.assertNotEqual(
            relation["vertical_at_cpa_m"], relation["current_vertical_m"]
        )

    def test_pair_relation_flags_missing_vertical_rate_assumption(self) -> None:
        target = {
            "latitude": 31.0,
            "longitude": 121.0,
            "track_heading_deg": 90.0,
            "ground_speed_mps": 100.0,
            "altitude_m": 3000.0,
            "vertical_rate_mps": float("nan"),
        }
        other = {
            "latitude": 31.0,
            "longitude": 121.1,
            "track_heading_deg": 270.0,
            "ground_speed_mps": 100.0,
            "altitude_m": 3600.0,
            "vertical_rate_mps": 0.0,
        }
        relation = pair_relation(target, other, 300.0)
        self.assertTrue(relation["target_vertical_rate_assumed_zero"])
        self.assertFalse(relation["other_vertical_rate_assumed_zero"])
        self.assertTrue(math.isfinite(relation["vertical_at_cpa_m"]))

    def test_flatten_preserves_raw_values_without_overwrite(self) -> None:
        flattened = flatten(
            "state_raw_", {"requested_flight_level": "S0600", "sid": None}
        )
        self.assertEqual(flattened["state_raw_requested_flight_level"], "S0600")
        self.assertIsNone(flattened["state_raw_sid"])


if __name__ == "__main__":
    unittest.main()
