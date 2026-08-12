import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_shanghai_training_data_v3 import (  # noqa: E402
    active_clearance_context,
    boundary_suffix,
    route_context_v3,
    semantic_label,
    strict_bool,
)


CONFIG = json.loads((ROOT / "configs" / "shanghai_training_data_v3.json").read_text(encoding="utf-8"))


class SemanticTests(unittest.TestCase):
    def test_repeated_speed_span_detects_later_bound(self):
        text = "速度三洞洞，稍后速度三洞洞以上"
        self.assertEqual(boundary_suffix(text, "速度三洞洞"), "above")

    def test_speed_above_becomes_lower_bound(self):
        label = semantic_label({
            "transcript": "东方六五拐拐速度三洞洞以上",
            "raw_span": "速度三洞洞",
            "intent_family": "speed",
            "action": "set_speed",
            "target_value": 300,
            "unit": "kt",
        }, CONFIG)
        self.assertEqual(label["action"], "maintain_min_speed")
        self.assertEqual(label["constraint_operator"], ">=")
        self.assertEqual(label["target_semantics"], "lower_bound")
        self.assertEqual(label["expanded_raw_span"], "速度三洞洞以上")

    def test_speed_normal_is_valid_categorical_label(self):
        label = semantic_label({
            "transcript": "东方一二三速度正常",
            "raw_span": "速度正常",
            "intent_family": "speed",
            "action": "speed_as_procedure",
            "target_value": None,
            "unit": None,
        }, CONFIG)
        self.assertEqual(label["action"], "resume_normal_speed")
        self.assertIsNone(label["target_value"])
        self.assertFalse(label["target_required"])
        self.assertEqual(label["target_semantics"], "cancel_active_speed_constraint")

    def test_speed_below_becomes_upper_bound(self):
        label = semantic_label({
            "transcript": "东方一二三减速二五洞以下",
            "raw_span": "减速二五洞",
            "intent_family": "speed",
            "action": "reduce_speed",
            "target_value": 250,
            "unit": "kt",
        }, CONFIG)
        self.assertEqual(label["action"], "reduce_to_max_speed")
        self.assertEqual(label["constraint_operator"], "<=")
        self.assertEqual(label["target_semantics"], "upper_bound")

    def test_vertical_rate_constraint_is_preserved(self):
        label = semantic_label({
            "transcript": "上升高度六千，上升率两千以上",
            "raw_span": "上升高度六千",
            "intent_family": "altitude",
            "action": "climb",
            "target_value": 6000,
            "unit": "m",
        }, CONFIG)
        self.assertEqual(label["vertical_rate_constraints"][0]["operator"], ">=")
        self.assertEqual(label["vertical_rate_constraints"][0]["target_value"], 2000)

    def test_nan_is_not_true(self):
        self.assertFalse(strict_bool(np.nan))


class HistoryTests(unittest.TestCase):
    def test_newest_clearance_wins(self):
        history = [
            {"intent_family": "speed", "action": "reduce_speed", "constraint_operator": "=", "target_value": 250, "target_unit": "kt", "target_semantics": "exact_value", "seconds_since_previous_event": 10, "previous_reference_event_id": "new", "previous_operational_cues": [], "full_previous_transcript": "速度二五洞"},
            {"intent_family": "speed", "action": "set_speed", "constraint_operator": "=", "target_value": 300, "target_unit": "kt", "target_semantics": "exact_value", "seconds_since_previous_event": 100, "previous_reference_event_id": "old", "previous_operational_cues": [], "full_previous_transcript": "速度三洞洞"},
        ]
        result = active_clearance_context(history)
        self.assertEqual(result["active_speed_constraint"]["previous_reference_event_id"], "new")

    def test_resume_normal_speed_cancels_older_restriction(self):
        history = [
            {"intent_family": "speed", "action": "resume_normal_speed", "constraint_operator": None, "target_value": None, "target_unit": None, "target_semantics": "cancel_active_speed_constraint", "seconds_since_previous_event": 10, "previous_reference_event_id": "normal", "previous_operational_cues": [], "full_previous_transcript": "速度正常"},
            {"intent_family": "speed", "action": "reduce_speed", "constraint_operator": "=", "target_value": 250, "target_unit": "kt", "target_semantics": "exact_value", "seconds_since_previous_event": 100, "previous_reference_event_id": "old", "previous_operational_cues": [], "full_previous_transcript": "速度二五洞"},
        ]
        result = active_clearance_context(history)
        self.assertIsNone(result["active_speed_constraint"])
        self.assertTrue(result["normal_speed_resumed"])
        self.assertEqual(result["most_recent_speed_event"]["previous_reference_event_id"], "normal")


class RouteTests(unittest.TestCase):
    def test_stale_ispass_is_replaced_by_eto_bracket(self):
        record = {
            "quality": {"plan_features_causally_available": True},
            "pre_command_state_raw": {"event_time_epoch": 1000.0, "latitude": 31.0, "longitude": 121.0, "plan_filtim_utc": "1970-01-01T00:16:30+00:00"},
            "route_points_raw_and_resolved": [
                {"ptid_raw": "PD302", "ispass_raw": "Y", "eto_delta_sec": -300.0, "resolved_latitude": None, "resolved_longitude": None, "planned_level_m": None, "planned_level_source_unit": None, "planned_level_parse_status": None},
                {"ptid_raw": "PD303", "ispass_raw": "N", "eto_delta_sec": -168.0, "resolved_latitude": None, "resolved_longitude": None, "planned_level_m": 3000, "planned_level_source_unit": "m", "planned_level_parse_status": "ok"},
                {"ptid_raw": "SS303", "ispass_raw": "N", "eto_delta_sec": 0.3, "resolved_latitude": None, "resolved_longitude": None, "planned_level_m": 3000, "planned_level_source_unit": "m", "planned_level_parse_status": "ok"},
                {"ptid_raw": "SS304", "ispass_raw": "N", "eto_delta_sec": 120.0, "resolved_latitude": None, "resolved_longitude": None, "planned_level_m": 3000, "planned_level_source_unit": "m", "planned_level_parse_status": "ok"},
            ],
        }
        route = route_context_v3(record, CONFIG)
        self.assertEqual(route["active_leg_method"], "eto_time_bracket")
        self.assertEqual(route["active_leg"], "PD303->SS303")
        self.assertGreater(route["raw_eto_delta_to_next_fix_sec"], 0)
        self.assertTrue(route["ispass_eto_conflict"])

    def test_stale_plan_snapshot_is_masked(self):
        record = {
            "quality": {"plan_features_causally_available": True},
            "pre_command_state_raw": {"event_time_epoch": 5000.0, "latitude": 31.0, "longitude": 121.0, "plan_filtim_utc": "1970-01-01T00:16:40+00:00"},
            "route_points_raw_and_resolved": [
                {"ptid_raw": "A", "ispass_raw": "Y", "eto_delta_sec": -10.0, "resolved_latitude": 31.0, "resolved_longitude": 120.9},
                {"ptid_raw": "B", "ispass_raw": "N", "eto_delta_sec": 10.0, "resolved_latitude": 31.0, "resolved_longitude": 121.1},
            ],
        }
        route = route_context_v3(record, CONFIG)
        self.assertFalse(route["route_feature_mask"])
        self.assertEqual(route["active_leg_method"], "stale_plan_snapshot")


if __name__ == "__main__":
    unittest.main()
