import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_shanghai_causal_alignment_v4 import (  # noqa: E402
    PlanVersion,
    CausalPlanIndex,
    normalize_callsign,
    parse_filtim_epoch,
    parse_message,
    point_in_polygon,
    route_context,
)


CONFIG = {
    "flight_plan": {"maximum_eobt_distance_hours": 48, "maximum_future_eobt_hours": 1, "prefer_source_ifplid": True},
    "route": {"at_fix_tolerance_sec": 5, "maximum_unbracketed_next_fix_sec": 600, "maximum_geometric_cross_track_nm": 20, "maximum_geometric_leg_length_nm": 100},
}


class PlanAlignmentTests(unittest.TestCase):
    def test_both_receive_and_filtim_must_be_causal(self):
        versions = [
            PlanVersion("old", 90, 90, "ABC1", "P1", {"IFPLID": "P1", "ARCID": "ABC1", "CFL": "S0300"}, [], False),
            PlanVersion("future_receive", 110, 95, "ABC1", "P1", {"IFPLID": "P1", "ARCID": "ABC1", "CFL": "S0600"}, [], False),
            PlanVersion("future_filtim", 95, 110, "ABC1", "P1", {"IFPLID": "P1", "ARCID": "ABC1", "CFL": "S0900"}, [], False),
        ]
        result = CausalPlanIndex(versions, CONFIG).match("ABC1", 100, "P1")
        self.assertEqual(result["fields"]["CFL"], "S0300")
        self.assertEqual(result["version_count_replayed"], 1)

    def test_explicit_empty_update_clears_old_star(self):
        versions = [
            PlanVersion("a", 80, 80, "ABC1", "P1", {"IFPLID": "P1", "ARCID": "ABC1", "STAR": "OLD1"}, [], False),
            PlanVersion("b", 90, 90, "ABC1", "P1", {"STAR": ""}, [], False),
        ]
        result = CausalPlanIndex(versions, CONFIG).match("ABC1", 100, "P1")
        self.assertEqual(result["fields"]["STAR"], "")

    def test_message_route_parser(self):
        fields, points, present = parse_message("-FILTIM 20250701000000\n-IFPLID 1\n-ARCID ABC1\n-BEGIN RTEPTS\n-PT -PTID A-FL S0300-ETO 20250701000100-ISPASS N\n-END RTEPTS")
        self.assertEqual(fields["ARCID"], "ABC1")
        self.assertTrue(present)
        self.assertEqual(points[0]["ptid"], "A")

    def test_six_digit_filtim_uses_nearest_receive_day(self):
        receive = 1751299476.463  # 2025-06-30 16:04:36 UTC
        parsed = parse_filtim_epoch("160431", receive)
        self.assertAlmostEqual(receive-parsed, 5.463, places=3)

    def test_six_digit_filtim_handles_midnight(self):
        receive = 1751328003.0  # 2025-07-01 00:00:03 UTC
        parsed = parse_filtim_epoch("235959", receive)
        self.assertAlmostEqual(receive-parsed, 4.0, places=3)


class RouteAndSectorTests(unittest.TestCase):
    def test_negative_next_fix_is_advanced(self):
        points = [
            {"route_point_index": 0, "ptid_raw": "A", "eto_delta_sec": -20, "planned_level_m": 3000, "ispass_raw": "N"},
            {"route_point_index": 1, "ptid_raw": "B", "eto_delta_sec": 30, "planned_level_m": 3600, "ispass_raw": "N"},
        ]
        result = route_context(points, {"selected_filtim_epoch": 90, "selected_receive_epoch": 90}, 100, CONFIG)
        self.assertEqual(result["active_leg_next_fix"], "B")
        self.assertGreaterEqual(result["time_to_next_fix_sec"], 0)

    def test_far_unbracketed_route_is_not_a_model_feature(self):
        points = [{"route_point_index": 0, "ptid_raw": "DEST", "eto_delta_sec": 4000, "planned_level_m": 0, "ispass_raw": "N"}]
        result = route_context(points, {"selected_filtim_epoch": 90, "selected_receive_epoch": 90}, 100, CONFIG)
        self.assertFalse(result["route_feature_mask"])
        self.assertEqual(result["active_leg_method"], "eto_not_bracketed")

    def test_polygon_membership(self):
        square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self.assertTrue(point_in_polygon(0.5, 0.5, square))
        self.assertFalse(point_in_polygon(2, 2, square))

    def test_callsign_normalization(self):
        self.assertEqual(normalize_callsign(" csh-9240 "), "CSH9240")


if __name__ == "__main__":
    unittest.main()
