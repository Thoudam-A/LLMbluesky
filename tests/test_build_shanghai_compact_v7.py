from __future__ import annotations

import unittest

from scripts.build_shanghai_compact_v7 import compact_features, compact_traffic


class CompactV7Tests(unittest.TestCase):
    def sample(self) -> dict:
        return {
            "instruction_bundle_id": "b1",
            "provenance": {"event_time_epoch": 1_751_299_480, "callsign": "CCA001"},
            "model_input": {
                "target_aircraft_state": {
                    "latitude": 31.0,
                    "longitude": 121.0,
                    "altitude_m": 3000,
                    "ground_speed_kt": 250,
                    "track_heading_deg": 90,
                    "vertical_rate_mps": 0,
                },
                "flight_plan_context": {
                    "plan_features_causally_available": True,
                    "plan_adep": "ZSPD",
                    "plan_ades": "ZSSS",
                },
                "route_context": {
                    "route_feature_mask": True,
                    "active_leg_next_fix": "FIX1",
                    "time_to_next_fix_sec": 60,
                },
                "vertical_context": {
                    "effective_prior_cleared_level_m": 2700,
                    "causal_plan_cleared_level": {"value_m": 3000},
                },
                "surveillance_intent_context": {
                    "model_feature_mask": True,
                    "observation_age_sec": 2,
                    "fields": {"selected_altitude_ft": 10000, "indicated_airspeed_kt": 240},
                },
                "surveillance_quality_context": {
                    "quality_tier": "high",
                    "mode_of_movement": {"altitude_discrepancy": False},
                },
                "traffic_summary": {"traffic_count": 2, "predicted_conflict_count": 1},
                "traffic_context": [
                    {
                        "other_trajectory_id": "far",
                        "relation_model_eligible": True,
                        "predicted_pair_conflict": False,
                        "relation_risk_rank": 2,
                        "cpa_horizontal_nm": 8,
                    },
                    {
                        "other_trajectory_id": "conflict",
                        "relation_model_eligible": True,
                        "predicted_pair_conflict": True,
                        "relation_risk_rank": 1,
                        "cpa_horizontal_nm": 3,
                    },
                ],
            },
        }

    def test_compact_features_keep_semantics_without_raw_nested_payload(self):
        result = compact_features(self.sample())
        self.assertEqual(result["next_fix"], "FIX1")
        self.assertEqual(result["prior_cleared_level_m"], 2700)
        self.assertAlmostEqual(result["surveillance_selected_altitude_m"], 3048)
        self.assertEqual(result["indicated_airspeed_kt"], 240)
        self.assertNotIn("surveillance_intent_context", result)

    def test_stale_surveillance_payload_is_masked(self):
        sample = self.sample()
        sample["model_input"]["surveillance_intent_context"]["model_feature_mask"] = False
        result = compact_features(sample)
        self.assertIsNone(result["indicated_airspeed_kt"])
        self.assertIsNone(result["surveillance_selected_altitude_m"])

    def test_traffic_is_bounded_and_conflict_first(self):
        result = compact_traffic("b1", self.sample(), top_k=1)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["predicted_pair_conflict"])
        self.assertEqual(result[0]["neighbor_rank"], 1)


if __name__ == "__main__":
    unittest.main()
