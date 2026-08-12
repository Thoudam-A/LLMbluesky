from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from score_seu_imitation import align_sequence, score  # noqa: E402
from convert_decision_log_for_imitation import convert_rows, parse_command  # noqa: E402
from seu_imitation_common import (  # noqa: E402
    classify_speaker_role,
    extract_seu_callsign,
    parse_vad_segment_time,
    target_matches,
)


class CallsignAndTimeTests(unittest.TestCase):
    def test_alias_number_is_adjacent_not_frequency(self) -> None:
        parsed = extract_seu_callsign("联系幺二五点三，吉祥幺幺洞八")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["callsign"], "DKH1108")

    def test_number_then_alias_readback(self) -> None:
        parsed = extract_seu_callsign("右转飞POMOK五幺六五东方")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["callsign"], "CES5165")

    def test_vad_local_to_utc(self) -> None:
        parsed = parse_vad_segment_time(
            r"x\OpenASR20250701000021_000880_002880_spk0.wav"
        )
        self.assertEqual(parsed["event_time_utc"], "2025-06-30T16:00:22.880Z")
        self.assertEqual(parsed["speaker_cluster"], 0)

    def test_vad_fractional_base_time(self) -> None:
        parsed = parse_vad_segment_time(
            r"x\20250701000021.017_000880_002880_spk1.wav",
            "OpenASR20250701000021",
        )
        self.assertEqual(parsed["event_time_utc"], "2025-06-30T16:00:22.897Z")
        self.assertEqual(parsed["speaker_cluster"], 1)

    def test_role_position(self) -> None:
        start = extract_seu_callsign("吉祥幺幺洞八高度下四两")
        end = extract_seu_callsign("高度下四两吉祥幺幺洞八")
        self.assertEqual(
            classify_speaker_role("吉祥幺幺洞八高度下四两", start)["speaker_role_rule"],
            "controller_candidate",
        )
        self.assertEqual(
            classify_speaker_role("高度下四两吉祥幺幺洞八", end)["speaker_role_rule"],
            "readback_candidate",
        )


class MetricTests(unittest.TestCase):
    def test_bluesky_converter_accepts_registration_callsign(self) -> None:
        parsed = parse_command("ALT_M N7777U,6000,CLIMB")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["callsign"], "N7777U")

    def test_one_prediction_cannot_match_two_references(self) -> None:
        refs = [
            {"_epoch": 100.0, "_family": "altitude"},
            {"_epoch": 110.0, "_family": "altitude"},
        ]
        preds = [{"_epoch": 105.0, "_family": "altitude"}]
        pairs, ref_misses, pred_misses = align_sequence(refs, preds, 60.0)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(ref_misses), 1)
        self.assertEqual(pred_misses, [])

    def test_parameter_tolerance_and_heading_wrap(self) -> None:
        self.assertTrue(
            target_matches(
                {"intent_type": "heading_change", "target_value": 355, "unit": "deg"},
                {"intent_type": "heading", "target_value": 2, "unit": "deg"},
            )
        )
        self.assertFalse(
            target_matches(
                {"intent_type": "speed_adjust", "target_value": 220, "unit": "kt"},
                {"intent_type": "speed", "target_value": 235, "unit": "kt"},
            )
        )

    def test_macro_recall_is_aircraft_balanced(self) -> None:
        references = [
            {
                "reference_event_id": "r1",
                "callsign": "CES100",
                "event_time_epoch": 100,
                "intent_family": "altitude",
            },
            {
                "reference_event_id": "r2",
                "callsign": "CES100",
                "event_time_epoch": 200,
                "intent_family": "speed",
            },
            {
                "reference_event_id": "r3",
                "callsign": "CSN200",
                "event_time_epoch": 100,
                "intent_family": "heading",
            },
        ]
        predictions = [
            {
                "event_id": "p1",
                "callsign": "CES100",
                "event_time_epoch": 101,
                "intent_type": "altitude_change",
            }
        ]
        result = score(references, predictions, 60.0, "unit-test")
        self.assertEqual(result["metrics"]["controller_imitation_micro_recall"], 0.333333)
        self.assertEqual(result["metrics"]["controller_imitation_macro_recall"], 0.25)

    def test_bluesky_command_conversion(self) -> None:
        altitude = parse_command("ALT CES5165,FL120,2000")
        altitude_m = parse_command("ALT_M CES5165,3600,DESCEND")
        speed = parse_command("SPD CES5165,250")
        self.assertEqual(altitude["target_value"], 3657.6)
        self.assertEqual(altitude["unit"], "m")
        self.assertEqual(altitude_m["target_value"], 3600)
        self.assertEqual(altitude_m["unit"], "m")
        self.assertEqual(altitude_m["action"], "descend")
        self.assertEqual(speed["intent_family"], "speed")
        self.assertEqual(speed["target_value"], 250)

    def test_sim_time_conversion(self) -> None:
        outputs, rejected = convert_rows(
            [{"simt": 8, "commands": ["SPD CES5165,250", "NOOP"]}],
            100.0,
        )
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["event_time_epoch"], 108.0)
        self.assertEqual(len(rejected), 1)

    def test_strict_parameter_metric_rejects_wrong_level(self) -> None:
        references = [
            {
                "callsign": "CES100",
                "event_time_epoch": 100,
                "intent_family": "altitude",
                "intent_type": "altitude_change",
                "action": "descend",
                "target_value": 3000,
                "unit": "m",
            }
        ]
        predictions = [
            {
                "callsign": "CES100",
                "event_time_epoch": 110,
                "intent_family": "altitude",
                "intent_type": "altitude_change",
                "action": "descend",
                "target_value": 3900,
                "unit": "m",
            }
        ]
        result = score(references, predictions, 60.0, "strict-test")
        self.assertEqual(result["metrics"]["controller_imitation_micro_recall"], 1.0)
        self.assertEqual(result["metrics"]["strict_parameter_micro_recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
