from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluation_platform.metrics.command_execution_acceptance.scorer import score as score_acceptance
from evaluation_platform.metrics.dynamic_separation_adjustment.scorer import score as score_separation


class HPPOEvaluationMetricTests(unittest.TestCase):
    def _events(self, root: Path, rows: list[dict]) -> Path:
        path = root / "events.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_command_acceptance_excludes_noop_and_duplicate_targets(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            result = score_acceptance(self._events(root, [
                {"event": "action_executed", "episode": 1, "sim_time_s": 1.0, "acid": "A1", "macro_action": 0, "command_applied": False},
                {"event": "action_executed", "episode": 1, "sim_time_s": 2.0, "acid": "A1", "macro_action": 3, "command": "ALT A1 37000", "command_applied": True},
                {"event": "action_executed", "episode": 1, "sim_time_s": 3.0, "acid": "A1", "macro_action": 3, "command": "", "command_applied": False},
                {"event": "command_failed", "episode": 1, "sim_time_s": 4.0, "acid": "A1", "macro_action": 2, "command": "HDG A1", "message": "HDG failed"},
            ]))
            self.assertEqual(result["counts"], {"submitted": 2, "accepted": 1, "rejected": 1, "ignored": 1})
            self.assertEqual(result["metrics"]["command_execution_acceptance_rate"], 0.5)

    def test_dynamic_separation_uses_real_change_event_and_episode_result(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            events = self._events(root, [{
                "event": "separation_updated", "episode": 2, "sim_time_s": 12.0,
                "previous_horizontal_km": 7.408, "horizontal_km": 9.26,
            }])
            diagnostics = root / "validation_diagnostics.csv"
            with diagnostics.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["episode", "safe_success"])
                writer.writeheader(); writer.writerow({"episode": 2, "safe_success": 1})
            result = score_separation(events, diagnostics)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["counts"]["eligible_events"], 1)
            self.assertEqual(result["metrics"]["dynamic_separation_adjustment_success_rate"], 1.0)

    def test_dynamic_separation_has_no_rate_without_runtime_change(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            events = self._events(root, [])
            diagnostics = root / "validation_diagnostics.csv"
            diagnostics.write_text("episode,safe_success\n1,1\n", encoding="utf-8")
            result = score_separation(events, diagnostics)
            self.assertEqual(result["status"], "not_evaluable")
            self.assertIsNone(result["primary"]["value"])


if __name__ == "__main__":
    unittest.main()
