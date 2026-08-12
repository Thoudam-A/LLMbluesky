from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluation_platform.metrics.command_execution_acceptance.scorer import score as score_acceptance
from evaluation_platform.metrics.command_acceptability_proxy.scorer import score as score_acceptability_proxy
from evaluation_platform.metrics.autonomous_command_response_time.scorer import score as score_response
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

    def test_dynamic_separation_formal_outcome_requires_stable_compliance(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            events = self._events(root, [
                {
                    "event": "separation_updated", "episode": 3, "sim_time_s": 12.0,
                    "adjustment_id": "sep-3-12-1", "config_applied": True,
                    "previous_horizontal_km": 7.408, "horizontal_km": 9.26,
                    "affected_pair_count": 1,
                },
                {
                    "event": "separation_adjustment_outcome", "episode": 3, "sim_time_s": 52.0,
                    "adjustment_id": "sep-3-12-1", "state": "SUCCESS", "success": True,
                    "reason": "stable_compliance_confirmed", "response_time_s": 40.0,
                    "stable_for_s": 30.0, "secondary_pair_count": 0,
                },
            ])
            result = score_separation(events)
            self.assertEqual(result["counts"]["formal_eligible_events"], 1)
            self.assertEqual(result["metrics"]["formal_dynamic_separation_adjustment_success_rate"], 1.0)
            self.assertEqual(result["details"][0]["evaluation_level"], "formal")

    def test_dynamic_separation_formal_secondary_conflict_is_not_success(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            events = self._events(root, [
                {
                    "event": "separation_updated", "episode": 4, "sim_time_s": 12.0,
                    "adjustment_id": "sep-4-12-1", "config_applied": True,
                    "previous_horizontal_km": 7.408, "horizontal_km": 9.26,
                },
                {
                    "event": "separation_adjustment_outcome", "episode": 4, "sim_time_s": 62.0,
                    "adjustment_id": "sep-4-12-1", "state": "FAILED", "success": False,
                    "reason": "secondary_conflict_after_adjustment", "secondary_pair_count": 1,
                },
            ])
            result = score_separation(events)
            self.assertEqual(result["metrics"]["dynamic_separation_adjustment_success_rate"], 0.0)
            self.assertEqual(result["failure_reasons"][0]["reason"], "secondary_conflict_after_adjustment")

    def test_command_acceptability_proxy_rejects_duplicate_and_scores_safe_command(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            result = score_acceptability_proxy(self._events(root, [
                {
                    "event": "action_executed", "episode": 1, "sim_time_s": 10.0,
                    "acid": "A1", "macro_action": 2,
                    "command": "HDG A1 230", "command_applied": True,
                    "parameter_target": {"heading_deg": 230.0},
                    "target_ids": ["A2"], "conflict_severity": 0.8,
                },
                {
                    "event": "action_executed", "episode": 1, "sim_time_s": 15.0,
                    "acid": "A1", "macro_action": 2,
                    "command": "HDG A1 230", "command_applied": True,
                    "parameter_target": {"heading_deg": 230.0},
                    "target_ids": ["A2"], "conflict_severity": 0.7,
                },
                {
                    "event": "action_executed", "episode": 1, "sim_time_s": 25.0,
                    "acid": "A1", "macro_action": 2,
                    "command": "HDG A1 245", "command_applied": True,
                    "parameter_target": {"heading_deg": 245.0},
                    "target_ids": ["A2"], "conflict_severity": 0.3,
                },
            ]), None)
            self.assertEqual(result["counts"]["eligible_commands"], 3)
            self.assertEqual(result["counts"]["rejected_commands"], 1)
            self.assertEqual(result["failure_reasons"][0]["reason"], "duplicate_target_within_hold_window")

    def test_command_acceptability_proxy_is_not_evaluable_without_commands(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            result = score_acceptability_proxy(self._events(root, [
                {"event": "manager_initialized", "episode": 0, "sim_time_s": 0.0},
            ]), None)
            self.assertEqual(result["status"], "not_evaluable")
            self.assertIsNone(result["primary"]["value"])

    def test_response_time_uses_sample_weighted_episode_means(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            diagnostics = root / "validation_diagnostics.csv"
            with diagnostics.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "episode", "scenario", "decision_count", "command_executions",
                    "decision_response_samples", "mean_decision_response_ms", "max_decision_response_ms",
                ])
                writer.writeheader()
                writer.writerow({"episode": 1, "scenario": "S1", "decision_count": 2, "command_executions": 1, "decision_response_samples": 2, "mean_decision_response_ms": 10, "max_decision_response_ms": 15})
                writer.writerow({"episode": 2, "scenario": "S1", "decision_count": 2, "command_executions": 1, "decision_response_samples": 1, "mean_decision_response_ms": 20, "max_decision_response_ms": 20})
            result = score_response(diagnostics)
            self.assertEqual(result["status"], "complete")
            self.assertAlmostEqual(result["primary"]["value"], 40 / 3, places=6)
            self.assertEqual(result["metrics"]["max_decision_response_ms"], 20)
            self.assertEqual(result["metrics"]["response_sample_coverage"], 0.75)
            self.assertEqual(result["metrics"]["amortized_ms_per_applied_command"], 20)

    def test_response_time_is_not_evaluable_without_samples(self):
        with tempfile.TemporaryDirectory() as raw:
            diagnostics = Path(raw) / "training_diagnostics.csv"
            diagnostics.write_text(
                "episode,decision_count,command_executions,decision_response_samples,mean_decision_response_ms,max_decision_response_ms\n"
                "1,3,0,0,0,0\n",
                encoding="utf-8",
            )
            result = score_response(diagnostics)
            self.assertEqual(result["status"], "not_evaluable")
            self.assertIsNone(result["primary"]["value"])

if __name__ == "__main__":
    unittest.main()
