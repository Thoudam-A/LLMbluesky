from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from evaluation_platform.catalog import load_catalog
from evaluation_platform.registry import MetricRegistry
from evaluation_platform import run_manager


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "tests/_evaluation_platform_runs"


class PortablePlatformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(parents=True, exist_ok=True)

    def test_missing_catalog_is_a_supported_setup_state(self):
        datasets, models, status = load_catalog(WORK / "missing-catalog.json")
        self.assertEqual(datasets, {})
        self.assertEqual(models, {})
        self.assertFalse(status["configured"])

    def test_relative_catalog_paths_resolve_from_catalog_directory(self):
        refs = WORK / "refs.jsonl"
        refs.write_text("", encoding="utf-8")
        catalog_path = WORK / "catalog.json"
        catalog_path.write_text(json.dumps({
            "datasets": [{"id": "d", "references": "refs.jsonl"}],
            "models": [{"id": "m"}],
        }), encoding="utf-8")
        datasets, models, status = load_catalog(catalog_path)
        self.assertEqual(datasets["d"]["references"], refs.resolve())
        self.assertIn("m", models)
        self.assertTrue(status["configured"])

    def test_metric_registry_contains_the_five_ready_metrics(self):
        registry = MetricRegistry(ROOT / "evaluation_platform/metrics")
        registry.load()
        self.assertEqual(
            {x["metric_id"] for x in registry.public_items()},
            {
                "controller_imitation",
                "dynamic_separation_adjustment",
                "command_execution_acceptance",
                "autonomous_command_response_time",
                "controller_intent_understanding_accuracy",
            },
        )

    def test_default_web_page_has_no_metric_result_numbers(self):
        text = (ROOT / "evaluation_platform/static/index.html").read_text(encoding="utf-8")
        for forbidden in ("53.68", "48.72", "43.45", "0.981", "24.42"):
            self.assertNotIn(forbidden, text)
        self.assertIn("待计算", text)

    def test_extension_metrics_switch_inside_the_registry_page(self):
        text = (ROOT / "evaluation_platform/static/index.html").read_text(encoding="utf-8")
        self.assertIn('data-metric-link="dynamic_separation_adjustment"', text)
        self.assertIn('data-metric-link="command_execution_acceptance"', text)
        self.assertIn('data-metric-link="autonomous_command_response_time"', text)
        self.assertIn('data-metric-link="controller_intent_understanding_accuracy"', text)
        self.assertIn('id="hppoRunConfig"', text)
        self.assertIn('id="intentRunConfig"', text)
        self.assertIn("controller_instructions.jsonl", text)
        self.assertIn("intent_reference_events.jsonl", text)
        self.assertIn("intent_predictions.jsonl", text)
        self.assertIn("同 event_id · 完整语义框架", text)
        self.assertNotIn("location.href='hppo_metrics.html?metric=dynamic_separation_adjustment'", text)
        self.assertNotIn("location.href='hppo_metrics.html?metric=command_execution_acceptance'", text)
        self.assertNotIn("location.href='hppo_metrics.html?metric=autonomous_command_response_time'", text)

    def test_response_time_page_only_shows_computable_outputs_and_actual_inputs(self):
        text = (ROOT / "evaluation_platform/static/index.html").read_text(encoding="utf-8")
        self.assertNotIn("P95", text)
        self.assertNotIn("P99", text)
        self.assertNotIn("当前不可计算", text)
        self.assertIn("validation_diagnostics.csv / training_diagnostics.csv", text)
        self.assertIn("decision_response_samples", text)
        self.assertIn("Σ(样本数 × 回合均值) / Σ样本数", text)

    def test_archived_run_uses_only_allowlisted_artifacts(self):
        refs = WORK / "archived-refs.jsonl"
        metric = WORK / "archived-metric.json"
        refs.write_text("", encoding="utf-8")
        metric.write_text(json.dumps({"metrics": {"controller_imitation_macro_recall": None}}), encoding="utf-8")
        old_datasets = dict(run_manager.DATASETS)
        old_models = dict(run_manager.MODELS)
        try:
            run_manager.DATASETS.clear()
            run_manager.DATASETS["d"] = {"id": "d", "references": refs}
            run_manager.MODELS.clear()
            run_manager.MODELS["m"] = {"id": "m", "archived_metric": metric}
            manager = run_manager.RunManager(WORK / "runs")
            state = manager.create({"dataset_id": "d", "model_id": "m", "mode": "archived_result"})
            deadline = time.time() + 5
            while state["status"] not in {"complete", "failed"} and time.time() < deadline:
                time.sleep(0.05)
                state = manager.get(state["run_id"])
            self.assertEqual(state["status"], "complete", state.get("error"))
            self.assertTrue((WORK / "runs" / state["run_id"] / "hashes.json").exists())
        finally:
            run_manager.DATASETS.clear(); run_manager.DATASETS.update(old_datasets)
            run_manager.MODELS.clear(); run_manager.MODELS.update(old_models)

    def test_response_time_runs_through_hppo_orchestration(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "hppo" / "validation-run"
            source.mkdir(parents=True)
            (source / "events.jsonl").write_text("", encoding="utf-8")
            (source / "validation_diagnostics.csv").write_text(
                "episode,decision_count,command_executions,decision_response_samples,mean_decision_response_ms,max_decision_response_ms\n"
                "1,2,1,2,12.5,18\n",
                encoding="utf-8",
            )
            old_hppo_root = run_manager.HPPO_OUTPUT_ROOT
            try:
                run_manager.HPPO_OUTPUT_ROOT = root / "hppo"
                manager = run_manager.RunManager(root / "runs")
                state = manager.create({"metric_id": "autonomous_command_response_time", "source_run": str(source)})
                state = self._wait(manager, state)
                self.assertEqual(state["status"], "complete", state.get("error"))
                self.assertEqual(manager.result(state["run_id"])["primary"]["value"], 12.5)
            finally:
                run_manager.HPPO_OUTPUT_ROOT = old_hppo_root

    @staticmethod
    def _wait(manager: run_manager.RunManager, state: dict) -> dict:
        deadline = time.time() + 5
        while state["status"] not in {"complete", "failed"} and time.time() < deadline:
            time.sleep(0.05)
            state = manager.get(state["run_id"])
        return state


if __name__ == "__main__":
    unittest.main()
