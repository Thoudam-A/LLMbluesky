from __future__ import annotations

import json
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

    def test_only_controller_imitation_is_registered(self):
        registry = MetricRegistry(ROOT / "evaluation_platform/metrics")
        registry.load()
        self.assertEqual([x["metric_id"] for x in registry.public_items()], ["controller_imitation"])

    def test_default_web_page_has_no_metric_result_numbers(self):
        text = (ROOT / "evaluation_platform/static/index.html").read_text(encoding="utf-8")
        for forbidden in ("53.68", "48.72", "43.45", "0.981", "24.42"):
            self.assertNotIn(forbidden, text)
        self.assertIn("待计算", text)

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


if __name__ == "__main__":
    unittest.main()
