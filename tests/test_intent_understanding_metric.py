from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from evaluation_platform import run_manager
from evaluation_platform.metrics.controller_intent_understanding.scorer import score


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


class IntentUnderstandingMetricTests(unittest.TestCase):
    def test_scores_direct_event_id_pairs_and_full_semantic_frames(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            instructions = root / "controller_instructions.jsonl"
            references = root / "intent_reference_events.jsonl"
            predictions = root / "intent_predictions.jsonl"
            write_jsonl(instructions, [
                {"event_id": "e1", "instruction_text": "CCA123下降到三千米"},
                {"event_id": "e2", "instruction_text": "CES456速度二百五"},
                {"event_id": "e3", "instruction_text": "CSN789航向三五零"},
            ])
            write_jsonl(references, [
                {"event_id": "e1", "callsign": "CCA123", "intent_family": "altitude", "intent_type": "altitude_change", "action": "descend", "target_value": 3000, "unit": "m"},
                {"event_id": "e2", "callsign": "CES456", "intent_family": "speed", "intent_type": "speed_adjust", "action": "set_speed", "target_value": 250, "unit": "kt"},
                {"event_id": "e3", "callsign": "CSN789", "intent_family": "heading", "intent_type": "heading_change", "action": "fly_heading", "target_value": 350, "unit": "deg"},
            ])
            write_jsonl(predictions, [
                {"event_id": "e1", "callsign": "CCA123", "intent_family": "altitude", "intent_type": "altitude_change", "action": "descend", "target_value": 3050, "unit": "m"},
                {"event_id": "e2", "callsign": "CES456", "intent_family": "heading", "intent_type": "heading_change", "action": "fly_heading", "target_value": 250, "unit": "deg"},
                {"event_id": "e3", "callsign": "CSN789", "intent_family": "heading", "intent_type": "heading_change", "action": "fly_heading", "target_value": 5, "unit": "deg"},
            ])
            result = score(instructions, references, predictions)
            self.assertEqual(result["status"], "complete")
            self.assertAlmostEqual(result["primary"]["value"], 1 / 3, places=6)
            self.assertAlmostEqual(result["metrics"]["intent_type_accuracy"], 2 / 3, places=6)
            self.assertAlmostEqual(result["metrics"]["action_accuracy"], 2 / 3, places=6)
            self.assertAlmostEqual(result["metrics"]["parameter_accuracy"], 1 / 3, places=6)
            self.assertEqual(result["counts"]["full_intent_hits"], 1)

    def test_missing_classifier_output_counts_as_incorrect(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            instructions = root / "instructions.jsonl"
            references = root / "references.jsonl"
            predictions = root / "predictions.jsonl"
            write_jsonl(instructions, [{"event_id": "e1", "instruction_text": "CCA123下降到三千米"}])
            write_jsonl(references, [{"event_id": "e1", "callsign": "CCA123", "intent_family": "altitude", "intent_type": "altitude_change", "action": "descend", "target_value": 3000, "unit": "m"}])
            predictions.write_text("", encoding="utf-8")
            result = score(instructions, references, predictions)
            self.assertEqual(result["primary"]["value"], 0)
            self.assertEqual(result["counts"]["missing_predictions"], 1)
            self.assertEqual(result["counts"]["parameter_comparable_events"], 1)

    def test_runs_through_platform_orchestration(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            instructions = root / "instructions.jsonl"
            references = root / "references.jsonl"
            predictions = root / "predictions.jsonl"
            write_jsonl(instructions, [{"event_id": "e1", "instruction_text": "CCA123下降到三千米"}])
            gold = {"event_id": "e1", "callsign": "CCA123", "intent_family": "altitude", "intent_type": "altitude_change", "action": "descend", "target_value": 3000, "unit": "m"}
            write_jsonl(references, [gold])
            write_jsonl(predictions, [gold])
            old_datasets = dict(run_manager.DATASETS)
            old_models = dict(run_manager.MODELS)
            try:
                run_manager.DATASETS.clear()
                run_manager.DATASETS["intent-d"] = {
                    "id": "intent-d",
                    "controller_instructions": instructions,
                    "intent_references": references,
                }
                run_manager.MODELS.clear()
                run_manager.MODELS["intent-m"] = {"id": "intent-m", "intent_predictions": predictions}
                manager = run_manager.RunManager(root / "runs")
                state = manager.create({
                    "metric_id": "controller_intent_understanding_accuracy",
                    "dataset_id": "intent-d",
                    "model_id": "intent-m",
                })
                deadline = time.time() + 5
                while state["status"] not in {"complete", "failed"} and time.time() < deadline:
                    time.sleep(0.05)
                    state = manager.get(state["run_id"])
                self.assertEqual(state["status"], "complete", state.get("error"))
                self.assertEqual(manager.result(state["run_id"])["primary"]["value"], 1)
            finally:
                run_manager.DATASETS.clear()
                run_manager.DATASETS.update(old_datasets)
                run_manager.MODELS.clear()
                run_manager.MODELS.update(old_models)


if __name__ == "__main__":
    unittest.main()
