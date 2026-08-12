#!/usr/bin/env python3
"""Rule-only ATC intent prediction baselines.

This script does not call an LLM. It has two modes:
- pattern: apply the hand-written text-pattern rules used by post-processing.
- candidate_sweep: emit schema-derived candidate intents so the current
  intent_type metric can be stress-tested without model output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from postprocess_atc_intent_predictions import callsign_from_text, postprocess_row, read_jsonl, write_jsonl


def base_row(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "annotation_id": source.get("annotation_id") or "",
        "utterance_id": source.get("utterance_id") or source.get("atc_instruction_id") or source.get("uid") or "",
        "source_file": source.get("source_file"),
        "audio": source.get("audio"),
        "start": source.get("start"),
        "end": source.get("end"),
        "language": source.get("language"),
        "text": source.get("text"),
        "annotation_status": "skip",
        "intents": [],
        "note": "rule_only",
    }


def pattern_predict(source: dict[str, Any]) -> dict[str, Any]:
    return postprocess_row(base_row(source))


def candidate_sweep_predict(source: dict[str, Any], schema: dict[str, Any], duplicates: int) -> dict[str, Any]:
    row = base_row(source)
    text = str(source.get("text") or "")
    callsign = callsign_from_text(text)
    intents: list[dict[str, Any]] = []
    for intent_type, spec in schema.get("intent_types", {}).items():
        for action in spec.get("actions", []):
            for _ in range(max(1, duplicates)):
                intent = {
                    "callsign": callsign,
                    "intent_type": intent_type,
                    "action": action,
                    "target_value": None,
                    "unit": None,
                    "frequency": None,
                    "runway": None,
                    "approach_type": None,
                    "fix": None,
                    "restriction_type": None,
                    "report_type": None,
                    "raw_value": "",
                    "raw_span": "",
                }
                if intent_type in {"altitude_change", "altitude_maintain"}:
                    intent["unit"] = "m"
                elif intent_type == "speed_adjust":
                    intent["unit"] = "kt"
                elif intent_type == "heading_change":
                    intent["unit"] = "degree"
                elif intent_type == "qnh_setting":
                    intent["unit"] = "hPa"
                elif intent_type == "restriction_cancel":
                    intent["restriction_type"] = "unspecified"
                elif intent_type == "report_requirement":
                    intent["report_type"] = "unspecified"
                elif intent_type == "approach_clearance":
                    intent["approach_type"] = "unknown"
                intents.append(intent)
    row["annotation_status"] = "done"
    row["intents"] = intents
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["pattern", "candidate_sweep"], default="pattern")
    parser.add_argument("--duplicates", type=int, default=4)
    args = parser.parse_args()

    schema = json.load(open(args.schema, encoding="utf-8"))
    rows = read_jsonl(Path(args.sample))
    if args.mode == "pattern":
        predictions = [pattern_predict(row) for row in rows]
    else:
        predictions = [candidate_sweep_predict(row, schema, args.duplicates) for row in rows]
    write_jsonl(Path(args.out), predictions)
    print(
        json.dumps(
            {
                "sample": args.sample,
                "rows": len(predictions),
                "mode": args.mode,
                "duplicates": args.duplicates,
                "out": args.out,
                "predicted_intents": sum(len(row.get("intents", [])) for row in predictions),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
