#!/usr/bin/env python3
"""Write a development-only recalibrated threshold into a trained policy bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", type=Path, required=True)
    parser.add_argument("--threshold-record", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record = json.loads(args.threshold_record.read_text(encoding="utf-8"))
    if record.get("test_labels_used") is not False:
        raise SystemExit("threshold record must explicitly confirm test_labels_used=false")
    threshold = float(record["selected_threshold"])
    bundle = joblib.load(args.input_model)
    bundle["decision_threshold"] = threshold
    bundle["threshold_recalibration"] = record
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output_model)
    print(
        json.dumps(
            {
                "input_model": str(args.input_model.resolve()),
                "output_model": str(args.output_model.resolve()),
                "decision_threshold": threshold,
                "test_labels_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
