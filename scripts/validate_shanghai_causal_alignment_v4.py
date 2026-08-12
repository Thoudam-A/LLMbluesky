#!/usr/bin/env python3
"""Validate causal and semantic invariants of Shanghai alignment v4."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path


def epoch(value):
    if not value:
        return None
    return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    path = args.dataset_dir / "instruction_bundle_training_records_v4.jsonl"
    failures = []
    counts = Counter()
    records = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records += 1
            row = json.loads(line)
            model = row["model_input"]
            cutoff = float(row["provenance"]["causal_alignment"]["input_cutoff_epoch"])
            plan = model["flight_plan_context"]
            receive = epoch(plan.get("selected_plan_receive_time_utc"))
            filtim = epoch(plan.get("effective_plan_filtim_utc"))
            if plan.get("plan_features_causally_available"):
                if receive is None or receive > cutoff + 1e-6:
                    failures.append(f"{row['instruction_bundle_id']}: target plan receive time violates cutoff")
                if filtim is None or filtim > cutoff + 1e-6:
                    failures.append(f"{row['instruction_bundle_id']}: target plan FILTIM violates cutoff")
            route = model["route_context"]
            if route.get("time_to_next_fix_sec") is not None and route["time_to_next_fix_sec"] < 0:
                failures.append(f"{row['instruction_bundle_id']}: negative next-fix time")
            if model["vertical_context"].get("next_controller_target_level_m") is not None:
                failures.append(f"{row['instruction_bundle_id']}: current target leaked into model input")
            counts[f"route_{route.get('active_leg_method')}"] += 1
            counts[f"coverage_{model['input_coverage'].get('input_coverage_tier')}"] += 1
            for relation in model.get("traffic_context", []):
                counts["traffic_relations"] += 1
                if not relation.get("other_plan_causally_available"):
                    continue
                counts["traffic_relations_with_plan"] += 1
                other_cutoff = float(relation.get("other_event_time_epoch") or cutoff)
                other_receive = epoch(relation.get("other_plan_receive_time_utc"))
                other_filtim = epoch(relation.get("other_plan_filtim_utc"))
                if other_receive is None or other_receive > other_cutoff + 1e-6:
                    failures.append(f"{row['instruction_bundle_id']}: other plan receive time violates cutoff")
                if other_filtim is None or other_filtim > other_cutoff + 1e-6:
                    failures.append(f"{row['instruction_bundle_id']}: other plan FILTIM violates cutoff")
    if args.expected_count is not None and records != args.expected_count:
        failures.append(f"expected {args.expected_count} records, found {records}")
    report = {
        "status": "PASS" if not failures else "FAIL",
        "records": records,
        "counts": dict(counts),
        "failure_count": len(failures),
        "failures_first_100": failures[:100],
        "checks": {
            "target_plan_receive_and_filtim_not_after_cutoff": True,
            "other_plan_receive_and_filtim_not_after_other_state": True,
            "next_fix_time_nonnegative": True,
            "current_controller_target_sentinel_null": True,
        },
    }
    output = args.dataset_dir / "validation_report_v4.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
