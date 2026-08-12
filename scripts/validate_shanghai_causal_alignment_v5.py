#!/usr/bin/env python3
"""Validate hard causal and semantic invariants of a v5 alignment build."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path, required=True)
    args = p.parse_args()
    records_path = args.dataset_dir / "instruction_bundle_training_records_v5.jsonl"
    audit_path = args.dataset_dir / "alignment_audit_v5.parquet"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    audit = pd.read_parquet(audit_path)
    failures: list[str] = []
    if len(records) != len(audit):
        failures.append(f"record/audit count mismatch: {len(records)} != {len(audit)}")
    if not (audit["prevoice_age_sec"] > 0).all():
        failures.append("at least one selected state is not strictly before voice start")
    if not (audit["plan_link_status"] != "unlinked").all():
        failures.append("unlinked plans found")
    for rec in records:
        p5 = rec.get("provenance", {}).get("causal_alignment_v5", {})
        if p5.get("selected_prevoice_state_epoch", float("inf")) >= p5.get("decision_cutoff_epoch", float("-inf")):
            failures.append(f"non-causal state: {rec.get('instruction_bundle_id')}")
    summary = {
        "records": len(records),
        "strict_prevoice": bool((audit["prevoice_age_sec"] > 0).all()),
        "prevoice_age_sec": {"min": float(audit["prevoice_age_sec"].min()), "max": float(audit["prevoice_age_sec"].max())},
        "plan_link_status": dict(Counter(audit["plan_link_status"])),
        "cat062_observation": dict(Counter(audit["cat062_observation_available"])),
        "cat062_fresh_model_feature": dict(Counter(audit["cat062_model_feature_fresh"])),
        "same_window_mutable_field_masks": int((audit["masked_mutable_fields"].fillna("") != "").sum()),
        "direction_conflict_isolated": int(audit["direction_conflict"].sum()),
        "failures": failures,
        "passed": not failures,
    }
    (args.dataset_dir / "validation_report_v5.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
