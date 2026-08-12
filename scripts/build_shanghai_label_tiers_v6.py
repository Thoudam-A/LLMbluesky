#!/usr/bin/env python3
"""Create explicit train/weak/audit label tiers from quality-gated v6 records."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v6-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def valid_actions(row: dict[str, Any]) -> bool:
    actions = row.get("label", {}).get("actions") or []
    return bool(actions) and all(a.get("intent_family") in {"altitude", "speed", "heading"} and isinstance(a.get("target_value"), (int, float)) and math.isfinite(float(a["target_value"])) for a in actions)


def main() -> int:
    a = parse_args()
    source = a.v6_dir / "instruction_bundle_training_records_v6.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    core, weak, audit, ledger = [], [], [], []
    counts: Counter[str] = Counter()
    for row in rows:
        actions = row.get("label", {}).get("actions") or []
        control = row.get("training_control", {})
        issues = set(control.get("state_quality_issues") or [])
        quality = row.get("model_input", {}).get("surveillance_quality_context", {})
        plan = row.get("provenance", {}).get("causal_alignment_v5", {}).get("plan_link", {})
        base_audit = str(control.get("recommended_use") or "").startswith("audit_only")
        reasons: list[str] = []
        if base_audit:
            reasons.append("upstream_audit_only")
        if not valid_actions(row):
            reasons.append("invalid_or_out_of_scope_label")
        if "altitude_label_direction_conflicts_with_prevoice_kinematics" in issues:
            reasons.append("label_direction_conflict")
        if quality.get("quality_tier") in {"stale_cat062_audit_only", "no_cat062_observation"}:
            reasons.append("no_fresh_cat062_quality_context")
        if plan.get("status") not in {"high", "medium"}:
            reasons.append("plan_link_not_reliable")
        if reasons:
            tier = "audit_only"; audit.append(row)
        elif len(actions) == 1 and quality.get("quality_tier") == "high":
            tier = "train_core"; core.append(row)
        else:
            tier = "train_weak"; weak.append(row)
        row.setdefault("training_control", {})["label_tier_v6"] = tier
        row["training_control"]["label_tier_reasons"] = reasons or ["single_action_fresh_quality"] if tier == "train_core" else reasons or ["multi_action_or_quality_weak"]
        ledger.append({
            "instruction_bundle_id": row["instruction_bundle_id"],
            "reference_event_id": (row.get("member_reference_event_ids") or [None])[0],
            "label_tier": tier, "action_count": len(actions),
            "intent_families": ",".join(str(x.get("intent_family")) for x in actions),
            "quality_tier": quality.get("quality_tier"), "plan_link_status": plan.get("status"),
            "reasons": ",".join(reasons),
        })
        counts[tier] += 1
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("train_core_v6.jsonl", core), ("train_weak_v6.jsonl", weak), ("audit_only_v6.jsonl", audit)):
        with (a.output_dir / name).open("w", encoding="utf-8", newline="\n") as f:
            for row in values:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    pd.DataFrame(ledger).to_parquet(a.output_dir / "label_tier_ledger_v6.parquet", index=False)
    report = {"source": str(source.resolve()), "counts": dict(counts), "contract": {"train_core": "single action, valid structured label, fresh high CAT062 quality, reliable plan link, no upstream audit issue", "train_weak": "valid but multi-action or quality-weak", "audit_only": "label/state/quality link unsafe"}}
    (a.output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
