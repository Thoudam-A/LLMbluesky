#!/usr/bin/env python3
"""Add CAT062 quality gates to the independent Shanghai v5.1 corpus."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from build_shanghai_causal_alignment_v5 import load_cat062_enrichment, norm_hex


VERSION = "shanghai_quality_gated_builder_v6.0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v5-dir", type=Path, required=True)
    p.add_argument("--raw-sqlite", type=Path, required=True)
    p.add_argument("--base-decoder", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cat-lookback-sec", type=float, default=30.0)
    p.add_argument("--cat-fresh-max-age-sec", type=float, default=15.0)
    p.add_argument("--critical-data-age-max-sec", type=float, default=8.0)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def quality_summary(cat: dict[str, Any] | None, max_observation_age: float, max_critical_age: float) -> dict[str, Any]:
    if cat is None:
        return {"quality_tier": "no_cat062_observation", "model_feature_mask": False, "quality_reasons": ["cat062_not_found_within_lookback"]}
    fields = cat.get("fields", {})
    age = float(cat["observation_age_sec"])
    if age > max_observation_age:
        return {"quality_tier": "stale_cat062_audit_only", "model_feature_mask": False, "observation_age_sec": age, "quality_reasons": ["cat062_observation_stale"]}
    system = fields.get("system_track_update_ages_sec") or {}
    track = fields.get("track_data_ages_sec") or {}
    critical_age = fields.get("maximum_critical_track_data_age_sec")
    reasons: list[str] = []
    if fields.get("track_status_coasting"):
        reasons.append("track_coasting")
    if (fields.get("mode_of_movement") or {}).get("altitude_discrepancy"):
        reasons.append("altitude_discrepancy")
    if critical_age is not None and float(critical_age) > max_critical_age:
        reasons.append("critical_track_data_age_exceeds_threshold")
    tier = "weak" if reasons else "high"
    return {
        "quality_tier": tier,
        "model_feature_mask": True,
        "observation_age_sec": age,
        "minimum_system_track_update_age_sec": None if not system else min(float(v) for v in system.values()),
        "maximum_critical_track_data_age_sec": critical_age,
        "track_status_coasting": bool(fields.get("track_status_coasting")),
        "track_status_flight_plan_coupled": fields.get("track_status_flight_plan_coupled"),
        "mode_of_movement": fields.get("mode_of_movement"),
        "available_track_data_age_fields": sorted(track),
        "quality_reasons": reasons,
    }


def main() -> int:
    a = parse_args()
    src = a.v5_dir / "instruction_bundle_training_records_v5.jsonl"
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    states, cutoffs, refs = {}, {}, []
    for row in rows:
        ref = (row.get("member_reference_event_ids") or [None])[0]
        meta = row.get("provenance", {}).get("causal_alignment_v5", {})
        state = row.get("model_input", {}).get("target_aircraft_state", {})
        states[ref] = {"target_address": norm_hex(meta.get("identity", {}).get("target_address") or state.get("target_address"))}
        cutoffs[ref] = float(meta["decision_cutoff_epoch"])
        refs.append(ref)
    cat = load_cat062_enrichment(a.raw_sqlite, a.base_decoder, states, cutoffs, a.cat_lookback_sec)
    output, audit = [], []
    counts: Counter[str] = Counter()
    for row, ref in zip(rows, refs):
        result = copy.deepcopy(row)
        raw = cat.get(ref)
        summary = quality_summary(raw, a.cat_fresh_max_age_sec, a.critical_data_age_max_sec)
        mi = result["model_input"]
        mi["surveillance_quality_context"] = summary
        if summary["model_feature_mask"] and raw is not None:
            # Refresh fields using the richer v6 decoder. They remain strictly
            # pre-voice and only the fresh observation is placed in model_input.
            mi["surveillance_intent_context"] = {**raw, "observation_status": "fresh_prevoice", "model_feature_mask": True}
        elif raw is not None:
            result.setdefault("source_audit", {})["cat062_quality_observation"] = raw
        control = result.setdefault("training_control", {})
        if summary["quality_tier"] == "weak" and control.get("recommended_training_weight", 1.0) > 0:
            control["recommended_training_weight"] = min(float(control.get("recommended_training_weight") or 1.0), 0.5)
            control.setdefault("state_quality_issues", []).extend(summary["quality_reasons"])
        result["schema_version"] = "shanghai_quality_gated_v6.0"
        result.setdefault("provenance", {})["cat062_quality_gate_v6"] = {
            "lookback_sec": a.cat_lookback_sec, "fresh_max_age_sec": a.cat_fresh_max_age_sec,
            "critical_data_age_max_sec": a.critical_data_age_max_sec,
        }
        audit.append({
            "instruction_bundle_id": result["instruction_bundle_id"], "reference_event_id": ref,
            "quality_tier": summary["quality_tier"], "model_feature_mask": summary["model_feature_mask"],
            "observation_age_sec": summary.get("observation_age_sec"),
            "minimum_system_track_update_age_sec": summary.get("minimum_system_track_update_age_sec"),
            "maximum_critical_track_data_age_sec": summary.get("maximum_critical_track_data_age_sec"),
            "coasting": summary.get("track_status_coasting"),
            "quality_reasons": ",".join(summary["quality_reasons"]),
            "recommended_use": control.get("recommended_use"),
            "training_weight": control.get("recommended_training_weight"),
        })
        counts["records"] += 1
        counts[f"quality_{summary['quality_tier']}"] += 1
        counts["quality_feature_masked"] += int(not summary["model_feature_mask"])
        counts["quality_weak_flagged"] += int(summary["quality_tier"] == "weak")
        output.append(result)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    out = a.output_dir / "instruction_bundle_training_records_v6.jsonl"
    write_jsonl(out, output)
    write_jsonl(a.output_dir / "samples_first5_v6.jsonl", output[:5])
    pd.DataFrame(audit).to_parquet(a.output_dir / "quality_audit_v6.parquet", index=False)
    manifest = {
        "builder_version": VERSION, "dataset_version": "shanghai_quality_gated_v6.0", "counts": dict(counts),
        "quality_contract": {"strictly_prevoice_cat062": True, "fresh_observation_required_for_model_features": True, "coasting_or_stale_critical_data_downweighted": True},
        "sources": {"v5_1": str(src.resolve()), "v5_1_sha256": sha256(src), "raw_sqlite": str(a.raw_sqlite.resolve()), "base_decoder": str(a.base_decoder.resolve())},
        "outputs": {"training_records": out.name, "quality_audit": "quality_audit_v6.parquet", "samples": "samples_first5_v6.jsonl"},
    }
    (a.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
