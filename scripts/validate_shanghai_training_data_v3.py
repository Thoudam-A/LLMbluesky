#!/usr/bin/env python3
"""Validate semantic, causal, route, and schema invariants for Shanghai v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def is_null(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(walk_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(walk_keys(item))
    return keys


def main() -> int:
    args = parse_args()
    root = args.dataset_dir
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = load_jsonl(root / "assembled_training_records_v3.jsonl")
    bundles = load_jsonl(root / "instruction_bundle_training_records_v3.jsonl")
    main = pd.read_parquet(root / "supervised_main_v3.parquet")
    traffic = pd.read_parquet(root / "traffic_context_v3.parquet")
    history = pd.read_parquet(root / "instruction_history_v3.parquet")
    labels = pd.read_parquet(root / "semantic_label_audit_v3.parquet")

    n = len(records)
    expected = args.expected_count if args.expected_count is not None else manifest["counts"]["reference_records_preserved"]
    require(n == expected, f"record count {n} != {expected}")
    require(len(main) == n == len(labels), "main/assembled/label audit counts differ")
    require(main["reference_event_id"].is_unique, "main reference ids are not unique")
    require(labels["reference_event_id"].is_unique, "label audit reference ids are not unique")
    require(len({record["instruction_bundle_id"] for record in bundles}) == len(bundles), "bundle ids are not unique")
    require(sum(len(record["member_reference_event_ids"]) for record in bundles) == n, "bundle membership does not preserve all references")

    forbidden = {"transcript", "raw_span", "original_action", "original_target_value", "cleared_flight_level"}
    for record in records:
        require(not (forbidden & set(walk_keys(record["model_input"]))), f"current-label field leaked into input: {record['reference_event_id']}")
        require(record["training_control"]["label_human_verified"] is False, "weak label incorrectly marked human verified")
        event_epoch = float(record["provenance"]["event_time_epoch"])
        history_items = record["model_input"]["instruction_history"]
        require([item["history_rank"] for item in history_items] == list(range(1, len(history_items) + 1)), "history rank gap")
        for item in history_items:
            require(float(item["previous_event_time_epoch"]) < event_epoch, "non-causal history item")
            require(float(item["seconds_since_previous_event"]) > 0, "invalid history delta")
        clearance = record["model_input"]["active_clearance_context"]
        if clearance["normal_speed_resumed"]:
            require(clearance["active_speed_constraint"] is None, "normal speed incorrectly retained as active constraint")
            require(clearance["most_recent_speed_event"]["target_semantics"] == "cancel_active_speed_constraint", "normal-speed flag lacks cancellation event")
        for item in record["model_input"]["traffic_context"]:
            if not item.get("other_plan_causally_available"):
                require(all(is_null(item.get(key)) for key in ["other_plan_adep", "other_plan_ades", "other_sid", "other_star", "other_departure_runway", "other_arrival_runway"]), "unsafe other plan leaked")

    for bundle in bundles:
        require(not (forbidden & set(walk_keys(bundle["model_input"]))), f"label field leaked into bundle input: {bundle['instruction_bundle_id']}")
        require(bundle["label"]["action_count"] == len(bundle["label"]["actions"]), "bundle action count mismatch")
        require(bundle["label"]["action_count"] == len(bundle["member_reference_event_ids"]), "bundle member/action mismatch")
        if bundle["training_control"]["label_alternative_conflict"]:
            require(bundle["training_control"]["recommended_use"] == "audit_only", "conflicting bundle not audit-only")
            require(bundle["training_control"]["recommended_training_weight"] == 0, "conflicting bundle has training weight")

    plan_hidden = main[~main["plan_features_causally_available"].fillna(False)]
    for column in ["plan_adep", "plan_ades", "sid", "star", "departure_runway", "arrival_runway", "requested_level_m", "planned_speed_kt"]:
        require(plan_hidden[column].isna().all(), f"plan column {column} leaked when unavailable")
    require((~plan_hidden["route_feature_mask"].fillna(False)).all(), "route visible without causal plan")

    route_visible = main[main["route_feature_mask"].fillna(False)]
    require((route_visible["route_context_confidence"] != "none").all(), "visible route has no confidence")
    require(route_visible["plan_snapshot_fresh"].fillna(False).all(), "route visible from stale/unknown plan snapshot")
    stale_methods = route_visible[route_visible["active_leg_method"].isin(["geometric_temporal_consistent", "ispass_recent_fallback"])]
    require((stale_methods["raw_eto_delta_to_next_fix_sec"].fillna(-60) >= -60).all(), "stale next fix used by geometric/ispass route")
    bracket = route_visible[route_visible["active_leg_method"] == "eto_time_bracket"]
    require((bracket["raw_eto_delta_to_next_fix_sec"] > 0).all(), "ETO bracket next fix is not future")

    above = labels[labels["transcript"].str.contains("速度", na=False) & labels["transcript"].str.contains("以上", na=False)]
    # Only require this for rows whose parsed target span is immediately bound;
    # the audit flag identifies those corrected by the builder.
    corrected_above = above[above["semantic_correction_reasons_json"].str.contains("preserved_speed_lower_bound_modifier", na=False)]
    if len(corrected_above):
        require((corrected_above["constraint_operator"] == ">=").all(), "speed above operator lost")
        require(corrected_above["corrected_action"].isin(["maintain_min_speed", "increase_to_min_speed"]).all(), "speed above action invalid")
        require(corrected_above["expanded_raw_span"].str.endswith("以上").all(), "expanded speed span lost modifier")

    normal = main[main["target_semantics"] == "cancel_active_speed_constraint"]
    if len(normal):
        require((normal["action"] == "resume_normal_speed").all(), "normal speed action invalid")
        require(normal["target_value"].isna().all(), "normal speed should not have numeric target")
        require((normal["recommended_training_weight"] > 0).all(), "valid nonnumeric speed labels excluded")
    require((main["speed_command_observation_comparable"] == False).all(), "ground speed was marked comparable to command speed")  # noqa: E712

    if not traffic.empty:
        require((traffic["relation_model_eligible"] == True).all(), "non-causal traffic relation included")  # noqa: E712
        require(not traffic.duplicated(["reference_event_id", "relation_source_row_index"]).any(), "duplicate traffic relation")
    if not history.empty:
        require((history["seconds_since_previous_event"] > 0).all(), "history table contains future/current event")
        require(not history.duplicated(["reference_event_id", "history_rank"]).any(), "duplicate history rank")

    for name, meta in manifest["outputs"].items():
        path = root / name
        require(path.exists(), f"missing output {name}")
        require(sha256_file(path) == meta["sha256"], f"hash mismatch {name}")

    report = {
        "status": "PASS",
        "records": n,
        "instruction_bundles": len(bundles),
        "multi_action_bundles": sum(record["label"]["action_count"] > 1 for record in bundles),
        "conflicting_bundles_audit_only": sum(record["training_control"]["label_alternative_conflict"] for record in bundles),
        "traffic_rows": len(traffic),
        "history_rows": len(history),
        "speed_lower_bound_corrections": len(corrected_above),
        "resume_normal_speed_labels": len(normal),
        "route_methods": main["active_leg_method"].value_counts(dropna=False).to_dict(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
