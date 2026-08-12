#!/usr/bin/env python3
"""Validate lossless preservation, causality and model-input isolation for v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


PLAN_COLUMNS = [
    "requested_level_m",
    "requested_level_source_unit",
    "requested_level_parse_status",
    "planned_speed_kt",
    "plan_adep",
    "plan_ades",
    "sid",
    "star",
    "departure_runway",
    "arrival_runway",
    "active_leg",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def finite_series(series: pd.Series) -> bool:
    return bool(
        series.apply(
            lambda value: value is not None and math.isfinite(float(value))
        ).all()
    )


def check(name: str, condition: bool, details: Any = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(condition), "details": details}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.dataset_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    audit = pd.read_parquet(root / "state_audit_v2.parquet")
    safe = pd.read_parquet(root / "model_safe_supervised_records_v2.parquet")
    relation_audit = pd.read_parquet(root / "traffic_relations_audit_v2.parquet")
    relation_safe = pd.read_parquet(root / "model_safe_traffic_relations_v2.parquet")
    route = pd.read_parquet(root / "route_points_audit_v2.parquet")
    history = pd.read_parquet(root / "instruction_history_v2.parquet")

    expected_rows = int(manifest["counts"]["reference_events_preserved"])
    source_columns = manifest["information_preservation"][
        "source_parquet_columns_preserved_with_state_raw_prefix"
    ]
    missing_raw_columns = [
        column for column in source_columns if f"state_raw_{column}" not in audit.columns
    ]
    schema = manifest["model_safe_schema"]
    input_columns = set(schema["model_input_columns"])
    label_columns = set(schema["label_columns"])
    missing_inputs = sorted(input_columns - set(safe.columns))
    forbidden_feature_hits = sorted(
        column
        for column in input_columns
        if column.startswith("state_raw_")
        or column in {"transcript", "raw_span", "cleared_flight_level"}
        or column.startswith("target_")
        or column in {"action", "intent_family", "reference_event_id"}
    )
    unsafe_plan = ~safe["plan_features_causally_available"].fillna(False)
    unsafe_plan_visible = int(
        safe.loc[unsafe_plan, PLAN_COLUMNS].notna().any(axis=1).sum()
    )
    f_mask = (
        audit["state_raw_requested_flight_level"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.startswith("F")
    )
    f_parse_failures = int(
        (
            audit.loc[f_mask, "requested_level_m"].isna()
            | (
                audit.loc[f_mask, "requested_level_parse_status"]
                != "parsed_feet_code"
            )
        ).sum()
    )
    valid_relations = relation_safe["relation_valid"].fillna(False)
    relation_finite = all(
        finite_series(relation_safe.loc[valid_relations, column])
        for column in [
            "current_horizontal_nm",
            "current_vertical_m",
            "tcpa_sec",
            "cpa_horizontal_nm",
            "vertical_at_cpa_m",
        ]
    )
    if history.empty:
        history_causal = True
    else:
        history_with_current = history.merge(
            safe[["reference_event_id", "event_time_epoch"]],
            on="reference_event_id",
            how="left",
            validate="many_to_one",
        )
        history_causal = bool(
            (
                history_with_current["previous_event_time_epoch"]
                < history_with_current["event_time_epoch"]
            ).all()
        )
    audit_rank_complete = all(
        sorted(group["relation_risk_rank"].astype(int).tolist())
        == list(range(1, len(group) + 1))
        for _, group in relation_audit.groupby("reference_event_id")
    )
    safe_rank_complete = all(
        sorted(group["relation_risk_rank"].astype(int).tolist())
        == list(range(1, len(group) + 1))
        for _, group in relation_safe.groupby("reference_event_id")
    )
    safe_one_per_trajectory = not relation_safe.duplicated(
        ["reference_event_id", "other_trajectory_id"]
    ).any()
    safe_relation_causal = bool(
        (
            relation_safe["other_event_time_epoch"]
            <= relation_safe["target_state_event_time_epoch"] + 1e-9
        ).all()
        and (
            relation_safe["other_event_time_epoch"]
            < relation_safe["command_event_time_epoch"]
        ).all()
    )

    checks = [
        check("all_reference_rows_preserved_in_audit", len(audit) == expected_rows),
        check("all_reference_rows_preserved_in_model_safe", len(safe) == expected_rows),
        check(
            "all_reference_rows_preserved_in_nested_jsonl",
            jsonl_count(root / "event_state_records.jsonl") == expected_rows,
        ),
        check(
            "all_reference_rows_preserved_in_quality_ledger",
            jsonl_count(root / "quality_ledger.jsonl") == expected_rows,
        ),
        check("all_source_columns_preserved_in_audit", not missing_raw_columns, missing_raw_columns),
        check("all_states_strictly_pre_command", bool((safe["state_event_time_epoch"] < safe["event_time_epoch"]).all())),
        check("all_declared_model_inputs_exist", not missing_inputs, missing_inputs),
        check("labels_are_not_model_inputs", not (label_columns & input_columns), sorted(label_columns & input_columns)),
        check("forbidden_fields_are_not_model_inputs", not forbidden_feature_hits, forbidden_feature_hits),
        check("causally_unavailable_plan_fields_are_hidden", unsafe_plan_visible == 0, unsafe_plan_visible),
        check("feet_level_codes_are_parsed", f_parse_failures == 0, {"rows": int(f_mask.sum()), "failures": f_parse_failures}),
        check("model_safe_relations_are_strictly_causal", safe_relation_causal),
        check("model_safe_relations_have_one_row_per_trajectory", safe_one_per_trajectory),
        check(
            "audit_same_bucket_relation_count_matches_manifest",
            len(relation_audit)
            == int(manifest["information_preservation"]["expected_same_bucket_relation_rows"]),
        ),
        check("valid_relation_values_are_finite", relation_finite),
        check("audit_relation_risk_ranks_are_complete", audit_rank_complete),
        check("model_relation_risk_ranks_are_complete", safe_rank_complete),
        check("instruction_history_is_strictly_previous", history_causal),
        check("route_estimates_are_nonnegative", not (safe["estimated_seconds_to_next_fix"] < 0).any()),
        check("reserved_test_labels_not_loaded", manifest["scope"]["reserved_test_labels_loaded"] is False),
    ]
    output_hash_failures = []
    for filename, item in manifest["outputs"].items():
        path = root / filename
        if not path.exists() or sha256_file(path) != item["sha256"]:
            output_hash_failures.append(filename)
    checks.append(check("output_hashes_match_manifest", not output_hash_failures, output_hash_failures))

    passed = all(item["passed"] for item in checks)
    report = {
        "dataset_dir": str(root),
        "passed": passed,
        "counts": {
            "reference_rows": expected_rows,
            "traffic_relation_rows": len(relation_safe),
            "route_point_rows": len(route),
            "instruction_history_rows": len(history),
            "feet_level_rows": int(f_mask.sum()),
            "vertical_rate_assumed_zero_relation_rows": int(
                (
                    relation_safe["target_vertical_rate_assumed_zero"].fillna(False)
                    | relation_safe["other_vertical_rate_assumed_zero"].fillna(False)
                ).sum()
            ),
        },
        "checks": checks,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
