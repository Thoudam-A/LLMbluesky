#!/usr/bin/env python3
"""Build a compact causal model view from Shanghai quality-gated v6 records.

The v6 JSONL remains the audit source.  This builder intentionally separates:

* one fixed-width model/label table;
* a bounded top-K traffic-relation table; and
* an audit index that also records excluded rows.

It does not create NOOP/HOLD states.  The resulting corpus is therefore a
positive-command target-selection corpus, not a complete decision-trigger
training set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


VERSION = "shanghai_compact_model_view_v7.0"
SHANGHAI_TZ = timezone(timedelta(hours=8))
FT_TO_M = 0.3048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiers-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--traffic-top-k", type=int, default=5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                yield line_number, json.loads(line)


def nested(value: Any, *keys: str, default: Any = None) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def flag(value: Any) -> bool | None:
    return None if value is None else bool(value)


def level_value(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("value_m")
    return finite(value)


def constraint_value(value: Any) -> float | None:
    if not isinstance(value, dict):
        return finite(value)
    for key in ("target_value", "value", "value_m", "target_speed_kt"):
        candidate = finite(value.get(key))
        if candidate is not None:
            return candidate
    return None


def surveillance_field(model_input: dict[str, Any], name: str) -> Any:
    context = model_input.get("surveillance_intent_context") or {}
    if not context.get("model_feature_mask"):
        return None
    return nested(context, "fields", name)


def cyclical_time(epoch: float | None) -> tuple[float | None, float | None]:
    if epoch is None:
        return None, None
    local = datetime.fromtimestamp(epoch, tz=SHANGHAI_TZ)
    seconds = local.hour * 3600 + local.minute * 60 + local.second
    angle = 2.0 * math.pi * seconds / 86400.0
    return math.sin(angle), math.cos(angle)


def compact_features(record: dict[str, Any]) -> dict[str, Any]:
    mi = record.get("model_input") or {}
    state = mi.get("target_aircraft_state") or {}
    plan = mi.get("flight_plan_context") or {}
    procedure = mi.get("procedure_context") or {}
    route = mi.get("route_context") or {}
    sector = mi.get("sector_context") or {}
    speed = mi.get("speed_context") or {}
    vertical = mi.get("vertical_context") or {}
    clearance = mi.get("active_clearance_context") or {}
    quality = mi.get("surveillance_quality_context") or {}
    traffic = mi.get("traffic_summary") or {}
    provenance = record.get("provenance") or {}
    epoch = finite(provenance.get("event_time_epoch"))
    time_sin, time_cos = cyclical_time(epoch)

    selected_altitude_ft = finite(surveillance_field(mi, "selected_altitude_ft"))
    final_selected_altitude_ft = finite(
        surveillance_field(mi, "final_selected_altitude_ft")
    )
    observed_altitude_m = finite(state.get("altitude_m"))
    selected_altitude_m = (
        selected_altitude_ft * FT_TO_M if selected_altitude_ft is not None else None
    )

    geometry_candidates = sector.get("geometry_sector_candidates") or []
    history = mi.get("instruction_history") or []
    movement = quality.get("mode_of_movement") or {}

    return {
        # Time and target state.
        "local_time_sin": time_sin,
        "local_time_cos": time_cos,
        "latitude": finite(state.get("latitude")),
        "longitude": finite(state.get("longitude")),
        "altitude_m": observed_altitude_m,
        "ground_speed_kt": finite(state.get("ground_speed_kt")),
        "track_heading_deg": finite(state.get("track_heading_deg")),
        "vertical_rate_mps": finite(state.get("vertical_rate_mps")),
        "time_from_sector_entry_sec": finite(state.get("time_from_sector_entry_sec")),
        "inside_source_sector": flag(state.get("inside_sector")),
        "aircraft_type": text(plan.get("aircraft_type") or state.get("aircraft_type")),
        "wake_turbulence_category": text(
            plan.get("wake_turbulence_category")
            or state.get("wake_turbulence_category")
        ),
        # Flight plan and declared procedure.
        "plan_available": flag(plan.get("plan_features_causally_available")),
        "plan_adep": text(plan.get("plan_adep")),
        "plan_ades": text(plan.get("plan_ades")),
        "requested_level_m": finite(plan.get("requested_level_m")),
        "planned_speed_kt": finite(plan.get("planned_speed_kt")),
        "sid": text(plan.get("sid")),
        "star": text(plan.get("star")),
        "departure_runway": text(plan.get("departure_runway")),
        "arrival_runway": text(plan.get("arrival_runway")),
        "plan_information_age_sec": finite(plan.get("plan_information_age_at_cutoff_sec")),
        "plan_filtim_age_sec": finite(plan.get("plan_filtim_age_at_cutoff_sec")),
        "flight_direction": text(procedure.get("flight_direction")),
        "flight_phase": text(procedure.get("flight_phase")),
        "declared_procedure_type": text(procedure.get("declared_procedure_type")),
        "declared_procedure_name": text(procedure.get("declared_procedure_name")),
        "procedure_alignment_status": text(procedure.get("procedure_alignment_status")),
        "procedure_relevance_confidence": text(
            procedure.get("procedure_relevance_confidence")
        ),
        # Route progress.  Every value remains masked by route_feature_mask.
        "route_feature_mask": flag(route.get("route_feature_mask")),
        "route_context_confidence": text(route.get("route_context_confidence")),
        "active_leg_method": text(route.get("active_leg_method")),
        "previous_fix": text(route.get("active_leg_previous_fix")),
        "next_fix": text(route.get("active_leg_next_fix")),
        "time_to_next_fix_sec": finite(route.get("time_to_next_fix_sec")),
        "distance_to_next_fix_nm": finite(route.get("distance_to_next_fix_nm")),
        "active_leg_fraction": finite(route.get("active_leg_fraction")),
        "active_leg_cross_track_nm": finite(route.get("active_leg_cross_track_nm")),
        "next_planned_level_m": finite(route.get("next_route_point_planned_level_m")),
        "route_point_count": finite(route.get("route_point_count")),
        "geometry_eto_conflict": flag(route.get("geometry_eto_conflict")),
        "ispass_eto_conflict": flag(route.get("ispass_eto_conflict")),
        # Active controller/electronic state kept semantically separate.
        "prior_cleared_level_m": finite(vertical.get("effective_prior_cleared_level_m")),
        "prior_cleared_level_source": text(
            vertical.get("effective_prior_cleared_level_source")
        ),
        "plan_cleared_level_m": level_value(vertical.get("causal_plan_cleared_level")),
        "plan_exit_level_m": level_value(vertical.get("causal_plan_exit_level")),
        "surveillance_selected_altitude_m": selected_altitude_m,
        "surveillance_final_selected_altitude_m": (
            final_selected_altitude_ft * FT_TO_M
            if final_selected_altitude_ft is not None
            else None
        ),
        "selected_altitude_delta_m": (
            selected_altitude_m - observed_altitude_m
            if selected_altitude_m is not None and observed_altitude_m is not None
            else None
        ),
        "prior_speed_constraint_kt": constraint_value(
            speed.get("prior_controller_speed_constraint")
            or clearance.get("active_speed_constraint")
        ),
        "normal_speed_resumed": flag(clearance.get("normal_speed_resumed")),
        "instruction_history_count": len(history),
        # Fresh CAT062 features only.  Raw fields remain in the v6 audit source.
        "surveillance_feature_mask": flag(
            nested(mi, "surveillance_intent_context", "model_feature_mask")
        ),
        "surveillance_observation_age_sec": finite(
            nested(mi, "surveillance_intent_context", "observation_age_sec")
        ),
        "indicated_airspeed_kt": finite(surveillance_field(mi, "indicated_airspeed_kt")),
        "true_airspeed_kt": finite(surveillance_field(mi, "true_airspeed_kt")),
        "mach_number": finite(surveillance_field(mi, "mach_number")),
        "barometric_vertical_rate_fpm": finite(
            surveillance_field(mi, "barometric_vertical_rate_fpm")
        ),
        "track_angle_rate_degps": finite(
            surveillance_field(mi, "track_angle_rate_degps")
        ),
        "roll_angle_deg": finite(surveillance_field(mi, "roll_angle_deg")),
        "quality_tier": text(quality.get("quality_tier")),
        "quality_observation_age_sec": finite(quality.get("observation_age_sec")),
        "maximum_critical_track_data_age_sec": finite(
            quality.get("maximum_critical_track_data_age_sec")
        ),
        "track_status_coasting": flag(quality.get("track_status_coasting")),
        "altitude_discrepancy": flag(movement.get("altitude_discrepancy")),
        "movement_vertical_code": finite(movement.get("vertical_code")),
        "movement_turn_code": finite(movement.get("turn_code")),
        # Sector and bounded traffic aggregates.
        "primary_sector_code": text(sector.get("primary_sector_code")),
        "sector_alignment_status": text(sector.get("sector_alignment_status")),
        "geometry_sector_candidate_count": len(geometry_candidates),
        "dynamic_sector_configuration_available": flag(
            sector.get("dynamic_sector_configuration_available")
        ),
        "traffic_count": finite(traffic.get("traffic_count")),
        "inside_sector_traffic_count": finite(traffic.get("inside_sector_traffic_count")),
        "nearest_horizontal_nm": finite(traffic.get("nearest_horizontal_nm")),
        "minimum_cpa_horizontal_nm": finite(traffic.get("minimum_cpa_horizontal_nm")),
        "minimum_vertical_at_cpa_m": finite(traffic.get("minimum_vertical_at_cpa_m")),
        "predicted_conflict_count": finite(traffic.get("predicted_conflict_count")),
        "same_sid_traffic_count": finite(traffic.get("same_sid_traffic_count")),
        "same_star_traffic_count": finite(traffic.get("same_star_traffic_count")),
        "same_departure_runway_traffic_count": finite(
            traffic.get("same_departure_runway_traffic_count")
        ),
    }


def traffic_sort_key(relation: dict[str, Any]) -> tuple[Any, ...]:
    risk_rank = finite(relation.get("relation_risk_rank"))
    cpa = finite(relation.get("cpa_horizontal_nm"))
    current = finite(relation.get("current_horizontal_nm"))
    tcpa = finite(relation.get("tcpa_sec"))
    return (
        0 if relation.get("relation_model_eligible") else 1,
        0 if relation.get("predicted_pair_conflict") else 1,
        risk_rank if risk_rank is not None else float("inf"),
        cpa if cpa is not None else float("inf"),
        abs(tcpa) if tcpa is not None else float("inf"),
        current if current is not None else float("inf"),
        text(relation.get("other_trajectory_id")) or "",
    )


def compact_traffic(bundle_id: str, record: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    relations = list(nested(record, "model_input", "traffic_context", default=[]) or [])
    relations.sort(key=traffic_sort_key)
    result: list[dict[str, Any]] = []
    for rank, relation in enumerate(relations[:top_k], 1):
        result.append(
            {
                "instruction_bundle_id": bundle_id,
                "neighbor_rank": rank,
                "relation_model_eligible": flag(relation.get("relation_model_eligible")),
                "predicted_pair_conflict": flag(relation.get("predicted_pair_conflict")),
                "other_flight_phase": text(relation.get("other_flight_phase")),
                "other_inside_sector": flag(relation.get("other_inside_sector")),
                "current_horizontal_nm": finite(relation.get("current_horizontal_nm")),
                "current_vertical_m": finite(relation.get("current_vertical_m")),
                "cpa_horizontal_nm": finite(relation.get("cpa_horizontal_nm")),
                "vertical_at_cpa_m": finite(relation.get("vertical_at_cpa_m")),
                "tcpa_sec": finite(relation.get("tcpa_sec")),
                "closing_speed_kt": finite(relation.get("closing_speed_kt")),
                "heading_difference_deg": finite(relation.get("heading_difference_deg")),
                "relative_bearing_deg": finite(relation.get("relative_bearing_deg")),
                "same_sid": flag(relation.get("same_sid")),
                "same_star": flag(relation.get("same_star")),
                "same_departure_runway": flag(relation.get("same_departure_runway")),
                "same_arrival_runway": flag(relation.get("same_arrival_runway")),
            }
        )
    return result


def main() -> int:
    args = parse_args()
    if args.traffic_top_k < 0:
        raise SystemExit("--traffic-top-k must be non-negative")

    sources = {
        "train_core": args.tiers_dir / "train_core_v6.jsonl",
        "train_weak": args.tiers_dir / "train_weak_v6.jsonl",
        "audit_only": args.tiers_dir / "audit_only_v6.jsonl",
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing v6 tier inputs: " + ", ".join(missing))

    model_rows: list[dict[str, Any]] = []
    traffic_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()

    for tier, source in sources.items():
        for line_number, record in rows(source):
            bundle_id = text(record.get("instruction_bundle_id"))
            if not bundle_id:
                raise ValueError(f"{source}:{line_number}: missing instruction_bundle_id")
            actions = nested(record, "label", "actions", default=[]) or []
            reference_ids = record.get("member_reference_event_ids") or []
            control = record.get("training_control") or {}
            audit_rows.append(
                {
                    "instruction_bundle_id": bundle_id,
                    "source_tier": tier,
                    "source_file": str(source.resolve()),
                    "source_line": line_number,
                    "reference_event_ids": ",".join(str(value) for value in reference_ids),
                    "action_count": len(actions),
                    "recommended_use": text(control.get("recommended_use")),
                    "state_quality_issues": ",".join(control.get("state_quality_issues") or []),
                    "label_tier_reasons": ",".join(control.get("label_tier_reasons") or []),
                }
            )
            counts[f"records_{tier}"] += 1
            if tier == "audit_only":
                continue

            features = compact_features(record)
            training_weight = finite(control.get("recommended_training_weight"))
            if training_weight is None:
                training_weight = 1.0 if tier == "train_core" else 0.5
            provenance = record.get("provenance") or {}
            for action_index, action in enumerate(actions):
                family = text(action.get("intent_family"))
                sample_id = f"{bundle_id}:{action_index}"
                model_rows.append(
                    {
                        "sample_id": sample_id,
                        "instruction_bundle_id": bundle_id,
                        "reference_event_id": (
                            text(reference_ids[action_index])
                            if action_index < len(reference_ids)
                            else text(reference_ids[0]) if reference_ids else None
                        ),
                        "callsign": text(provenance.get("callsign")),
                        "event_time_epoch": finite(provenance.get("event_time_epoch")),
                        "label_tier": tier,
                        "training_weight": training_weight,
                        "label_action_index": action_index,
                        "label_intent_family": family,
                        "label_action": text(action.get("action")),
                        "label_target_value": finite(action.get("target_value")),
                        "label_target_unit": text(action.get("target_unit")),
                        "label_target_semantics": text(action.get("target_semantics")),
                        "label_constraint_operator": text(action.get("constraint_operator")),
                        **features,
                    }
                )
                family_counts[family or "missing"] += 1
            traffic_rows.extend(compact_traffic(bundle_id, record, args.traffic_top_k))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model_samples_v7.parquet"
    traffic_path = args.output_dir / "traffic_topk_v7.parquet"
    audit_path = args.output_dir / "audit_index_v7.parquet"
    pd.DataFrame(model_rows).to_parquet(model_path, index=False)
    pd.DataFrame(traffic_rows).to_parquet(traffic_path, index=False)
    pd.DataFrame(audit_rows).to_parquet(audit_path, index=False)

    sample_path = args.output_dir / "compact_samples_first5_v7.jsonl"
    with sample_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in model_rows[:5]:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    feature_columns = sorted(
        set(model_rows[0])
        - {
            "sample_id",
            "instruction_bundle_id",
            "reference_event_id",
            "callsign",
            "event_time_epoch",
            "label_tier",
            "training_weight",
            "label_action_index",
            "label_intent_family",
            "label_action",
            "label_target_value",
            "label_target_unit",
            "label_target_semantics",
            "label_constraint_operator",
        }
    )
    manifest = {
        "builder_version": VERSION,
        "scope": "positive command target-selection records; NOOP/HOLD states are not included",
        "traffic_top_k": args.traffic_top_k,
        "counts": {
            **dict(counts),
            "model_action_rows": len(model_rows),
            "traffic_rows": len(traffic_rows),
            "audit_index_rows": len(audit_rows),
            "families": dict(family_counts),
            "model_feature_columns": len(feature_columns),
        },
        "contracts": {
            "audit_source_preserved": True,
            "current_instruction_text_in_model_view": False,
            "raw_traffic_relations_in_model_view": False,
            "fresh_surveillance_feature_mask_respected": True,
            "full_runtime_feature_parity_complete": False,
        },
        "sources": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in sources.items()
        },
        "outputs": {
            "model_samples": model_path.name,
            "traffic_topk": traffic_path.name,
            "audit_index": audit_path.name,
            "samples": sample_path.name,
        },
        "model_feature_columns": feature_columns,
        "warnings": [
            "Labels remain rule-derived weak labels, not human-reviewed gold.",
            "2025-07-02 has been repeatedly inspected and is not a pristine final blind test.",
            "This v7 view cannot train a command trigger until causal NOOP/HOLD states are built with the same schema.",
            "Dynamic sector configuration and authoritative procedure charts remain unavailable.",
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
