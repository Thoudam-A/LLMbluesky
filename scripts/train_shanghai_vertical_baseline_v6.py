#!/usr/bin/env python3
"""Chronological diagnostic baseline for v6 altitude-controller actions.

This is deliberately limited to altitude actions because the current core set
has only 74 speed and 2 heading labels.  It reports a within-day chronological
holdout diagnostic, not a final controller-imitation metric.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_KIN = ["altitude_m", "ground_speed_kt", "track_heading_deg", "vertical_rate_mps", "latitude", "longitude", "time_from_sector_entry_sec"]
NUMERIC_ENRICHED = NUMERIC_KIN + ["requested_level_m", "next_route_level_m", "planned_speed_kt", "traffic_min_cpa_nm", "traffic_count", "cat_observation_age_sec", "cat_selected_altitude_ft", "cat_indicated_airspeed_kt", "cat_true_airspeed_kt", "cat_baro_vertical_rate_fpm", "cat_critical_data_age_sec"]
CATEGORICAL_ENRICHED = ["flight_phase", "primary_sector", "plan_adep", "plan_ades", "aircraft_type", "sid", "star", "cat_selected_altitude_source", "movement_vertical_code", "movement_turn_code"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--test-fraction", type=float, default=0.2)
    return p.parse_args()


def number(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (ValueError, TypeError):
        return None


def build_frame(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        action = record["label"]["actions"][0]
        if action.get("intent_family") != "altitude" or action.get("action") not in {"climb", "descend", "maintain_altitude"}:
            continue
        mi = record["model_input"]
        state, fp = mi.get("target_aircraft_state", {}), mi.get("flight_plan_context", {})
        route, traffic = mi.get("route_context", {}), mi.get("traffic_summary", {})
        cat = mi.get("surveillance_intent_context", {})
        cat_fields, q = cat.get("fields", {}), mi.get("surveillance_quality_context", {})
        proc, sector = mi.get("procedure_context", {}), mi.get("sector_context", {})
        movement = q.get("mode_of_movement") or {}
        rows.append({
            "instruction_bundle_id": record["instruction_bundle_id"],
            "epoch": record["provenance"]["causal_alignment_v5"]["decision_cutoff_epoch"],
            "action": action["action"], "target_m": number(action.get("target_value")),
            "altitude_m": number(state.get("altitude_m")), "ground_speed_kt": number(state.get("ground_speed_kt")),
            "track_heading_deg": number(state.get("track_heading_deg")), "vertical_rate_mps": number(state.get("vertical_rate_mps")),
            "latitude": number(state.get("latitude")), "longitude": number(state.get("longitude")), "time_from_sector_entry_sec": number(state.get("time_from_sector_entry_sec")),
            "requested_level_m": number(fp.get("requested_level_m")), "next_route_level_m": number(route.get("next_route_point_planned_level_m")),
            "planned_speed_kt": number(fp.get("planned_speed_kt")), "traffic_min_cpa_nm": number(traffic.get("minimum_cpa_horizontal_nm")), "traffic_count": number(traffic.get("traffic_count")),
            "cat_observation_age_sec": number(cat.get("observation_age_sec")), "cat_selected_altitude_ft": number(cat_fields.get("selected_altitude_ft")),
            "cat_indicated_airspeed_kt": number(cat_fields.get("indicated_airspeed_kt")), "cat_true_airspeed_kt": number(cat_fields.get("true_airspeed_kt")), "cat_baro_vertical_rate_fpm": number(cat_fields.get("barometric_vertical_rate_fpm")), "cat_critical_data_age_sec": number(q.get("maximum_critical_track_data_age_sec")),
            "flight_phase": proc.get("flight_phase"), "primary_sector": sector.get("primary_sector_code"), "plan_adep": fp.get("plan_adep"), "plan_ades": fp.get("plan_ades"), "aircraft_type": fp.get("aircraft_type"), "sid": fp.get("sid"), "star": fp.get("star"), "cat_selected_altitude_source": cat_fields.get("selected_altitude_source"), "movement_vertical_code": None if movement.get("vertical_code") is None else str(movement.get("vertical_code")), "movement_turn_code": None if movement.get("turn_code") is None else str(movement.get("turn_code")),
        })
    return pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)


def make_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False))]), categorical),
    ])
    return Pipeline([("preprocess", pre), ("classifier", HistGradientBoostingClassifier(max_iter=160, learning_rate=0.06, max_leaf_nodes=12, min_samples_leaf=12, l2_regularization=1.0, random_state=42))])


def run_variant(frame: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, name: str, numeric: list[str], categorical: list[str], output: Path) -> dict[str, Any]:
    model = make_pipeline(numeric, categorical)
    counts = train["action"].value_counts()
    weights = train["action"].map(lambda value: len(train) / (len(counts) * counts[value])).to_numpy()
    model.fit(train[numeric + categorical], train["action"], classifier__sample_weight=weights)
    predicted = model.predict(test[numeric + categorical])
    target_model = make_pipeline(numeric, categorical)
    target_counts = train["target_m"].value_counts()
    target_weights = train["target_m"].map(lambda value: len(train) / (len(target_counts) * target_counts[value])).to_numpy()
    target_model.fit(train[numeric + categorical], train["target_m"], classifier__sample_weight=target_weights)
    predicted_target = target_model.predict(test[numeric + categorical]).astype(float)
    target_within_100m = np.abs(predicted_target - test["target_m"].to_numpy(dtype=float)) <= 100.0
    joint = (predicted == test["action"].to_numpy()) & target_within_100m
    result = {
        "variant": name, "train_rows": int(len(train)), "test_rows": int(len(test)),
        "test_action_accuracy": float(accuracy_score(test["action"], predicted)),
        "test_action_balanced_accuracy": float(balanced_accuracy_score(test["action"], predicted)),
        "test_action_macro_recall_present_classes": float(recall_score(test["action"], predicted, average="macro", zero_division=0)),
        "test_target_within_100m_accuracy": float(target_within_100m.mean()),
        "test_joint_action_and_target_within_100m_accuracy": float(joint.mean()),
        "train_class_counts": {str(k): int(v) for k, v in counts.items()},
        "test_class_counts": {str(k): int(v) for k, v in test["action"].value_counts().items()},
    }
    probs = model.predict_proba(test[numeric + categorical])
    predictions = test[["instruction_bundle_id", "epoch", "action", "target_m"]].copy()
    predictions["predicted_action"] = predicted
    predictions["predicted_action_confidence"] = probs.max(axis=1)
    predictions["predicted_target_m"] = predicted_target
    predictions["target_within_100m"] = target_within_100m
    predictions["joint_action_target_within_100m"] = joint
    predictions.to_parquet(output / f"{name}_chronological_holdout_predictions.parquet", index=False)
    joblib.dump({"action_model": model, "target_model": target_model, "numeric_features": numeric, "categorical_features": categorical, "scope": "altitude_action_and_target_only"}, output / f"{name}_vertical_action_model.joblib")
    return result


def main() -> int:
    a = parse_args()
    frame = build_frame(a.core_jsonl)
    if frame.empty:
        raise SystemExit("no core altitude actions")
    split = int(len(frame) * (1.0 - a.test_fraction))
    train, test = frame.iloc[:split].copy(), frame.iloc[split:].copy()
    if train["action"].nunique() < 2 or test["action"].nunique() < 2:
        raise SystemExit("chronological split has insufficient action classes")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model_version": "shanghai_vertical_diagnostic_baseline_v6.0", "scope": "single-day chronological holdout diagnostic; not final controller-imitation precision", "source": str(a.core_jsonl.resolve()),
        "dataset": {"altitude_core_rows": int(len(frame)), "time_range_epoch": [float(frame.epoch.min()), float(frame.epoch.max())], "chronological_split": {"train": int(len(train)), "test": int(len(test)), "test_fraction": a.test_fraction}},
        "variants": [
            run_variant(frame, train, test, "kinematic_only", NUMERIC_KIN, [], a.output_dir),
            run_variant(frame, train, test, "quality_plan_enriched", NUMERIC_ENRICHED, CATEGORICAL_ENRICHED, a.output_dir),
        ],
        "data_gates": {"speed_core_labels": 74, "heading_core_labels": 2, "speed_heading_model_trained": False, "reason": "insufficient class support for reliable chronological holdout"},
    }
    (a.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
