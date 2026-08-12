#!/usr/bin/env python3
"""Train the target-selection stage for the Shanghai hybrid imitation policy.

The existing v1.2 candidate ranker remains the command-trigger stage.  This
model is trained only on positive command records and predicts:

1. altitude versus speed; and
2. the target value within the selected family.

Only features that the frozen replay runtime can reproduce are used.  The
script deliberately excludes heading because the compact v7 corpus has too few
examples for a defensible learned heading policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


VERSION = "shanghai_hybrid_target_policy_v1.1_empirical_family_prior"

NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "altitude_m",
    "ground_speed_kt",
    "track_heading_deg",
    "vertical_rate_mps",
    "time_from_sector_entry_sec",
    "requested_level_m",
    "planned_speed_kt",
    "next_planned_level_m",
    "seconds_to_next_fix",
    "route_progress",
    "traffic_count",
    "nearest_horizontal_nm",
    "min_cpa_horizontal_nm",
]

CATEGORICAL_FEATURES = [
    "phase",
    "plan_adep",
    "plan_ades",
    "sid",
    "star",
    "departure_runway",
    "arrival_runway",
    "next_fix",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260810)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_schema_sha256() -> str:
    payload = json.dumps(
        {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pipeline(seed: int, min_samples_leaf: int) -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore", min_frequency=2, sparse_output=False
                ),
            ),
        ]
    )
    transform = ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=2.0,
        random_state=seed,
    )
    return Pipeline([("preprocess", transform), ("classifier", classifier)])


def balanced_weights(labels: pd.Series, base: pd.Series) -> np.ndarray:
    counts = labels.value_counts()
    factors = {label: len(labels) / (len(counts) * count) for label, count in counts.items()}
    return np.asarray([float(weight) * factors[label] for label, weight in zip(labels, base)], dtype=float)


def prepare_frame(raw: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = raw.copy()
    frame["phase"] = frame["flight_phase"]
    frame["seconds_to_next_fix"] = frame["time_to_next_fix_sec"]
    frame["route_progress"] = frame["active_leg_fraction"]
    frame["min_cpa_horizontal_nm"] = frame["minimum_cpa_horizontal_nm"]
    allowed_altitude = {int(value) for value in config["candidates"]["altitude_levels_m"]}
    allowed_speed = {int(value) for value in config["candidates"]["speed_targets_kt"]}
    frame["target_int"] = frame["label_target_value"].round().astype("Int64")
    frame["target_id"] = frame.apply(
        lambda row: (
            f"ALT_{int(row.target_int)}"
            if row.label_intent_family == "altitude" and pd.notna(row.target_int)
            else f"SPD_{int(row.target_int)}"
            if row.label_intent_family == "speed" and pd.notna(row.target_int)
            else None
        ),
        axis=1,
    )
    in_scope = (
        ((frame.label_intent_family == "altitude") & frame.target_int.isin(allowed_altitude))
        | ((frame.label_intent_family == "speed") & frame.target_int.isin(allowed_speed))
    )
    counts = {
        "input_rows": int(len(frame)),
        "heading_or_other_excluded": int((~frame.label_intent_family.isin(["altitude", "speed"])).sum()),
        "target_outside_runtime_candidates": int(
            (frame.label_intent_family.isin(["altitude", "speed"]) & ~in_scope).sum()
        ),
    }
    frame = frame[in_scope].copy()
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"compact data missing runtime-compatible features: {missing}")
    counts["training_rows"] = int(len(frame))
    return frame, counts


def cross_validated_metrics(
    frame: pd.DataFrame,
    label_column: str,
    seed: int,
    folds: int,
    min_samples_leaf: int,
    class_balance: bool,
) -> dict[str, Any]:
    labels = frame[label_column].astype(str)
    groups = frame["callsign"].fillna(frame["instruction_bundle_id"]).astype(str)
    unique_groups = groups.nunique()
    n_splits = min(folds, unique_groups)
    predictions = pd.Series(index=frame.index, dtype="object")
    splitter = GroupKFold(n_splits=n_splits)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for train_index, validation_index in splitter.split(frame, labels, groups):
        train = frame.iloc[train_index]
        validation = frame.iloc[validation_index]
        model = pipeline(seed, min_samples_leaf)
        weights = (
            balanced_weights(
                train[label_column].astype(str), train["training_weight"].astype(float)
            )
            if class_balance
            else train["training_weight"].astype(float).to_numpy()
        )
        model.fit(
            train[features],
            train[label_column].astype(str),
            classifier__sample_weight=weights,
        )
        predictions.loc[validation.index] = model.predict(validation[features])
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    return {
        "rows": int(len(frame)),
        "groups": int(unique_groups),
        "folds": int(n_splits),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1": float(values["f1-score"]),
                "support": int(values["support"]),
            }
            for label, values in report.items()
            if label in set(labels)
        },
        "label_counts": labels.value_counts().to_dict(),
    }


def fit_final(
    frame: pd.DataFrame,
    label_column: str,
    seed: int,
    min_samples_leaf: int,
    class_balance: bool,
) -> Pipeline:
    model = pipeline(seed, min_samples_leaf)
    weights = (
        balanced_weights(
            frame[label_column].astype(str), frame["training_weight"].astype(float)
        )
        if class_balance
        else frame["training_weight"].astype(float).to_numpy()
    )
    model.fit(
        frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        frame[label_column].astype(str),
        classifier__sample_weight=weights,
    )
    return model


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw = pd.read_parquet(args.compact_data)
    frame, counts = prepare_frame(raw, config)
    if frame.empty:
        raise SystemExit("no in-scope altitude/speed target rows")

    family_metrics = cross_validated_metrics(
        frame, "label_intent_family", args.random_seed, args.folds, 10, False
    )
    altitude = frame[frame.label_intent_family == "altitude"].copy()
    speed = frame[frame.label_intent_family == "speed"].copy()
    altitude_metrics = cross_validated_metrics(
        altitude, "target_id", args.random_seed + 1, args.folds, 10, True
    )
    speed_metrics = cross_validated_metrics(
        speed, "target_id", args.random_seed + 2, min(args.folds, 3), 5, True
    )

    family_model = fit_final(frame, "label_intent_family", args.random_seed, 10, False)
    altitude_model = fit_final(altitude, "target_id", args.random_seed + 1, 10, True)
    speed_model = fit_final(speed, "target_id", args.random_seed + 2, 5, True)

    bundle = {
        "model_version": VERSION,
        "role": "target_selection_only; trigger remains program_policy_v1.2",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_schema_sha256": feature_schema_sha256(),
        "family_model": family_model,
        "target_models": {"altitude": altitude_model, "speed": speed_model},
        "runtime_candidates": config["candidates"],
        "source_compact_data_sha256": sha256(args.compact_data),
        "source_config_sha256": sha256(args.config),
        "claim_boundary": "weak-label positive-command target selector; not a complete ATC policy",
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output_model)

    report = {
        "model_version": VERSION,
        "counts": counts,
        "family_cross_validation": family_metrics,
        "altitude_target_cross_validation": altitude_metrics,
        "speed_target_cross_validation": speed_metrics,
        "model_sha256": sha256(args.output_model),
        "feature_schema_sha256": feature_schema_sha256(),
        "warnings": [
            "Cross-validation uses weak labels from one development date.",
            "The model does not decide command timing and cannot be scored alone with the imitation metric.",
            "Heading is excluded because only three compact v7 action rows are available.",
            "Speed target estimates are high variance because only about ninety in-scope rows are available.",
        ],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
