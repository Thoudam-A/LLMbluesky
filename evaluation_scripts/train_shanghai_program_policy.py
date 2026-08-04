#!/usr/bin/env python3
"""Train and calibrate the Shanghai program-aware candidate ranker."""

from __future__ import annotations

import argparse
import hashlib
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shanghai_program_policy import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    record_runtime_candidate,
    runtime_candidate_rejection,
)


VERSION = "shanghai_program_policy_ranker_v1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-data", type=Path, required=True)
    parser.add_argument("--continuous-calibration-data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def feature_schema_sha256(config: dict[str, Any]) -> str:
    payload = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "candidates": config["candidates"],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def build_pipeline(config: dict[str, Any]) -> Pipeline:
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
                    handle_unknown="ignore",
                    min_frequency=2,
                    sparse_output=False,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    settings = config["model"]
    classifier = HistGradientBoostingClassifier(
        learning_rate=float(settings["learning_rate"]),
        max_iter=int(settings["max_iter"]),
        max_leaf_nodes=int(settings["max_leaf_nodes"]),
        min_samples_leaf=int(settings["min_samples_leaf"]),
        l2_regularization=float(settings["l2_regularization"]),
        random_state=int(config["data"]["random_seed"]),
    )
    return Pipeline([("preprocess", preprocessor), ("classifier", classifier)])


def scored_states(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    scored = frame[
        [
            "state_id",
            "split",
            "sample_kind",
            "true_candidate_id",
            "candidate_id",
            "candidate_kind",
            "candidate_target",
            "callsign",
            "time_bucket_epoch",
            "phase",
            "ground_speed_kt",
        ]
    ].copy()
    scored["candidate_score"] = scores
    rows: list[dict[str, Any]] = []
    for state_id, group in scored.groupby("state_id", sort=False):
        hold = group[group["candidate_id"] == "HOLD"].sort_values(
            "candidate_score", ascending=False
        )
        non_hold = group[group["candidate_id"] != "HOLD"].sort_values(
            ["candidate_score", "candidate_id"], ascending=[False, True]
        )
        hold_score = float(hold.iloc[0]["candidate_score"])
        best = non_hold.iloc[0]
        rows.append(
            {
                "state_id": state_id,
                "split": str(group.iloc[0]["split"]),
                "sample_kind": str(group.iloc[0]["sample_kind"]),
                "true_candidate_id": str(group.iloc[0]["true_candidate_id"]),
                "callsign": str(group.iloc[0]["callsign"]),
                "time_bucket_epoch": int(group.iloc[0]["time_bucket_epoch"]),
                "phase": str(group.iloc[0]["phase"]),
                "ground_speed_kt": float(group.iloc[0]["ground_speed_kt"]),
                "best_non_hold_candidate_id": str(best["candidate_id"]),
                "best_non_hold_kind": str(best["candidate_kind"]),
                "best_non_hold_target": float(best["candidate_target"]),
                "best_non_hold_score": float(best["candidate_score"]),
                "hold_score": hold_score,
                "margin": float(best["candidate_score"]) - hold_score,
            }
        )
    return pd.DataFrame(rows)


def threshold_metrics(states: pd.DataFrame, threshold: float) -> dict[str, float]:
    predicted = np.where(
        states["margin"].to_numpy() >= threshold,
        states["best_non_hold_candidate_id"].to_numpy(),
        "HOLD",
    )
    truth = states["true_candidate_id"].to_numpy()
    positive = truth != "HOLD"
    hold = ~positive
    predicted_command = predicted != "HOLD"
    exact = predicted == truth
    family_truth = np.array([value.split("_", 1)[0] for value in truth])
    family_prediction = np.array([value.split("_", 1)[0] for value in predicted])

    positive_exact = float(exact[positive].mean()) if positive.any() else 0.0
    positive_family = (
        float((family_truth[positive] == family_prediction[positive]).mean())
        if positive.any()
        else 0.0
    )
    hold_specificity = (
        float((predicted[hold] == "HOLD").mean()) if hold.any() else 0.0
    )
    command_true_positive = int((predicted_command & positive).sum())
    command_precision = (
        float(command_true_positive / predicted_command.sum())
        if predicted_command.any()
        else 0.0
    )
    command_recall = (
        float(command_true_positive / positive.sum()) if positive.any() else 0.0
    )
    objective = (
        2.0 * positive_exact * hold_specificity / (positive_exact + hold_specificity)
        if positive_exact + hold_specificity > 0
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "objective_hmean_positive_exact_hold_specificity": objective,
        "sampled_exact_accuracy": float(exact.mean()),
        "positive_exact_accuracy": positive_exact,
        "positive_family_accuracy": positive_family,
        "hold_specificity": hold_specificity,
        "command_precision_on_sampled_states": command_precision,
        "command_recall_on_positive_states": command_recall,
        "predicted_command_count": int(predicted_command.sum()),
        "true_positive_state_count": int(positive.sum()),
        "hold_state_count": int(hold.sum()),
    }


def calibrate_threshold(
    states: pd.DataFrame, config: dict[str, Any]
) -> tuple[float, list[dict[str, float]], bool]:
    margins = states["margin"].to_numpy(dtype=float)
    quantiles = np.linspace(0.0, 1.0, 201)
    thresholds = sorted(
        set(
            [-1.0, 1.0]
            + [float(value) for value in np.quantile(margins, quantiles)]
        )
    )
    sweep = [threshold_metrics(states, value) for value in thresholds]
    minimum_precision = float(
        config["model"]["minimum_sampled_command_precision"]
    )
    maximum_ratio = float(
        config["model"]["maximum_sampled_command_to_positive_ratio"]
    )
    feasible = [
        row
        for row in sweep
        if row["command_precision_on_sampled_states"] >= minimum_precision
        and row["predicted_command_count"]
        <= maximum_ratio * row["true_positive_state_count"]
    ]
    pool = feasible or sweep
    best = max(
        pool,
        key=lambda row: (
            row["positive_exact_accuracy"],
            row["objective_hmean_positive_exact_hold_specificity"],
            row["command_precision_on_sampled_states"],
            row["threshold"],
        ),
    )
    return float(best["threshold"]), sweep, bool(feasible)


def align_predictions(
    references: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    window_sec: float = 60.0,
) -> tuple[int, int]:
    used: set[int] = set()
    broad_matches = 0
    strict_matches = 0
    for reference in sorted(
        references, key=lambda row: (row["callsign"], row["epoch"])
    ):
        family = reference["candidate_id"].split("_", 1)[0]
        eligible: list[tuple[float, int]] = []
        for index, prediction in enumerate(predictions):
            if index in used or prediction["callsign"] != reference["callsign"]:
                continue
            if prediction["candidate_id"].split("_", 1)[0] != family:
                continue
            delta = abs(float(prediction["epoch"]) - float(reference["epoch"]))
            if delta <= window_sec:
                eligible.append((delta, index))
        if not eligible:
            continue
        _, selected_index = min(eligible)
        used.add(selected_index)
        broad_matches += 1
        if (
            predictions[selected_index]["candidate_id"]
            == reference["candidate_id"]
        ):
            strict_matches += 1
    return broad_matches, strict_matches


def scored_continuous_candidates(
    frame: pd.DataFrame,
    scores: np.ndarray,
) -> pd.DataFrame:
    """Retain ranked fallback candidates for runtime-faithful calibration."""

    scored = frame[
        [
            "state_id",
            "true_candidate_id",
            "candidate_id",
            "candidate_kind",
            "candidate_target",
            "callsign",
            "time_bucket_epoch",
            "phase",
            "ground_speed_kt",
        ]
    ].copy()
    scored["candidate_score"] = scores
    hold_scores = (
        scored[scored["candidate_id"] == "HOLD"]
        .groupby("state_id")["candidate_score"]
        .max()
        .rename("hold_score")
    )
    non_hold = scored[scored["candidate_id"] != "HOLD"].copy()
    non_hold = non_hold.join(hold_scores, on="state_id", how="inner")
    non_hold["margin"] = (
        non_hold["candidate_score"] - non_hold["hold_score"]
    )
    return non_hold.sort_values(
        [
            "time_bucket_epoch",
            "margin",
            "candidate_score",
            "candidate_id",
            "callsign",
        ],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)


def continuous_threshold_metrics(
    candidates: pd.DataFrame,
    threshold: float,
    config: dict[str, Any],
) -> dict[str, float]:
    runtime = config["runtime"]
    command_cooldown: dict[str, float] = {}
    axis_cooldown: dict[tuple[str, str], float] = {}
    last_target: dict[tuple[str, str], tuple[str, float]] = {}
    predictions: list[dict[str, Any]] = []
    selected_epoch: float | None = None
    for row in candidates.itertuples(index=False):
        epoch = float(row.time_bucket_epoch)
        if selected_epoch == epoch or float(row.margin) < threshold:
            continue
        candidate = {
            "callsign": str(row.callsign),
            "candidate_kind": str(row.candidate_kind),
            "candidate_id": str(row.candidate_id),
            "candidate_target": float(row.candidate_target),
            "phase": str(row.phase),
            "ground_speed_kt": float(row.ground_speed_kt),
        }
        if runtime_candidate_rejection(
            candidate,
            epoch,
            command_cooldown,
            axis_cooldown,
            last_target,
            runtime,
        ) is not None:
            continue
        record_runtime_candidate(
            candidate,
            epoch,
            command_cooldown,
            axis_cooldown,
            last_target,
            runtime,
        )
        selected_epoch = epoch
        predictions.append(
            {
                "callsign": candidate["callsign"],
                "epoch": epoch,
                "candidate_id": candidate["candidate_id"],
            }
        )
    states = candidates.drop_duplicates("state_id")
    references = [
        {
            "callsign": str(row["callsign"]),
            "epoch": float(row["time_bucket_epoch"]),
            "candidate_id": str(row["true_candidate_id"]),
        }
        for _, row in states[
            ~states["true_candidate_id"].isin(["HOLD", "UNLABELED"])
        ].iterrows()
    ]
    broad_matches, strict_matches = align_predictions(references, predictions)
    reference_count = len(references)
    prediction_count = len(predictions)
    broad_recall = broad_matches / reference_count if reference_count else 0.0
    broad_precision = broad_matches / prediction_count if prediction_count else 0.0
    strict_recall = strict_matches / reference_count if reference_count else 0.0
    strict_precision = strict_matches / prediction_count if prediction_count else 0.0
    f1 = (
        2.0 * broad_precision * broad_recall / (broad_precision + broad_recall)
        if broad_precision + broad_recall > 0
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "reference_count": reference_count,
        "prediction_count": prediction_count,
        "command_to_reference_ratio": (
            prediction_count / reference_count if reference_count else 0.0
        ),
        "broad_matches": broad_matches,
        "strict_matches": strict_matches,
        "broad_recall": broad_recall,
        "broad_precision": broad_precision,
        "strict_recall": strict_recall,
        "strict_precision": strict_precision,
        "broad_f1": f1,
    }


def calibrate_continuous_threshold(
    candidates: pd.DataFrame, config: dict[str, Any]
) -> tuple[float, list[dict[str, float]], bool]:
    margins = (
        candidates.groupby("state_id")["margin"].max().to_numpy(dtype=float)
    )
    thresholds = sorted(
        set(
            [-1.0, 1.0]
            + [
                float(value)
                # Continuous replay is substantially more expensive than
                # sampled-state scoring.  Twenty-one deterministic quantiles
                # are enough to calibrate the operating point without
                # repeatedly replaying almost identical thresholds.
                for value in np.quantile(margins, np.linspace(0.0, 1.0, 21))
            ]
        )
    )
    sweep = [
        continuous_threshold_metrics(candidates, value, config)
        for value in thresholds
    ]
    minimum_precision = float(config["model"]["minimum_sampled_command_precision"])
    maximum_ratio = float(
        config["model"]["maximum_sampled_command_to_positive_ratio"]
    )
    feasible = [
        row
        for row in sweep
        if row["broad_precision"] >= minimum_precision
        and row["command_to_reference_ratio"] <= maximum_ratio
    ]
    if feasible:
        best = max(
            feasible,
            key=lambda row: (
                row["strict_recall"],
                row["broad_recall"],
                row["strict_precision"],
                row["threshold"],
            ),
        )
    else:
        bounded = [
            row
            for row in sweep
            if row["command_to_reference_ratio"] <= maximum_ratio
        ]
        best = max(
            bounded or sweep,
            key=lambda row: (
                row["broad_f1"],
                row["strict_recall"],
                row["broad_precision"],
                row["threshold"],
            ),
        )
    return float(best["threshold"]), sweep, bool(feasible)


def predict_scores_in_chunks(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    features: list[str],
    chunk_size: int = 100000,
) -> np.ndarray:
    parts = []
    for start in range(0, len(frame), chunk_size):
        chunk = frame.iloc[start : start + chunk_size]
        parts.append(pipeline.predict_proba(chunk[features])[:, 1])
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def split_metrics(
    pipeline: Pipeline, frame: pd.DataFrame, split: str, threshold: float
) -> tuple[dict[str, float], pd.DataFrame]:
    part = frame[frame["split"] == split].copy()
    scores = pipeline.predict_proba(
        part[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    )[:, 1]
    states = scored_states(part, scores)
    return threshold_metrics(states, threshold), states


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    frame = pd.read_parquet(args.candidate_data)
    required = set(
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [
            "state_id",
            "split",
            "candidate_id",
            "true_candidate_id",
            "is_selected",
            "training_weight",
        ]
    )
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"candidate data missing columns: {sorted(missing)}")
    train = frame[frame["split"] == "train"].copy()
    calibration = frame[frame["split"] == "calibration"].copy()
    if train.empty or calibration.empty:
        raise SystemExit("both train and calibration rows are required")

    pipeline = build_pipeline(config)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    pipeline.fit(
        train[features],
        train["is_selected"].astype(int),
        classifier__sample_weight=train["training_weight"].astype(float),
    )
    calibration_scores = pipeline.predict_proba(calibration[features])[:, 1]
    calibration_states = scored_states(calibration, calibration_scores)
    sampled_threshold, sampled_sweep, sampled_constrained_found = calibrate_threshold(
        calibration_states, config
    )
    continuous = pd.read_parquet(args.continuous_calibration_data)
    continuous_scores = predict_scores_in_chunks(pipeline, continuous, features)
    continuous_candidates = scored_continuous_candidates(
        continuous, continuous_scores
    )
    threshold, continuous_sweep, constrained_threshold_found = (
        calibrate_continuous_threshold(continuous_candidates, config)
    )
    train_metrics, train_states = split_metrics(pipeline, frame, "train", threshold)
    calibration_metrics = threshold_metrics(calibration_states, threshold)

    bundle = {
        "model_version": VERSION,
        "config_version": config["config_version"],
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "decision_threshold": threshold,
        "config_sha256": sha256_file(args.config),
        "training_candidate_data_sha256": sha256_file(args.candidate_data),
        "continuous_calibration_data_sha256": sha256_file(
            args.continuous_calibration_data
        ),
        "feature_schema_sha256": feature_schema_sha256(config),
        "pipeline": pipeline,
        "claim_boundary": (
            "candidate-ranking model trained from rule-extracted positive instructions "
            "and weak HOLD negatives; not an operational ATC policy"
        ),
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output_model)

    best_sweep = sorted(
        sampled_sweep,
        key=lambda row: (
            row["objective_hmean_positive_exact_hold_specificity"],
            row["command_precision_on_sampled_states"],
        ),
        reverse=True,
    )[:20]
    metrics = {
        "model_version": VERSION,
        "config_version": config["config_version"],
        "random_seed": config["data"]["random_seed"],
        "data": {
            "candidate_rows": int(len(frame)),
            "train_candidate_rows": int(len(train)),
            "calibration_candidate_rows": int(len(calibration)),
            "train_states": int(train["state_id"].nunique()),
            "calibration_states": int(calibration["state_id"].nunique()),
            "train_positive_states": int(
                train.loc[train["true_candidate_id"] != "HOLD", "state_id"].nunique()
            ),
            "calibration_positive_states": int(
                calibration.loc[
                    calibration["true_candidate_id"] != "HOLD", "state_id"
                ].nunique()
            ),
            "trajectory_overlap": int(
                len(
                    set(
                        train["trajectory_id"]
                    )
                    & set(calibration["trajectory_id"])
                )
            ),
        },
        "calibration": {
            "selected_threshold": threshold,
            "constraints": {
                "minimum_sampled_command_precision": config["model"][
                    "minimum_sampled_command_precision"
                ],
                "maximum_sampled_command_to_positive_ratio": config["model"][
                    "maximum_sampled_command_to_positive_ratio"
                ],
                "feasible_threshold_found": constrained_threshold_found,
            },
            "selected_metrics": calibration_metrics,
            "sampled_state_only_threshold": sampled_threshold,
            "sampled_state_only_feasible_threshold_found": sampled_constrained_found,
            "top_thresholds": best_sweep,
        },
        "continuous_calibration": {
            "candidate_rows": int(len(continuous)),
            "states": int(continuous["state_id"].nunique()),
            "positive_states": int(
                continuous.loc[
                    ~continuous["true_candidate_id"].isin(
                        ["HOLD", "UNLABELED"]
                    ),
                    "state_id",
                ].nunique()
            ),
            "unlabeled_states": int(
                continuous.loc[
                    continuous["true_candidate_id"] == "UNLABELED",
                    "state_id",
                ].nunique()
            ),
            "selected_threshold": threshold,
            "selected_metrics": continuous_threshold_metrics(
                continuous_candidates, threshold, config
            ),
            "feasible_precision_and_inflation_threshold_found": constrained_threshold_found,
            "top_thresholds": sorted(
                continuous_sweep,
                key=lambda row: (
                    row["strict_recall"],
                    row["broad_recall"],
                    row["broad_precision"],
                ),
                reverse=True,
            )[:20],
        },
        "diagnostic_metrics": {
            "train_at_selected_threshold": train_metrics,
            "calibration_at_selected_threshold": calibration_metrics,
        },
        "artifacts": {
            "candidate_data": str(args.candidate_data.resolve()),
            "candidate_data_sha256": sha256_file(args.candidate_data),
            "continuous_calibration_data": str(
                args.continuous_calibration_data.resolve()
            ),
            "continuous_calibration_data_sha256": sha256_file(
                args.continuous_calibration_data
            ),
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
            "model": str(args.output_model.resolve()),
            "model_sha256": sha256_file(args.output_model),
        },
        "warnings": [
            "Calibration metrics use sampled weak HOLD states, not the full continuous replay distribution.",
            "No 2025-07-02 labels or states are used for fitting or threshold selection.",
            "Reported values are diagnostic model metrics, not controller imitation accuracy.",
        ],
    }
    args.output_metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
