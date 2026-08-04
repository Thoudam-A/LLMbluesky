#!/usr/bin/env python3
"""Run the trained program-aware Shanghai policy on frozen shadow replay."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import pyarrow.dataset as ds

from seu_imitation_common import jsonl_rows
from shanghai_program_policy import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PARQUET_COLUMNS,
    candidate_safety,
    command_for_candidate,
    expand_candidate_rows,
    record_runtime_candidate,
    runtime_candidate_rejection,
    state_features,
)
from train_shanghai_program_policy import feature_schema_sha256


VERSION = "shanghai_program_policy_replay_v1.0"
LOCAL_TZ = ZoneInfo("Asia/Shanghai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--trajectory-parquet", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--local-date")
    parser.add_argument("--decision-threshold", type=float)
    parser.add_argument("--start-tick", type=int, default=0)
    parser.add_argument("--max-ticks", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def day_bounds(local_date: str) -> tuple[int, int]:
    start = dt.datetime.fromisoformat(local_date).replace(tzinfo=LOCAL_TZ)
    return int(start.timestamp()), int((start + dt.timedelta(days=1)).timestamp())


def load_program_rows(
    parquet_path: Path,
    local_date: str,
    minimum_altitude_m: float,
    maximum_altitude_m: float,
) -> dict[tuple[str, int], dict[str, Any]]:
    dataset = ds.dataset(str(parquet_path), format="parquet")
    start, end = day_bounds(local_date)
    expression = (
        (ds.field("time_bucket_epoch") >= start)
        & (ds.field("time_bucket_epoch") < end)
        & (ds.field("inside_sector") == True)  # noqa: E712
        & (ds.field("altitude_m") >= minimum_altitude_m)
        & (ds.field("altitude_m") <= maximum_altitude_m)
    )
    table = dataset.scanner(
        columns=PARQUET_COLUMNS,
        filter=expression,
        batch_size=65536,
    ).to_table()
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for row in table.to_pylist():
        key = (str(row["trajectory_id"]), int(row["time_bucket_epoch"]))
        prior = best.get(key)
        if prior is None or float(row.get("event_time_epoch") or 0.0) > float(
            prior.get("event_time_epoch") or 0.0
        ):
            best[key] = row
    return best


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return float(ordered[index])


def valid_traffic_rows(tick: dict[str, Any]) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for row in tick.get("aircraft", []):
        try:
            values = [
                float(row["latitude"]),
                float(row["longitude"]),
                float(row["altitude_m"]),
                float(row["ground_speed_mps"]),
                float(row["track_heading_deg"]),
            ]
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in values):
            valid.append(row)
    return valid


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    bundle = joblib.load(args.model)
    if bundle["config_version"] != config["config_version"]:
        raise SystemExit(
            f"model/config mismatch: {bundle['config_version']} != {config['config_version']}"
        )
    actual_config_sha256 = sha256_file(args.config)
    if bundle.get("config_sha256") != actual_config_sha256:
        raise SystemExit(
            "model/config SHA256 mismatch: "
            f"{bundle.get('config_sha256')} != {actual_config_sha256}"
        )
    actual_schema_sha256 = feature_schema_sha256(config)
    if bundle.get("feature_schema_sha256") != actual_schema_sha256:
        raise SystemExit(
            "model feature/candidate schema mismatch: "
            f"{bundle.get('feature_schema_sha256')} != {actual_schema_sha256}"
        )
    if bundle.get("numeric_features") != NUMERIC_FEATURES or bundle.get(
        "categorical_features"
    ) != CATEGORICAL_FEATURES:
        raise SystemExit("model feature lists do not match runtime code")
    pipeline = bundle["pipeline"]
    threshold = (
        float(args.decision_threshold)
        if args.decision_threshold is not None
        else float(bundle["decision_threshold"])
    )
    local_date = args.local_date or config["data"]["test_local_date"]
    minimum_altitude = float(config["data"]["minimum_state_altitude_m"])
    maximum_altitude = float(config["data"]["maximum_state_altitude_m"])
    program_rows = load_program_rows(
        args.trajectory_parquet,
        local_date,
        minimum_altitude,
        maximum_altitude,
    )
    runtime = config["runtime"]
    safety_config = config["safety"]
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    command_cooldown: dict[str, float] = {}
    axis_cooldown: dict[tuple[str, str], float] = {}
    last_target: dict[tuple[str, str], tuple[str, float]] = {}
    command_counts: collections.Counter[str] = collections.Counter()
    selection_counts: collections.Counter[str] = collections.Counter()
    rejection_counts: collections.Counter[str] = collections.Counter()
    phase_counts: collections.Counter[str] = collections.Counter()
    margins: list[float] = []
    safety_durations: list[float] = []
    prediction_durations: list[float] = []
    processed = 0
    ticks_with_scored_aircraft = 0
    ticks_above_threshold = 0
    ticks_with_command = 0
    scored_aircraft_count = 0
    eligible_candidate_count = 0
    safety_checks = 0
    safety_rejections = 0
    start_wall = time.perf_counter()
    args.output_log.parent.mkdir(parents=True, exist_ok=True)

    with args.output_log.open("w", encoding="utf-8", newline="\n") as output:
        for source_index, tick in enumerate(jsonl_rows(args.replay)):
            if source_index < args.start_tick:
                continue
            if args.max_ticks is not None and processed >= args.max_ticks:
                break
            processed += 1
            epoch = float(tick["time_bucket_epoch"])
            traffic = valid_traffic_rows(tick)
            program_tick_rows: list[dict[str, Any]] = []
            target_rows: list[dict[str, Any]] = []
            for replay_row in traffic:
                key = (
                    str(replay_row.get("trajectory_id") or ""),
                    int(tick["time_bucket_epoch"]),
                )
                program_row = program_rows.get(key)
                if program_row is None:
                    continue
                program_tick_rows.append(program_row)
                callsign = str(program_row.get("callsign") or "").upper()
                if not callsign or epoch < command_cooldown.get(
                    callsign, float("-inf")
                ):
                    continue
                target_rows.append(program_row)
            if not target_rows:
                continue
            ticks_with_scored_aircraft += 1
            scored_aircraft_count += len(target_rows)

            candidate_rows: list[dict[str, Any]] = []
            state_by_id: dict[str, dict[str, Any]] = {}
            row_by_id: dict[str, dict[str, Any]] = {}
            for program_row in target_rows:
                # Match the training feature scope: inside-sector aircraft in
                # the configured 1400-6200 m band.  The broader replay traffic
                # remains visible only to the safety gate below.
                state = state_features(program_row, program_tick_rows)
                state_id = f"{state['trajectory_id']}:{int(epoch)}"
                state["state_id"] = state_id
                state_by_id[state_id] = state
                row_by_id[state_id] = program_row
                candidate_rows.extend(expand_candidate_rows(state, config))
                phase_counts[state["phase"]] += 1
            frame = pd.DataFrame(candidate_rows)
            predict_start = time.perf_counter()
            frame["candidate_score"] = pipeline.predict_proba(frame[features])[:, 1]
            prediction_durations.append(time.perf_counter() - predict_start)

            eligible: list[dict[str, Any]] = []
            audit_aircraft: list[dict[str, Any]] = []
            for state_id, group in frame.groupby("state_id", sort=False):
                state = state_by_id[state_id]
                callsign = state["callsign"]
                hold_score = float(
                    group.loc[group["candidate_id"] == "HOLD", "candidate_score"].max()
                )
                non_hold = group[group["candidate_id"] != "HOLD"].copy()
                non_hold["margin"] = non_hold["candidate_score"] - hold_score
                non_hold.sort_values(
                    ["margin", "candidate_score", "candidate_id"],
                    ascending=[False, False, True],
                    inplace=True,
                )
                top = non_hold.head(5)
                audit_aircraft.append(
                    {
                        "callsign": callsign,
                        "trajectory_id": state["trajectory_id"],
                        "phase": state["phase"],
                        "active_leg": state["active_leg"],
                        "next_fix": state["next_fix"],
                        "hold_score": round(hold_score, 6),
                        "top_non_hold": [
                            {
                                "candidate_id": str(row["candidate_id"]),
                                "score": round(float(row["candidate_score"]), 6),
                                "margin": round(float(row["margin"]), 6),
                            }
                            for _, row in top.iterrows()
                        ],
                    }
                )
                for _, row in non_hold.iterrows():
                    margin = float(row["margin"])
                    if margin < threshold:
                        break
                    kind = str(row["candidate_kind"])
                    row["phase"] = state["phase"]
                    row["ground_speed_kt"] = state["ground_speed_kt"]
                    rejection = runtime_candidate_rejection(
                        row,
                        epoch,
                        command_cooldown,
                        axis_cooldown,
                        last_target,
                        runtime,
                    )
                    if rejection is not None:
                        rejection_counts[rejection] += 1
                        continue
                    item = row.to_dict()
                    item["state_id"] = state_id
                    item["callsign"] = callsign
                    item["margin"] = margin
                    item["hold_score"] = hold_score
                    eligible.append(item)
            if not eligible:
                continue
            ticks_above_threshold += 1
            eligible_candidate_count += len(eligible)
            eligible.sort(
                key=lambda row: (
                    -float(row["margin"]),
                    -float(row["candidate_score"]),
                    str(row["candidate_id"]),
                    str(row["callsign"]),
                )
            )
            selected: dict[str, Any] | None = None
            selected_safety: dict[str, Any] | None = None
            checked_candidates: list[dict[str, Any]] = []
            for candidate in eligible[: int(runtime["candidate_fallback_limit"])]:
                safety_start = time.perf_counter()
                if bool(safety_config["enabled"]):
                    safety_result = candidate_safety(
                        row_by_id[candidate["state_id"]],
                        candidate,
                        traffic,
                        safety_config,
                    )
                else:
                    safety_result = {
                        "safe": True,
                        "checked_pairs": 0,
                        "unsafe_pairs": [],
                    }
                safety_durations.append(time.perf_counter() - safety_start)
                safety_checks += 1
                checked_candidates.append(
                    {
                        "callsign": candidate["callsign"],
                        "candidate_id": candidate["candidate_id"],
                        "margin": round(float(candidate["margin"]), 6),
                        "safe": bool(safety_result["safe"]),
                        "checked_pairs": safety_result.get("checked_pairs"),
                        "unsafe_pairs": safety_result.get("unsafe_pairs", [])[:3],
                    }
                )
                if safety_result["safe"]:
                    selected = candidate
                    selected_safety = safety_result
                    break
                safety_rejections += 1
                rejection_counts["finite_horizon_safety"] += 1

            commands: list[str] = []
            if selected is not None:
                command = command_for_candidate(selected["callsign"], selected)
                if command:
                    commands.append(command)
                    callsign = str(selected["callsign"])
                    kind = str(selected["candidate_kind"])
                    record_runtime_candidate(
                        selected,
                        epoch,
                        command_cooldown,
                        axis_cooldown,
                        last_target,
                        runtime,
                    )
                    command_counts[kind] += 1
                    selection_counts[str(selected["candidate_id"])] += 1
                    margins.append(float(selected["margin"]))
                    ticks_with_command += 1
            output.write(
                json.dumps(
                    {
                        "policy_version": VERSION,
                        "model_version": bundle["model_version"],
                        "config_version": config["config_version"],
                        "source_tick_index": source_index,
                        "replay_tick_id": tick.get("replay_tick_id"),
                        "event_time_utc": tick.get("event_time_utc"),
                        "event_time_epoch": epoch,
                        "reference_hidden": True,
                        "decision_threshold": threshold,
                        "scored_aircraft_count": len(target_rows),
                        "eligible_candidate_count": len(eligible),
                        "commands": commands,
                        "selected_candidate": (
                            None
                            if selected is None
                            else {
                                "callsign": selected["callsign"],
                                "trajectory_id": state_by_id[selected["state_id"]][
                                    "trajectory_id"
                                ],
                                "candidate_id": selected["candidate_id"],
                                "candidate_kind": selected["candidate_kind"],
                                "candidate_target": selected["candidate_target"],
                                "candidate_score": round(
                                    float(selected["candidate_score"]), 6
                                ),
                                "hold_score": round(
                                    float(selected["hold_score"]), 6
                                ),
                                "margin": round(float(selected["margin"]), 6),
                                "phase": state_by_id[selected["state_id"]]["phase"],
                                "active_leg": state_by_id[selected["state_id"]][
                                    "active_leg"
                                ],
                                "next_fix": state_by_id[selected["state_id"]][
                                    "next_fix"
                                ],
                            }
                        ),
                        "safety_result": selected_safety,
                        "checked_candidates": checked_candidates,
                        "aircraft_audit_top": sorted(
                            audit_aircraft,
                            key=lambda item: (
                                -(
                                    item["top_non_hold"][0]["margin"]
                                    if item["top_non_hold"]
                                    else -999
                                ),
                                item["callsign"],
                            ),
                        )[:5],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    elapsed = time.perf_counter() - start_wall
    summary = {
        "policy_version": VERSION,
        "model_version": bundle["model_version"],
        "config_version": config["config_version"],
        "local_date": local_date,
        "reference_hidden": True,
        "historical_states_frozen": True,
        "commands_applied_to_trajectory": False,
        "decision_threshold": threshold,
        "processed_ticks": processed,
        "ticks_with_scored_aircraft": ticks_with_scored_aircraft,
        "ticks_above_threshold": ticks_above_threshold,
        "ticks_with_command": ticks_with_command,
        "scored_aircraft_count": scored_aircraft_count,
        "eligible_candidate_count": eligible_candidate_count,
        "commands_by_kind": dict(sorted(command_counts.items())),
        "selected_targets": dict(selection_counts.most_common()),
        "scored_aircraft_by_phase": dict(sorted(phase_counts.items())),
        "rejections": dict(sorted(rejection_counts.items())),
        "safety": {
            "enabled": bool(safety_config["enabled"]),
            "checks": safety_checks,
            "rejections": safety_rejections,
            "selected_commands_passed": ticks_with_command,
            "selected_command_safe_rate": 1.0 if ticks_with_command else None,
            "scope": safety_config["scope"],
            "claim_boundary": safety_config["claim_boundary"],
        },
        "selected_margin": {
            "mean": round(statistics.mean(margins), 6) if margins else None,
            "median": round(statistics.median(margins), 6) if margins else None,
            "p90": round(percentile(margins, 0.9), 6) if margins else None,
        },
        "runtime_sec": round(elapsed, 3),
        "prediction_duration_sec": {
            "sum": round(sum(prediction_durations), 3),
            "median_per_scored_tick": round(
                statistics.median(prediction_durations), 6
            )
            if prediction_durations
            else None,
        },
        "safety_duration_sec": {
            "sum": round(sum(safety_durations), 3),
            "median_per_check": round(statistics.median(safety_durations), 6)
            if safety_durations
            else None,
        },
        "artifacts": {
            "replay": str(args.replay.resolve()),
            "replay_sha256": sha256_file(args.replay),
            "trajectory_parquet": str(args.trajectory_parquet.resolve()),
            "trajectory_parquet_sha256": sha256_file(args.trajectory_parquet),
            "model": str(args.model.resolve()),
            "model_sha256": sha256_file(args.model),
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
            "output_log": str(args.output_log.resolve()),
            "output_log_sha256": sha256_file(args.output_log),
        },
        "warnings": [
            "This is an offline frozen-trajectory shadow replay, not a live BlueSky closed loop.",
            "The safety rate refers only to the configured five-minute simplified constant-track pairwise check.",
            "The policy was trained from rule-extracted positives and weak HOLD negatives.",
        ],
    }
    summary_path = args.output_log.with_suffix(args.output_log.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
