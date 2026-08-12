#!/usr/bin/env python3
"""Build leakage-bounded candidate-ranking data for the Shanghai policy."""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.dataset as ds

from seu_imitation_common import jsonl_rows, write_jsonl
from shanghai_program_policy import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PARQUET_COLUMNS,
    expand_candidate_rows,
    historical_candidate_id,
    state_features,
)


VERSION = "shanghai_program_policy_dataset_v1.0"
LOCAL_TZ = ZoneInfo("Asia/Shanghai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-parquet", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def load_day_rows(
    dataset: ds.Dataset,
    local_date: str,
    minimum_altitude_m: float,
    maximum_altitude_m: float,
) -> list[dict[str, Any]]:
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
    return table.to_pylist()


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        trajectory_id = str(row.get("trajectory_id") or "")
        if not trajectory_id:
            continue
        key = (trajectory_id, int(row["time_bucket_epoch"]))
        prior = best.get(key)
        if prior is None or float(row.get("event_time_epoch") or 0.0) > float(
            prior.get("event_time_epoch") or 0.0
        ):
            best[key] = row
    return sorted(
        best.values(),
        key=lambda row: (int(row["time_bucket_epoch"]), str(row["trajectory_id"])),
    )


def reference_date(row: dict[str, Any]) -> str:
    return str(row.get("event_time_local") or "")[:10]


def build_reference_audit(
    references: list[dict[str, Any]],
    rows_by_day: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    settings = config["data"]
    step = int(settings["step_sec"])
    offset = float(settings["positive_state_offset_sec"])
    trajectory_rows: dict[str, dict[str, tuple[list[float], list[dict[str, Any]]]]] = {}
    for day, rows in rows_by_day.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["trajectory_id"])].append(row)
        trajectory_rows[day] = {}
        for trajectory_id, values in grouped.items():
            ordered = sorted(values, key=lambda row: float(row["event_time_epoch"]))
            trajectory_rows[day][trajectory_id] = (
                [float(row["event_time_epoch"]) for row in ordered],
                ordered,
            )
    bucket_rows = {
        day: defaultdict(list)
        for day in rows_by_day
    }
    for day, rows in rows_by_day.items():
        for row in rows:
            bucket_rows[day][int(row["time_bucket_epoch"])].append(row)

    audit: list[dict[str, Any]] = []
    linked: list[dict[str, Any]] = []
    for reference in references:
        day = reference_date(reference)
        if day not in trajectory_rows or reference.get("intent_family") not in {
            "altitude",
            "speed",
        }:
            continue
        trajectory_id = str(
            (reference.get("target_state") or {}).get("trajectory_id") or ""
        )
        event_epoch = float(reference["event_time_epoch"])
        cutoff = event_epoch - offset
        trajectory = trajectory_rows[day].get(trajectory_id)
        row = None
        pre_state_age_sec = None
        if trajectory is not None:
            times, values = trajectory
            index = bisect.bisect_right(times, cutoff) - 1
            if index >= 0:
                candidate_row = values[index]
                age = cutoff - float(candidate_row["event_time_epoch"])
                if age <= float(settings["maximum_pre_state_age_sec"]):
                    row = candidate_row
                    pre_state_age_sec = age
        bucket = (
            int(row["time_bucket_epoch"])
            if row is not None
            else int(math.floor(cutoff / step) * step)
        )
        truth = historical_candidate_id(reference)
        item = {
            "reference_event_id": reference.get("reference_event_id"),
            "event_time_epoch": event_epoch,
            "local_date": day,
            "trajectory_id": trajectory_id,
            "callsign": reference.get("callsign"),
            "intent_family": reference.get("intent_family"),
            "target_value": reference.get("target_value"),
            "true_candidate_id": truth,
            "pre_state_bucket_epoch": bucket,
            "pre_state_age_sec": pre_state_age_sec,
            "pre_state_linked": row is not None,
            "candidate_available": False,
            "candidate_count": 0,
            "exclusion_reason": None,
        }
        if truth is None:
            item["exclusion_reason"] = "reference_missing_numeric_target"
            audit.append(item)
            continue
        if row is None:
            item["exclusion_reason"] = "no_in_scope_pre_state"
            audit.append(item)
            continue
        feature = state_features(row, bucket_rows[day][bucket])
        candidates = expand_candidate_rows(feature, config)
        candidate_by_id = {
            candidate["candidate_id"]: candidate for candidate in candidates
        }
        candidate_ids = set(candidate_by_id)
        item["candidate_count"] = len(candidates)
        item["phase"] = feature["phase"]
        item["candidate_available"] = truth in candidate_ids
        if truth not in candidate_ids:
            item["exclusion_reason"] = "historical_action_not_in_program_legal_candidates"
            audit.append(item)
            continue
        historical_action = str(reference.get("action") or "")
        expected_direction = {
            "climb": "climb",
            "descend": "descend",
            "maintain_altitude": "maintain",
            "increase_speed": "accelerate",
            "reduce_speed": "decelerate",
            "maintain_speed": "maintain_speed",
        }.get(historical_action)
        derived_direction = str(
            candidate_by_id[truth].get("candidate_direction") or ""
        )
        item["historical_action"] = historical_action
        item["expected_direction"] = expected_direction
        item["derived_candidate_direction"] = derived_direction
        item["direction_consistent"] = (
            expected_direction is None or expected_direction == derived_direction
        )
        if not item["direction_consistent"]:
            item["candidate_available"] = False
            item["exclusion_reason"] = "historical_direction_mismatch"
            audit.append(item)
            continue
        item["state_row"] = row
        item["state_features"] = feature
        item["reference"] = reference
        linked.append(item)
        audit.append({key: value for key, value in item.items() if key not in {"state_row", "state_features", "reference"}})
    return audit, linked


def assign_splits(
    positives: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, str], set[str], int]:
    train_date = config["data"]["train_local_date"]
    start, end = day_bounds(train_date)
    boundary = start + int(
        (end - start) * float(config["data"]["train_fraction_of_day"])
    )
    trajectory_sides: dict[str, set[str]] = defaultdict(set)
    for item in positives:
        if item["local_date"] != train_date:
            continue
        side = "train" if int(item["pre_state_bucket_epoch"]) < boundary else "calibration"
        trajectory_sides[item["trajectory_id"]].add(side)
    crossing = {
        trajectory_id
        for trajectory_id, sides in trajectory_sides.items()
        if len(sides) > 1
    }
    assignments = {
        trajectory_id: next(iter(sides))
        for trajectory_id, sides in trajectory_sides.items()
        if trajectory_id not in crossing
    }
    return assignments, crossing, boundary


def positive_samples(
    linked: list[dict[str, Any]],
    assignments: dict[str, str],
    train_date: str,
) -> tuple[list[dict[str, Any]], int]:
    by_state: dict[tuple[str, int], dict[str, Any]] = {}
    duplicate_count = 0
    for item in sorted(linked, key=lambda value: float(value["event_time_epoch"])):
        if item["local_date"] != train_date or item["trajectory_id"] not in assignments:
            continue
        key = (item["trajectory_id"], int(item["pre_state_bucket_epoch"]))
        if key in by_state:
            duplicate_count += 1
            continue
        by_state[key] = item
    samples: list[dict[str, Any]] = []
    for item in by_state.values():
        samples.append(
            {
                **item["state_features"],
                "state_id": (
                    f"{assignments[item['trajectory_id']]}:"
                    f"{item['trajectory_id']}:{item['pre_state_bucket_epoch']}"
                ),
                "split": assignments[item["trajectory_id"]],
                "sample_kind": "historical_instruction_positive",
                "true_candidate_id": item["true_candidate_id"],
                "reference_event_id": item["reference_event_id"],
                "label_source": "rule_extracted_reference_candidate",
            }
        )
    return samples, duplicate_count


def hold_samples(
    positives: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    assignments: dict[str, str],
    boundary: int,
    all_reference_epochs: dict[str, list[float]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    seed = int(config["data"]["random_seed"])
    rng = random.Random(seed)
    exclusion = float(config["data"]["hold_exclusion_sec"])
    max_per_positive = int(config["data"]["max_hold_samples_per_positive"])
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bucket: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        trajectory_id = str(row["trajectory_id"])
        if trajectory_id in assignments:
            by_trajectory[trajectory_id].append(row)
        by_bucket[int(row["time_bucket_epoch"])].append(row)
    positive_counts: Counter[str] = Counter()
    for sample in positives:
        positive_counts[sample["trajectory_id"]] += 1

    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for trajectory_id, count in positive_counts.items():
        split = assignments[trajectory_id]
        candidates = []
        for row in by_trajectory[trajectory_id]:
            bucket = int(row["time_bucket_epoch"])
            if (bucket < boundary) != (split == "train"):
                continue
            if any(
                abs(bucket - epoch) <= exclusion
                for epoch in all_reference_epochs.get(trajectory_id, [])
            ):
                continue
            candidates.append(row)
        rng.shuffle(candidates)
        for row in candidates[: max_per_positive * count]:
            key = (trajectory_id, int(row["time_bucket_epoch"]))
            selected[key] = row

    holds: list[dict[str, Any]] = []
    for (trajectory_id, bucket), row in sorted(
        selected.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        split = assignments[trajectory_id]
        feature = state_features(row, by_bucket[bucket])
        holds.append(
            {
                **feature,
                "state_id": f"{split}:{trajectory_id}:{bucket}",
                "split": split,
                "sample_kind": "weak_hold_far_from_extracted_instruction",
                "true_candidate_id": "HOLD",
                "reference_event_id": None,
                "label_source": "absence_of_extracted_instruction_with_120s_guard",
            }
        )
    return holds


def continuous_calibration_candidate_rows(
    positives: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    assignments: dict[str, str],
    boundary: int,
    all_reference_epochs: dict[str, list[float]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the actual post-boundary all-aircraft decision distribution.

    Targets whose trajectory appeared in the training split are excluded from
    threshold calibration, while they remain in traffic features so the
    feature scope matches runtime.
    """

    positive_map = {
        (sample["trajectory_id"], int(sample["time_bucket_epoch"])): sample[
            "true_candidate_id"
        ]
        for sample in positives
        if sample["split"] == "calibration"
    }
    train_trajectories = {
        trajectory_id
        for trajectory_id, split in assignments.items()
        if split == "train"
    }
    by_bucket: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        bucket = int(row["time_bucket_epoch"])
        if bucket >= boundary:
            by_bucket[bucket].append(row)

    rows: list[dict[str, Any]] = []
    state_count = 0
    positive_state_count = 0
    excluded_train_trajectory_states = 0
    for bucket in sorted(by_bucket):
        traffic = by_bucket[bucket]
        for row in traffic:
            trajectory_id = str(row["trajectory_id"])
            if trajectory_id in train_trajectories:
                excluded_train_trajectory_states += 1
                continue
            truth = positive_map.get((trajectory_id, bucket), "HOLD")
            near_any_instruction = truth == "HOLD" and any(
                abs(bucket - epoch)
                <= float(config["data"]["hold_exclusion_sec"])
                for epoch in all_reference_epochs.get(trajectory_id, [])
            )
            if near_any_instruction:
                # Preserve the decision opportunity for continuous threshold
                # replay, but do not pretend that absence of an exact T-4
                # positive makes this state a reliable HOLD negative.
                truth = "UNLABELED"
            if truth not in {"HOLD", "UNLABELED"}:
                positive_state_count += 1
            feature = state_features(row, traffic)
            feature.update(
                {
                    "state_id": f"continuous_calibration:{trajectory_id}:{bucket}",
                    "split": "continuous_calibration",
                    "sample_kind": (
                        "historical_instruction_positive"
                        if truth not in {"HOLD", "UNLABELED"}
                        else (
                            "continuous_unlabeled_near_instruction"
                            if truth == "UNLABELED"
                            else "continuous_weak_hold"
                        )
                    ),
                    "true_candidate_id": truth,
                }
            )
            rows.extend(
                expand_candidate_rows(feature, config, true_candidate_id=truth)
            )
            state_count += 1
    summary = {
        "state_count": state_count,
        "positive_state_count": positive_state_count,
        "hold_state_count": sum(
            1 for row in rows if row.get("candidate_id") == "HOLD"
            and row.get("true_candidate_id") == "HOLD"
        ),
        "unlabeled_state_count": sum(
            1 for row in rows if row.get("candidate_id") == "HOLD"
            and row.get("true_candidate_id") == "UNLABELED"
        ),
        "candidate_rows": len(rows),
        "excluded_states_from_train_trajectories": excluded_train_trajectory_states,
        "time_bucket_count": len(by_bucket),
    }
    return rows, summary


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train_date = config["data"]["train_local_date"]
    minimum_altitude = float(config["data"]["minimum_state_altitude_m"])
    maximum_altitude = float(config["data"]["maximum_state_altitude_m"])
    dataset = ds.dataset(str(args.trajectory_parquet), format="parquet")

    rows_by_day = {
        train_date: dedupe_rows(
            load_day_rows(dataset, train_date, minimum_altitude, maximum_altitude)
        ),
    }
    references = list(jsonl_rows(args.references))
    audit, linked = build_reference_audit(references, rows_by_day, config)
    assignments, crossing, boundary = assign_splits(linked, config)
    positives, duplicate_positive_count = positive_samples(
        linked, assignments, train_date
    )
    # HOLD means that no controller operation is observed around the state,
    # not merely that no altitude/speed operation is observed.  Use every
    # linked historical instruction on the development date as a guard,
    # including heading and other families that this first policy does not
    # predict.
    all_reference_epochs: dict[str, list[float]] = defaultdict(list)
    for reference in references:
        if reference_date(reference) != train_date:
            continue
        trajectory_id = str(
            (reference.get("target_state") or {}).get("trajectory_id") or ""
        )
        event_epoch = reference.get("event_time_epoch")
        if trajectory_id and event_epoch is not None:
            all_reference_epochs[trajectory_id].append(float(event_epoch))
    holds = hold_samples(
        positives,
        rows_by_day[train_date],
        assignments,
        boundary,
        all_reference_epochs,
        config,
    )
    state_samples = positives + holds
    continuous_rows, continuous_summary = continuous_calibration_candidate_rows(
        positives,
        rows_by_day[train_date],
        assignments,
        boundary,
        all_reference_epochs,
        config,
    )

    candidate_rows: list[dict[str, Any]] = []
    for state in state_samples:
        expanded = expand_candidate_rows(
            state, config, true_candidate_id=state["true_candidate_id"]
        )
        if sum(int(row["is_selected"]) for row in expanded) != 1:
            raise RuntimeError(f"state does not have exactly one selected candidate: {state['state_id']}")
        state_weight = (
            float(config["model"]["hold_state_weight"])
            if state["true_candidate_id"] == "HOLD"
            else 1.0
        )
        negative_count = max(1, len(expanded) - 1)
        for row in expanded:
            row["state_weight"] = state_weight
            row["training_weight"] = (
                state_weight * 0.5
                if row["is_selected"]
                else state_weight * 0.5 / negative_count
            )
            candidate_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "state_samples.parquet"
    candidate_path = args.output_dir / "candidate_rows.parquet"
    audit_path = args.output_dir / "reference_candidate_oracle.jsonl"
    continuous_path = args.output_dir / "continuous_calibration_candidate_rows.parquet"
    state_frame = pd.DataFrame(state_samples)
    candidate_frame = pd.DataFrame(candidate_rows)
    state_frame.to_parquet(state_path, index=False)
    candidate_frame.to_parquet(candidate_path, index=False)
    pd.DataFrame(continuous_rows).to_parquet(continuous_path, index=False)
    write_jsonl(audit_path, audit)

    audit_frame = pd.DataFrame(audit)
    strict_audit = audit_frame[audit_frame["true_candidate_id"].notna()]
    linked_audit = strict_audit[strict_audit["pre_state_linked"]]
    manifest = {
        "dataset_version": VERSION,
        "config_version": config["config_version"],
        "input": {
            "trajectory_parquet": str(args.trajectory_parquet.resolve()),
            "trajectory_parquet_sha256": sha256_file(args.trajectory_parquet),
            "references": str(args.references.resolve()),
            "references_sha256": sha256_file(args.references),
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
        },
        "scope": {
            "train_local_date": train_date,
            "test_local_date_reserved_for_post_freeze_evaluation": config["data"][
                "test_local_date"
            ],
            "oracle_dates": [train_date],
            "positive_state_offset_sec": config["data"]["positive_state_offset_sec"],
            "maximum_pre_state_age_sec": config["data"]["maximum_pre_state_age_sec"],
            "altitude_band_m": [minimum_altitude, maximum_altitude],
            "inside_sector_only": True,
            "split_boundary_epoch": boundary,
            "split_boundary_local": dt.datetime.fromtimestamp(
                boundary, tz=LOCAL_TZ
            ).isoformat(),
            "trajectory_grouped": True,
        },
        "oracle": {
            "reference_events_considered": int(len(audit_frame)),
            "references_with_numeric_target": int(len(strict_audit)),
            "pre_state_linked": int(strict_audit["pre_state_linked"].sum()),
            "candidate_available": int(strict_audit["candidate_available"].sum()),
            "coverage_of_all_considered": round(
                float(strict_audit["candidate_available"].mean()), 6
            ),
            "coverage_given_pre_state_linked": round(
                float(linked_audit["candidate_available"].mean()), 6
            )
            if len(linked_audit)
            else None,
            "exclusion_reasons": {
                str(key): int(value)
                for key, value in Counter(
                    value or "included" for value in audit_frame["exclusion_reason"]
                ).items()
            },
        },
        "training_data": {
            "trajectory_crossing_time_boundary_excluded": len(crossing),
            "same_bucket_later_positive_excluded": duplicate_positive_count,
            "state_samples": len(state_frame),
            "positive_states": int(
                (state_frame["true_candidate_id"] != "HOLD").sum()
            ),
            "weak_hold_states": int(
                (state_frame["true_candidate_id"] == "HOLD").sum()
            ),
            "candidate_rows": len(candidate_frame),
            "states_by_split": {
                str(key): int(value)
                for key, value in Counter(state_frame["split"]).items()
            },
            "positives_by_split": {
                split: int(
                    (
                        (state_frame["split"] == split)
                        & (state_frame["true_candidate_id"] != "HOLD")
                    ).sum()
                )
                for split in sorted(set(state_frame["split"]))
            },
            "candidate_targets": {
                str(key): int(value)
                for key, value in Counter(
                    state_frame.loc[
                        state_frame["true_candidate_id"] != "HOLD",
                        "true_candidate_id",
                    ]
                ).items()
            },
            "continuous_calibration": continuous_summary,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        "warnings": [
            "Reference actions are rule-extracted candidates, not human-reviewed gold labels.",
            "HOLD states are weak negatives inferred from absence of an extracted instruction and a 120-second guard.",
            "Cleared flight level is intentionally excluded from features because it may encode a recently issued clearance.",
            "Candidate coverage oracle is development-date only; test labels are not loaded.",
        ],
        "outputs": {},
    }
    for path in [state_path, candidate_path, continuous_path, audit_path]:
        manifest["outputs"][path.name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
