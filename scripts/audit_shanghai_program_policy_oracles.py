#!/usr/bin/env python3
"""Audit historical-action candidate coverage and finite-horizon verifier pass rate."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.dataset as ds

from seu_imitation_common import jsonl_rows, write_jsonl
from shanghai_program_policy import candidate_safety


VERSION = "shanghai_program_policy_oracle_audit_v1.0"
STATE_COLUMNS = [
    "trajectory_id",
    "callsign",
    "time_bucket_epoch",
    "event_time_epoch",
    "inside_sector",
    "latitude",
    "longitude",
    "altitude_m",
    "ground_speed_mps",
    "track_heading_deg",
    "vertical_rate_mps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--trajectory-parquet", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_from_id(value: str, current_altitude_m: float) -> dict[str, Any]:
    if value.startswith("ALT_"):
        target = float(value.split("_", 1)[1])
        delta = target - current_altitude_m
        direction = "maintain" if abs(delta) < 150 else ("climb" if delta > 0 else "descend")
        return {
            "candidate_id": value,
            "candidate_kind": "altitude",
            "candidate_target": target,
            "candidate_direction": direction,
        }
    if value.startswith("SPD_"):
        return {
            "candidate_id": value,
            "candidate_kind": "speed",
            "candidate_target": float(value.split("_", 1)[1]),
            "candidate_direction": "set_speed",
        }
    raise ValueError(f"unsupported candidate id: {value}")


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    oracle = [
        row
        for row in jsonl_rows(args.oracle)
        if row.get("candidate_available") and row.get("true_candidate_id")
    ]
    buckets = sorted({int(row["pre_state_bucket_epoch"]) for row in oracle})
    dataset = ds.dataset(str(args.trajectory_parquet), format="parquet")
    table = dataset.scanner(
        columns=STATE_COLUMNS,
        filter=ds.field("time_bucket_epoch").isin(buckets),
        batch_size=65536,
    ).to_table()
    grouped: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    target_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in table.to_pylist():
        bucket = int(row["time_bucket_epoch"])
        grouped[bucket].append(row)
        key = (str(row["trajectory_id"]), bucket)
        prior = target_rows.get(key)
        if prior is None or float(row.get("event_time_epoch") or 0.0) > float(
            prior.get("event_time_epoch") or 0.0
        ):
            target_rows[key] = row

    details: list[dict[str, Any]] = []
    horizons = [300, 480, 600]
    for item in oracle:
        key = (str(item["trajectory_id"]), int(item["pre_state_bucket_epoch"]))
        target = target_rows.get(key)
        detail = {
            "reference_event_id": item["reference_event_id"],
            "local_date": item["local_date"],
            "trajectory_id": item["trajectory_id"],
            "callsign": item["callsign"],
            "intent_family": item["intent_family"],
            "true_candidate_id": item["true_candidate_id"],
            "pre_state_bucket_epoch": item["pre_state_bucket_epoch"],
            "target_state_found": target is not None,
        }
        if target is None:
            details.append(detail)
            continue
        candidate = candidate_from_id(
            str(item["true_candidate_id"]), float(target["altitude_m"])
        )
        for horizon in horizons:
            safety_config = {**config["safety"], "horizon_sec": horizon}
            result = candidate_safety(
                target,
                candidate,
                grouped[int(item["pre_state_bucket_epoch"])],
                safety_config,
            )
            detail[f"safe_{horizon}_sec"] = bool(result["safe"])
            detail[f"unsafe_pair_count_{horizon}_sec"] = len(
                result.get("unsafe_pairs", [])
            )
        details.append(detail)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, details)
    summary: dict[str, Any] = {
        "audit_version": VERSION,
        "config_version": config["config_version"],
        "historical_candidate_events": len(oracle),
        "target_state_found": sum(bool(row["target_state_found"]) for row in details),
        "horizons": {},
        "by_family": {},
        "artifacts": {
            "oracle": str(args.oracle.resolve()),
            "oracle_sha256": sha256_file(args.oracle),
            "trajectory_parquet": str(args.trajectory_parquet.resolve()),
            "trajectory_parquet_sha256": sha256_file(args.trajectory_parquet),
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
            "details": str(args.output.resolve()),
            "details_sha256": sha256_file(args.output),
        },
        "claim_boundary": config["safety"]["claim_boundary"],
        "warning": (
            "A rejection does not prove the historical command was unsafe; it may reflect "
            "frozen traffic, constant-track extrapolation, missing intent, or incomplete state."
        ),
    }
    found = [row for row in details if row["target_state_found"]]
    for horizon in horizons:
        key = f"safe_{horizon}_sec"
        passed = sum(bool(row.get(key)) for row in found)
        summary["horizons"][str(horizon)] = {
            "passed": passed,
            "evaluated": len(found),
            "pass_rate": round(passed / len(found), 6) if found else None,
        }
    for family in ["altitude", "speed"]:
        family_rows = [row for row in found if row["intent_family"] == family]
        summary["by_family"][family] = {}
        for horizon in horizons:
            key = f"safe_{horizon}_sec"
            passed = sum(bool(row.get(key)) for row in family_rows)
            summary["by_family"][family][str(horizon)] = {
                "passed": passed,
                "evaluated": len(family_rows),
                "pass_rate": (
                    round(passed / len(family_rows), 6) if family_rows else None
                ),
            }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
