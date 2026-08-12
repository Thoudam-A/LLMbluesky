#!/usr/bin/env python3
"""Build an auditable SEU historical-instruction/state association dataset.

The script does not claim human ground truth.  It produces conservative
controller-command candidates, associates them to the nearest trajectory state,
and writes same-time sector snapshots for shadow-replay evaluation.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

from build_historical_habit_events import RULE_VERSION, parse_rule_intents
from seu_imitation_common import (
    MAIN_INTENT_TYPES,
    classify_speaker_role,
    extract_seu_callsign,
    family_for_intent,
    parse_vad_segment_time,
    write_jsonl,
)


PIPELINE_VERSION = "seu_imitation_link_v0.1"
STATE_COLUMNS = [
    "trajectory_id",
    "sector_code",
    "callsign",
    "target_address",
    "sac",
    "sic",
    "track_number",
    "event_time_utc",
    "event_time_epoch",
    "time_bucket_epoch",
    "time_from_track_start_sec",
    "time_from_sector_entry_sec",
    "sector_entry_utc",
    "sector_exit_utc",
    "inside_sector",
    "sector_hit",
    "longitude",
    "latitude",
    "altitude_m",
    "geometric_altitude_m",
    "barometric_altitude_fl",
    "measured_flight_level",
    "vx_mps",
    "vy_mps",
    "ground_speed_mps",
    "track_heading_deg",
    "vertical_rate_mps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vad-json", type=Path, required=True)
    parser.add_argument("--trajectory-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-state-delta-sec", type=float, default=8.0)
    parser.add_argument(
        "--include-uncertain-role",
        action="store_true",
        help="Include uncertain speaker-role events in the primary reference set.",
    )
    parser.add_argument(
        "--skip-snapshots",
        action="store_true",
        help="Skip same-time all-aircraft snapshots (useful for a quick smoke run).",
    )
    return parser.parse_args()


def require_pyarrow() -> tuple[Any, Any]:
    if importlib.util.find_spec("pyarrow") is None:
        raise SystemExit(
            "pyarrow>=20 is required. Install the dependencies from "
            "scripts/requirements-shanghai-data-pipeline.txt."
        )
    import pyarrow as pa
    import pyarrow.dataset as ds

    major = int(pa.__version__.split(".", 1)[0])
    if major < 20:
        raise SystemExit(
            f"pyarrow {pa.__version__} is installed, but this parquet failed under pyarrow 19 "
            "in the current environment. Use an environment with pyarrow>=20."
        )
    return pa, ds


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def source_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path}: expected a top-level list of objects")
    return value


def canonical_intents(text: str, callsign_info: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    intents, metadata = parse_rule_intents(text)
    # “马上高度两拐” means “current altitude 2700”, not “climb to 2700”.
    # The generic rule sees the substring “上高度”; suppress this documented
    # Shanghai-ASR false positive before constructing a reference event.
    if "马上高度" in text:
        intents = [
            intent
            for intent in intents
            if not (
                intent.get("action") == "climb"
                and str(intent.get("raw_span") or "").startswith("上高度")
            )
        ]
    if not callsign_info:
        return intents, metadata
    for intent in intents:
        intent["callsign"] = callsign_info["callsign"]
        intent["callsign_raw"] = callsign_info["raw"]
        intent["callsign_rule_id"] = callsign_info["rule"]
    metadata["callsign"] = callsign_info["callsign"]
    metadata["callsign_raw"] = callsign_info["raw"]
    metadata["callsign_rule_id"] = callsign_info["rule"]
    return intents, metadata


def build_instruction_events(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], collections.Counter[str]]:
    events: list[dict[str, Any]] = []
    counters: collections.Counter[str] = collections.Counter()
    for row_index, row in enumerate(rows):
        counters["segments_total"] += 1
        text = str(row.get("transcript") or "").strip()
        if not text:
            counters["segments_empty"] += 1
            continue
        try:
            time_fields = parse_vad_segment_time(row.get("path"), row.get("utt"))
        except ValueError:
            counters["segments_bad_time"] += 1
            continue
        callsign = extract_seu_callsign(text)
        if callsign:
            counters["segments_with_callsign"] += 1
        intents, parse_metadata = canonical_intents(text, callsign)
        role = classify_speaker_role(text, callsign)
        if intents:
            counters["segments_with_any_intent"] += 1

        base = {
            "source_row_index": row_index,
            "utt": row.get("utt"),
            "audio_path": row.get("path"),
            "transcript": text,
            "source_class": row.get("class"),
            **time_fields,
            **role,
            "callsign": callsign["callsign"] if callsign else None,
            "callsign_raw": callsign["raw"] if callsign else None,
            "callsign_rule_id": callsign["rule"] if callsign else None,
            "callsign_candidate_count": callsign["candidate_count"] if callsign else 0,
            "parse_review_required": parse_metadata.get("review_required"),
            "parse_review_reasons": parse_metadata.get("review_reasons", []),
            "parse_rule_confidence": parse_metadata.get("rule_confidence"),
            "rule_version": RULE_VERSION,
            "pipeline_version": PIPELINE_VERSION,
        }
        if not intents:
            events.append(
                {
                    **base,
                    "event_id": f"seg-{row_index:06d}-none",
                    "intent_index": None,
                    "intent_type": None,
                    "intent_family": None,
                    "action": None,
                    "target_value": None,
                    "unit": None,
                    "rule_eligible_main": False,
                    "validation_flags": [],
                }
            )
            continue

        for intent_index, intent in enumerate(intents):
            family = family_for_intent(intent.get("intent_type"))
            eligible = (
                callsign is not None
                and intent.get("intent_type") in MAIN_INTENT_TYPES
                and not intent.get("validation_flags")
            )
            counters["intent_events_total"] += 1
            if eligible:
                counters["intent_events_rule_eligible"] += 1
            events.append(
                {
                    **base,
                    "event_id": f"seg-{row_index:06d}-intent-{intent_index}",
                    "intent_index": intent_index,
                    "intent_type": intent.get("intent_type"),
                    "intent_family": family,
                    "action": intent.get("action"),
                    "target_value": intent.get("target_value"),
                    "unit": intent.get("unit"),
                    "constraint_operator": intent.get("constraint_operator"),
                    "target_semantics": intent.get("target_semantics"),
                    "raw_span": intent.get("raw_span"),
                    "rule_id": intent.get("rule_id"),
                    "unit_inferred": intent.get("unit_inferred", False),
                    "rule_eligible_main": eligible,
                    "validation_flags": intent.get("validation_flags", []),
                }
            )
    return events, counters


def scan_relevant_states(dataset: Any, callsigns: set[str]) -> dict[str, list[dict[str, Any]]]:
    by_callsign: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    if not callsigns:
        return by_callsign
    import pyarrow.dataset as ds

    scanner = dataset.scanner(
        columns=STATE_COLUMNS,
        filter=ds.field("callsign").isin(sorted(callsigns)),
        batch_size=65536,
    )
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            callsign = str(row.get("callsign") or "").upper()
            if callsign:
                by_callsign[callsign].append(row)
    for states in by_callsign.values():
        states.sort(key=lambda row: float(row["event_time_epoch"]))
    return by_callsign


def associate_events(
    events: list[dict[str, Any]],
    by_callsign: dict[str, list[dict[str, Any]]],
    max_delta_sec: float,
) -> tuple[list[dict[str, Any]], collections.Counter[str], list[float]]:
    counters: collections.Counter[str] = collections.Counter()
    deltas: list[float] = []
    times_by_callsign = {
        callsign: [float(row["event_time_epoch"]) for row in states]
        for callsign, states in by_callsign.items()
    }
    linked: list[dict[str, Any]] = []
    for event in events:
        if not event.get("rule_eligible_main"):
            continue
        counters["eligible_events"] += 1
        callsign = str(event["callsign"])
        states = by_callsign.get(callsign, [])
        times = times_by_callsign.get(callsign, [])
        if not states:
            counters["no_callsign_trajectory"] += 1
            linked.append({**event, "state_link_status": "no_callsign_trajectory"})
            continue
        target = float(event["event_time_epoch"])
        insertion = bisect.bisect_left(times, target)
        candidates = [idx for idx in (insertion - 1, insertion) if 0 <= idx < len(times)]
        nearest = min(candidates, key=lambda idx: abs(times[idx] - target))
        state = states[nearest]
        delta = abs(float(state["event_time_epoch"]) - target)
        if delta > max_delta_sec:
            counters["nearest_state_outside_window"] += 1
            linked.append(
                {
                    **event,
                    "state_link_status": "nearest_state_outside_window",
                    "nearest_state_delta_sec": round(delta, 6),
                }
            )
            continue
        counters["state_linked"] += 1
        deltas.append(delta)
        snapshot_id = f"bucket-{int(state['time_bucket_epoch'])}"
        linked.append(
            {
                **event,
                "state_link_status": "linked",
                "state_time_delta_sec": round(delta, 6),
                "state_snapshot_id": snapshot_id,
                "state_time_bucket_epoch": int(state["time_bucket_epoch"]),
                "target_state": state,
            }
        )
    return linked, counters, deltas


def dedupe_primary_events(linked: list[dict[str, Any]], include_uncertain: bool) -> tuple[list[dict[str, Any]], int]:
    role_allowed = {"controller_candidate"}
    if include_uncertain:
        role_allowed.add("uncertain")
    candidates = [
        row
        for row in linked
        if row.get("state_link_status") == "linked" and row.get("speaker_role_rule") in role_allowed
    ]
    candidates.sort(
        key=lambda row: (
            str(row.get("callsign")),
            float(row.get("event_time_epoch")),
            str(row.get("intent_family")),
        )
    )
    kept: list[dict[str, Any]] = []
    duplicate_count = 0
    for row in candidates:
        key = (
            row.get("callsign"),
            row.get("intent_family"),
            row.get("action"),
            row.get("target_value"),
            row.get("unit"),
        )
        duplicate = None
        for prior in reversed(kept[-12:]):
            prior_key = (
                prior.get("callsign"),
                prior.get("intent_family"),
                prior.get("action"),
                prior.get("target_value"),
                prior.get("unit"),
            )
            if prior_key != key:
                continue
            if abs(float(row["event_time_epoch"]) - float(prior["event_time_epoch"])) <= 12.0:
                duplicate = prior
            break
        if duplicate is not None:
            duplicate_count += 1
            continue
        kept.append(row)

    kept.sort(key=lambda row: (float(row["event_time_epoch"]), str(row["callsign"])))
    for index, row in enumerate(kept):
        row["reference_event_id"] = f"ref-{index:06d}"
    return kept, duplicate_count


def snapshot_rows(dataset: Any, buckets: set[int]) -> Iterable[dict[str, Any]]:
    if not buckets:
        return
    import pyarrow.dataset as ds

    scanner = dataset.scanner(
        columns=STATE_COLUMNS,
        filter=ds.field("time_bucket_epoch").isin(sorted(buckets)),
        batch_size=65536,
    )
    grouped: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            bucket = int(row["time_bucket_epoch"])
            if bucket in buckets:
                grouped[bucket].append(row)
    for bucket in sorted(buckets):
        aircraft = sorted(
            grouped.get(bucket, []),
            key=lambda row: (str(row.get("callsign") or ""), str(row.get("trajectory_id") or "")),
        )
        yield {
            "snapshot_id": f"bucket-{bucket}",
            "time_bucket_epoch": bucket,
            "sector_code": "ZSSSAP01",
            "state_scope": "sector_context_window",
            "aircraft_count": len(aircraft),
            "inside_sector_aircraft_count": sum(bool(row.get("inside_sector")) for row in aircraft),
            "aircraft": aircraft,
        }


def sequence_rows(primary: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in primary:
        local_date = str(row["event_time_local"])[:10]
        grouped[(local_date, str(row["callsign"]))].append(row)
    for (local_date, callsign), events in sorted(grouped.items()):
        events.sort(key=lambda row: float(row["event_time_epoch"]))
        yield {
            "sequence_id": f"{local_date}-{callsign}",
            "local_date": local_date,
            "callsign": callsign,
            "event_count": len(events),
            "events": [
                {
                    key: event.get(key)
                    for key in (
                        "reference_event_id",
                        "event_time_utc",
                        "event_time_epoch",
                        "state_snapshot_id",
                        "intent_type",
                        "intent_family",
                        "action",
                        "target_value",
                        "unit",
                        "transcript",
                    )
                }
                for event in events
            ],
        }


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p90": None, "max": None}
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))
    return {
        "median": round(statistics.median(ordered), 6),
        "p90": round(ordered[p90_index], 6),
        "max": round(max(ordered), 6),
    }


def main() -> int:
    args = parse_args()
    _, ds = require_pyarrow()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vad_rows = source_rows(args.vad_json)
    events, parse_counts = build_instruction_events(vad_rows)
    write_jsonl(args.output_dir / "instruction_events.jsonl", events)

    relevant_callsigns = {
        str(row["callsign"]) for row in events if row.get("rule_eligible_main") and row.get("callsign")
    }
    dataset = ds.dataset(str(args.trajectory_parquet), format="parquet")
    by_callsign = scan_relevant_states(dataset, relevant_callsigns)
    linked, link_counts, deltas = associate_events(events, by_callsign, args.max_state_delta_sec)
    write_jsonl(args.output_dir / "linked_main_events_all_roles.jsonl", linked)

    primary, duplicate_count = dedupe_primary_events(linked, args.include_uncertain_role)
    write_jsonl(args.output_dir / "reference_events.jsonl", primary)
    write_jsonl(args.output_dir / "reference_sequences.jsonl", sequence_rows(primary))
    split_counts: dict[str, int] = {}
    for local_date in sorted({str(row["event_time_local"])[:10] for row in primary}):
        split_rows = [row for row in primary if str(row["event_time_local"]).startswith(local_date)]
        split_counts[local_date] = write_jsonl(
            args.output_dir / f"reference_events_{local_date}.jsonl",
            split_rows,
        )

    snapshot_count = 0
    if not args.skip_snapshots:
        buckets = {int(row["state_time_bucket_epoch"]) for row in primary}
        snapshot_count = write_jsonl(
            args.output_dir / "state_snapshots.jsonl",
            snapshot_rows(dataset, buckets),
        )

    by_family = collections.Counter(str(row["intent_family"]) for row in primary)
    by_role = collections.Counter(str(row["speaker_role_rule"]) for row in linked if row.get("state_link_status") == "linked")
    eligible = int(link_counts["eligible_events"])
    linked_count = int(link_counts["state_linked"])
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "rule_version": RULE_VERSION,
        "definition": {
            "reference_event": (
                "rule-valid main maneuver, standardized ICAO callsign, probable controller role, "
                "and nearest same-callsign trajectory point within the configured window"
            ),
            "time_alignment": "VAD filename is Asia/Shanghai local time; parquet is UTC; association uses epoch seconds",
            "primary_role_policy": (
                "controller_candidate plus uncertain"
                if args.include_uncertain_role
                else "controller_candidate only"
            ),
            "human_ground_truth": False,
        },
        "inputs": {
            "vad_json": str(args.vad_json.resolve()),
            "trajectory_parquet": str(args.trajectory_parquet.resolve()),
        },
        "parameters": {
            "max_state_delta_sec": args.max_state_delta_sec,
            "include_uncertain_role": args.include_uncertain_role,
            "snapshot_scope": "all trajectories in the parquet sector-context window",
        },
        "counts": {
            **dict(parse_counts),
            **dict(link_counts),
            "reference_events_after_role_and_dedup": len(primary),
            "near_duplicate_events_removed": duplicate_count,
            "unique_reference_callsigns": len({row["callsign"] for row in primary}),
            "unique_reference_snapshots": len({row["state_snapshot_id"] for row in primary}),
            "snapshot_rows_written": snapshot_count,
            "reference_events_by_local_date": split_counts,
        },
        "rates": {
            "state_association_rate_over_rule_eligible": (
                round(linked_count / eligible, 6) if eligible else None
            ),
            "primary_reference_yield_over_vad_segments": (
                round(len(primary) / len(vad_rows), 6) if vad_rows else None
            ),
        },
        "state_time_delta_sec": quantiles(deltas),
        "reference_by_family": dict(sorted(by_family.items())),
        "linked_by_role_rule": dict(sorted(by_role.items())),
        "warnings": [
            "speaker_role_rule is heuristic and is not equivalent to controller-verified labeling",
            "association coverage is not controller-imitation accuracy",
            "state snapshots are event-context audit artifacts, not a model invocation schedule",
            "continuous shadow replay must invoke the system on every independent 4-second tick",
            "the initial rule parser covers the Chinese-language subset; English radiotelephony remains out of scope",
        ],
    }
    (args.output_dir / "association_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    output_files = [
        path
        for path in (
            args.output_dir / "instruction_events.jsonl",
            args.output_dir / "linked_main_events_all_roles.jsonl",
            args.output_dir / "reference_events.jsonl",
            args.output_dir / "reference_sequences.jsonl",
            args.output_dir / "state_snapshots.jsonl",
            args.output_dir / "association_summary.json",
            *(args.output_dir / f"reference_events_{local_date}.jsonl" for local_date in split_counts),
        )
        if path.exists()
    ]
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "source_sha256": {
            str(args.vad_json.resolve()): sha256_file(args.vad_json),
            str(args.trajectory_parquet.resolve()): sha256_file(args.trajectory_parquet),
        },
        "output_sha256": {
            path.name: sha256_file(path)
            for path in output_files
        },
        "command_note": (
            "Regenerate with build_seu_imitation_dataset.py using the parameters in association_summary.json"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
