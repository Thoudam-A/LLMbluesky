#!/usr/bin/env python3
"""Convert decision-system command logs into the imitation scorer JSONL contract.

Accepted input rows may contain one ``command`` string or a ``commands`` /
``issued_commands`` list.  Timestamps may be absolute UTC, epoch seconds, or
simulation seconds combined with ``--replay-start-utc``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Iterable

from seu_imitation_common import UTC, iso_to_epoch, jsonl_rows, normalize_callsign, write_jsonl


CONVERTER_VERSION = "decision_log_to_imitation_v0.1"
ALT_RE = re.compile(
    r"^\s*ALT\s+(?P<callsign>[A-Z0-9]{2,8})\s*,\s*FL(?P<fl>\d{1,3})(?:\s*,.*)?$",
    re.IGNORECASE,
)
ALT_M_RE = re.compile(
    r"^\s*ALT_M\s+(?P<callsign>[A-Z0-9]{2,8})\s*,\s*(?P<altitude_m>\d{3,5})(?:\s*,\s*(?P<direction>CLIMB|DESCEND))?$",
    re.IGNORECASE,
)
SPD_RE = re.compile(
    r"^\s*SPD\s+(?P<callsign>[A-Z0-9]{2,8})\s*,\s*(?P<speed>\d{2,3})(?:\s*,.*)?$",
    re.IGNORECASE,
)
HDG_RE = re.compile(
    r"^\s*(?:HDG|HDGTRK)\s+(?P<callsign>[A-Z0-9]{2,8})\s*,\s*(?P<heading>\d{1,3})(?:\s*,.*)?$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--replay-start-utc",
        help="Required when rows provide only simt/sim_time_sec; ISO-8601 UTC.",
    )
    return parser.parse_args()


def row_epoch(row: dict[str, Any], replay_start_epoch: float | None) -> float:
    for key in ("event_time_epoch", "time_bucket_epoch", "timestamp_epoch"):
        if row.get(key) is not None:
            return float(row[key])
    for key in ("event_time_utc", "timestamp_utc"):
        if row.get(key):
            return iso_to_epoch(row[key])
    for key in ("simt", "sim_time_sec"):
        if row.get(key) is not None:
            if replay_start_epoch is None:
                raise ValueError(
                    f"row contains {key} but --replay-start-utc was not supplied"
                )
            return replay_start_epoch + float(row[key])
    raise ValueError("row has no absolute timestamp or simulation time")


def commands_from_row(row: dict[str, Any]) -> list[str]:
    if isinstance(row.get("command"), str):
        return [row["command"]]
    for key in ("commands", "issued_commands"):
        value = row.get(key)
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
    solver = row.get("decision")
    if isinstance(solver, dict) and isinstance(solver.get("commands"), list):
        return [str(item) for item in solver["commands"] if str(item).strip()]
    return []


def parse_command(command: str) -> dict[str, Any] | None:
    match = ALT_M_RE.match(command)
    if match:
        altitude_m = int(match.group("altitude_m"))
        direction = str(match.group("direction") or "").lower()
        return {
            "callsign": normalize_callsign(match.group("callsign")),
            "intent_type": "altitude_change",
            "intent_family": "altitude",
            "action": direction or "set_altitude",
            "target_value": altitude_m,
            "unit": "m",
            "raw_target_value": altitude_m,
            "raw_unit": "m",
        }
    match = ALT_RE.match(command)
    if match:
        callsign = normalize_callsign(match.group("callsign"))
        flight_level = int(match.group("fl"))
        return {
            "callsign": callsign,
            "intent_type": "altitude_change",
            "intent_family": "altitude",
            "action": "set_flight_level",
            # Historical Shanghai instructions use metres.  Preserve the raw
            # flight level and provide a physically equivalent metre target.
            "target_value": round(flight_level * 100.0 * 0.3048, 1),
            "unit": "m",
            "raw_target_value": flight_level,
            "raw_unit": "FL",
        }
    match = SPD_RE.match(command)
    if match:
        return {
            "callsign": normalize_callsign(match.group("callsign")),
            "intent_type": "speed_adjust",
            "intent_family": "speed",
            "action": "set_speed",
            "target_value": int(match.group("speed")),
            "unit": "kt",
        }
    match = HDG_RE.match(command)
    if match:
        return {
            "callsign": normalize_callsign(match.group("callsign")),
            "intent_type": "heading_change",
            "intent_family": "heading",
            "action": "fly_heading",
            "target_value": int(match.group("heading")) % 360,
            "unit": "degree",
        }
    return None


def convert_rows(
    rows: Iterable[dict[str, Any]],
    replay_start_epoch: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outputs: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        commands = commands_from_row(row)
        if not commands:
            continue
        try:
            epoch = row_epoch(row, replay_start_epoch)
        except ValueError as exc:
            rejected.extend(
                {
                    "source_row_index": row_index,
                    "command": command,
                    "reason": str(exc),
                }
                for command in commands
            )
            continue
        timestamp = dt.datetime.fromtimestamp(epoch, tz=UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        for command_index, command in enumerate(commands):
            parsed = parse_command(command)
            if parsed is None or parsed.get("callsign") is None:
                rejected.append(
                    {
                        "source_row_index": row_index,
                        "command": command,
                        "reason": "unsupported_command_schema",
                    }
                )
                continue
            outputs.append(
                {
                    "event_id": f"system-{row_index:06d}-{command_index}",
                    "event_time_utc": timestamp,
                    "event_time_epoch": epoch,
                    "source_row_index": row_index,
                    "source_command": command,
                    "converter_version": CONVERTER_VERSION,
                    **parsed,
                }
            )
    return outputs, rejected


def main() -> int:
    args = parse_args()
    replay_start_epoch = iso_to_epoch(args.replay_start_utc) if args.replay_start_utc else None
    outputs, rejected = convert_rows(jsonl_rows(args.input), replay_start_epoch)
    write_jsonl(args.output, outputs)
    rejected_path = args.output.with_suffix(args.output.suffix + ".rejected.jsonl")
    write_jsonl(rejected_path, rejected)
    summary = {
        "converter_version": CONVERTER_VERSION,
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "converted_command_count": len(outputs),
        "rejected_command_count": len(rejected),
        "rejected_output": str(rejected_path.resolve()),
        "replay_start_utc": args.replay_start_utc,
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
