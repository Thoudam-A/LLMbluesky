#!/usr/bin/env python3
"""Export continuous 4-second SEU traffic ticks for leakage-free shadow replay."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from seu_imitation_common import LOCAL_TZ, UTC


EXPORT_VERSION = "seu_shadow_replay_v0.1"
REPLAY_STATE_COLUMNS = [
    "trajectory_id",
    "sector_code",
    "callsign",
    "target_address",
    "event_time_utc",
    "event_time_epoch",
    "time_bucket_epoch",
    "inside_sector",
    "sector_hit",
    "longitude",
    "latitude",
    "altitude_m",
    "barometric_altitude_fl",
    "ground_speed_mps",
    "track_heading_deg",
    "vertical_rate_mps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-parquet", type=Path, required=True)
    parser.add_argument("--local-date", required=True, help="Evaluation date in Asia/Shanghai, YYYY-MM-DD")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-sec", type=int, default=4)
    parser.add_argument(
        "--inside-sector-only",
        action="store_true",
        help="Exclude the parquet's 15-minute pre/post-sector trajectory context.",
    )
    return parser.parse_args()


def require_pyarrow() -> Any:
    if importlib.util.find_spec("pyarrow") is None:
        raise SystemExit(
            "pyarrow>=20 is required; install the data-pipeline requirements."
        )
    import pyarrow as pa
    import pyarrow.dataset as ds

    if int(pa.__version__.split(".", 1)[0]) < 20:
        raise SystemExit(f"pyarrow {pa.__version__} is too old for this SEU parquet; use >=20")
    return ds


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.step_sec <= 0:
        raise SystemExit("--step-sec must be positive")
    local_day = dt.date.fromisoformat(args.local_date)
    start_local = dt.datetime.combine(local_day, dt.time.min, tzinfo=LOCAL_TZ)
    end_local = start_local + dt.timedelta(days=1)
    start_epoch = int(start_local.astimezone(UTC).timestamp())
    end_epoch = int(end_local.astimezone(UTC).timestamp())

    ds = require_pyarrow()
    dataset = ds.dataset(str(args.trajectory_parquet), format="parquet")
    filter_expr = (
        (ds.field("time_bucket_epoch") >= start_epoch)
        & (ds.field("time_bucket_epoch") < end_epoch)
    )
    if args.inside_sector_only:
        filter_expr = filter_expr & (ds.field("inside_sector") == True)  # noqa: E712

    grouped: dict[int, list[dict[str, Any]]] = {}
    scanner = dataset.scanner(columns=REPLAY_STATE_COLUMNS, filter=filter_expr, batch_size=65536)
    source_state_rows = 0
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            bucket = int(row["time_bucket_epoch"])
            grouped.setdefault(bucket, []).append(row)
            source_state_rows += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tick_count = 0
    nonempty_count = 0
    aircraft_count_sum = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for bucket in range(start_epoch, end_epoch, args.step_sec):
            aircraft = sorted(
                grouped.get(bucket, []),
                key=lambda row: (str(row.get("callsign") or ""), str(row.get("trajectory_id") or "")),
            )
            tick_count += 1
            if aircraft:
                nonempty_count += 1
            aircraft_count_sum += len(aircraft)
            timestamp = dt.datetime.fromtimestamp(bucket, tz=UTC).isoformat().replace("+00:00", "Z")
            row = {
                "replay_tick_id": f"{args.local_date}-tick-{tick_count - 1:05d}",
                "time_bucket_epoch": bucket,
                "event_time_utc": timestamp,
                "sector_code": "ZSSSAP01",
                "historical_instruction_visible_to_model": False,
                "state_scope": (
                    "inside_sector_only" if args.inside_sector_only else "sector_context_window"
                ),
                "aircraft_count": len(aircraft),
                "inside_sector_aircraft_count": sum(bool(x.get("inside_sector")) for x in aircraft),
                "aircraft": aircraft,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "export_version": EXPORT_VERSION,
        "local_date": args.local_date,
        "timezone": "Asia/Shanghai",
        "utc_interval": [
            start_local.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            end_local.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        ],
        "step_sec": args.step_sec,
        "tick_count": tick_count,
        "nonempty_tick_count": nonempty_count,
        "source_state_rows": source_state_rows,
        "mean_aircraft_per_tick": round(aircraft_count_sum / tick_count, 6),
        "state_scope": "inside_sector_only" if args.inside_sector_only else "sector_context_window",
        "leakage_control": (
            "No historical command is included. The decision system must run at every tick "
            "or use a trigger frozen independently of historical command times."
        ),
        "trajectory_parquet": str(args.trajectory_parquet.resolve()),
        "trajectory_parquet_sha256": sha256_file(args.trajectory_parquet),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
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
