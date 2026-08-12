"""Inspect the SEU replay SQLite and teacher Parquet without mutating sources."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--parquet", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    connection = sqlite3.connect(f"file:{args.sqlite.as_posix()}?mode=ro", uri=True)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        result: dict[str, object] = {"sqlite": str(args.sqlite), "tables": {}}
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            columns = connection.execute(f"PRAGMA table_info({quoted})").fetchall()
            names = {row[1] for row in columns}
            summary: dict[str, object] = {
                "row_count": connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
            }
            if "receive_time_ms" in names:
                minimum, maximum = connection.execute(
                    f"SELECT MIN(receive_time_ms), MAX(receive_time_ms) FROM {quoted}"
                ).fetchone()
                summary["receive_time_ms_min"] = minimum
                summary["receive_time_ms_max"] = maximum
            if "partition_dt" in names:
                summary["partition_counts"] = connection.execute(
                    f"SELECT partition_dt, COUNT(*) FROM {quoted} GROUP BY partition_dt ORDER BY partition_dt"
                ).fetchall()
            if "source_file" in names:
                summary["source_file_count"] = connection.execute(
                    f"SELECT COUNT(DISTINCT source_file) FROM {quoted}"
                ).fetchone()[0]
            result["tables"][table] = {
                "columns": [
                    {
                        "position": row[0],
                        "name": row[1],
                        "type": row[2],
                        "not_null": bool(row[3]),
                        "primary_key": bool(row[5]),
                    }
                    for row in columns
                ],
                "sample": connection.execute(f"SELECT * FROM {quoted} LIMIT 1").fetchone(),
                "summary": summary,
            }
    finally:
        connection.close()

    if args.parquet:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(args.parquet)
        result["parquet"] = {
            "path": str(args.parquet),
            "rows": parquet.metadata.num_rows,
            "row_groups": parquet.metadata.num_row_groups,
            "schema": str(parquet.schema_arrow),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
