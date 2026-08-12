#!/usr/bin/env python3
"""Overlay recovered intent predictions onto a complete prediction ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def keyed(rows: list[dict[str, Any]], source: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, 1):
        utterance_id = str(row.get("utterance_id") or "").strip()
        if not utterance_id:
            raise ValueError(f"{source}:{line_number}: utterance_id is required")
        if utterance_id in result:
            raise ValueError(f"{source}:{line_number}: duplicate utterance_id {utterance_id}")
        result[utterance_id] = row
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--recovered", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    base_rows = read_jsonl(args.base)
    recovered_rows = read_jsonl(args.recovered)
    base_by_id = keyed(base_rows, args.base)
    recovered_by_id = keyed(recovered_rows, args.recovered)
    unknown = sorted(set(recovered_by_id) - set(base_by_id))
    if unknown:
        raise ValueError(f"recovered rows are absent from base ledger: {unknown[:10]}")

    merged_rows = [
        recovered_by_id.get(str(row["utterance_id"]), row)
        for row in base_rows
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in merged_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "base_rows": len(base_rows),
        "recovered_rows": len(recovered_rows),
        "merged_rows": len(merged_rows),
        "replaced_utterance_ids": sorted(recovered_by_id),
        "inputs": {
            "base_sha256": sha256(args.base),
            "recovered_sha256": sha256(args.recovered),
        },
        "output_sha256": sha256(args.output),
    }
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
