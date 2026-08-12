#!/usr/bin/env python3
"""Audit source-field retention and CAT062 lookback-window sensitivity for v5.

The script is read-only for raw sources.  It enumerates every MH4029 field
actually present in the corpus and measures how many v5 target aircraft would
have a pre-voice CAT062 observation under each candidate lookback.
"""
from __future__ import annotations

import argparse
import base64
import collections
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / ".deps" / "pyarrow20").exists():
    sys.path.insert(0, str(ROOT / ".deps" / "pyarrow20"))

from build_shanghai_causal_alignment_v4 import parse_message
from build_shanghai_causal_alignment_v5 import merge_intervals, norm_hex
from cat062_enrichment_v5 import load_enriched_decoder


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--sqlite", type=Path, required=True)
    p.add_argument("--decoder", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--windows-sec", type=float, nargs="+", default=[8, 16, 30, 60])
    return p.parse_args()


def v5_index(path: Path) -> tuple[dict[str, tuple[float, str]], collections.Counter]:
    targets: dict[str, tuple[float, str]] = {}
    fields: collections.Counter = collections.Counter()
    for line in (path / "instruction_bundle_training_records_v5.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ref = (row.get("member_reference_event_ids") or [None])[0]
        meta = row.get("provenance", {}).get("causal_alignment_v5", {})
        address = norm_hex(meta.get("identity", {}).get("target_address"))
        if ref and address and meta.get("decision_cutoff_epoch") is not None:
            targets[ref] = (float(meta["decision_cutoff_epoch"]), address)
        for section, value in row.get("model_input", {}).items():
            if isinstance(value, dict):
                for name, item in value.items():
                    if item is not None:
                        fields[f"{section}.{name}"] += 1
    return targets, fields


def plan_inventory(con: sqlite3.Connection) -> tuple[int, collections.Counter, dict[str, str]]:
    counts: collections.Counter = collections.Counter()
    example: dict[str, str] = {}
    messages = 0
    for (outer_json,) in con.execute("SELECT outer_json FROM mh4029_raw"):
        try:
            fields, _, _ = parse_message(json.loads(outer_json).get("raw_value", ""))
        except Exception:
            continue
        messages += 1
        for key, value in fields.items():
            counts[key] += 1
            if key not in example and value:
                example[key] = str(value)[:120]
    return messages, counts, example


def cat_window_sensitivity(con: sqlite3.Connection, decoder: Any, targets: dict[str, tuple[float, str]], windows: list[float]) -> dict[str, Any]:
    largest = max(windows)
    by_address: dict[str, list[tuple[str, float]]] = collections.defaultdict(list)
    for ref, (cutoff, address) in targets.items():
        by_address[address].append((ref, cutoff))
    # Read each raw message at most once, under the largest candidate window.
    best: dict[str, float] = {}
    decoded_records = raw_rows = failures = 0
    for low, high in merge_intervals((cutoff for cutoff, _ in targets.values()), before=largest, after=0.0):
        for receive_ms, outer_json in con.execute("SELECT receive_time_ms, outer_json FROM cat062_raw WHERE receive_time_ms BETWEEN ? AND ? ORDER BY receive_time_ms", (low, high)):
            raw_rows += 1
            receive_epoch = float(receive_ms) / 1000.0
            try:
                raw_value = json.loads(outer_json).get("raw_value")
                payload = base64.b64decode("".join(str(raw_value).split()), validate=True)
                records = []
                for block in decoder.iter_cat062_blocks(payload):
                    if block and block[0] == 0x3E:
                        records.extend(decoder.parse_cat062_data_block(block))
            except Exception:
                failures += 1
                continue
            decoded_records += len(records)
            for record in records:
                for ref, cutoff in by_address.get(norm_hex(record.get("target_address")), []):
                    age = cutoff - receive_epoch
                    if 0 < age <= largest and (ref not in best or age < best[ref]):
                        best[ref] = age
    result = {"raw_rows_examined": raw_rows, "decoded_records": decoded_records, "parse_failures": failures, "matched_age_sec": best}
    for window in windows:
        ages = [value for value in best.values() if value <= window]
        result[str(int(window) if window.is_integer() else window)] = {
            "lookback_sec": window, "matched": len(ages), "coverage_pct": round(100 * len(ages) / len(targets), 3),
            "edge_80pct_to_100pct_gain": None,
            "age_p95_sec": None if not ages else sorted(ages)[int(0.95 * (len(ages)-1))],
            "age_max_sec": None if not ages else max(ages),
        }
    return result


def main() -> int:
    a = args()
    targets, v5_fields = v5_index(a.dataset_dir)
    con = sqlite3.connect(f"file:{a.sqlite}?mode=ro", uri=True)
    try:
        message_count, plan_counts, plan_example = plan_inventory(con)
        sensitivity = cat_window_sensitivity(con, load_enriched_decoder(a.decoder), targets, a.windows_sec)
    finally:
        con.close()
    output = {
        "v5_target_count": len(targets),
        "mh4029": {"messages_parsed": message_count, "field_count": len(plan_counts), "fields": {k: {"present_messages": v, "coverage_pct": round(100*v/message_count, 3), "example": plan_example.get(k)} for k, v in sorted(plan_counts.items())}},
        "v5_model_input_non_null": dict(sorted(v5_fields.items())),
        "cat062_window_sensitivity": sensitivity,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"v5_target_count": len(targets), "mh_fields": len(plan_counts), "window": {k: v for k, v in sensitivity.items() if k != "matched_age_sec"}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
