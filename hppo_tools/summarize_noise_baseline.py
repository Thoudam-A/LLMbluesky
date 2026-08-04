"""Summarize fixed-test-set evaluation results by source scenario template."""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate H-PPO noisy-test evaluation results by scenario.")
    parser.add_argument("--evaluation-root", type=Path, required=True)
    return parser.parse_args()


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0.0)
    except ValueError:
        return 0.0


def main() -> int:
    root = parse_args().evaluation_root.expanduser().resolve()
    stats_file = root / "validation_stats.csv"
    if not stats_file.is_file():
        raise FileNotFoundError(f"Missing evaluation results: {stats_file}")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with stats_file.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            match = re.match(r"(\d{2})_", row.get("scenario", ""))
            groups[match.group(1) if match else "unknown"].append(row)
    summary_rows: list[dict[str, object]] = []
    for scenario, rows in sorted(groups.items()):
        count = len(rows)
        summary_rows.append({
            "scenario": scenario,
            "samples": count,
            "safe_success_rate": sum(number(row, "safe_success") > 0 for row in rows) / count,
            "arrival_rate": sum(number(row, "arrival") > 0 for row in rows) / count,
            "collision_rate": sum(number(row, "collision_events") > 0 for row in rows) / count,
            "mean_safety_violations": sum(number(row, "safety_violations") for row in rows) / count,
            "mean_command_count": sum(number(row, "command_count") for row in rows) / count,
            "mean_episode_return": sum(number(row, "episode_return") for row in rows) / count,
        })
    output = root / "noise_baseline_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]) if summary_rows else ["scenario"])
        writer.writeheader()
        writer.writerows(summary_rows)
    for row in summary_rows:
        print(
            f"scenario {row['scenario']}: safe={row['safe_success_rate']:.1%}, "
            f"collision={row['collision_rate']:.1%}, commands={row['mean_command_count']:.1f}"
        )
    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
