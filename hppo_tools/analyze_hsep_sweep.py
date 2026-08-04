"""Summarise horizontal-separation evaluation runs without changing their data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


KM_PER_NM = 1.852


def _number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _diagnostics_file(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("validation_diagnostics.csv", "training_diagnostics.csv"):
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No diagnostics CSV found under {path}")


def summarise(path: Path) -> dict[str, object]:
    diagnostics = _diagnostics_file(path)
    with diagnostics.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No episode rows in {diagnostics}")

    episode_count = len(rows)
    hsep_km = _mean([_number(row, "separation_horizontal_km") for row in rows])
    actual_min_km = [_number(row, "min_contact_horizontal_km", -1.0) for row in rows]
    actual_min_km = [value for value in actual_min_km if value >= 0.0]
    predicted_min_km = [_number(row, "min_predicted_cpa_km", -1.0) for row in rows]
    predicted_min_km = [value for value in predicted_min_km if value >= 0.0]
    first_interventions = [_number(row, "first_intervention_s", -1.0) for row in rows]
    first_interventions = [value for value in first_interventions if value >= 0.0]
    arrivals = sum(row.get("reason") == "all_arrived" for row in rows)
    collisions = sum(_number(row, "collision_events") > 0.0 for row in rows)
    timeouts = sum(row.get("reason") == "timeout" for row in rows)
    safe_successes = sum(_number(row, "safe_success") > 0.0 for row in rows)

    return {
        "run": str(path),
        "episodes": episode_count,
        "hsep_nm": hsep_km / KM_PER_NM,
        "arrived_pct": 100.0 * arrivals / episode_count,
        "safe_success_pct": 100.0 * safe_successes / episode_count,
        "collision_pct": 100.0 * collisions / episode_count,
        "timeout_pct": 100.0 * timeouts / episode_count,
        "mean_violations": _mean([_number(row, "safety_violations") for row in rows]),
        "mean_commands": _mean([_number(row, "command_count") for row in rows]),
        "mean_first_intervention_s": _mean(first_interventions),
        "mean_min_contact_nm": _mean(actual_min_km) / KM_PER_NM,
        "worst_min_contact_nm": min(actual_min_km, default=-1.0) / KM_PER_NM,
        "mean_min_predicted_cpa_nm": _mean(predicted_min_km) / KM_PER_NM,
        "worst_min_predicted_cpa_nm": min(predicted_min_km, default=-1.0) / KM_PER_NM,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare H-PPO horizontal-separation evaluation runs.")
    parser.add_argument("runs", nargs="+", help="Evaluation output folders or diagnostics CSV files.")
    parser.add_argument("--output", type=Path, help="Optional CSV path for the aggregate table.")
    args = parser.parse_args()

    rows = [summarise(Path(item).resolve()) for item in args.runs]
    fields = list(rows[0])
    print(",".join(fields))
    for row in rows:
        print(",".join(str(row[key]) if key == "run" else f"{float(row[key]):.3f}" for key in fields))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
