"""Create a reproducible, held-out noisy initial-condition test set for H-PPO."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "bluesky_project" / "routes" / "hppo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed noisy H-PPO evaluation samples from route templates.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenarios", default="01,02,03,04,05,06,07,08,09")
    parser.add_argument("--samples-per-scenario", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--position-km", type=float, default=0.5)
    parser.add_argument("--heading-deg", type=float, default=3.0)
    parser.add_argument("--speed-kt", type=float, default=5.0)
    parser.add_argument("--altitude-ft", type=float, default=200.0)
    parser.add_argument("--entry-time-s", type=float, default=5.0)
    return parser.parse_args()


def scenario_files(source_root: Path, selectors: str) -> list[Path]:
    available = sorted(
        path
        for path in source_root.iterdir()
        if path.suffix in {".npy", ".npz"} and len(path.name) >= 3 and path.name[:2].isdigit() and path.name[2] == "_"
    )
    selected: list[Path] = []
    for token in (part.strip() for part in selectors.split(",")):
        if not token:
            continue
        key = token.zfill(2) if token.isdigit() else token
        matches = [path for path in available if path.name.startswith(f"{key}_")]
        if len(matches) != 1:
            raise ValueError(f"Scenario selector '{token}' matched {len(matches)} templates")
        selected.append(matches[0])
    return selected


def load_template(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load both legacy route arrays and richer NPZ scenario templates."""
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            routes = np.asarray(loaded["routes"], dtype=np.float32)
            count = len(routes)
            speed_kt = np.asarray(loaded.get("speed_kt", np.full(count, 450.0)), dtype=np.float32)
            altitude_ft = np.asarray(loaded.get("altitude_ft", np.full(count, 35000.0)), dtype=np.float32)
            entry_time_s = np.asarray(loaded.get("entry_time_s", np.zeros(count)), dtype=np.float32)
        finally:
            loaded.close()
    else:
        routes = np.asarray(loaded, dtype=np.float32)
        count = len(routes)
        speed_kt = np.full(count, 450.0, dtype=np.float32)
        altitude_ft = np.full(count, 35000.0, dtype=np.float32)
        entry_time_s = np.zeros(count, dtype=np.float32)
    if routes.ndim != 2 or routes.shape[1] != 5:
        raise ValueError(f"Invalid route template {path}: expected [N, 5], got {routes.shape}")
    for name, values in (("speed_kt", speed_kt), ("altitude_ft", altitude_ft), ("entry_time_s", entry_time_s)):
        if values.shape != (count,):
            raise ValueError(f"Invalid route template {path}: {name} must have shape ({count},)")
    return routes, speed_kt, altitude_ft, entry_time_s


def perturb_starts(routes: np.ndarray, rng: np.random.Generator, position_km: float, heading_deg: float) -> np.ndarray:
    result = np.asarray(routes, dtype=np.float64).copy()
    east_km = rng.uniform(-position_km, position_km, size=len(result))
    north_km = rng.uniform(-position_km, position_km, size=len(result))
    result[:, 0] += north_km / 111.32
    result[:, 1] += east_km / (111.32 * np.maximum(np.cos(np.deg2rad(result[:, 0])), 0.1))
    result[:, 4] = (result[:, 4] + rng.uniform(-heading_deg, heading_deg, size=len(result))) % 360.0
    return result.astype(np.float32)


def main() -> int:
    args = parse_args()
    if args.samples_per_scenario < 1:
        raise ValueError("--samples-per-scenario must be positive")
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source scenario directory is missing: {source_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for template in scenario_files(source_root, args.scenarios):
        routes, base_speed_kt, base_altitude_ft, base_entry_time_s = load_template(template)
        scenario_id = template.name[:2]
        for sample_index in range(1, args.samples_per_scenario + 1):
            sample_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            sample_rng = np.random.default_rng(sample_seed)
            sample_routes = perturb_starts(routes, sample_rng, args.position_km, args.heading_deg)
            speed_kt = (base_speed_kt + sample_rng.uniform(-args.speed_kt, args.speed_kt, size=len(routes))).astype(np.float32)
            altitude_ft = (base_altitude_ft + sample_rng.uniform(-args.altitude_ft, args.altitude_ft, size=len(routes))).astype(np.float32)
            entry_time_s = (base_entry_time_s + sample_rng.uniform(0.0, args.entry_time_s, size=len(routes))).astype(np.float32)
            name = f"{scenario_id}_s{sample_index:03d}.npz"
            metadata = {
                "template": template.name,
                "scenario_id": scenario_id,
                "sample_index": sample_index,
                "sample_seed": sample_seed,
                "noise": {
                    "position_km": args.position_km,
                    "heading_deg": args.heading_deg,
                    "speed_kt": args.speed_kt,
                    "altitude_ft": args.altitude_ft,
                    "entry_time_s": args.entry_time_s,
                },
            }
            np.savez_compressed(
                output_root / name,
                routes=sample_routes,
                speed_kt=speed_kt,
                altitude_ft=altitude_ft,
                entry_time_s=entry_time_s,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            rows.append({"file": name, **metadata})
    with (output_root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "template", "scenario_id", "sample_index", "sample_seed", "noise"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} samples in {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
