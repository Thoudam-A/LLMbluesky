"""Generate a fixed eight-aircraft, two-wave sector-capacity evaluation case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "output" / "H_PPO" / "testsets" / "sector_8ac_two_wave_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an eight-aircraft sector-capacity H-PPO test case.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help="Replace only the generated NPZ in an existing output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "10_sector_8ac_two_wave.npz"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Generated scenario already exists: {output_path}")

    # Four directional corridors through one sector. Each corridor has a lead
    # aircraft and a follower entering 120 s later, so all eight aircraft are
    # concurrently monitored while the two four-aircraft conflict waves remain
    # temporally separable under the configured 90 s time-spacing standard.
    routes = np.asarray(
        [
            [39.55, 116.50, 40.60, 116.50, 0.0],
            [39.55, 116.50, 40.60, 116.50, 0.0],
            [40.45, 116.50, 39.40, 116.50, 180.0],
            [40.45, 116.50, 39.40, 116.50, 180.0],
            [40.00, 115.88, 40.00, 117.12, 90.0],
            [40.00, 115.88, 40.00, 117.12, 90.0],
            [40.00, 117.12, 40.00, 115.88, 270.0],
            [40.00, 117.12, 40.00, 115.88, 270.0],
        ],
        dtype=np.float32,
    )
    speed_kt = np.full(8, 450.0, dtype=np.float32)
    altitude_ft = np.full(8, 35000.0, dtype=np.float32)
    entry_time_s = np.asarray([0.0, 120.0, 0.0, 120.0, 0.0, 120.0, 0.0, 120.0], dtype=np.float32)
    metadata = {
        "scenario_id": "10_sector_8ac_two_wave",
        "aircraft_count": 8,
        "design": "four directional corridors, two aircraft per corridor, two temporally separated four-aircraft crossing waves",
        "lead_entry_s": 0.0,
        "follower_entry_s": 120.0,
        "initial_speed_kt": 450.0,
        "initial_altitude_ft": 35000.0,
    }
    np.savez_compressed(
        output_path,
        routes=routes,
        speed_kt=speed_kt,
        altitude_ft=altitude_ft,
        entry_time_s=entry_time_s,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"Generated {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
