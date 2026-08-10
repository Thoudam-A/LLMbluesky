"""Generate an eight-aircraft distributed sector-capacity evaluation case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "output" / "H_PPO" / "testsets" / "sector_8ac_distributed_pairs_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an eight-aircraft distributed sector-capacity H-PPO test case.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help="Replace only the generated NPZ in an existing output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "11_sector_8ac_distributed_pairs.npz"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Generated scenario already exists: {output_path}")

    # Four spatially separated 90-degree crossing pairs. All aircraft enter at
    # t=0, but each starts roughly 94 km from its local crossing point. This
    # verifies sector-wide eight-aircraft monitoring without turning the test
    # into an artificial eight-aircraft single-point collision.
    routes = np.asarray(
        [
            # Northwest pair, crossing near (40.55, 116.00).
            [40.55, 114.90, 40.55, 116.40, 90.0],
            [39.70, 116.00, 40.95, 116.00, 0.0],
            # Northeast pair, crossing near (40.55, 117.00).
            [40.55, 118.10, 40.55, 116.60, 270.0],
            [39.70, 117.00, 40.95, 117.00, 0.0],
            # Southwest pair, crossing near (39.45, 116.00).
            [39.45, 114.90, 39.45, 116.40, 90.0],
            [40.30, 116.00, 39.05, 116.00, 180.0],
            # Southeast pair, crossing near (39.45, 117.00).
            [39.45, 118.10, 39.45, 116.60, 270.0],
            [40.30, 117.00, 39.05, 117.00, 180.0],
        ],
        dtype=np.float32,
    )
    metadata = {
        "scenario_id": "11_sector_8ac_distributed_pairs",
        "aircraft_count": 8,
        "design": "four spatially separated 90-degree crossing pairs; eight aircraft simultaneously present",
        "initial_distance_to_local_crossing_km": 94.0,
        "initial_speed_kt": 450.0,
        "initial_altitude_ft": 35000.0,
    }
    np.savez_compressed(
        output_path,
        routes=routes,
        speed_kt=np.full(8, 450.0, dtype=np.float32),
        altitude_ft=np.full(8, 35000.0, dtype=np.float32),
        entry_time_s=np.zeros(8, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"Generated {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
