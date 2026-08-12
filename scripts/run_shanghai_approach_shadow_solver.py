#!/usr/bin/env python3
"""Run the current conflict-resolution solver on frozen Shanghai replay states.

This adapter reuses the decision/search/verification methods from
``headless_dynamic_sector_validation.py`` while replacing the Chengdu projection,
FL270-FL390 action space, and speed limits with an explicit Shanghai-approach
configuration. Historical traffic states remain frozen; emitted commands are
not applied to the trajectory.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from seu_imitation_common import jsonl_rows


ADAPTER_VERSION = "shanghai_current_solver_adapter_v0.1"
M_TO_FT = 1.0 / 0.3048
MPS_TO_KT = 1.0 / 0.514444


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--solver-source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument(
        "--state-scope",
        choices=("inside_sector", "sector_context"),
        required=True,
    )
    parser.add_argument("--start-tick", type=int, default=0)
    parser.add_argument("--max-ticks", type=int)
    parser.add_argument(
        "--target-memory-sec",
        type=float,
        default=900.0,
        help="Forget stored clearances after an aircraft has been absent this long.",
    )
    parser.add_argument(
        "--aircraft-command-cooldown-sec",
        type=float,
        default=60.0,
        help="After a command, wait this long before resolving another pair involving that aircraft.",
    )
    parser.add_argument(
        "--pair-retry-interval-sec",
        type=float,
        default=60.0,
        help="Minimum interval before re-solving the same callsign pair in frozen replay.",
    )
    return parser.parse_args()


def load_solver_module(path: Path) -> Any:
    """Import the current solver and undo its import-time cwd change."""

    module_name = "_current_headless_sector_solver"
    previous_cwd = Path.cwd()
    spec = importlib.util.spec_from_file_location(module_name, path.resolve())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import solver source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous_cwd)
    return module


def configure_solver(module: Any, config: dict[str, Any]) -> list[float]:
    altitude = config["altitude"]
    speed = config["speed"]
    conflict = config["conflict"]
    center = config["projection_center"]
    safe_levels_fl = [float(value) * M_TO_FT / 100.0 for value in altitude["candidate_levels_m"]]

    module.CENTER_LAT = float(center["latitude"])
    module.CENTER_LON = float(center["longitude"])
    module.SAFE_LEVELS = safe_levels_fl
    module.MIN_SPEED_KT = int(speed["minimum_kt"])
    module.MAX_SPEED_KT = int(speed["maximum_kt"])
    module.SPEED_DELTAS_KT = [int(value) for value in speed["candidate_deltas_kt"]]
    module.LOOKAHEAD_MIN = float(conflict["lookahead_min"])
    module.HSEP_NM = float(conflict["loss_horizontal_nm"])
    module.PREDICT_GATE_NM = float(conflict["predict_gate_nm"])
    module.VERIFY_HSEP_NM = float(conflict["verify_horizontal_nm"])
    vertical_ft = float(conflict["vertical_separation_m"]) * M_TO_FT
    module.VSEP_FT = vertical_ft
    module.VERIFY_VSEP_FT = vertical_ft
    module.VERIFY_DT_SEC = int(conflict["verify_step_sec"])
    module.SEARCH_TIME_LIMIT_SEC = float(conflict.get("search_time_limit_sec", 1.0))
    module.RESOLUTION_PREFERENCE = "speed_first"
    module.TEACHER_POLICY = "speed_first"
    module.ACTION_API_URL = ""
    module.ENABLE_UNVERIFIED_FALLBACK = False
    return safe_levels_fl


def make_runner(module: Any, safe_levels_fl: list[float]) -> Any:
    class ShanghaiRunner(module.HeadlessSectorRunner):
        def __init__(self) -> None:
            # Avoid BlueSky/log initialization; only the pure decision state is
            # required for historical shadow replay.
            self.active_meta = {}
            self.last_targets = {}
            self.last_speed_targets = {}
            self.safe_levels_fl = list(safe_levels_fl)

        def current_target_action(self, state: Any) -> Any:
            return module.CandidateAction(
                acid=state.acid,
                kind="hold",
                target_fl=float(self.effective_alt_ft(state) / 100.0),
                target_speed_kt=self.effective_speed_kt(state),
                command=None,
                label="current_target",
            )

        def generate_candidate_actions(self, state: Any, suppress_speed: bool = False) -> list[Any]:
            current_fl = float(state.alt_ft / 100.0)
            effective_fl = float(self.effective_alt_ft(state) / 100.0)
            current_speed = self.effective_speed_kt(state)
            active_target_fl = self.last_targets.get(state.acid)
            altitude_locked = active_target_fl is not None and self.pending_alt_dir(state) != "none"
            candidates = [
                module.CandidateAction(
                    acid=state.acid,
                    kind="hold",
                    target_fl=effective_fl,
                    target_speed_kt=current_speed,
                    command=None,
                    label="hold",
                )
            ]
            if not altitude_locked:
                for target_fl in sorted(
                    (value for value in self.safe_levels_fl if abs(value - effective_fl) > 1e-6),
                    key=lambda value: (abs(value - effective_fl), abs(value - current_fl)),
                ):
                    target_m = int(round(target_fl * 100.0 * 0.3048))
                    direction = "CLIMB" if target_fl > current_fl else "DESCEND"
                    candidates.append(
                        module.CandidateAction(
                            acid=state.acid,
                            kind="altitude",
                            target_fl=target_fl,
                            target_speed_kt=current_speed,
                            command=f"ALT_M {state.acid},{target_m},{direction}",
                            label=f"altitude:M{target_m}",
                        )
                    )
            if module.ALLOW_SPEED_ACTIONS and not suppress_speed and not altitude_locked:
                seen: set[int] = set()
                for delta in module.SPEED_DELTAS_KT:
                    target = max(module.MIN_SPEED_KT, min(module.MAX_SPEED_KT, current_speed + delta))
                    if target == current_speed or target in seen:
                        continue
                    seen.add(target)
                    candidates.append(
                        module.CandidateAction(
                            acid=state.acid,
                            kind="speed",
                            target_fl=effective_fl,
                            target_speed_kt=target,
                            command=f"SPD {state.acid},{target}",
                            label=f"speed:{target}",
                        )
                    )
            order = {"hold": 0, "speed": 1, "altitude": 2}
            return sorted(
                candidates,
                key=lambda action: (
                    order.get(action.kind, 9),
                    abs(float(action.target_fl) - effective_fl),
                    abs(action.target_speed_kt - current_speed),
                ),
            )

        def commands_from_solution(self, solution: dict[str, Any], state_by_id: dict[str, Any]) -> list[str]:
            commands: list[str] = []
            for acid, action in sorted(solution.items()):
                state = state_by_id[acid]
                if action.kind == "hold":
                    continue
                if action.kind == "altitude":
                    if self.last_targets.get(acid) == action.target_fl:
                        continue
                    commands.append(str(action.command))
                    self.last_targets[acid] = float(action.target_fl)
                elif action.kind == "speed":
                    if (
                        self.last_speed_targets.get(acid) == action.target_speed_kt
                        or action.target_speed_kt == self.effective_speed_kt(state)
                    ):
                        continue
                    commands.append(str(action.command))
                    self.last_speed_targets[acid] = int(action.target_speed_kt)
            return commands

    return ShanghaiRunner()


def states_from_tick(
    module: Any,
    tick: dict[str, Any],
    state_scope: str,
    minimum_altitude_m: float,
    maximum_altitude_m: float,
) -> tuple[list[Any], int]:
    by_callsign: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for row in tick.get("aircraft", []):
        if state_scope == "inside_sector" and not bool(row.get("inside_sector")):
            continue
        try:
            altitude_m = float(row.get("altitude_m"))
        except (TypeError, ValueError):
            continue
        if not minimum_altitude_m <= altitude_m <= maximum_altitude_m:
            continue
        callsign = str(row.get("callsign") or "").upper()
        if not callsign:
            continue
        values = (
            row.get("latitude"),
            row.get("longitude"),
            row.get("altitude_m"),
            row.get("track_heading_deg"),
            row.get("ground_speed_mps"),
        )
        try:
            numeric = [float(value) for value in values]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in numeric):
            continue
        if callsign in by_callsign:
            duplicate_count += 1
            prior = by_callsign[callsign]
            # Prefer the geometrically inside record, then the newer event.
            if bool(prior.get("inside_sector")) and not bool(row.get("inside_sector")):
                continue
            if float(prior.get("event_time_epoch") or 0) > float(row.get("event_time_epoch") or 0):
                continue
        by_callsign[callsign] = row

    states = [
        module.AircraftState(
            acid=callsign,
            lat=float(row["latitude"]),
            lon=float(row["longitude"]),
            alt_ft=float(row["altitude_m"]) * M_TO_FT,
            trk=float(row["track_heading_deg"]) % 360.0,
            gs_mps=float(row["ground_speed_mps"]),
        )
        for callsign, row in sorted(by_callsign.items())
    ]
    return states, duplicate_count


def conflict_detections(
    module: Any,
    runner: Any,
    states: list[Any],
    epoch: float,
    command_cooldown_until: dict[str, float],
    pair_last_attempt: dict[tuple[str, str], float],
    pair_retry_interval_sec: float,
) -> tuple[list[tuple], int, int]:
    detections: list[tuple] = []
    cooldown_skips = 0
    retry_skips = 0
    for index, a in enumerate(states):
        for b in states[index + 1 :]:
            tcpa, hsep, vsep = module.cpa(a, b)
            target_vsep = abs(runner.effective_alt_ft(a) - runner.effective_alt_ft(b))
            current_vsep = abs(a.alt_ft - b.alt_ft)
            if hsep >= module.PREDICT_GATE_NM:
                continue
            if target_vsep >= module.VERIFY_VSEP_FT and current_vsep >= module.VERIFY_VSEP_FT:
                continue
            pair = tuple(sorted((a.acid, b.acid)))
            if any(epoch < command_cooldown_until.get(acid, float("-inf")) for acid in pair):
                cooldown_skips += 1
                continue
            if epoch - pair_last_attempt.get(pair, float("-inf")) < pair_retry_interval_sec:
                retry_skips += 1
                continue
            if runner.current_targets_are_safe(a, b):
                continue
            runner.last_targets.pop(a.acid, None)
            runner.last_targets.pop(b.acid, None)
            detections.append((tcpa, hsep, vsep, a, b, pair))
    detections.sort(key=lambda item: (item[0], item[1], item[5]))
    return detections, cooldown_skips, retry_skips


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))
    return ordered[index]


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    module = load_solver_module(args.solver_source)
    safe_levels_fl = configure_solver(module, config)
    runner = make_runner(module, safe_levels_fl)

    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    last_seen: dict[str, float] = {}
    processed = 0
    ticks_with_conflict = 0
    ticks_with_commands = 0
    total_detections = 0
    duplicate_states = 0
    solve_durations: list[float] = []
    method_counts: collections.Counter[str] = collections.Counter()
    command_counts: collections.Counter[str] = collections.Counter()
    command_cooldown_until: dict[str, float] = {}
    pair_last_attempt: dict[tuple[str, str], float] = {}
    cooldown_conflict_skips = 0
    retry_conflict_skips = 0
    start_wall = time.perf_counter()
    minimum_altitude_m = float(config["evaluation"]["minimum_state_altitude_m"])
    maximum_altitude_m = float(config["evaluation"]["maximum_state_altitude_m"])

    with args.output_log.open("w", encoding="utf-8", newline="\n") as output:
        for source_index, tick in enumerate(jsonl_rows(args.replay)):
            if source_index < args.start_tick:
                continue
            if args.max_ticks is not None and processed >= args.max_ticks:
                break
            processed += 1
            epoch = float(tick["time_bucket_epoch"])
            states, duplicates = states_from_tick(
                module,
                tick,
                args.state_scope,
                minimum_altitude_m,
                maximum_altitude_m,
            )
            duplicate_states += duplicates
            active = {state.acid for state in states}
            for acid in active:
                last_seen[acid] = epoch
            expired = [
                acid
                for acid, seen_epoch in last_seen.items()
                if epoch - seen_epoch > args.target_memory_sec
            ]
            for acid in expired:
                last_seen.pop(acid, None)
                runner.last_targets.pop(acid, None)
                runner.last_speed_targets.pop(acid, None)
                command_cooldown_until.pop(acid, None)
            expired_pairs = [
                pair
                for pair, attempt_epoch in pair_last_attempt.items()
                if epoch - attempt_epoch > max(args.target_memory_sec, args.pair_retry_interval_sec)
            ]
            for pair in expired_pairs:
                pair_last_attempt.pop(pair, None)

            detections, cooldown_skips, retry_skips = conflict_detections(
                module,
                runner,
                states,
                epoch,
                command_cooldown_until,
                pair_last_attempt,
                args.pair_retry_interval_sec,
            )
            cooldown_conflict_skips += cooldown_skips
            retry_conflict_skips += retry_skips
            if not detections:
                continue
            for detection in detections:
                pair_last_attempt[detection[5]] = epoch
            ticks_with_conflict += 1
            total_detections += len(detections)
            state_by_id = {state.acid: state for state in states}
            solve_start = time.perf_counter()
            commands, solver_info = runner.build_resolution_plan(state_by_id, detections)
            duration = time.perf_counter() - solve_start
            solve_durations.append(duration)
            method = str(solver_info.get("method") or "unknown")
            method_counts[method] += 1
            if commands:
                ticks_with_commands += 1
                for command in commands:
                    command_counts[command.split(" ", 1)[0]] += 1
                    parts = command.split()
                    if len(parts) >= 2:
                        acid = parts[1].split(",", 1)[0].upper()
                        command_cooldown_until[acid] = epoch + args.aircraft_command_cooldown_sec
            output.write(
                json.dumps(
                    {
                        "adapter_version": ADAPTER_VERSION,
                        "config_version": config.get("config_version"),
                        "source_tick_index": source_index,
                        "replay_tick_id": tick.get("replay_tick_id"),
                        "event_time_utc": tick.get("event_time_utc"),
                        "event_time_epoch": epoch,
                        "state_scope": args.state_scope,
                        "aircraft_count": len(states),
                        "conflict_pair_count": len(detections),
                        "conflict_pairs": [
                            {
                                "callsigns": list(item[5]),
                                "tcpa_min": round(float(item[0]), 4),
                                "cpa_hsep_nm": round(float(item[1]), 4),
                                "current_vsep_m": round(abs(item[3].alt_ft - item[4].alt_ft) * 0.3048, 1),
                            }
                            for item in detections
                        ],
                        "commands": commands,
                        "solver_method": method,
                        "solver_success": bool(solver_info.get("success", commands)),
                        "solver_search_nodes": solver_info.get("search_nodes"),
                        "solver_search_limited": solver_info.get("search_limited"),
                        "solve_duration_sec": round(duration, 6),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    elapsed = time.perf_counter() - start_wall
    summary = {
        "adapter_version": ADAPTER_VERSION,
        "config_version": config.get("config_version"),
        "solver_source": str(args.solver_source.resolve()),
        "replay": str(args.replay.resolve()),
        "state_scope": args.state_scope,
        "start_tick": args.start_tick,
        "requested_max_ticks": args.max_ticks,
        "processed_ticks": processed,
        "ticks_with_conflict": ticks_with_conflict,
        "ticks_with_commands": ticks_with_commands,
        "total_conflict_detections": total_detections,
        "commands_by_type": dict(sorted(command_counts.items())),
        "solver_methods": dict(sorted(method_counts.items())),
        "duplicate_callsign_states_resolved": duplicate_states,
        "conflict_candidates_skipped_by_aircraft_cooldown": cooldown_conflict_skips,
        "conflict_candidates_skipped_by_pair_retry_interval": retry_conflict_skips,
        "runtime_sec": round(elapsed, 3),
        "solve_duration_sec": {
            "mean": round(statistics.mean(solve_durations), 6) if solve_durations else None,
            "median": round(statistics.median(solve_durations), 6) if solve_durations else None,
            "p90": round(percentile(solve_durations, 0.9), 6) if solve_durations else None,
            "max": round(max(solve_durations), 6) if solve_durations else None,
        },
        "adapter_parameters": {
            "projection_center": config["projection_center"],
            "candidate_levels_m": config["altitude"]["candidate_levels_m"],
            "vertical_separation_m": config["conflict"]["vertical_separation_m"],
            "speed_range_kt": [
                config["speed"]["minimum_kt"],
                config["speed"]["maximum_kt"],
            ],
            "historical_states_frozen": True,
            "commands_applied_to_trajectory": False,
            "target_memory_sec": args.target_memory_sec,
            "aircraft_command_cooldown_sec": args.aircraft_command_cooldown_sec,
            "pair_retry_interval_sec": args.pair_retry_interval_sec,
            "state_altitude_band_m": [minimum_altitude_m, maximum_altitude_m],
        },
        "warnings": [
            "This is an offline Shanghai-unit adaptation of the current conflict-resolution core, not a live BlueSky run.",
            "The solver's internal target memory is retained, but emitted commands do not alter frozen historical trajectories.",
        ],
    }
    summary_path = args.output_log.with_suffix(args.output_log.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
