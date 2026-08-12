"""Offline proxy evaluation for the processed Shanghai approach handoff data.

This script intentionally does *not* report human controller acceptance or
dynamic separation-adjustment success.  The handoff contains historical
instructions and radar snapshots, but no pilot readback and no explicit change
of separation standard.  It therefore reports two clearly named proxies:

* historical_instruction_operational_plausibility_rate
* post_instruction_separation_maintenance_rate

Both are useful diagnostics for the historical replay data and must remain
separate from the corresponding H-PPO/BlueSky runtime metrics.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EARTH_RADIUS_NM = 3440.065
METERS_TO_FEET = 3.280839895
ACTION_ALIASES = {
    "climb": "climb", "descend": "descend",
    "set_speed": "speed", "reduce_speed": "decelerate", "increase_speed": "accelerate",
    "fly_heading": "heading", "turn_left_heading": "turn_left", "turn_right_degrees": "turn_right",
    "maintain_altitude": "hold", "maintain_speed": "hold", "speed_as_procedure": "hold",
}
SUPPORTED_ACTIONS = set(ACTION_ALIASES.values())


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def haversine_nm(a: dict[str, Any], b: dict[str, Any]) -> float:
    lat1, lon1 = math.radians(float(a["latitude"])), math.radians(float(a["longitude"]))
    lat2, lon2 = math.radians(float(b["latitude"])), math.radians(float(b["longitude"]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return EARTH_RADIUS_NM * 2.0 * math.asin(min(1.0, math.sqrt(h)))


def signed_heading_delta(start: float, end: float) -> float:
    return (end - start + 180.0) % 360.0 - 180.0


def snapshot_index(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    by_callsign: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_time: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for snapshot in rows(path):
        stamp = int(snapshot.get("time_bucket_epoch", 0))
        for aircraft in snapshot.get("aircraft", []):
            callsign = str(aircraft.get("callsign") or "").strip()
            if not callsign or not stamp:
                continue
            record = dict(aircraft)
            record["_time"] = stamp
            by_callsign[callsign].append(record)
            by_time[stamp].append(record)
    for values in by_callsign.values():
        values.sort(key=lambda item: item["_time"])
    return by_callsign, by_time


def first_after(track: list[dict[str, Any]], start: float, min_delay: float, max_delay: float) -> dict[str, Any] | None:
    times = [item["_time"] for item in track]
    index = bisect.bisect_left(times, start + min_delay)
    if index < len(track) and track[index]["_time"] <= start + max_delay:
        return track[index]
    return None


def valid_target(event: dict[str, Any]) -> bool:
    if ACTION_ALIASES.get(str(event.get("action") or "").lower()) == "hold":
        return True
    try:
        return math.isfinite(float(event.get("target_value")))
    except (TypeError, ValueError):
        return False


def response_observed(event: dict[str, Any], before: dict[str, Any], after: dict[str, Any]) -> bool | None:
    """Return None when action direction cannot be inferred from this schema."""
    action = ACTION_ALIASES.get(str(event.get("action") or "").lower(), "")
    if action == "hold":
        return None
    target = float(event["target_value"])
    if action in {"climb", "descend"}:
        initial, final = float(before["altitude_m"]), float(after["altitude_m"])
        return final > initial + 30.0 if action == "climb" else final < initial - 30.0
    if action in {"speed", "accelerate", "decelerate"}:
        initial, final = float(before["ground_speed_mps"]), float(after["ground_speed_mps"])
        # Target units vary across extracts.  Direction is reliable only for
        # explicitly accelerate/decelerate actions.
        if action == "accelerate":
            return final > initial + 1.0
        if action == "decelerate":
            return final < initial - 1.0
        return None
    if action in {"turn_left", "turn_right", "heading"}:
        delta = signed_heading_delta(float(before["track_heading_deg"]), float(after["track_heading_deg"]))
        if action == "turn_left":
            return delta < -2.0
        if action == "turn_right":
            return delta > 2.0
        return abs(delta) >= 2.0
    return None


def closest_pair_at_time(ownship: dict[str, Any], aircraft: list[dict[str, Any]]) -> tuple[float, float] | None:
    best: tuple[float, float] | None = None
    for intruder in aircraft:
        if intruder.get("callsign") == ownship.get("callsign"):
            continue
        try:
            horizontal = haversine_nm(ownship, intruder)
            vertical = abs(float(ownship["altitude_m"]) - float(intruder["altitude_m"])) * METERS_TO_FEET
        except (KeyError, TypeError, ValueError):
            continue
        if best is None or horizontal < best[0]:
            best = horizontal, vertical
    return best


def score(dataset_root: Path, output_dir: Path, window_s: float, risk_nm: float, hsep_nm: float, vsep_ft: float) -> dict[str, Any]:
    base = dataset_root / "datasets" / "seu_controller_imitation_v06"
    references_path = base / "reference_events.jsonl"
    snapshots_path = base / "state_snapshots.jsonl"
    if not references_path.is_file() or not snapshots_path.is_file():
        raise FileNotFoundError("Expected seu_controller_imitation_v06 reference_events.jsonl and state_snapshots.jsonl")

    tracks, snapshots = snapshot_index(snapshots_path)
    plausibility = Counter()
    separation = Counter()
    action_counts = Counter()
    report_rows: list[dict[str, Any]] = []

    for event in rows(references_path):
        raw_action = str(event.get("action") or "").lower()
        action = ACTION_ALIASES.get(raw_action, raw_action)
        callsign = str(event.get("callsign") or "").strip()
        stamp = float(event.get("event_time_epoch") or 0.0)
        action_counts[raw_action] += 1
        if action not in SUPPORTED_ACTIONS or not callsign or not stamp or not valid_target(event):
            plausibility["rejected_invalid_instruction"] += 1
            continue
        track = tracks.get(callsign, [])
        before = first_after(track, stamp, -5.0, 10.0)
        after = first_after(track, stamp, 20.0, window_s)
        if before is None:
            plausibility["rejected_no_state_link"] += 1
            continue
        if after is None:
            plausibility["conditional_no_post_instruction_track"] += 1
            continue
        observed = response_observed(event, before, after)
        if observed is True:
            plausibility["accepted_observed_response"] += 1
            status = "accepted"
        elif observed is False:
            plausibility["rejected_no_observed_response"] += 1
            status = "rejected"
        else:
            plausibility["conditional_direction_unavailable"] += 1
            status = "conditional"

        # Historical post-instruction separation proxy.  Eligible means the
        # target aircraft had a nearby aircraft at the instruction timestamp.
        initial_group = snapshots.get(int(before["_time"]), [])
        initial_pair = closest_pair_at_time(before, initial_group)
        separation_status = "not_at_risk"
        stable = None
        min_horizontal = None
        if initial_pair is not None and initial_pair[0] <= risk_nm:
            separation["nearby_instruction"] += 1
            samples = [item for item in track if stamp + 20.0 <= item["_time"] <= stamp + window_s]
            pair_samples = []
            for item in samples:
                pair = closest_pair_at_time(item, snapshots.get(int(item["_time"]), []))
                if pair is not None:
                    pair_samples.append(pair)
            if not pair_samples:
                separation["inconclusive_no_pair_observation"] += 1
                separation_status = "inconclusive"
            else:
                min_horizontal = min(pair[0] for pair in pair_samples)
                stable = all(horizontal >= hsep_nm or vertical >= vsep_ft for horizontal, vertical in pair_samples)
                initially_lost = initial_pair[0] < hsep_nm and initial_pair[1] < vsep_ft
                if initially_lost:
                    separation["initial_loss_of_separation"] += 1
                if stable:
                    separation["nearby_maintained"] += 1
                    if initially_lost:
                        separation["loss_recovered"] += 1
                    separation_status = "maintained"
                else:
                    separation["nearby_not_maintained"] += 1
                    if initially_lost:
                        separation["loss_not_recovered"] += 1
                    separation_status = "not_maintained"

        report_rows.append({
            "event_id": event.get("event_id"), "callsign": callsign, "action": action,
            "event_time_epoch": stamp, "plausibility_status": status,
            "separation_status": separation_status, "post_window_min_horizontal_nm": min_horizontal,
        })

    accepted = plausibility["accepted_observed_response"]
    rejected = plausibility["rejected_invalid_instruction"] + plausibility["rejected_no_state_link"] + plausibility["rejected_no_observed_response"]
    conditional = plausibility["conditional_no_post_instruction_track"] + plausibility["conditional_direction_unavailable"]
    observable = accepted + rejected
    nearby = separation["nearby_instruction"]
    nearby_maintained = separation["nearby_maintained"]
    initial_loss = separation["initial_loss_of_separation"]
    loss_recovered = separation["loss_recovered"]
    result = {
        "metric_scope": "historical_atc_proxy_only",
        "limitations": [
            "No pilot readback or controller rating is present; plausibility is not human acceptance.",
            "No explicit separation-standard change is present; maintenance is not dynamic separation-adjustment success.",
            "Radar sampling and aircraft association can make a post-instruction response inconclusive.",
        ],
        "configuration": {"post_instruction_window_s": window_s, "risk_horizontal_nm": risk_nm, "horizontal_separation_nm": hsep_nm, "vertical_separation_ft": vsep_ft},
        "historical_instruction_operational_plausibility": {
            "rate": accepted / observable if observable else None,
            "accepted_observed_response": accepted, "rejected": rejected, "conditional": conditional,
            "observable_denominator": observable, "breakdown": dict(plausibility),
        },
        "post_instruction_separation_maintenance": {
            "nearby_traffic_maintenance_rate": nearby_maintained / nearby if nearby else None,
            "nearby_instruction_count": nearby, "nearby_maintained": nearby_maintained,
            "initial_loss_recovery_rate": loss_recovered / initial_loss if initial_loss else None,
            "initial_loss_of_separation_count": initial_loss, "loss_recovered": loss_recovered,
            "breakdown": dict(separation),
        },
        "action_counts": dict(action_counts),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "historical_atc_proxy_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "historical_atc_proxy_events.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "callsign", "action", "event_time_epoch", "plausibility_status", "separation_status", "post_window_min_horizontal_nm"])
        writer.writeheader()
        writer.writerows(report_rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Score clearly labelled offline proxies on the processed historical ATC handoff.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-s", type=float, default=60.0)
    parser.add_argument("--risk-nm", type=float, default=5.0)
    parser.add_argument("--hsep-nm", type=float, default=3.0)
    parser.add_argument("--vsep-ft", type=float, default=1000.0)
    args = parser.parse_args()
    result = score(args.dataset_root, args.output, args.window_s, args.risk_nm, args.hsep_nm, args.vsep_ft)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
