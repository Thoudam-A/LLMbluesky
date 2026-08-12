#!/usr/bin/env python3
"""Build an auditable, multi-key and pre-voice aligned Shanghai ATC corpus.

This builder is deliberately additive: it reads v4 labels and v2 provenance,
re-selects every state strictly before the VAD segment starts, replays MH4029
plans at that cutoff using address/SSR/callsign evidence, and enriches the
state with CAT062 I062/380 and I062/390 fields.  It never overwrites v4.

The output distinguishes: (a) observed surveillance intent, (b) electronic
flight-plan state, and (c) previous spoken controller clearance.  In
particular, a same-window CFL is *not* exposed as a prior clearance.
"""
from __future__ import annotations

import argparse
import base64
import copy
import datetime as dt
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
PYARROW20 = ROOT / ".deps" / "pyarrow20"
if PYARROW20.exists() and str(PYARROW20) not in sys.path:
    sys.path.insert(0, str(PYARROW20))

import pandas as pd
import pyarrow.dataset as ds

from build_shanghai_causal_alignment_v4 import (
    CausalPlanIndex, PlanVersion, iso_utc, normalize_callsign, parse_filtim_epoch,
    parse_message, plan_context, route_context, resolve_route_points,
    sector_context, SectorIndex, sha256_file, vertical_speed_context, coverage,
)
from build_shanghai_lossless_training_data_v2 import load_nav_index
from cat062_enrichment_v5 import load_enriched_decoder


VERSION = "shanghai_causal_alignment_builder_v5.0"
MUTABLE_PLAN_FIELDS = {"CFL", "XFL", "SID", "STAR", "DRWY", "ARWY", "SECTOR", "ROUTE", "SPEED"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v4-dir", type=Path, required=True)
    p.add_argument("--v2-dir", type=Path, required=True)
    p.add_argument("--trajectory-parquet", type=Path, required=True)
    p.add_argument("--raw-sqlite", type=Path, required=True)
    p.add_argument("--base-decoder", type=Path, required=True)
    p.add_argument("--sector-xml", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cutoff_guard_sec", type=float, default=0.1,
                   help="strict buffer before VAD segment start")
    p.add_argument("--mutable_field_guard_sec", type=float, default=30.0,
                   help="mask same-decision electronic plan updates")
    p.add_argument("--cat-lookback-sec", type=float, default=30.0,
                   help="raw CAT062 retrieval lookback before the decision cutoff")
    p.add_argument("--cat-feature-max-age-sec", type=float, default=15.0,
                   help="maximum CAT062 observation age allowed in model input")
    p.add_argument("--state-fresh-max-age-sec", type=float, default=10.0,
                   help="maximum strictly-prevoice trajectory age for full training weight")
    p.add_argument("--state-weak-max-age-sec", type=float, default=15.0,
                   help="maximum strictly-prevoice trajectory age for weak training")
    p.add_argument("--limit", type=int)
    return p.parse_args()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as h:
        for row in rows:
            h.write(json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def norm_hex(value: Any) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch in "0123456789ABCDEF")


def norm_ssr(value: Any) -> str:
    text = str(value or "").upper().strip()
    if text.startswith("A"):
        text = text[1:]
    return "".join(ch for ch in text if ch in "01234567")


def parse_segment_start(instruction: dict[str, Any]) -> float | None:
    value = instruction.get("segment_start_local")
    if value:
        try:
            return dt.datetime.fromisoformat(str(value)).timestamp()
        except ValueError:
            pass
    event = instruction.get("event_time_epoch")
    duration = instruction.get("segment_duration_sec")
    return None if event is None else float(event) - float(duration or 0) / 2.0


@dataclass
class Identity:
    callsign: str
    target_address: str
    mode3a: str
    preferred_ifplid: str
    trajectory_id: str


class MultiKeyPlanIndex(CausalPlanIndex):
    """Causal MH4029 replay with immutable identity evidence, not callsign-only."""

    def __init__(self, versions: list[PlanVersion], config: dict[str, Any]):
        super().__init__(versions, config)
        self.address_groups: dict[str, set[str]] = defaultdict(set)
        self.ssr_groups: dict[str, set[str]] = defaultdict(set)
        for ifplid, values in self.by_ifplid.items():
            for v in values:
                for address in (v.fields.get("ARCADDR"),):
                    if norm_hex(address):
                        self.address_groups[norm_hex(address)].add(ifplid)
                for ssr in (v.fields.get("PSSRCODE"), v.fields.get("SSRCODE")):
                    if norm_ssr(ssr):
                        self.ssr_groups[norm_ssr(ssr)].add(ifplid)

    @classmethod
    def load(cls, sqlite_path: Path, identities: list[Identity], low: float, high: float,
             config: dict[str, Any]) -> "MultiKeyPlanIndex":
        lookback = float(config["flight_plan"]["scan_lookback_hours"]) * 3600
        wanted_names = {x.callsign for x in identities if x.callsign}
        wanted_addr = {x.target_address for x in identities if x.target_address}
        wanted_ssr = {x.mode3a for x in identities if x.mode3a}
        versions: list[PlanVersion] = []
        con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        try:
            q = "SELECT raw_data_md5id, receive_time_ms, outer_json FROM mh4029_raw WHERE receive_time_ms BETWEEN ? AND ? ORDER BY receive_time_ms"
            for md5id, ms, outer_json in con.execute(q, (int((low - lookback) * 1000), int(high * 1000))):
                try:
                    raw = json.loads(outer_json).get("raw_value")
                    if not isinstance(raw, str):
                        continue
                    fields, points, present = parse_message(raw)
                    name = normalize_callsign(fields.get("ARCID"))
                    addr = norm_hex(fields.get("ARCADDR"))
                    ssrs = {norm_ssr(fields.get("PSSRCODE")), norm_ssr(fields.get("SSRCODE"))}
                    if not (name in wanted_names or addr in wanted_addr or bool(ssrs & wanted_ssr)):
                        continue
                    ifplid = str(fields.get("IFPLID") or "")
                    if ifplid:
                        versions.append(PlanVersion(str(md5id), float(ms) / 1000.0,
                            parse_filtim_epoch(fields.get("FILTIM"), float(ms) / 1000.0),
                            name, ifplid, fields, points, present))
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
        finally:
            con.close()
        return cls(versions, config)

    def match_identity(self, identity: Identity, cutoff: float) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        candidates_ids = set()
        if identity.callsign:
            candidates_ids |= self.callsign_groups.get(identity.callsign, set())
        if identity.target_address:
            candidates_ids |= self.address_groups.get(identity.target_address, set())
        if identity.mode3a:
            candidates_ids |= self.ssr_groups.get(identity.mode3a, set())
        if identity.preferred_ifplid:
            candidates_ids.add(identity.preferred_ifplid)
        scored: list[tuple[int, dict[str, Any], list[str]]] = []
        for ifplid in candidates_ids:
            replay = self._replay(ifplid, cutoff)
            if replay is None:
                continue
            f = replay["fields"]
            evidence: list[str] = []
            score = 0
            if identity.target_address and norm_hex(f.get("ARCADDR")) == identity.target_address:
                score += 100; evidence.append("target_address")
            ssrs = {norm_ssr(f.get("PSSRCODE")), norm_ssr(f.get("SSRCODE"))}
            if identity.mode3a and identity.mode3a in ssrs:
                score += 45; evidence.append("mode3a")
            if identity.callsign and normalize_callsign(f.get("ARCID")) == identity.callsign:
                score += 25; evidence.append("callsign")
            if identity.preferred_ifplid and ifplid == identity.preferred_ifplid:
                score += 15; evidence.append("source_ifplid")
            if score:
                scored.append((score, replay, evidence))
        if not scored:
            return None, {"status": "unlinked", "score": 0, "evidence": [], "candidate_count": 0}
        scored.sort(key=lambda item: (item[0], item[1]["selected_receive_epoch"]), reverse=True)
        best_score, best, evidence = scored[0]
        ties = sum(1 for score, _, _ in scored if score == best_score)
        status = "high" if "target_address" in evidence and ("mode3a" in evidence or "callsign" in evidence) else "medium" if "target_address" in evidence or ("mode3a" in evidence and "callsign" in evidence) else "low"
        if ties > 1:
            status = "ambiguous"
        return best, {"status": status, "score": best_score, "evidence": evidence,
                      "candidate_count": len(scored), "top_score_tie_count": ties}


def load_prevoice_states(parquet: Path, identities: dict[str, Identity], cutoffs: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Read only command trajectories and choose latest state strictly before speech start."""
    cols = ["trajectory_id", "event_time_epoch", "callsign", "target_address", "mode3a_octal", "track_number", "latitude", "longitude", "altitude_m", "geometric_altitude_m", "barometric_altitude_fl", "measured_flight_level", "ground_speed_mps", "track_heading_deg", "vertical_rate_mps", "vx_mps", "vy_mps", "inside_sector", "sector_hit", "sector_code", "time_bucket_epoch", "time_from_sector_entry_sec", "time_from_track_start_sec", "trajectory_phase", "sac", "sic"]
    trajectory_ids = sorted({x.trajectory_id for x in identities.values() if x.trajectory_id})
    table = ds.dataset(str(parquet), format="parquet").to_table(columns=cols, filter=ds.field("trajectory_id").isin(trajectory_ids))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table.to_pylist():
        grouped[str(row["trajectory_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: float(r["event_time_epoch"]))
    selected = {}
    for ref, identity in identities.items():
        rows = grouped.get(identity.trajectory_id, [])
        cutoff = cutoffs[ref]
        eligible = [r for r in rows if float(r["event_time_epoch"]) < cutoff]
        if eligible:
            selected[ref] = eligible[-1]
    return selected


def merge_intervals(cutoffs: Iterable[float], before: float = 14.0, after: float = 2.0) -> list[tuple[int, int]]:
    spans = sorted((int((x-before)*1000), int((x+after)*1000)) for x in cutoffs)
    merged: list[list[int]] = []
    for low, high in spans:
        if merged and low <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return [(a, b) for a, b in merged]


def load_cat062_enrichment(sqlite_path: Path, decoder_path: Path, states: dict[str, dict[str, Any]], cutoffs: dict[str, float], lookback_sec: float) -> dict[str, dict[str, Any]]:
    """Decode only raw records near decision cutoffs, keyed by target address.

    CAT timestamps are reconciled against the raw message receive time; fields
    are retained only if the decoded observation is strictly pre-voice.
    """
    by_addr: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for ref, state in states.items():
        address = norm_hex(state.get("target_address"))
        if address:
            by_addr[address].append((ref, cutoffs[ref]))
    decoder = load_enriched_decoder(decoder_path)
    found: dict[str, tuple[float, dict[str, Any]]] = {}
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        for low, high in merge_intervals(cutoffs.values(), before=lookback_sec, after=0.0):
            q = "SELECT receive_time_ms, outer_json FROM cat062_raw WHERE receive_time_ms BETWEEN ? AND ? ORDER BY receive_time_ms"
            for receive_ms, outer_json in con.execute(q, (low, high)):
                try:
                    outer = json.loads(outer_json)
                    raw_value = outer.get("raw_value")
                    if not isinstance(raw_value, str):
                        continue
                    payload = base64.b64decode("".join(raw_value.split()), validate=True)
                    records = []
                    for block in decoder.iter_cat062_blocks(payload):
                        if block and block[0] == 0x3E:
                            records.extend(decoder.parse_cat062_data_block(block))
                    receive_epoch = float(receive_ms) / 1000.0
                except Exception:
                    continue
                for rec in records:
                    address = norm_hex(rec.get("target_address"))
                    if address not in by_addr:
                        continue
                    # CAT time-of-day may be absent; raw receipt is the conservative fallback.
                    event_epoch = receive_epoch
                    for ref, cutoff in by_addr[address]:
                        if event_epoch >= cutoff or cutoff-event_epoch > lookback_sec:
                            continue
                        old = found.get(ref)
                        if old is None or event_epoch > old[0]:
                            found[ref] = (event_epoch, rec)
    finally:
        con.close()
    result = {}
    keep = {"target_address", "target_identification", "track_number", "selected_altitude_ft", "selected_altitude_source", "final_selected_altitude_ft", "managed_vertical_mode_active", "altitude_hold_active", "approach_mode_active", "mode_s_communications_capability", "mode_s_flight_status", "adsb_acas_status_code", "adsb_navigation_mode_status_code", "adsb_emergency_status_code", "aircraft_track_angle_deg", "aircraft_ground_speed_nm_s", "true_airspeed_kt", "indicated_airspeed_kt", "mach_number", "barometric_pressure_setting_hpa", "barometric_vertical_rate_fpm", "geometric_vertical_rate_fpm", "roll_angle_deg", "turn_indicator_code", "track_angle_rate_degps", "trajectory_intent_status", "trajectory_intent_points", "i390_current_cleared_flight_level", "i390_current_cleared_altitude_m", "i390_sid", "i390_star", "i390_departure_airport", "i390_destination_airport", "i390_runway_designation", "i390_control_centre_code", "i390_control_position_code", "i390_aircraft_type", "i390_wake_turbulence_category", "i390_ifps_flight_id_number", "mode_s_bds_reports", "system_track_update_ages_sec", "track_data_ages_sec", "maximum_system_track_update_age_sec", "maximum_critical_track_data_age_sec", "track_status_coasting", "track_status_flight_plan_coupled", "mode_of_movement", "estimated_position_accuracy_cartesian_m", "estimated_position_covariance_xy_raw", "estimated_position_accuracy_wgs84_raw_hex", "estimated_geometric_altitude_accuracy_m", "estimated_barometric_altitude_accuracy_m", "estimated_velocity_accuracy_mps", "estimated_acceleration_accuracy_mps2", "estimated_rate_of_climb_accuracy_fpm", "last_measurement_sensor", "last_measured_polar_position", "last_measured_3d_height_raw", "last_measured_mode_c", "last_measured_mode3a", "last_measurement_report_type_raw"}
    for ref, (epoch, row) in found.items():
        result[ref] = {"observation_epoch": epoch, "observation_utc": iso_utc(epoch),
                       "observation_age_sec": cutoffs[ref] - epoch,
                       "fields": {k: row.get(k) for k in keep if row.get(k) is not None}}
    return result


def mask_mutable_plan_fields(plan: dict[str, Any] | None, cutoff: float, guard: float) -> tuple[dict[str, Any] | None, list[str]]:
    if plan is None:
        return None, []
    safe = copy.deepcopy(plan)
    masked = []
    fields, field_times = safe["fields"], safe["field_update_epochs"]
    for name in MUTABLE_PLAN_FIELDS:
        age = cutoff - float(field_times.get(name, -1e99))
        if name in fields and age < guard:
            fields.pop(name, None)
            masked.append(name)
    safe["masked_mutable_fields"] = masked
    return safe, masked


def check_label_direction(record: dict[str, Any], state: dict[str, Any]) -> list[str]:
    issues = []
    altitude = float(state.get("altitude_m") or 0)
    rate = float(state.get("vertical_rate_mps") or 0)
    for action in record.get("label", {}).get("actions", []):
        if action.get("intent_family") != "altitude" or action.get("target_value") is None:
            continue
        target = float(action["target_value"])
        if action.get("action") == "descend" and target > altitude + 150 and rate > 1:
            issues.append("altitude_label_direction_conflicts_with_prevoice_kinematics")
        if action.get("action") == "climb" and target < altitude - 150 and rate < -1:
            issues.append("altitude_label_direction_conflicts_with_prevoice_kinematics")
    return list(dict.fromkeys(issues))


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    v4_path = args.v4_dir / "instruction_bundle_training_records_v4.jsonl"
    v2_path = args.v2_dir / "event_state_records.jsonl"
    records = [json.loads(x) for x in v4_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        records = records[:args.limit]
    v2_by_ref = {row["reference_event_id"]: row for row in (json.loads(x) for x in v2_path.read_text(encoding="utf-8").splitlines() if x.strip())}
    identities, cutoffs = {}, {}
    for rec in records:
        ref = (rec.get("member_reference_event_ids") or [None])[0]
        source = v2_by_ref[ref]
        raw = source["pre_command_state_raw"]
        cutoff = parse_segment_start(source["instruction"])
        if cutoff is None:
            raise ValueError(f"no VAD start for {ref}")
        cutoffs[ref] = cutoff - args.cutoff_guard_sec
        identities[ref] = Identity(normalize_callsign(raw.get("callsign")), norm_hex(raw.get("target_address")), norm_ssr(raw.get("mode3a_octal")), str(raw.get("plan_ifplid") or ""), str(raw.get("trajectory_id") or ""))
    states = load_prevoice_states(args.trajectory_parquet, identities, cutoffs)
    plan_index = MultiKeyPlanIndex.load(args.raw_sqlite, list(identities.values()), min(cutoffs.values()), max(cutoffs.values()), config)
    cat = load_cat062_enrichment(args.raw_sqlite, args.base_decoder, states, cutoffs, args.cat_lookback_sec)
    sector_index = SectorIndex(args.sector_xml, config["sector"]["code_prefix"])
    nav_index = load_nav_index(Path(config["route"]["waypoint_file"]), Path(config["route"]["airport_file"]))
    local_airports = set(config["airports"]["local"])
    outputs, audits, route_rows = [], [], []
    counts: Counter[str] = Counter()
    for rec in records:
        out = copy.deepcopy(rec)
        ref = (out.get("member_reference_event_ids") or [None])[0]
        cutoff, identity, state = cutoffs[ref], identities[ref], states.get(ref)
        control = out.setdefault("training_control", {})
        issues = list(control.get("state_quality_issues") or [])
        if state is None:
            issues.append("no_strict_prevoice_trajectory_state")
            control.update(recommended_use="audit_only_prevoice_state_missing", recommended_training_weight=0.0)
            out["schema_version"] = "shanghai_causal_alignment_v5.0"
            control["state_quality_issues"] = list(dict.fromkeys(issues))
            outputs.append(out); counts["no_prevoice_state"] += 1
            continue
        # Identity is refreshed from the selected observation before matching the plan.
        identity = Identity(normalize_callsign(state.get("callsign")) or identity.callsign,
                            norm_hex(state.get("target_address")) or identity.target_address,
                            norm_ssr(state.get("mode3a_octal")) or identity.mode3a,
                            identity.preferred_ifplid, identity.trajectory_id)
        plan_raw, link = plan_index.match_identity(identity, cutoff)
        plan, masked = mask_mutable_plan_fields(plan_raw, cutoff, args.mutable_field_guard_sec)
        aligned_plan, procedure, reported = plan_context(plan, cutoff, local_airports)
        resolved = resolve_route_points([] if plan is None else plan["route_points"], state, nav_index, config)
        route = route_context(resolved, plan, cutoff, config, state)
        if procedure.get("active_procedure_name"):
            procedure.update(current_route_fix=route.get("active_leg_previous_fix"), next_route_fix=route.get("active_leg_next_fix"),
                             progress_source=route.get("active_leg_method"))
        # No SID/STAR chart was delivered with the source. Declaration is retained, activation is unknown.
        procedure["procedure_source_status"] = "plan_declared_only_no_procedure_chart_source"
        procedure["procedure_activity_observed"] = False
        vertical, speed = vertical_speed_context(plan, out["model_input"])
        vertical["electronic_plan_cfl_masked_same_window"] = "CFL" in masked
        vertical["surveillance_final_state_selected_altitude_ft"] = cat.get(ref, {}).get("fields", {}).get("final_selected_altitude_ft")
        vertical["surveillance_selected_altitude_ft"] = cat.get(ref, {}).get("fields", {}).get("selected_altitude_ft")
        vertical["clearance_semantics"] = "CAT062 final-state selected altitude is surveillance intent, not spoken-clearance proof"
        sectors = sector_context(state, sector_index, reported["plan_reported_sector"], config["sector"]["source_sector_code"])
        sectors["dynamic_sector_configuration_available"] = False
        sectors["dynamic_sector_limitation"] = "only static AIXM geometry supplied; opening/combined-sector roster absent"
        mi = out["model_input"]
        replacement_state = {k: state.get(k) for k in ("latitude", "longitude", "altitude_m", "geometric_altitude_m", "track_heading_deg", "vertical_rate_mps", "inside_sector", "sector_hit", "sector_code", "time_from_sector_entry_sec")}
        replacement_state["observed_ground_speed_mps"] = state.get("ground_speed_mps")
        replacement_state["ground_speed_kt"] = None if state.get("ground_speed_mps") is None else float(state["ground_speed_mps"]) * 1.9438444924406
        mi["target_aircraft_state"] = {**mi.get("target_aircraft_state", {}), **replacement_state}
        mi["flight_plan_context"], mi["route_context"], mi["procedure_context"], mi["vertical_context"], mi["speed_context"], mi["sector_context"] = aligned_plan, route, procedure, vertical, speed, sectors
        cat_row = cat.get(ref)
        if cat_row is None:
            mi["surveillance_intent_context"] = {"observation_status": "not_found_within_prevoice_window", "model_feature_mask": False}
        elif float(cat_row["observation_age_sec"]) <= args.cat_feature_max_age_sec:
            mi["surveillance_intent_context"] = {**cat_row, "observation_status": "fresh_prevoice", "model_feature_mask": True}
        else:
            # Keep stale raw evidence outside model_input.  It is useful for
            # audit, but a language model consuming the full input must not
            # mistake a 15--30 s old intent for the current aircraft intent.
            out.setdefault("source_audit", {})["cat062_prevoice_observation"] = cat_row
            mi["surveillance_intent_context"] = {"observation_status": "stale_prevoice_audit_only", "model_feature_mask": False,
                                                   "observation_age_sec": cat_row["observation_age_sec"], "observation_utc": cat_row["observation_utc"]}
        mi["input_coverage"] = coverage(mi)
        direction_issues = check_label_direction(out, state)
        issues.extend(direction_issues)
        if link["status"] in {"unlinked", "ambiguous", "low"}:
            issues.append(f"plan_identity_link_{link['status']}")
        state_age = cutoff - float(state["event_time_epoch"])
        if state_age > args.state_weak_max_age_sec:
            issues.append("prevoice_trajectory_state_stale_gt_weak_threshold")
            control.update(recommended_use="audit_only_prevoice_state_stale", recommended_training_weight=0.0)
        elif state_age > args.state_fresh_max_age_sec:
            issues.append("prevoice_trajectory_state_weak_freshness")
            control["recommended_training_weight"] = min(float(control.get("recommended_training_weight") or 1.0), 0.5)
        if direction_issues:
            control.update(recommended_use="audit_only_label_direction_conflict", recommended_training_weight=0.0)
        elif link["status"] in {"unlinked", "ambiguous"}:
            control.update(recommended_use="weak_train_plan_ambiguous", recommended_training_weight=0.25)
        else:
            control["recommended_training_weight"] = min(float(control.get("recommended_training_weight") or 1.0), 0.75 if masked else 1.0)
        control["state_quality_issues"] = list(dict.fromkeys(issues))
        control["input_cutoff_policy"] = "strictly_before_vad_segment_start_minus_guard"
        control["current_instruction_text_is_model_input"] = False
        out["schema_version"] = "shanghai_causal_alignment_v5.0"
        out.setdefault("provenance", {})["causal_alignment_v5"] = {
            "decision_cutoff_epoch": cutoff, "decision_cutoff_utc": iso_utc(cutoff),
            "selected_prevoice_state_epoch": state["event_time_epoch"], "selected_prevoice_state_utc": iso_utc(float(state["event_time_epoch"])),
            "state_age_before_voice_start_sec": cutoff-float(state["event_time_epoch"]),
            "identity": {"callsign": identity.callsign, "target_address": identity.target_address, "mode3a": identity.mode3a},
            "plan_link": link, "selected_ifplid": None if plan_raw is None else plan_raw["ifplid"],
            "masked_mutable_plan_fields": masked,
        }
        audits.append({"instruction_bundle_id": out["instruction_bundle_id"], "reference_event_id": ref, "callsign": identity.callsign,
                       "cutoff_epoch": cutoff, "prevoice_state_epoch": state["event_time_epoch"], "prevoice_age_sec": cutoff-float(state["event_time_epoch"]),
                       "plan_link_status": link["status"], "plan_link_score": link["score"], "plan_link_evidence": ",".join(link["evidence"]),
                       "selected_ifplid": None if plan_raw is None else plan_raw["ifplid"], "masked_mutable_fields": ",".join(masked),
                       "cat062_observation_available": cat_row is not None,
                       "cat062_model_feature_fresh": bool(cat_row is not None and float(cat_row["observation_age_sec"]) <= args.cat_feature_max_age_sec),
                       "direction_conflict": bool(direction_issues),
                       "recommended_use": control.get("recommended_use")})
        for point in resolved:
            route_rows.append({"instruction_bundle_id": out["instruction_bundle_id"], "reference_event_id": ref, **point})
        counts["records"] += 1; counts[f"plan_link_{link['status']}"] += 1
        counts["cat062_observed_within_lookback"] += int(cat_row is not None)
        counts["cat062_fresh_model_feature"] += int(cat_row is not None and float(cat_row["observation_age_sec"]) <= args.cat_feature_max_age_sec)
        counts["cat062_stale_audit_only"] += int(cat_row is not None and float(cat_row["observation_age_sec"]) > args.cat_feature_max_age_sec)
        counts["state_weak_freshness"] += int(state_age > args.state_fresh_max_age_sec and state_age <= args.state_weak_max_age_sec)
        counts["state_stale_audit_only"] += int(state_age > args.state_weak_max_age_sec)
        counts["masked_cfl"] += int("CFL" in masked)
        counts["label_direction_conflict"] += int(bool(direction_issues))
        outputs.append(out)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "instruction_bundle_training_records_v5.jsonl"
    audit_path = args.output_dir / "alignment_audit_v5.parquet"
    route_path = args.output_dir / "route_points_audit_v5.parquet"
    write_jsonl(out_path, outputs); write_jsonl(args.output_dir / "aligned_samples_first5_v5.jsonl", outputs[:5])
    pd.DataFrame(audits).to_parquet(audit_path, index=False); pd.DataFrame(route_rows).to_parquet(route_path, index=False)
    manifest = {"builder_version": VERSION, "dataset_version": "shanghai_causal_alignment_v5.0", "counts": dict(counts),
                "causal_contract": {"state_strictly_before_vad_start": True, "current_instruction_label_not_model_input": True, "plan_receive_and_filtim_not_after_cutoff": True, "mutable_electronic_fields_guarded": True, "cat062_retrieval_lookback_sec": args.cat_lookback_sec, "cat062_model_feature_max_age_sec": args.cat_feature_max_age_sec, "state_fresh_max_age_sec": args.state_fresh_max_age_sec, "state_weak_max_age_sec": args.state_weak_max_age_sec},
                "semantic_contract": {"CAT062_selected_altitude_not_spoken_clearance": True, "CAT062_final_state_selected_altitude_is_surveillance_intent": True, "SID_STAR_activation_not_claimed_without_chart": True, "dynamic_sector_not_claimed_without_roster": True},
                "sources": {"v4": str(v4_path.resolve()), "v4_sha256": sha256_file(v4_path), "v2": str(v2_path.resolve()), "raw_sqlite": str(args.raw_sqlite.resolve()), "trajectory_parquet": str(args.trajectory_parquet.resolve()), "base_decoder": str(args.base_decoder.resolve())},
                "outputs": {"training_records": out_path.name, "audit": audit_path.name, "route_audit": route_path.name, "samples": "aligned_samples_first5_v5.jsonl"}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
