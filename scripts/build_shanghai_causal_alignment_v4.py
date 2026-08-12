#!/usr/bin/env python3
"""Re-align Shanghai ATC training inputs at each causal instruction cutoff.

This v4 builder is additive: it keeps v3 labels, but replaces model input
flight-plan/route context with a plan replay selected independently for every
instruction bundle.  A plan version is usable only when both its receive time
and FILTIM are no later than the selected pre-command state.  It also aligns
SID/STAR, runway, vertical/speed clearances, and Shanghai approach sector
geometry at the same cutoff.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import datetime as dt
import hashlib
import json
import math
import re
import sqlite3
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from build_shanghai_lossless_training_data_v2 import (
    distance_nm,
    load_nav_index,
    parse_flight_level,
    resolve_navpoint,
    segment_projection_nm,
)
from shanghai_program_policy import parse_planned_speed_kt


VERSION = "shanghai_causal_alignment_builder_v4.0"
FIELD_RE = re.compile(r"^-([A-Z0-9]+)(?:\s+(.*))?$", re.MULTILINE)
RTEPT_RE = re.compile(r"-([A-Z0-9]+)\s+([^\s-]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-dir", type=Path, required=True)
    parser.add_argument("--v2-dir", type=Path, required=True)
    parser.add_argument("--raw-sqlite", type=Path, required=True)
    parser.add_argument("--sector-xml", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_callsign(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def parse_utc_epoch(value: Any, fmt: str | None = None) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = dt.datetime.strptime(text, fmt).replace(tzinfo=dt.timezone.utc) if fmt else dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.timestamp()
    except ValueError:
        return None


def iso_utc(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_filtim_epoch(value: Any, receive_epoch: float) -> float | None:
    """Parse MH4029 FILTIM, which is commonly only HHMMSS in this corpus."""
    text = str(value or "").strip()
    if re.fullmatch(r"\d{14}", text):
        return parse_utc_epoch(text, "%Y%m%d%H%M%S")
    if not re.fullmatch(r"\d{6}", text):
        return None
    received = dt.datetime.fromtimestamp(receive_epoch, dt.timezone.utc)
    clock = dt.datetime.strptime(text, "%H%M%S").time()
    candidates = [
        dt.datetime.combine((received + dt.timedelta(days=offset)).date(), clock, tzinfo=dt.timezone.utc).timestamp()
        for offset in (-1, 0, 1)
    ]
    return min(candidates, key=lambda epoch: abs(epoch-receive_epoch))


def parse_message(raw: str) -> tuple[dict[str, str], list[dict[str, Any]], bool]:
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    fields: dict[str, str] = {}
    points: list[dict[str, Any]] = []
    in_points = False
    points_present = False
    for line in normalized.splitlines():
        stripped = line.strip().lstrip("\ufeff")
        if stripped.startswith("ZCZC"):
            stripped = stripped[4:].lstrip()
        upper = stripped.upper()
        if upper.startswith("-BEGIN RTEPTS"):
            in_points, points_present = True, True
            continue
        if upper.startswith("-END RTEPTS"):
            in_points = False
            continue
        if in_points and upper.startswith("-PT"):
            values = {m.group(1).upper(): m.group(2).strip() for m in RTEPT_RE.finditer(stripped[3:].strip())}
            points.append({
                "ptid": values.get("PTID", ""),
                "sector": values.get("SECT", ""),
                "flight_level": values.get("FL", ""),
                "eto_utc": iso_utc(parse_utc_epoch(values.get("ETO"), "%Y%m%d%H%M%S")),
                "ispass": values.get("ISPASS", ""),
            })
            continue
        if not in_points and stripped.startswith("-"):
            body = stripped[1:]
            name, _, value = body.partition(" ")
            fields[name.upper()] = value.strip()
    return fields, points, points_present


@dataclass
class PlanVersion:
    md5id: str
    receive_epoch: float
    filtim_epoch: float | None
    callsign: str
    ifplid: str
    fields: dict[str, str]
    points: list[dict[str, Any]]
    points_present: bool


class CausalPlanIndex:
    def __init__(self, versions: list[PlanVersion], config: dict[str, Any]):
        self.config = config
        by_ifplid: dict[str, list[PlanVersion]] = defaultdict(list)
        callsign_groups: dict[str, set[str]] = defaultdict(set)
        for version in versions:
            if version.ifplid:
                by_ifplid[version.ifplid].append(version)
                if version.callsign:
                    callsign_groups[version.callsign].add(version.ifplid)
        ifplid_callsign: dict[str, str] = {}
        for ifplid, values in by_ifplid.items():
            names = [v.callsign for v in values if v.callsign]
            if names:
                ifplid_callsign[ifplid] = names[-1]
        for ifplid, callsign in ifplid_callsign.items():
            callsign_groups[callsign].add(ifplid)
        self.by_ifplid = {
            key: sorted(values, key=lambda v: ((v.filtim_epoch or v.receive_epoch), v.receive_epoch))
            for key, values in by_ifplid.items()
        }
        self.callsign_groups = callsign_groups
        self.cache: dict[tuple[str, int, str], dict[str, Any] | None] = {}

    @classmethod
    def load(cls, sqlite_path: Path, callsigns: set[str], minimum_epoch: float, maximum_epoch: float, config: dict[str, Any]) -> "CausalPlanIndex":
        lookback = float(config["flight_plan"]["scan_lookback_hours"]) * 3600
        connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        versions: list[PlanVersion] = []
        try:
            query = "SELECT raw_data_md5id, receive_time_ms, outer_json FROM mh4029_raw WHERE receive_time_ms BETWEEN ? AND ? ORDER BY receive_time_ms"
            for md5id, receive_ms, outer_json in connection.execute(query, (int((minimum_epoch-lookback)*1000), int(maximum_epoch*1000))):
                try:
                    outer = json.loads(outer_json)
                    raw = outer.get("raw_value")
                    if not isinstance(raw, str):
                        continue
                    fields, points, present = parse_message(raw)
                    callsign = normalize_callsign(fields.get("ARCID"))
                    ifplid = str(fields.get("IFPLID") or "")
                    versions.append(PlanVersion(
                        md5id=str(md5id), receive_epoch=float(receive_ms)/1000.0,
                        filtim_epoch=parse_filtim_epoch(fields.get("FILTIM"), float(receive_ms)/1000.0),
                        callsign=callsign, ifplid=ifplid, fields=fields,
                        points=points, points_present=present,
                    ))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        finally:
            connection.close()
        relevant_ifplids = {v.ifplid for v in versions if v.callsign in callsigns and v.ifplid}
        relevant = [v for v in versions if v.ifplid in relevant_ifplids]
        return cls(relevant, config)

    def _replay(self, ifplid: str, cutoff: float) -> dict[str, Any] | None:
        eligible = [v for v in self.by_ifplid.get(ifplid, []) if v.receive_epoch <= cutoff + 1e-6 and (v.filtim_epoch is None or v.filtim_epoch <= cutoff + 1e-6)]
        if not eligible:
            return None
        fields: dict[str, str] = {}
        field_times: dict[str, float] = {}
        points: list[dict[str, Any]] = []
        last = eligible[-1]
        point_version = None
        for version in eligible:
            fields.update(version.fields)
            stamp = version.filtim_epoch or version.receive_epoch
            field_times.update({key: stamp for key in version.fields})
            if version.points_present:
                points = version.points
                point_version = version
        eobt = None
        if fields.get("EOBD") and fields.get("EOBT"):
            eobt = parse_utc_epoch(fields["EOBD"] + fields["EOBT"], "%Y%m%d%H%M")
        return {
            "ifplid": ifplid,
            "fields": fields,
            "field_update_epochs": field_times,
            "route_points": points,
            "selected_md5id": last.md5id,
            "selected_receive_epoch": last.receive_epoch,
            "selected_filtim_epoch": last.filtim_epoch,
            "effective_filtim_epoch": last.filtim_epoch,
            "route_receive_epoch": None if point_version is None else point_version.receive_epoch,
            "route_filtim_epoch": None if point_version is None else point_version.filtim_epoch,
            "version_count_replayed": len(eligible),
            "eobt_epoch": eobt,
        }

    def match(self, callsign: Any, cutoff: float, preferred_ifplid: Any = None) -> dict[str, Any] | None:
        name = normalize_callsign(callsign)
        preferred = str(preferred_ifplid or "")
        key = (name, int(round(cutoff * 1000)), preferred)
        if key in self.cache:
            return self.cache[key]
        groups = list(self.callsign_groups.get(name, set()))
        candidates = [item for group in groups if (item := self._replay(group, cutoff)) is not None]
        max_distance = float(self.config["flight_plan"]["maximum_eobt_distance_hours"]) * 3600
        max_future = float(self.config["flight_plan"]["maximum_future_eobt_hours"]) * 3600
        filtered = [c for c in candidates if c["eobt_epoch"] is None or (-max_future <= cutoff-c["eobt_epoch"] <= max_distance)]
        candidates = filtered or candidates
        selected = None
        if preferred and self.config["flight_plan"].get("prefer_source_ifplid", True):
            selected = next((c for c in candidates if c["ifplid"] == preferred), None)
        if selected is None and candidates:
            selected = max(candidates, key=lambda c: (c["selected_filtim_epoch"] or c["selected_receive_epoch"], c["selected_receive_epoch"]))
        self.cache[key] = selected
        return selected


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def point_in_polygon(lon: float, lat: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        on_edge = abs((lat-yi)*(xj-xi) - (lon-xi)*(yj-yi)) < 1e-9 and min(xi,xj)-1e-9 <= lon <= max(xi,xj)+1e-9 and min(yi,yj)-1e-9 <= lat <= max(yi,yj)+1e-9
        if on_edge:
            return True
        if ((yi > lat) != (yj > lat)) and lon < (xj-xi)*(lat-yi)/(yj-yi) + xi:
            inside = not inside
        j = i
    return inside


class SectorIndex:
    def __init__(self, xml_path: Path, prefix: str):
        root = ET.parse(xml_path).getroot()
        self.sectors: list[dict[str, Any]] = []
        for sector in root.iter():
            if local_name(sector.tag) != "sector" or not str(sector.get("code") or "").startswith(prefix):
                continue
            item = {"code": sector.get("code"), "name": sector.get("name"), "volumes": []}
            for volume in sector:
                if local_name(volume.tag) != "volume":
                    continue
                polygons = []
                for polygon in volume:
                    if local_name(polygon.tag) != "polygon":
                        continue
                    points = [(float(p.get("d_longitude")), float(p.get("d_latitude"))) for p in polygon if local_name(p.tag) == "point" and p.get("d_longitude") and p.get("d_latitude")]
                    if len(points) >= 3:
                        polygons.append(points)
                item["volumes"].append({"lower": float(volume.get("lower") or -math.inf), "upper": float(volume.get("upper") or math.inf), "polygons": polygons})
            self.sectors.append(item)

    def match(self, latitude: float, longitude: float, altitude_m: float) -> list[dict[str, Any]]:
        result = []
        for sector in self.sectors:
            for volume in sector["volumes"]:
                if volume["lower"] <= altitude_m <= volume["upper"] and any(point_in_polygon(longitude, latitude, poly) for poly in volume["polygons"]):
                    result.append({"code": sector["code"], "name": sector["name"], "lower_m": volume["lower"], "upper_m": volume["upper"]})
                    break
        return result


def parse_level_context(value: Any) -> dict[str, Any]:
    parsed = parse_flight_level(value)
    return {"raw_code": parsed["raw"], "value_m": parsed["value_m"], "source_unit": parsed["source_unit"], "parse_status": parsed["parse_status"]}


def resolve_route_points(points: list[dict[str, Any]], state: dict[str, Any], nav_index: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    epoch = float(state["event_time_epoch"])
    lat, lon = float(state["latitude"]), float(state["longitude"])
    maximum = float(config["route"]["maximum_navpoint_distance_nm"])
    result = []
    for index, point in enumerate(points):
        eto_epoch = parse_utc_epoch(point.get("eto_utc"))
        level = parse_level_context(point.get("flight_level"))
        rlat, rlon, source = resolve_navpoint(point.get("ptid"), nav_index, lat, lon)
        if rlat is not None and distance_nm(lat, lon, rlat, rlon) > maximum:
            rlat, rlon, source = None, None, None
        result.append({
            "route_point_index": index, "ptid_raw": point.get("ptid"),
            "sector_raw": point.get("sector"), "flight_level_raw": point.get("flight_level"),
            "planned_level_m": level["value_m"], "planned_level_source_unit": level["source_unit"],
            "planned_level_parse_status": level["parse_status"], "eto_utc_raw": point.get("eto_utc"),
            "eto_epoch": eto_epoch, "eto_delta_sec": None if eto_epoch is None else eto_epoch-epoch,
            "ispass_raw": point.get("ispass"), "resolved_latitude": rlat,
            "resolved_longitude": rlon, "coordinate_source": source,
        })
    return result


def route_context(points: list[dict[str, Any]], plan: dict[str, Any] | None, cutoff: float, config: dict[str, Any], state: dict[str, Any] | None = None) -> dict[str, Any]:
    base = {
        "route_feature_mask": False, "route_context_confidence": "none",
        "active_leg_method": "no_causal_plan" if plan is None else "no_route_points",
        "active_leg_previous_fix": None, "active_leg_next_fix": None, "active_leg": None,
        "active_leg_cross_track_nm": None, "active_leg_fraction": None,
        "distance_to_next_fix_nm": None, "active_leg_length_nm": None,
        "time_to_next_fix_sec": None, "raw_eto_delta_to_next_fix_sec": None,
        "next_route_point_planned_level_m": None, "next_route_point_planned_level_source_unit": None,
        "next_route_point_planned_level_parse_status": None, "next_route_point_sector": None,
        "route_point_count": len(points), "ispass_eto_conflict": False,
        "plan_snapshot_age_sec": None if plan is None else cutoff-(plan.get("route_receive_epoch") or plan["selected_receive_epoch"]),
        "plan_information_age_sec": None if plan is None else cutoff-plan["selected_receive_epoch"],
        "plan_filtim_age_sec": None if plan is None or plan.get("effective_filtim_epoch") is None else cutoff-plan["effective_filtim_epoch"],
    }
    if not points:
        return base
    tolerance = float(config["route"]["at_fix_tolerance_sec"])
    timed = [(i, p["eto_delta_sec"]) for i, p in enumerate(points) if p["eto_delta_sec"] is not None]
    next_index = next((i for i, delta in timed if delta >= -tolerance), None)
    if next_index is None:
        base["active_leg_method"] = "route_eto_all_past"
        return base
    # Never retain a materially negative next-fix time. Advance to the next
    # route point; at-fix values inside tolerance are represented as zero.
    while next_index < len(points)-1 and points[next_index]["eto_delta_sec"] is not None and points[next_index]["eto_delta_sec"] < -tolerance:
        next_index += 1
    previous_index = next_index - 1 if next_index > 0 else None
    nxt = points[next_index]
    previous = None if previous_index is None else points[previous_index]
    delta = nxt["eto_delta_sec"]
    confidence = "medium"
    geometry_conflict = False
    geometry_available = False
    cross_track = fraction = distance_to_next = leg_length = None
    if previous is not None and state is not None and all(
        value is not None for value in (
            previous.get("resolved_latitude"), previous.get("resolved_longitude"),
            nxt.get("resolved_latitude"), nxt.get("resolved_longitude"),
            state.get("latitude"), state.get("longitude"),
        )
    ):
        geometry_available = True
        leg_length = distance_nm(previous["resolved_latitude"], previous["resolved_longitude"], nxt["resolved_latitude"], nxt["resolved_longitude"])
        cross_track, fraction, distance_to_next = segment_projection_nm(
            float(state["latitude"]), float(state["longitude"]),
            previous["resolved_latitude"], previous["resolved_longitude"],
            nxt["resolved_latitude"], nxt["resolved_longitude"],
        )
        geometry_conflict = (
            leg_length > float(config["route"]["maximum_geometric_leg_length_nm"])
            or cross_track > float(config["route"]["maximum_geometric_cross_track_nm"])
        )
        confidence = "low" if geometry_conflict else "high"
    unbracketed_far = previous is None and delta is not None and delta > float(config["route"]["maximum_unbracketed_next_fix_sec"])
    feature_mask = not geometry_conflict and not unbracketed_far
    method = "eto_geometry_conflict" if geometry_conflict else "eto_not_bracketed" if unbracketed_far else "causal_plan_eto_bracket"
    if unbracketed_far:
        confidence = "low"
    base.update({
        "route_feature_mask": feature_mask,
        "route_context_confidence": confidence,
        "active_leg_method": method,
        "active_leg_previous_fix": None if previous is None else previous.get("ptid_raw"),
        "active_leg_next_fix": nxt.get("ptid_raw"),
        "active_leg": f"{previous.get('ptid_raw')}->{nxt.get('ptid_raw')}" if previous is not None else f"BEFORE->{nxt.get('ptid_raw')}",
        "time_to_next_fix_sec": None if delta is None else max(0.0, delta),
        "raw_eto_delta_to_next_fix_sec": delta,
        "next_route_point_planned_level_m": nxt.get("planned_level_m"),
        "next_route_point_planned_level_source_unit": nxt.get("planned_level_source_unit"),
        "next_route_point_planned_level_parse_status": nxt.get("planned_level_parse_status"),
        "next_route_point_sector": nxt.get("sector_raw") or None,
        "active_leg_cross_track_nm": cross_track,
        "active_leg_fraction": fraction,
        "distance_to_next_fix_nm": distance_to_next,
        "active_leg_length_nm": leg_length,
        "geometry_cross_check_available": geometry_available,
        "geometry_eto_conflict": geometry_conflict,
    })
    eto_passed = {p["route_point_index"] for p in points if p.get("eto_delta_sec") is not None and p["eto_delta_sec"] < -tolerance}
    ispass_passed = {p["route_point_index"] for p in points if str(p.get("ispass_raw") or "").upper() == "Y"}
    base["ispass_eto_conflict"] = eto_passed != ispass_passed
    return base


def flight_phase(fields: dict[str, str], local_airports: set[str]) -> str:
    dep, arr = fields.get("ADEP", ""), fields.get("ADES", "")
    if dep in local_airports and arr in local_airports:
        return "local_transfer"
    if dep in local_airports:
        return "departure"
    if arr in local_airports:
        return "arrival"
    return "overflight"


def plan_context(plan: dict[str, Any] | None, cutoff: float, local_airports: set[str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if plan is None:
        empty_plan = {"plan_features_causally_available": False, "plan_ifplid": None, "plan_adep": None, "plan_ades": None, "aircraft_type": None, "wake_turbulence_category": None, "sid": None, "star": None, "departure_runway": None, "arrival_runway": None, "requested_level_m": None, "planned_speed_kt": None, "selected_plan_filtim_utc": None, "selected_plan_receive_time_utc": None, "version_count_replayed": 0}
        return empty_plan, {"flight_phase": "unknown", "active_procedure_type": None, "active_procedure_name": None, "procedure_alignment_status": "no_causal_plan"}, {"plan_reported_sector": None}
    f = plan["fields"]
    requested = parse_level_context(f.get("RFL"))
    speed = parse_planned_speed_kt(f.get("SPEED"))
    speed = None if not math.isfinite(speed) else speed
    phase = flight_phase(f, local_airports)
    sid, star = f.get("SID") or None, f.get("STAR") or None
    active_type = "SID" if phase == "departure" and sid else "STAR" if phase == "arrival" and star else None
    active_name = sid if active_type == "SID" else star if active_type == "STAR" else None
    procedure_status = "declared_and_phase_consistent" if active_type else "not_declared_or_not_applicable"
    context = {
        "plan_features_causally_available": True, "plan_ifplid": plan["ifplid"],
        "plan_adep": f.get("ADEP") or None, "plan_ades": f.get("ADES") or None,
        "aircraft_type": f.get("ARCTYP") or None, "wake_turbulence_category": f.get("WKTRC") or None,
        "sid": sid, "star": star, "departure_runway": f.get("DRWY") or None,
        "arrival_runway": f.get("ARWY") or None, "requested_level_m": requested["value_m"],
        "requested_level_source_unit": requested["source_unit"], "requested_level_parse_status": requested["parse_status"],
        "planned_speed_kt": speed, "planned_speed_raw": f.get("SPEED") or None,
        "selected_plan_filtim_utc": iso_utc(plan["selected_filtim_epoch"]),
        "effective_plan_filtim_utc": iso_utc(plan.get("effective_filtim_epoch")),
        "selected_plan_receive_time_utc": iso_utc(plan["selected_receive_epoch"]),
        "plan_information_age_at_cutoff_sec": cutoff-plan["selected_receive_epoch"],
        "plan_filtim_age_at_cutoff_sec": None if plan.get("effective_filtim_epoch") is None else cutoff-plan["effective_filtim_epoch"],
        "version_count_replayed": plan["version_count_replayed"],
        "field_update_times_utc": {
            name: iso_utc(plan["field_update_epochs"].get(name))
            for name in ("ADEP", "ADES", "ARCTYP", "CFL", "XFL", "RFL", "SPEED", "SID", "STAR", "DRWY", "ARWY", "SECTOR", "ROUTE")
            if name in plan["field_update_epochs"]
        },
    }
    procedure = {
        "flight_phase": phase, "flight_direction": phase,
        "declared_procedure_type": active_type, "declared_procedure_name": active_name,
        # Kept for backward compatibility. The confidence fields below make
        # clear this is inferred relevance, not an observed procedure clearance.
        "active_procedure_type": active_type, "active_procedure_name": active_name,
        "sid": sid, "star": star, "departure_runway": context["departure_runway"],
        "arrival_runway": context["arrival_runway"], "procedure_alignment_status": procedure_status,
        "procedure_activity_observed": False,
        "procedure_relevance_confidence": "medium" if active_type else "none",
        "procedure_relevance_basis": "declared_plan_and_local_flight_direction" if active_type else None,
    }
    return context, procedure, {"plan_reported_sector": f.get("SECTOR") or None}


def vertical_speed_context(plan: dict[str, Any] | None, model_input: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    fields = {} if plan is None else plan["fields"]
    history = model_input.get("active_clearance_context") or {}
    prior_alt = history.get("active_altitude_constraint")
    prior_speed = history.get("active_speed_constraint")
    cfl, xfl, rfl = (parse_level_context(fields.get(name)) for name in ("CFL", "XFL", "RFL"))
    effective_level = prior_alt.get("target_value") if prior_alt else cfl["value_m"]
    effective_source = "prior_controller_instruction" if prior_alt else "causal_flight_plan_cfl" if cfl["value_m"] is not None else None
    state = model_input.get("target_aircraft_state") or {}
    vertical = {
        "observed_altitude_m": state.get("altitude_m"), "observed_vertical_rate_mps": state.get("vertical_rate_mps"),
        "prior_controller_altitude_clearance": prior_alt, "causal_plan_cleared_level": cfl,
        "causal_plan_cleared_level_update_utc": None if plan is None else iso_utc(plan["field_update_epochs"].get("CFL")),
        "causal_plan_exit_level": xfl, "requested_cruise_level": rfl,
        "effective_prior_cleared_level_m": effective_level, "effective_prior_cleared_level_source": effective_source,
        "next_controller_target_level_m": None,
        "label_separation_note": "next controller target is label-only and is not present in model input",
    }
    speed = {
        "observed_ground_speed_kt": state.get("ground_speed_kt"), "observed_speed_basis": "ground_speed",
        "prior_controller_speed_constraint": prior_speed,
        "planned_speed_kt": None if plan is None else parse_planned_speed_kt(fields.get("SPEED")),
        "planned_speed_raw": fields.get("SPEED") or None,
        "planned_speed_update_utc": None if plan is None else iso_utc(plan["field_update_epochs"].get("SPEED")),
        "commanded_speed_observation_comparable": False,
        "comparability_note": "ground speed is not treated as indicated airspeed",
    }
    if isinstance(speed["planned_speed_kt"], float) and not math.isfinite(speed["planned_speed_kt"]):
        speed["planned_speed_kt"] = None
    return vertical, speed


def normalized_plan_sector(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    if text.startswith("ZSSS"):
        return text
    match = re.fullmatch(r"APP?(\d{2})(\([NS]\))?", text)
    if match:
        return "ZSSSAP" + match.group(1) + (match.group(2) or "")
    return text


def sector_context(state: dict[str, Any], sector_index: SectorIndex, reported: Any, source_sector: str) -> dict[str, Any]:
    matches = sector_index.match(float(state["latitude"]), float(state["longitude"]), float(state["altitude_m"]))
    codes = [m["code"] for m in matches]
    plan_code = normalized_plan_sector(reported)
    if plan_code in codes:
        primary, method, status = plan_code, "plan_and_geometry", "consistent"
    elif len(codes) == 1:
        primary, method, status = codes[0], "geometry", "plan_missing" if plan_code is None else "plan_geometry_conflict"
    elif codes:
        primary, method, status = codes[0], "geometry_ambiguous", "ambiguous_overlap"
    else:
        primary, method, status = plan_code, "plan_only" if plan_code else "unavailable", "outside_loaded_approach_sectors"
    return {
        "source_extraction_sector": source_sector, "source_inside_sector_flag": state.get("inside_sector"),
        "geometry_sector_candidates": matches, "primary_sector_code": primary,
        "primary_sector_selection_method": method, "plan_reported_sector": reported,
        "normalized_plan_reported_sector": plan_code, "sector_alignment_status": status,
        "inside_source_sector_recomputed": source_sector in codes,
    }


def coverage(model_input: dict[str, Any]) -> dict[str, Any]:
    groups = {
        "kinematic_state": bool(model_input.get("target_aircraft_state")),
        "causal_flight_plan": bool(model_input["flight_plan_context"].get("plan_features_causally_available")),
        "usable_route_progress": bool(model_input["route_context"].get("route_feature_mask")),
        "procedure_context": model_input["procedure_context"].get("flight_phase") in {"overflight", "local_transfer"} or bool(model_input["procedure_context"].get("active_procedure_name")),
        "vertical_context": model_input["vertical_context"].get("effective_prior_cleared_level_m") is not None,
        "speed_context": model_input["speed_context"].get("prior_controller_speed_constraint") is not None or model_input["speed_context"].get("planned_speed_kt") is not None,
        "sector_context": model_input["sector_context"].get("primary_sector_code") is not None,
        "causal_traffic": bool(model_input.get("traffic_context")),
    }
    available = sum(groups.values())
    return {"context_groups": groups, "available_context_groups": available, "total_context_groups": len(groups), "missing_context_groups": [k for k,v in groups.items() if not v], "input_coverage_tier": "high" if available >= 7 else "medium" if available >= 5 else "low", "not_a_decision_sufficiency_label": True}


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    v3_path = args.v3_dir / "instruction_bundle_training_records_v3.jsonl"
    records = []
    with v3_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if args.limit and len(records) >= args.limit:
                    break
    v2_by_ref = {}
    with (args.v2_dir / "event_state_records.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                v2_by_ref[row["reference_event_id"]] = row
    cutoffs = [
        float(v2_by_ref[(r.get("member_reference_event_ids") or [None])[0]]["pre_command_state_raw"]["event_time_epoch"])
        for r in records
    ]
    callsigns = {normalize_callsign(r["provenance"].get("callsign")) for r in records}
    for record in records:
        callsigns.update(normalize_callsign(x.get("other_callsign")) for x in record["model_input"].get("traffic_context", []))
    plan_index = CausalPlanIndex.load(args.raw_sqlite, callsigns, min(cutoffs), max(cutoffs), config)
    sector_index = SectorIndex(args.sector_xml, config["sector"]["code_prefix"])
    nav_index = load_nav_index(Path(config["route"]["waypoint_file"]), Path(config["route"]["airport_file"]))
    local_airports = set(config["airports"]["local"])
    output_records, audit_rows, route_rows = [], [], []
    counters: Counter[str] = Counter()
    for record in records:
        result = copy.deepcopy(record)
        member_refs = result.get("member_reference_event_ids") or []
        reference = member_refs[0]
        v2 = v2_by_ref[reference]
        raw_state = v2["pre_command_state_raw"]
        cutoff = float(raw_state["event_time_epoch"])
        preferred_ifplid = raw_state.get("plan_ifplid")
        old_plan_available = bool(result["model_input"].get("flight_plan_context", {}).get("plan_features_causally_available"))
        plan = plan_index.match(result["provenance"].get("callsign"), cutoff, preferred_ifplid)
        aligned_plan, procedure, reported = plan_context(plan, cutoff, local_airports)
        resolved_points = resolve_route_points([] if plan is None else plan["route_points"], raw_state, nav_index, config)
        route = route_context(resolved_points, plan, cutoff, config, raw_state)
        if procedure.get("active_procedure_name"):
            procedure["current_route_fix"] = route.get("active_leg_previous_fix")
            procedure["next_route_fix"] = route.get("active_leg_next_fix")
            procedure["progress_source"] = "causal_flight_plan_eto_bracket"
        vertical, speed = vertical_speed_context(plan, result["model_input"])
        sectors = sector_context(raw_state, sector_index, reported["plan_reported_sector"], config["sector"]["source_sector_code"])
        model_input = result["model_input"]
        model_input["flight_plan_context"] = aligned_plan
        model_input["route_context"] = route
        model_input["procedure_context"] = procedure
        model_input["vertical_context"] = vertical
        model_input["speed_context"] = speed
        model_input["sector_context"] = sectors
        for relation in model_input.get("traffic_context", []):
            other_epoch = float(relation.get("other_event_time_epoch") or cutoff)
            other_plan = plan_index.match(relation.get("other_callsign"), other_epoch)
            other_context, other_proc, other_reported = plan_context(other_plan, other_epoch, local_airports)
            other_state = {"latitude": relation.get("other_latitude"), "longitude": relation.get("other_longitude"), "altitude_m": relation.get("other_altitude_m"), "inside_sector": relation.get("other_inside_sector")}
            relation["other_plan_causally_available"] = other_context["plan_features_causally_available"]
            relation["other_plan_ifplid"] = other_context.get("plan_ifplid")
            relation["other_plan_receive_time_utc"] = other_context.get("selected_plan_receive_time_utc")
            relation["other_plan_filtim_utc"] = other_context.get("effective_plan_filtim_utc")
            relation["other_plan_information_age_sec"] = other_context.get("plan_information_age_at_cutoff_sec")
            relation["other_plan_adep"] = other_context.get("plan_adep")
            relation["other_plan_ades"] = other_context.get("plan_ades")
            relation["other_sid"] = other_context.get("sid")
            relation["other_star"] = other_context.get("star")
            relation["other_departure_runway"] = other_context.get("departure_runway")
            relation["other_arrival_runway"] = other_context.get("arrival_runway")
            relation["other_flight_phase"] = other_proc.get("flight_phase")
            relation["other_primary_sector_code"] = sector_context(other_state, sector_index, other_reported["plan_reported_sector"], config["sector"]["source_sector_code"])["primary_sector_code"]
            relation["same_sid"] = bool(aligned_plan.get("sid") and aligned_plan.get("sid") == relation.get("other_sid"))
            relation["same_star"] = bool(aligned_plan.get("star") and aligned_plan.get("star") == relation.get("other_star"))
            relation["same_departure_runway"] = bool(aligned_plan.get("departure_runway") and aligned_plan.get("departure_runway") == relation.get("other_departure_runway"))
            relation["same_arrival_runway"] = bool(aligned_plan.get("arrival_runway") and aligned_plan.get("arrival_runway") == relation.get("other_arrival_runway"))
        model_input["input_coverage"] = coverage(model_input)
        result["schema_version"] = config["dataset_version"]
        legacy_issues = list(result["training_control"].get("state_quality_issues") or [])
        new_issues = [x for x in legacy_issues if x not in {"plan_not_available_at_pre_state", "plan_update_after_pre_state"}]
        if plan is None:
            new_issues.append("no_causal_plan_v4")
        if not route["route_feature_mask"]:
            new_issues.append("route_progress_unavailable_v4")
        if sectors["sector_alignment_status"] == "plan_geometry_conflict":
            new_issues.append("sector_plan_geometry_conflict_v4")
        result["training_control"]["legacy_state_quality_issues_v3"] = legacy_issues
        result["training_control"]["state_quality_issues"] = list(dict.fromkeys(new_issues))
        result["training_control"]["input_cutoff_policy"] = config["input_cutoff_policy"]
        result["provenance"]["causal_alignment"] = {
            "input_cutoff_epoch": cutoff, "input_cutoff_utc": iso_utc(cutoff),
            "preferred_source_ifplid": preferred_ifplid,
            "selected_ifplid": None if plan is None else plan["ifplid"],
            "selected_plan_md5id": None if plan is None else plan["selected_md5id"],
        }
        recovered = not old_plan_available and plan is not None
        changed = plan is not None and str(preferred_ifplid or "") != str(plan["ifplid"] or "")
        audit_rows.append({
            "instruction_bundle_id": result["instruction_bundle_id"], "reference_event_id": reference,
            "callsign": result["provenance"].get("callsign"), "cutoff_epoch": cutoff,
            "old_plan_available": old_plan_available, "causal_plan_available_v4": plan is not None,
            "plan_recovered": recovered, "preferred_source_ifplid": preferred_ifplid,
            "selected_ifplid": None if plan is None else plan["ifplid"], "selected_ifplid_changed": changed,
            "selected_plan_age_sec": route["plan_snapshot_age_sec"],
            "route_available_v4": route["route_feature_mask"], "route_method_v4": route["active_leg_method"],
            "previous_fix_v4": route["active_leg_previous_fix"], "next_fix_v4": route["active_leg_next_fix"],
            "time_to_next_fix_sec_v4": route["time_to_next_fix_sec"],
            "next_route_point_planned_level_m_v4": route["next_route_point_planned_level_m"],
            "flight_phase_v4": procedure["flight_phase"], "active_procedure_v4": procedure.get("active_procedure_name"),
            "effective_prior_cleared_level_m_v4": vertical["effective_prior_cleared_level_m"],
            "primary_sector_v4": sectors["primary_sector_code"], "sector_alignment_status_v4": sectors["sector_alignment_status"],
            "input_coverage_tier_v4": model_input["input_coverage"]["input_coverage_tier"],
        })
        for point in resolved_points:
            route_rows.append({"instruction_bundle_id": result["instruction_bundle_id"], "reference_event_id": reference, **point})
        counters["records"] += 1
        counters["causal_plan_available"] += int(plan is not None)
        counters["plan_recovered"] += int(recovered)
        counters["route_available"] += int(route["route_feature_mask"])
        counters[f"coverage_{model_input['input_coverage']['input_coverage_tier']}"] += 1
        counters[f"phase_{procedure['flight_phase']}"] += 1
        counters[f"sector_status_{sectors['sector_alignment_status']}"] += 1
        output_records.append(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "instruction_bundle_training_records_v4.jsonl"
    audit_path = args.output_dir / "alignment_audit_v4.parquet"
    route_path = args.output_dir / "route_points_audit_v4.parquet"
    sample_path = args.output_dir / "aligned_samples_first5_v4.jsonl"
    write_jsonl(output_path, output_records)
    write_jsonl(sample_path, output_records[:5])
    pd.DataFrame(audit_rows).to_parquet(audit_path, index=False)
    pd.DataFrame(route_rows).to_parquet(route_path, index=False)
    manifest = {
        "builder_version": VERSION, "dataset_version": config["dataset_version"],
        "counts": dict(counters),
        "causal_contract": {"input_cutoff": config["input_cutoff_policy"], "plan_receive_time_not_after_cutoff": True, "plan_filtim_not_after_cutoff": True, "current_instruction_label_not_in_model_input": True},
        "semantic_contract": {"next_route_point_planned_level_is_not_controller_target": True, "ground_speed_is_not_commanded_ias": True, "sector_is_recomputed_from_position_altitude": True, "other_aircraft_plans_are_realigned": True},
        "sources": {"v3": str(v3_path.resolve()), "v3_sha256": sha256_file(v3_path), "v2": str((args.v2_dir/'event_state_records.jsonl').resolve()), "raw_sqlite": str(args.raw_sqlite.resolve()), "sector_xml": str(args.sector_xml.resolve()), "config": str(args.config.resolve())},
        "outputs": {"training_records": output_path.name, "alignment_audit": audit_path.name, "route_audit": route_path.name, "sample_records": sample_path.name},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
