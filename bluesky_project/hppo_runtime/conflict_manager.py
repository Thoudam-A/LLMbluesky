from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, radians, sin
from typing import Dict, List

import numpy as np

from .config import HPPOConfig


@dataclass(slots=True)
class ConflictContact:
    callsign: str
    horiz_km: float
    vert_ft: float
    # Predicted separation at the closest point of approach.  Keeping these
    # values with the contact makes runtime interval-management results
    # inspectable without changing the control policy.
    dcpa_km: float
    predicted_vert_ft: float
    rel_bearing_deg: float
    rel_bearing_sin: float
    rel_bearing_cos: float
    rel_alt_norm: float
    rel_speed_norm: float
    rel_track_sin: float
    rel_track_cos: float
    tcpa_s: float
    time_gap_s: float
    temporal_conflict: float
    conflict_flag: float
    severity: float


@dataclass(slots=True)
class AircraftConflictState:
    callsign: str
    active: bool = False
    severity: float = 0.0
    allow_resume: bool = False
    targets: list[ConflictContact] = field(default_factory=list)
    control_candidate: bool = False
    loss_of_separation: bool = False
    predicted_conflict: bool = False
    temporal_conflict: bool = False
    collision: bool = False
    danger: bool = False
    min_time_gap_s: float = float("inf")


class ConflictManager:
    def __init__(self, config: HPPOConfig):
        self.cfg = config

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2.0) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2.0) ** 2
        return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(max(1.0 - a, 1e-12)))

    @staticmethod
    def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        lat1_r = radians(lat1)
        lat2_r = radians(lat2)
        dlon = radians(lon2 - lon1)
        y = sin(dlon) * cos(lat2_r)
        x = cos(lat1_r) * sin(lat2_r) - sin(lat1_r) * cos(lat2_r) * cos(dlon)
        return (np.degrees(atan2(y, x)) + 360.0) % 360.0

    @staticmethod
    def _wrap_hdg_diff(a: float, b: float) -> float:
        diff = (a - b + 180.0) % 360.0 - 180.0
        return diff

    @staticmethod
    def _penetration(value: float, outer: float, inner: float) -> float:
        return float(np.clip((outer - value) / max(outer - inner, 1e-6), 0.0, 1.0))

    def _current_severity(self, horiz_km: float, vert_ft: float) -> float:
        separation = self.cfg.separation
        horizontal = self._penetration(
            horiz_km, separation.horizontal_km, self.cfg.collision_horizontal_km
        )
        if separation.mode == "horizontal":
            return horizontal
        vertical = self._penetration(
            vert_ft, separation.vertical_ft, self.cfg.collision_vertical_ft
        )
        return float(np.sqrt(horizontal * vertical))

    def _predicted_severity(self, dcpa_km: float, predicted_vert_ft: float, tcpa_s: float) -> float:
        separation = self.cfg.separation
        horizontal = self._penetration(
            dcpa_km, separation.horizontal_km, self.cfg.collision_horizontal_km
        )
        urgency = float(np.clip(1.0 - tcpa_s / max(self.cfg.conflict_lookahead_s, 1e-6), 0.0, 1.0))
        if separation.mode == "horizontal":
            return horizontal * (0.25 + 0.75 * urgency)
        vertical = self._penetration(
            predicted_vert_ft, separation.vertical_ft, self.cfg.collision_vertical_ft
        )
        spatial = float(np.sqrt(horizontal * vertical))
        return spatial * (0.25 + 0.75 * urgency)

    def _trajectory_time_gap(
        self,
        rel_pos: np.ndarray,
        own_vel: np.ndarray,
        other_vel: np.ndarray,
        own_alt_ft: float,
        other_alt_ft: float,
        own_vs_fps: float,
        other_vs_fps: float,
    ) -> tuple[float, bool]:
        matrix = np.column_stack((own_vel, -other_vel)).astype(np.float64)
        if abs(float(np.linalg.det(matrix))) <= 1e-9:
            return float("inf"), False
        own_time, other_time = np.linalg.solve(matrix, rel_pos.astype(np.float64))
        lookahead = self.cfg.conflict_lookahead_s
        if not (0.0 <= own_time <= lookahead and 0.0 <= other_time <= lookahead):
            return float("inf"), False
        own_crossing_alt = own_alt_ft + own_vs_fps * float(own_time)
        other_crossing_alt = other_alt_ft + other_vs_fps * float(other_time)
        altitude_compatible = abs(own_crossing_alt - other_crossing_alt) <= self.cfg.separation.vertical_ft
        return abs(float(own_time - other_time)), bool(altitude_compatible)

    @staticmethod
    def _to_xy_km(lat_ref: float, lon_ref: float, lat: float, lon: float) -> np.ndarray:
        mean_lat = np.radians((lat_ref + lat) / 2.0)
        return np.array(
            [
                (lon - lon_ref) * 111.32 * np.cos(mean_lat),
                (lat - lat_ref) * 110.57,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _velocity_kmps(track_deg: float, speed_kt: float) -> np.ndarray:
        speed_kmps = speed_kt * 1.852 / 3600.0
        track_rad = np.radians(track_deg)
        return np.array(
            [
                speed_kmps * np.sin(track_rad),
                speed_kmps * np.cos(track_rad),
            ],
            dtype=np.float32,
        )

    def detect(self, traf, max_aircraft: int) -> Dict[str, AircraftConflictState]:
        separation = self.cfg.separation
        spatial_enabled = separation.mode in {"horizontal", "spatial", "combined"}
        horizontal_only = separation.mode == "horizontal"
        temporal_enabled = separation.mode in {"temporal", "combined"}
        ids = list(traf.id)
        states = {
            acid: AircraftConflictState(callsign=acid)
            for acid in ids
        }
        for i, acid in enumerate(ids):
            if acid not in states:
                continue
            lat_i = float(traf.lat[i])
            lon_i = float(traf.lon[i])
            alt_i = float(traf.alt[i] * 3.28084)
            spd_i = float(traf.cas[i] * 1.94384)
            trk_i = float(traf.trk[i])
            pos_i = np.array([0.0, 0.0], dtype=np.float32)
            vel_i = self._velocity_kmps(trk_i, spd_i)
            contacts: List[ConflictContact] = []
            severity = 0.0
            active_loss = False
            predicted_conflict = False
            temporal_conflict = False
            collision = False
            danger = False
            allow_resume = True
            min_time_gap = float("inf")
            for j, other in enumerate(ids):
                if i == j:
                    continue
                other_lat = float(traf.lat[j])
                other_lon = float(traf.lon[j])
                other_alt = float(traf.alt[j] * 3.28084)
                other_spd = float(traf.cas[j] * 1.94384)
                other_trk = float(traf.trk[j])
                rel_pos = self._to_xy_km(lat_i, lon_i, other_lat, other_lon) - pos_i
                rel_vel = self._velocity_kmps(other_trk, other_spd) - vel_i
                rel_speed_sq = float(np.dot(rel_vel, rel_vel))
                raw_tcpa = float(-np.dot(rel_pos, rel_vel) / rel_speed_sq) if rel_speed_sq > 1e-9 else float("inf")
                tcpa = float(np.clip(raw_tcpa, 0.0, self.cfg.conflict_lookahead_s))
                closest_rel = rel_pos + rel_vel * tcpa
                dcpa = float(np.linalg.norm(closest_rel))
                horiz = self._haversine_km(lat_i, lon_i, other_lat, other_lon)
                vert = abs(alt_i - float(traf.alt[j] * 3.28084))
                vs_i_fps = float(getattr(traf, "vs", np.zeros(len(ids)))[i]) * 3.28084
                vs_j_fps = float(getattr(traf, "vs", np.zeros(len(ids)))[j]) * 3.28084
                predicted_vert = abs((other_alt - alt_i) + (vs_j_fps - vs_i_fps) * tcpa)
                time_gap, temporal_applicable = self._trajectory_time_gap(
                    rel_pos,
                    vel_i,
                    self._velocity_kmps(other_trk, other_spd),
                    alt_i,
                    other_alt,
                    vs_i_fps,
                    vs_j_fps,
                )
                temporal_loss = (
                    temporal_enabled
                    and temporal_applicable
                    and time_gap < separation.temporal_s
                )
                if temporal_applicable:
                    min_time_gap = min(min_time_gap, time_gap)
                bearing = self._bearing_deg(lat_i, lon_i, other_lat, other_lon)
                rel_bearing = self._wrap_hdg_diff(bearing, trk_i)
                rel_alt_norm = np.clip((other_alt - alt_i) / 3000.0, -1.0, 1.0)
                rel_speed_norm = np.clip((other_spd - spd_i) / 30.0, -1.0, 1.0)
                rel_track = other_trk
                rel_track_diff = self._wrap_hdg_diff(rel_track, trk_i)
                current_loss = spatial_enabled and horiz <= separation.horizontal_km and (
                    horizontal_only or vert <= separation.vertical_ft
                )
                lookahead_conflict = (
                    spatial_enabled
                    and
                    0.0 < raw_tcpa <= self.cfg.conflict_lookahead_s
                    and dcpa <= separation.horizontal_km
                    and (horizontal_only or predicted_vert <= separation.vertical_ft)
                )
                collision_pair = horiz <= self.cfg.collision_horizontal_km and vert <= self.cfg.collision_vertical_ft
                current_loss = current_loss or collision_pair
                if not (current_loss or lookahead_conflict or temporal_loss or horiz <= self.cfg.observation_radius_km):
                    continue
                spatial_clear = horiz > separation.horizontal_km * separation.release_factor
                if not horizontal_only:
                    spatial_clear = spatial_clear or vert > separation.vertical_ft * separation.release_factor
                temporal_clear = (
                    not temporal_applicable
                    or time_gap >= separation.temporal_s * separation.release_factor
                )
                if spatial_enabled:
                    allow_resume = allow_resume and spatial_clear
                if temporal_enabled:
                    allow_resume = allow_resume and temporal_clear
                conflict_flag = 1.0 if current_loss else 0.0
                active_loss = active_loss or current_loss
                predicted_conflict = predicted_conflict or lookahead_conflict
                temporal_conflict = temporal_conflict or temporal_loss
                collision = collision or collision_pair
                danger = danger or (
                    horiz <= self.cfg.danger_horizontal_km
                    and vert <= self.cfg.danger_vertical_ft
                )
                contact_severity = 0.0
                if current_loss:
                    contact_severity = max(contact_severity, self._current_severity(horiz, vert))
                if lookahead_conflict:
                    contact_severity = max(
                        contact_severity,
                        self._predicted_severity(dcpa, predicted_vert, tcpa),
                    )
                if temporal_loss:
                    temporal_penetration = float(
                        np.clip(1.0 - time_gap / max(separation.temporal_s, 1e-6), 0.0, 1.0)
                    )
                    # time_gap is the difference between both aircraft reaching
                    # their projected crossing point.  It can be zero even when
                    # that encounter is still several minutes away, so it must
                    # not itself determine urgency.
                    temporal_urgency = float(
                        np.clip(1.0 - tcpa / max(self.cfg.conflict_lookahead_s, 1e-6), 0.0, 1.0)
                    )
                    contact_severity = max(
                        contact_severity,
                        temporal_penetration * (0.25 + 0.75 * temporal_urgency),
                    )
                contact = ConflictContact(
                    callsign=other,
                    horiz_km=float(horiz),
                    vert_ft=float(vert),
                    dcpa_km=float(dcpa),
                    predicted_vert_ft=float(predicted_vert),
                    rel_bearing_deg=float(rel_bearing),
                    rel_bearing_sin=float(np.sin(np.radians(rel_bearing))),
                    rel_bearing_cos=float(np.cos(np.radians(rel_bearing))),
                    rel_alt_norm=float(rel_alt_norm),
                    rel_speed_norm=float(rel_speed_norm),
                    rel_track_sin=float(np.sin(np.radians(rel_track_diff))),
                    rel_track_cos=float(np.cos(np.radians(rel_track_diff))),
                    tcpa_s=float(np.clip(tcpa, 0.0, self.cfg.conflict_lookahead_s)),
                    time_gap_s=float(time_gap),
                    temporal_conflict=float(temporal_loss),
                    conflict_flag=float(conflict_flag),
                    severity=float(contact_severity),
                )
                contacts.append(contact)
                severity = max(severity, contact.severity)
            contacts.sort(key=lambda x: (-x.conflict_flag, -x.temporal_conflict, -x.severity, x.horiz_km))
            state = states[acid]
            state.targets = contacts[: self.cfg.max_intruders]
            state.severity = severity
            state.loss_of_separation = active_loss
            state.predicted_conflict = predicted_conflict
            state.temporal_conflict = temporal_conflict
            state.collision = collision
            state.danger = danger
            state.active = active_loss or predicted_conflict or temporal_conflict
            state.control_candidate = state.active
            state.allow_resume = bool(not state.active and allow_resume)
            state.min_time_gap_s = min_time_gap
        return states

    def select_targets(self, conflict_states: Dict[str, AircraftConflictState], max_targets: int | None = None) -> list[str]:
        max_targets = max_targets or self.cfg.network.max_aircraft
        ranked = sorted(
            [s for s in conflict_states.values() if s.control_candidate],
            key=lambda s: (-float(s.loss_of_separation), -s.severity, s.callsign),
        )
        return [item.callsign for item in ranked[:max_targets]]

    def can_resume_route(self, current_state: AircraftConflictState, command_state, current_conflict: bool) -> bool:
        if command_state is None or not command_state.has_snapshot:
            return False
        if current_conflict:
            return False
        if not current_state.allow_resume:
            return False
        if command_state.hold_until_s is None:
            return False
        if command_state.current_macro not in (2, 3, 4):
            return False
        return True

    def build_action_mask(self, conflict_state: AircraftConflictState, command_state, simt: float) -> np.ndarray:
        mask = np.zeros(5, dtype=np.float32)
        mask[0] = 1.0
        conflict_active = conflict_state.active
        if command_state is not None and command_state.current_macro in (2, 3, 4):
            can_break_hold = (
                self.cfg.runtime.emergency_override_enabled
                and conflict_state.severity >= self.cfg.runtime.emergency_override_severity
            )
            if conflict_active and (simt >= command_state.hold_until_s or can_break_hold):
                mask[2:] = 1.0
        elif conflict_active:
            mask[2:] = 1.0

        # A no-op is safe while an existing maneuver is being held. Without an
        # active maneuver, however, allowing it near loss of separation lets a
        # weak deterministic policy repeatedly choose inaction until collision.
        has_active_maneuver = command_state is not None and command_state.current_macro in (2, 3, 4)
        target_tcpa = min((contact.tcpa_s for contact in conflict_state.targets), default=float("inf"))
        temporal_urgent = (
            conflict_state.temporal_conflict
            and target_tcpa <= self.cfg.runtime.urgent_noop_threshold_s
            and conflict_state.min_time_gap_s <= self.cfg.runtime.urgent_noop_threshold_s
        )
        urgent = (
            conflict_state.loss_of_separation
            or target_tcpa <= self.cfg.runtime.urgent_noop_threshold_s
            or temporal_urgent
        )
        if (
            self.cfg.runtime.urgent_noop_mask_enabled
            and conflict_active
            and urgent
            and not has_active_maneuver
            and np.any(mask[2:] > 0.0)
        ):
            mask[0] = 0.0
        if (
            command_state is not None
            and command_state.current_macro in (2, 3, 4)
            and command_state.has_snapshot
            and simt >= command_state.hold_until_s
        ):
            if not conflict_active and conflict_state.allow_resume:
                mask[1] = 1.0
        return mask
