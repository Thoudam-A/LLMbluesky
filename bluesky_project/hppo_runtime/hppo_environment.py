from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import bluesky as bs
import numpy as np
from bluesky.tools import aero

from .command_executor import CommandExecutor, CommandPhase, CommandSnapshot, CommandState, ExecutionResult
from .config import HPPOConfig
from .conflict_manager import AircraftConflictState, ConflictManager
from .state_schema import validate_network_dimensions


@dataclass(slots=True)
class AircraftRuntime:
    callsign: str
    route_index: int = -1
    active: bool = False
    last_goal_distance_km: float = 0.0
    last_along_track_km: float = 0.0
    last_command_macro: int = 0
    last_command_time_s: float = -1.0
    pending_reward: float = 0.0
    resolved_conflicts: int = 0
    conflict_violations: int = 0
    secondary_conflicts: int = 0
    last_conflict_active: bool = False
    last_predicted_conflict: bool = False
    last_temporal_conflict: bool = False
    last_collision_active: bool = False
    last_risk_severity: float = 0.0
    arrival_rewarded: bool = False
    command_clear_since_s: float | None = None
    removed: bool = False
    command_history: list[int] = field(default_factory=list)
    command_state: CommandState = field(default_factory=CommandState)


class HPPOEnvironment:
    def __init__(self, config: HPPOConfig):
        self.cfg = config
        validate_network_dimensions(config)
        self.executor = CommandExecutor(config)
        self.conflict_manager = ConflictManager(config)
        self.scenario_files = tuple(Path(path).resolve() for path in (self.cfg.scenario_files or (self.cfg.route_file,)))
        self.positions = np.empty((0, 5), dtype=np.float32)
        self.start_speeds_kt = np.empty((0,), dtype=np.float32)
        self.start_altitudes_ft = np.empty((0,), dtype=np.float32)
        self.entry_times_s = np.empty((0,), dtype=np.float32)
        self.pending_spawn_indices: set[int] = set()
        self.spawned_aircraft_ids: set[str] = set()
        self.spawn_origin_s = 0.0
        self.route_shape_names: list[str] = []
        self.current_scenario_path = self.scenario_files[0]
        self.current_scenario_name = self.current_scenario_path.stem
        self.expected_aircraft_ids: set[str] = set()
        self.arrived_aircraft: set[str] = set()
        self._scenario_rng = np.random.default_rng(self.cfg.seed)
        self.runtime: Dict[str, AircraftRuntime] = {}
        self.episode_id = 0
        self.episode_step = 0
        self.episode_start_time_s = 0.0
        self.last_decision_time_s = -1.0
        self.last_state_update_time_s = -1.0
        self.reward_stats: Dict[str, float] = {}
        self.action_mask_reasons: Dict[str, str] = {}
        self._ensure_output_dirs()
        self.load_scenario(0)
        self.reset_episode()
        self._gui_zoom = 0.4

    def _ensure_output_dirs(self) -> None:
        self.cfg.logging.output_dir.mkdir(parents=True, exist_ok=True)
        (self.cfg.logging.output_dir / self.cfg.logging.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        (self.cfg.logging.output_dir / self.cfg.logging.scenario_dir).mkdir(parents=True, exist_ok=True)

    def reset_episode(self) -> None:
        self.runtime.clear()
        self.action_mask_reasons.clear()
        self.arrived_aircraft.clear()
        self.episode_step = 0
        self.episode_start_time_s = float(getattr(bs.sim, "simt", 0.0))
        self.last_decision_time_s = -1.0
        self.last_state_update_time_s = -1.0
        self.reward_stats = {
            "episode_return": 0.0,
            "conflict_resolved": 0.0,
            "safety_violations": 0.0,
            "secondary_conflicts": 0.0,
            "arrival": 0.0,
            "extra_distance": 0.0,
            "command_count": 0.0,
            "oscillation_count": 0.0,
            "command_failures": 0.0,
            "collision_events": 0.0,
            "conflict_detected": 0.0,
            "temporal_conflict_events": 0.0,
            "duplicate_commands": 0.0,
            "replan_count": 0.0,
            "risk_improvement_reward": 0.0,
            # Separation-management observability.  These are passive
            # episode metrics; they do not influence reward or policy output.
            "min_contact_horizontal_km": -1.0,
            "min_predicted_cpa_km": -1.0,
            "min_predicted_cpa_vertical_ft": -1.0,
            "first_intervention_s": -1.0,
            # Capacity-test observability. These values describe how many
            # aircraft were simultaneously present and selected for policy
            # processing at any point in the episode.
            "peak_active_aircraft": 0.0,
            "peak_policy_targets": 0.0,
        }

    @staticmethod
    def _validate_scenario(path: Path, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 5 or positions.shape[0] == 0:
            raise ValueError(f"Invalid scenario {path}: expected a non-empty [N,5] array, got {positions.shape}")
        if not np.isfinite(positions).all():
            raise ValueError(f"Invalid scenario {path}: contains NaN or Inf")
        if np.any(np.abs(positions[:, [0, 2]]) > 90.0) or np.any(np.abs(positions[:, [1, 3]]) > 180.0):
            raise ValueError(f"Invalid scenario {path}: latitude or longitude is out of range")
        positions[:, 4] %= 360.0
        return positions

    def _load_scenario_data(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                if "routes" not in loaded.files:
                    raise ValueError(f"Invalid scenario {path}: NPZ must contain 'routes'")
                routes = self._validate_scenario(path, loaded["routes"])
                count = len(routes)
                speeds = np.asarray(
                    loaded.get(
                        "speed_kt",
                        np.full(count, self.cfg.command_ranges.default_initial_speed_kt),
                    ),
                    dtype=np.float32,
                )
                altitudes = np.asarray(
                    loaded.get(
                        "altitude_ft",
                        np.full(count, self.cfg.command_ranges.default_initial_altitude_ft),
                    ),
                    dtype=np.float32,
                )
                entry_times = np.asarray(loaded.get("entry_time_s", np.zeros(count)), dtype=np.float32)
            finally:
                loaded.close()
        else:
            routes = self._validate_scenario(path, loaded)
            count = len(routes)
            speeds = np.full(
                count,
                self.cfg.command_ranges.default_initial_speed_kt,
                dtype=np.float32,
            )
            altitudes = np.full(
                count,
                self.cfg.command_ranges.default_initial_altitude_ft,
                dtype=np.float32,
            )
            entry_times = np.zeros(count, dtype=np.float32)
        for name, values in (("speed_kt", speeds), ("altitude_ft", altitudes), ("entry_time_s", entry_times)):
            if values.shape != (len(routes),):
                raise ValueError(f"Invalid scenario {path}: {name} must have shape ({len(routes)},)")
            if not np.isfinite(values).all():
                raise ValueError(f"Invalid scenario {path}: {name} contains NaN or Inf")
        if np.any(entry_times < 0.0):
            raise ValueError(f"Invalid scenario {path}: entry times must be non-negative")
        if np.any(speeds < self.cfg.command_ranges.min_speed_kt) or np.any(speeds > self.cfg.command_ranges.max_speed_kt):
            raise ValueError(f"Invalid scenario {path}: speed is outside configured command limits")
        if np.any(altitudes < self.cfg.command_ranges.min_altitude_ft) or np.any(altitudes > self.cfg.command_ranges.max_altitude_ft):
            raise ValueError(f"Invalid scenario {path}: altitude is outside configured command limits")
        return routes, speeds, altitudes, entry_times

    def load_scenario(self, episode_id: int) -> Path:
        if self.cfg.scenario_selection == "random":
            index = int(self._scenario_rng.integers(0, len(self.scenario_files)))
        elif self.cfg.scenario_selection == "cycle":
            index = episode_id % len(self.scenario_files)
        else:
            index = 0
        path = self.scenario_files[index]
        self.positions, self.start_speeds_kt, self.start_altitudes_ft, self.entry_times_s = self._load_scenario_data(path)
        if len(self.positions) > self.cfg.network.max_aircraft:
            raise ValueError(
                f"Scenario {path} has {len(self.positions)} aircraft, exceeding max_aircraft={self.cfg.network.max_aircraft}"
            )
        self.current_scenario_path = path
        self.current_scenario_name = path.stem
        self.expected_aircraft_ids = {f"KL{idx}" for idx in range(len(self.positions))}
        self.pending_spawn_indices.clear()
        self.spawned_aircraft_ids.clear()
        return path

    def clear_route_overlays(self) -> None:
        """删除上一场景在 QtGL 中绘制的路线。"""
        for shape_name in self.route_shape_names:
            bs.stack.stack(f"DEL {shape_name}")
        self.route_shape_names.clear()

    def draw_route_overlays(self) -> None:
        """根据当前 npy 场景绘制起点到终点的辅助线。"""
        self.clear_route_overlays()

        for idx, route in enumerate(
            self.positions[: self.cfg.network.max_aircraft]
        ):
            start_lat, start_lon, goal_lat, goal_lon, _ = map(float, route)

            # 不要与飞机呼号同名，避免 DEL 时误删飞机
            shape_name = f"HPPO_RTE_{idx:02d}"

            bs.stack.stack(
                f"LINE {shape_name},"
                f"{start_lat:.8f},{start_lon:.8f},"
                f"{goal_lat:.8f},{goal_lon:.8f}"
            )

            self.route_shape_names.append(shape_name)
    
    def clear_traffic(self) -> None:
        if len(bs.traf.id) > 0:
            bs.traf.delete(np.arange(len(bs.traf.id), dtype=int))
    
    def calculate_gui_view(self) -> tuple[float, float, float]:
        """根据当前场景全部起点和终点计算中心及目标缩放值。"""
        if self.positions.size == 0:
            return 0.0, 0.0, 1.0

        # 每行：
        # start_lat, start_lon, goal_lat, goal_lon, start_heading
        points = np.vstack(
            (
                self.positions[:, 0:2],
                self.positions[:, 2:4],
            )
        )

        lat_min = float(np.min(points[:, 0]))
        lat_max = float(np.max(points[:, 0]))
        lon_min = float(np.min(points[:, 1]))
        lon_max = float(np.max(points[:, 1]))

        center_lat = 0.5 * (lat_min + lat_max)
        center_lon = 0.5 * (lon_min + lon_max)

        min_span = self.cfg.runtime.gui_min_span_deg
        margin = self.cfg.runtime.gui_view_margin
        aspect_ratio = self.cfg.runtime.gui_aspect_ratio

        lat_span = max(lat_max - lat_min, min_span)
        lon_span = max(lon_max - lon_min, min_span)

        required_half_lat = 0.5 * lat_span * margin
        required_half_lon = 0.5 * lon_span * margin

        cos_lat = max(abs(np.cos(np.radians(center_lat))), 0.1)

        # QtGL:
        # half latitude range = 1 / (zoom * aspect_ratio)
        # half longitude range = 1 / (zoom * cos(latitude))
        zoom_for_lat = 1.0 / max(
            required_half_lat * aspect_ratio,
            1e-6,
        )
        zoom_for_lon = 1.0 / max(
            required_half_lon * cos_lat,
            1e-6,
        )

        target_zoom = min(zoom_for_lat, zoom_for_lon)

        target_zoom = float(
            np.clip(
                target_zoom,
                self.cfg.runtime.gui_min_zoom,
                self.cfg.runtime.gui_max_zoom,
            )
        )
        return center_lat, center_lon, target_zoom
    
    def frame_gui_to_scenario(self) -> dict:
        """自动将 QtGL 视图移动并缩放到当前场景。"""
        if not self.cfg.runtime.auto_frame_gui:
            return {}

        center_lat, center_lon, target_zoom = self.calculate_gui_view()

        current_zoom = max(self._gui_zoom, 1e-6)

        # ZOOM 数字是相对于当前视野的倍率
        zoom_factor = target_zoom / current_zoom

        bs.stack.stack(
            f"PAN {center_lat:.8f},{center_lon:.8f}"
        )
        bs.stack.stack(
            f"ZOOM {zoom_factor:.8f}"
        )

        self._gui_zoom = target_zoom

        return {
            "center_lat": center_lat,
            "center_lon": center_lon,
            "target_zoom": target_zoom,
            "zoom_factor": zoom_factor,
        }
    
    def spawn_episode_fleet(self, simt: float | None = None) -> int:
        if len(bs.traf.id) != 0:
            raise RuntimeError("Cannot spawn an episode while old traffic is still present")
        self.spawn_origin_s = float(bs.sim.simt if simt is None else simt)
        self.pending_spawn_indices = set(range(min(len(self.positions), self.cfg.network.max_aircraft)))
        self.spawned_aircraft_ids.clear()
        self.spawn_due_aircraft(self.spawn_origin_s)
        return len(self.expected_aircraft_ids)

    def spawn_due_aircraft(self, simt: float) -> list[str]:
        queued: list[str] = []
        elapsed = max(0.0, float(simt) - self.spawn_origin_s)
        due = sorted(index for index in self.pending_spawn_indices if self.entry_times_s[index] <= elapsed + 1e-6)
        for idx in due:
            lat, lon, goal_lat, goal_lon, start_hdg = map(float, self.positions[idx])
            start_alt = float(self.start_altitudes_ft[idx])
            start_speed = float(self.start_speeds_kt[idx])
            start_speed_cas = float(aero.tas2cas(start_speed * aero.kts, start_alt * aero.ft) / aero.kts)
            acid = f"KL{idx}"
            try:
                bs.stack.stack(f"CRE {acid}, B737, {lat}, {lon}, {start_hdg}, {start_alt}, {start_speed_cas}")
                bs.stack.stack(f"ADDWPT {acid} {goal_lat}, {goal_lon}")
                bs.stack.stack(f"LNAV {acid} ON")
                bs.stack.stack(f"VNAV {acid} ON")
                queued.append(acid)
                self.pending_spawn_indices.remove(idx)
            except Exception as exc:
                raise RuntimeError(f"Failed to spawn {acid}: {exc}") from exc
        return queued

    def spawn_complete(self) -> bool:
        if not set(bs.traf.id):
            return False
        for acid in bs.traf.id:
            idx = bs.traf.id2idx(acid)
            if idx < 0 or idx >= len(bs.traf.ap.route) or bs.traf.ap.route[idx].nwp <= 0:
                return False
        return True

    def all_expected_aircraft_spawned(self) -> bool:
        observed = self.spawned_aircraft_ids | set(bs.traf.id) | self.arrived_aircraft
        return not self.pending_spawn_indices and self.expected_aircraft_ids.issubset(observed)

    def cancel_pending_spawns(self) -> None:
        self.pending_spawn_indices.clear()

    def sync_runtime(self) -> Dict[str, AircraftRuntime]:
        current_ids = set(bs.traf.id)
        self.reward_stats["peak_active_aircraft"] = max(
            self.reward_stats.get("peak_active_aircraft", 0.0),
            float(len(current_ids)),
        )
        removed: Dict[str, AircraftRuntime] = {}
        for acid in list(self.runtime.keys()):
            if acid not in current_ids:
                runtime = self.runtime.pop(acid)
                runtime.removed = True
                removed[acid] = runtime
        for acid in current_ids:
            self.spawned_aircraft_ids.add(acid)
            if acid not in self.runtime:
                idx = bs.traf.id2idx(acid)
                route_index = int(acid[2:]) if acid[2:].isdigit() else idx
                runtime = AircraftRuntime(callsign=acid, route_index=route_index)
                self.runtime[acid] = runtime
                self._init_aircraft_state(runtime, idx)
        return removed

    def _init_aircraft_state(self, runtime: AircraftRuntime, idx: int) -> None:
        runtime.last_goal_distance_km = self._distance_to_goal(idx)
        runtime.last_along_track_km = self._along_track_distance(idx)
        runtime.command_state = CommandState()

    def _route(self, runtime: AircraftRuntime):
        route_idx = runtime.route_index
        if route_idx < 0 or route_idx >= len(self.positions):
            route_idx = min(max(route_idx, 0), len(self.positions) - 1)
        return self.positions[route_idx]

    @staticmethod
    def _latlon_to_xy(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
        mean_lat = np.radians((lat1 + lat2) / 2.0)
        dx = (lon2 - lon1) * 111.32 * np.cos(mean_lat)
        dy = (lat2 - lat1) * 110.57
        return dx, dy

    def _distance_to_goal(self, idx: int) -> float:
        route_idx = int(bs.traf.id[idx][2:]) if bs.traf.id[idx][2:].isdigit() else idx
        route = self.positions[min(max(route_idx, 0), len(self.positions) - 1)]
        return self._haversine_km(float(bs.traf.lat[idx]), float(bs.traf.lon[idx]), float(route[2]), float(route[3]))

    def _along_track_distance(self, idx: int) -> float:
        route_idx = int(bs.traf.id[idx][2:]) if bs.traf.id[idx][2:].isdigit() else idx
        route = self.positions[min(max(route_idx, 0), len(self.positions) - 1)]
        start_lat, start_lon, goal_lat, goal_lon = map(float, route[:4])
        dx1, dy1 = self._latlon_to_xy(start_lat, start_lon, float(bs.traf.lat[idx]), float(bs.traf.lon[idx]))
        dx2, dy2 = self._latlon_to_xy(start_lat, start_lon, goal_lat, goal_lon)
        route_len = max(np.hypot(dx2, dy2), 1e-6)
        proj = (dx1 * dx2 + dy1 * dy2) / route_len
        return float(proj)

    def _route_length_km(self, idx: int) -> float:
        route_idx = int(bs.traf.id[idx][2:]) if bs.traf.id[idx][2:].isdigit() else idx
        route = self.positions[min(max(route_idx, 0), len(self.positions) - 1)]
        dx, dy = self._latlon_to_xy(float(route[0]), float(route[1]), float(route[2]), float(route[3]))
        return float(np.hypot(dx, dy))

    def _has_reached_terminal(self, idx: int) -> bool:
        """Recognize a destination capture or a close forward pass of the terminal gate."""
        if self._distance_to_goal(idx) <= self.cfg.arrival_distance_km:
            return True
        return bool(
            self._distance_to_goal(idx) <= self.cfg.terminal_capture_distance_km
            and self._along_track_distance(idx) >= self._route_length_km(idx)
            and self._cross_track_distance_km(idx) <= self.cfg.terminal_pass_cross_track_km
        )

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
        return float(2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(max(1.0 - a, 1e-12))))

    @staticmethod
    def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        y = np.sin(np.radians(lon2 - lon1)) * np.cos(np.radians(lat2))
        x = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon2 - lon1))
        return float((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)

    @staticmethod
    def _wrap_angle_deg(value: float) -> float:
        return ((value + 180.0) % 360.0) - 180.0

    def _cross_track_distance_km(self, idx: int) -> float:
        route_idx = int(bs.traf.id[idx][2:]) if bs.traf.id[idx][2:].isdigit() else idx
        route = self.positions[min(max(route_idx, 0), len(self.positions) - 1)]
        start_lat, start_lon, goal_lat, goal_lon = map(float, route[:4])
        px, py = self._latlon_to_xy(start_lat, start_lon, float(bs.traf.lat[idx]), float(bs.traf.lon[idx]))
        gx, gy = self._latlon_to_xy(start_lat, start_lon, goal_lat, goal_lon)
        seg = np.array([gx, gy], dtype=np.float32)
        point = np.array([px, py], dtype=np.float32)
        seg_norm = float(np.dot(seg, seg))
        if seg_norm <= 1e-9:
            return float(np.linalg.norm(point))
        t = float(np.clip(np.dot(point, seg) / seg_norm, 0.0, 1.0))
        proj = seg * t
        return float(np.linalg.norm(point - proj))

    def _norm_lat(self, lat: float) -> float:
        return float(np.clip(lat / 90.0, -1.0, 1.0))

    def _norm_lon(self, lon: float) -> float:
        return float(np.clip(lon / 180.0, -1.0, 1.0))

    def _norm_speed(self, speed_kt: float) -> float:
        rng = self.cfg.command_ranges
        return float(np.clip(2.0 * (speed_kt - rng.min_speed_kt) / max(rng.max_speed_kt - rng.min_speed_kt, 1e-6) - 1.0, -1.0, 1.0))

    def _norm_alt(self, alt_ft: float) -> float:
        rng = self.cfg.command_ranges
        return float(np.clip(2.0 * (alt_ft - rng.min_altitude_ft) / max(rng.max_altitude_ft - rng.min_altitude_ft, 1e-6) - 1.0, -1.0, 1.0))

    def _own_features(self, idx: int) -> list[float]:
        acid = bs.traf.id[idx]
        route_idx = int(acid[2:]) if acid[2:].isdigit() else idx
        route = self.positions[min(max(route_idx, 0), len(self.positions) - 1)]
        lat = float(bs.traf.lat[idx])
        lon = float(bs.traf.lon[idx])
        speed = float(bs.traf.tas[idx] / aero.kts)
        altitude = float(bs.traf.alt[idx] * 3.28084)
        heading = float(bs.traf.hdg[idx])
        bearing_to_goal = self._bearing_deg(lat, lon, float(route[2]), float(route[3]))
        route_hdg = float(route[4])
        return [
            self._norm_lat(lat),
            self._norm_lon(lon),
            self._norm_speed(speed),
            self._norm_alt(altitude),
            float(np.sin(np.radians(heading))),
            float(np.cos(np.radians(heading))),
            self._norm_lat(float(route[0])),
            self._norm_lon(float(route[1])),
            self._norm_lat(float(route[2])),
            self._norm_lon(float(route[3])),
            float(np.sin(np.radians(route_hdg))),
            float(np.cos(np.radians(route_hdg))),
            float(np.sin(np.radians(bearing_to_goal))),
            float(np.cos(np.radians(bearing_to_goal))),
            float(np.clip(self._wrap_angle_deg(bearing_to_goal - heading) / 180.0, -1.0, 1.0)),
            float(np.clip(self._distance_to_goal(idx) / 200.0, 0.0, 1.0)),
            float(np.clip(self._along_track_distance(idx) / 200.0, -1.0, 1.0)),
            float(np.clip(self._distance_to_goal(idx) / max(self.cfg.arrival_distance_km, 1.0), 0.0, 1.0)),
        ]

    def _separation_standard_features(self) -> list[float]:
        separation = self.cfg.separation
        return [
            float(np.clip(separation.horizontal_km / separation.max_horizontal_km, 0.0, 1.0)),
            float(np.clip(separation.vertical_ft / separation.max_vertical_ft, 0.0, 1.0)),
            float(np.clip(separation.temporal_s / separation.max_temporal_s, 0.0, 1.0)),
        ]

    def set_horizontal_separation_km(self, horizontal_km: float) -> tuple[float, float]:
        """Apply an operator-selected horizontal separation standard at runtime."""
        value = float(horizontal_km)
        separation = self.cfg.separation
        max_supported = min(
            separation.max_horizontal_km,
            self.cfg.observation_radius_km / max(separation.release_factor, 1e-6),
        )
        if not 0.0 < value <= max_supported:
            raise ValueError(
                f"horizontal separation must be in (0, {max_supported:.3f}] km for the current H-PPO configuration"
            )
        previous = float(separation.horizontal_km)
        separation.horizontal_km = value
        separation.validate()
        self.action_mask_reasons.clear()
        self.last_decision_time_s = -1.0
        return previous, value

    def rebaseline_active_command_risk(self, conflict_states: Dict[str, AircraftConflictState], simt: float) -> None:
        """Prevent an old separation standard from biasing maneuver-effect checks."""
        for acid, runtime in self.runtime.items():
            command_state = runtime.command_state
            if not command_state.active:
                continue
            snapshot = self._risk_snapshot(conflict_states.get(acid))
            command_state.risk_severity_at_issue = float(snapshot.get("severity", 0.0))
            command_state.risk_tcpa_s_at_issue = float(snapshot.get("tcpa_s", float("inf")))
            command_state.risk_horizontal_km_at_issue = float(snapshot.get("horizontal_km", float("inf")))
            command_state.risk_vertical_ft_at_issue = float(snapshot.get("vertical_ft", float("inf")))
            command_state.risk_target_count_at_issue = int(snapshot.get("target_count", 0))
            command_state.risk_target_ids_at_issue = tuple(snapshot.get("target_ids", ()))
            command_state.last_effect_evaluation_s = float(simt)
            command_state.stalled_evaluations = 0

    def build_global_state(self, conflict_states: Dict[str, AircraftConflictState]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        slot_dim = self.cfg.network.global_slot_dim
        global_state = np.zeros((self.cfg.network.max_aircraft, slot_dim), dtype=np.float32)
        presence = np.zeros((self.cfg.network.max_aircraft,), dtype=np.float32)
        id_to_slot: dict[str, int] = {}
        for slot, acid in enumerate(sorted(bs.traf.id)):
            if slot >= self.cfg.network.max_aircraft:
                continue
            id_to_slot[acid] = slot
            idx = bs.traf.id2idx(acid)
            presence[slot] = 1.0
            cmd_state = self.runtime.get(acid).command_state if acid in self.runtime else CommandState()
            conflict_state = conflict_states.get(acid, AircraftConflictState(acid))
            own = self._own_features(idx)
            global_state[slot, :18] = np.asarray(own, dtype=np.float32)
            global_state[slot, 18] = 1.0 if conflict_state.active else 0.0
            global_state[slot, 19] = float(np.clip(conflict_state.severity, 0.0, 1.0))
            global_state[slot, 20] = 1.0 if cmd_state.current_macro != 0 else 0.0
            global_state[slot, 21] = float(np.clip((bs.sim.simt - cmd_state.issued_at_s) / max(self.cfg.runtime.min_command_hold_s, 1e-6), 0.0, 1.0)) if cmd_state.issued_at_s >= 0.0 else 0.0
            global_state[slot, 22:25] = np.asarray(self._separation_standard_features(), dtype=np.float32)
            time_gap = conflict_state.min_time_gap_s
            global_state[slot, 25] = (
                1.0
                if not np.isfinite(time_gap)
                else float(np.clip(time_gap / max(self.cfg.separation.temporal_s, 1e-6), 0.0, 2.0) / 2.0)
            )
        return global_state, presence, id_to_slot

    def build_local_observation(self, acid: str, conflict_states: Dict[str, AircraftConflictState]) -> np.ndarray:
        idx = bs.traf.id2idx(acid)
        if idx < 0:
            return np.zeros(self.cfg.network.local_obs_dim, dtype=np.float32)
        runtime = self.runtime[acid]
        conflict_state = conflict_states.get(acid, AircraftConflictState(acid))
        self._refresh_command_phase(acid, conflict_state, float(bs.sim.simt))
        own = self._own_features(idx)
        cmd = runtime.command_state
        macro_one_hot = np.zeros(5, dtype=np.float32)
        if 0 <= cmd.current_macro < 5:
            macro_one_hot[cmd.current_macro] = 1.0
        command_age = 0.0 if cmd.issued_at_s < 0.0 else float(np.clip((bs.sim.simt - cmd.issued_at_s) / max(self.cfg.runtime.min_command_hold_s, 1e-6), 0.0, 1.0))
        hold_remaining = 0.0 if cmd.hold_until_s < 0.0 else float(np.clip((cmd.hold_until_s - bs.sim.simt) / max(self.cfg.runtime.min_command_hold_s, 1e-6), 0.0, 1.0))
        command_context = np.array(
            [
                *macro_one_hot.tolist(),
                command_age,
                hold_remaining,
                1.0 if cmd.snapshot.has_snapshot else 0.0,
                1.0 if conflict_state.active else 0.0,
                1.0 if cmd.active else 0.0,
                *self._phase_one_hot(cmd.phase).tolist(),
                self._command_target_progress(acid, cmd),
                1.0 if cmd.last_risk_improved else 0.0,
                float(np.clip(cmd.stalled_evaluations / max(self.cfg.runtime.replan_stall_evaluations, 1), 0.0, 1.0)),
                *self._cooldown_features(cmd, float(bs.sim.simt)).tolist(),
            ],
            dtype=np.float32,
        )
        contacts = []
        for contact in conflict_state.targets[: self.cfg.max_intruders]:
            contacts.extend(
                [
                    float(np.clip(contact.horiz_km / self.cfg.observation_radius_km, 0.0, 1.0)),
                    contact.rel_bearing_sin,
                    contact.rel_bearing_cos,
                    float(np.clip(contact.rel_alt_norm, -1.0, 1.0)),
                    float(np.clip(contact.rel_speed_norm, -1.0, 1.0)),
                    contact.rel_track_sin,
                    contact.rel_track_cos,
                    float(np.clip(contact.tcpa_s / self.cfg.conflict_lookahead_s, 0.0, 1.0)),
                    contact.conflict_flag,
                    (
                        1.0
                        if not np.isfinite(contact.time_gap_s)
                        else float(
                            np.clip(
                                contact.time_gap_s / max(self.cfg.separation.temporal_s, 1e-6),
                                0.0,
                                2.0,
                            )
                            / 2.0
                        )
                    ),
                ]
            )
        while len(contacts) < self.cfg.max_intruders * 10:
            contacts.append(0.0)
        obs = np.asarray(
            own
            + command_context.tolist()
            + self._separation_standard_features()
            + contacts[: self.cfg.max_intruders * 10],
            dtype=np.float32,
        )
        if obs.shape[0] != self.cfg.network.local_obs_dim:
            raise ValueError(f"Local observation shape mismatch: expected {self.cfg.network.local_obs_dim}, got {obs.shape[0]}")
        return obs

    def compute_rewards(self, conflict_states: Dict[str, AircraftConflictState]) -> dict[str, float]:
        rewards = {acid: 0.0 for acid in bs.traf.id}
        if len(bs.traf.id) == 0:
            return rewards
        for acid in bs.traf.id:
            idx = bs.traf.id2idx(acid)
            if idx < 0:
                continue
            runtime = self.runtime[acid]
            own_reward = self.cfg.reward.step_cost
            prev_goal = runtime.last_goal_distance_km
            current_goal = self._distance_to_goal(idx)
            progress = prev_goal - current_goal
            own_reward += self.cfg.reward.progress * progress
            runtime.last_goal_distance_km = current_goal
            current_along = self._along_track_distance(idx)
            cross_track = self._cross_track_distance_km(idx)
            own_reward += self.cfg.reward.route_deviation * cross_track
            extra_distance = max(0.0, current_along - runtime.last_along_track_km - max(progress, 0.0))
            own_reward += self.cfg.reward.extra_distance * extra_distance
            runtime.last_along_track_km = current_along
            conf = conflict_states.get(acid, AircraftConflictState(acid))
            # Contacts are already restricted to relevant neighbours by the
            # conflict manager.  Track the closest observed and predicted
            # separation over the whole episode for interval experiments.
            for contact in conf.targets:
                current_min = self.reward_stats["min_contact_horizontal_km"]
                predicted_min = self.reward_stats["min_predicted_cpa_km"]
                predicted_vertical_min = self.reward_stats["min_predicted_cpa_vertical_ft"]
                if current_min < 0.0 or contact.horiz_km < current_min:
                    self.reward_stats["min_contact_horizontal_km"] = float(contact.horiz_km)
                if predicted_min < 0.0 or contact.dcpa_km < predicted_min:
                    self.reward_stats["min_predicted_cpa_km"] = float(contact.dcpa_km)
                if predicted_vertical_min < 0.0 or contact.predicted_vert_ft < predicted_vertical_min:
                    self.reward_stats["min_predicted_cpa_vertical_ft"] = float(contact.predicted_vert_ft)
            conflict_event = conf.active
            previous_event = (
                runtime.last_conflict_active
                or runtime.last_predicted_conflict
                or runtime.last_temporal_conflict
            )
            if conflict_event and not previous_event:
                self.reward_stats["conflict_detected"] += 1.0
                if conf.temporal_conflict:
                    self.reward_stats["temporal_conflict_events"] += 1.0
            if conf.loss_of_separation:
                own_reward += self.cfg.reward.intrusion * float(conf.severity)
                if not runtime.last_conflict_active:
                    runtime.conflict_violations += 1
                    self.reward_stats["safety_violations"] += 1.0
                if conf.collision:
                    own_reward += self.cfg.reward.collision
                    if not runtime.last_collision_active:
                        self.reward_stats["collision_events"] += 1.0
                elif conf.danger:
                    own_reward += self.cfg.reward.warning * float(conf.severity)
                secondary = max(len([item for item in conf.targets if item.conflict_flag > 0.0]) - 1, 0)
                if secondary > 0:
                    own_reward += self.cfg.reward.secondary_conflict * float(secondary)
                    if not runtime.last_conflict_active:
                        self.reward_stats["secondary_conflicts"] += 1.0
            elif conf.predicted_conflict or conf.temporal_conflict:
                own_reward += self.cfg.reward.warning * float(conf.severity)
            else:
                if previous_event and not conflict_event and runtime.command_state.current_macro in (2, 3, 4):
                    runtime.resolved_conflicts += 1
                    own_reward += self.cfg.reward.resolution_success
                    self.reward_stats["conflict_resolved"] += 1.0
            if conf.active and previous_event:
                severity_reduction = max(0.0, runtime.last_risk_severity - float(conf.severity))
                if severity_reduction > 0.0 and self.cfg.reward.risk_improvement != 0.0:
                    improvement_reward = self.cfg.reward.risk_improvement * severity_reduction
                    own_reward += improvement_reward
                    self.reward_stats["risk_improvement_reward"] += improvement_reward * self.cfg.reward.reward_scale
            if self._has_reached_terminal(idx) and not runtime.arrival_rewarded:
                own_reward += self.cfg.reward.arrival
                runtime.arrival_rewarded = True
                self.reward_stats["arrival"] += 1.0
                self.arrived_aircraft.add(acid)
            runtime.last_conflict_active = conf.loss_of_separation
            runtime.last_predicted_conflict = conf.predicted_conflict
            runtime.last_temporal_conflict = conf.temporal_conflict
            runtime.last_collision_active = conf.collision
            runtime.last_risk_severity = float(conf.severity)
            self.reward_stats["extra_distance"] += extra_distance
            rewards[acid] = float(own_reward * self.cfg.reward.reward_scale)
        return rewards

    def remove_arrived_aircraft(self) -> list[str]:
        removable = sorted(acid for acid in self.arrived_aircraft if acid in bs.traf.id)
        if not removable:
            return []
        indices = np.asarray([bs.traf.id2idx(acid) for acid in removable], dtype=int)
        indices = indices[indices >= 0]
        if indices.size:
            bs.traf.delete(indices)
        return removable

    def get_action_mask(self, acid: str, conflict_states: Dict[str, AircraftConflictState]) -> np.ndarray:
        runtime = self.runtime.get(acid)
        if runtime is None:
            return np.array([1, 0, 0, 0, 0], dtype=np.float32)
        conf = conflict_states.get(acid, AircraftConflictState(acid))
        simt = float(bs.sim.simt)
        cmd = runtime.command_state
        self._refresh_command_phase(acid, conf, simt)
        mask = self.conflict_manager.build_action_mask(conf, cmd, simt)
        reasons: list[str] = []
        if mask[1] > 0.0 and not self._resume_ready(runtime, conf, simt):
            mask[1] = 0.0
            reasons.append("resume_stability_window")
        if mask[3] > 0.0 and not self._altitude_maneuver_available(acid):
            mask[3] = 0.0
            reasons.append("altitude_boundary_guard")
        vertical_leaders = self._vertical_group_leaders(conflict_states)
        leader = vertical_leaders.get(acid)
        if leader is not None and leader != acid and mask[3] > 0.0:
            mask[3] = 0.0
            reasons.append(f"vertical_group_yield:{leader}")
        emergency = self._emergency_replan_required(conf, cmd, simt)
        for macro_action in (2, 3, 4):
            if (
                cmd.cooldown_until_by_macro.get(macro_action, -1.0) > simt
                and not emergency
            ):
                mask[macro_action] = 0.0
                reasons.append(f"replan_cooldown:{macro_action}")
        if cmd.active and cmd.phase in (CommandPhase.MANEUVERING, CommandPhase.MONITORING) and not emergency:
            # Keep observing the same effective command until it has had two
            # decision periods to demonstrate an effect. RESUME_ROUTE remains
            # available when the conflict manager has released the command.
            mask[2:] = 0.0
            reasons.append(f"command_{cmd.phase.value.lower()}")
        elif cmd.phase == CommandPhase.REPLAN_REQUIRED:
            if cmd.current_macro in (2, 3, 4) and not emergency:
                mask[cmd.current_macro] = 0.0
                reasons.append(f"failed_dimension:{cmd.current_macro}")
        if self.cfg.network.parameter_policy_type == "discrete":
            parameter_masks = self.get_parameter_masks(acid)
            for branch in range(3):
                if parameter_masks[branch].sum() <= 0.0:
                    mask[branch + 2] = 0.0
                    reasons.append(f"empty_parameter_branch:{branch}")
        if cmd.phase == CommandPhase.REPLAN_REQUIRED and np.any(mask[2:] > 0.0):
            # Replanning is meaningful only when the policy must select a new
            # effective dimension instead of extending the failed maneuver.
            mask[0] = 0.0
            reasons.append(f"replan_required:{cmd.replan_reason or 'stalled'}")
        if not np.any(mask > 0.0):
            # A malformed combination of rule masks must never create an
            # invalid categorical distribution. Preserve NO_NEW_COMMAND as a
            # final fail-safe and surface it in the trace.
            mask[0] = 1.0
            reasons.append("fallback_noop")
        self.action_mask_reasons[acid] = ";".join(reasons)
        return mask

    def get_action_mask_reason(self, acid: str) -> str:
        return self.action_mask_reasons.get(acid, "")

    def _parameter_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        heading_step = int(round(self.cfg.network.heading_bin_deg))
        altitude_step = int(round(self.cfg.network.altitude_bin_step_ft))
        speed_step = int(round(self.cfg.command_ranges.speed_bin_step_kt))
        headings = np.arange(0.0, 360.0, float(heading_step), dtype=np.float32)
        altitudes = np.arange(
            self.cfg.command_ranges.min_altitude_ft,
            self.cfg.command_ranges.max_altitude_ft + 0.5 * altitude_step,
            float(altitude_step),
            dtype=np.float32,
        )
        speeds = np.arange(
            self.cfg.command_ranges.min_speed_kt,
            self.cfg.command_ranges.max_speed_kt + 0.5 * speed_step,
            float(speed_step),
            dtype=np.float32,
        )
        return headings, altitudes, speeds

    def get_parameter_masks(self, acid: str) -> np.ndarray:
        """Return [heading, altitude, speed] masks over padded absolute bins."""
        headings, altitudes, speeds = self._parameter_values()
        max_dim = max(len(headings), len(altitudes), len(speeds))
        masks = np.zeros((3, max_dim), dtype=np.float32)
        idx = bs.traf.id2idx(acid)
        if idx < 0:
            return masks

        current_hdg = float(bs.traf.hdg[idx])
        heading_delta = np.asarray([self._wrap_angle_deg(float(value - current_hdg)) for value in headings])
        masks[0, : len(headings)] = (
            (np.abs(heading_delta) <= self.cfg.network.heading_emergency_limit_deg)
            & (np.abs(heading_delta) >= self.cfg.network.heading_min_effective_change_deg)
        ).astype(np.float32)

        current_alt = float(bs.traf.alt[idx] / aero.ft)
        perf = getattr(bs.traf, "perf", None)
        perf_max_alt = (
            float(getattr(perf, "hmax", [self.cfg.command_ranges.max_altitude_ft * aero.ft])[idx] / aero.ft)
            if perf is not None
            else self.cfg.command_ranges.max_altitude_ft
        )
        upper_alt = min(self.cfg.command_ranges.max_altitude_ft, perf_max_alt)
        masks[1, : len(altitudes)] = (
            (altitudes >= self.cfg.command_ranges.min_altitude_ft)
            & (altitudes <= upper_alt)
            & (np.abs(altitudes - current_alt) <= self.cfg.command_ranges.altitude_delta_ft + 1e-6)
            & (np.abs(altitudes - current_alt) >= 100.0)
        ).astype(np.float32)

        current_tas = float(bs.traf.tas[idx] / aero.kts)
        cas_targets = np.asarray([aero.tas2cas(float(value) * aero.kts, bs.traf.alt[idx]) for value in speeds])
        perf_min = float(getattr(perf, "vmin", [0.0])[idx]) if perf is not None else 0.0
        perf_max = float(getattr(perf, "vmax", [np.inf])[idx]) if perf is not None else np.inf
        masks[2, : len(speeds)] = (
            (np.abs(speeds - current_tas) <= self.cfg.command_ranges.speed_delta_kt + 1e-6)
            & (np.abs(speeds - current_tas) >= 1.0)
            & (cas_targets >= perf_min)
            & (cas_targets <= perf_max)
        ).astype(np.float32)
        runtime = self.runtime.get(acid)
        if runtime is not None:
            cmd = runtime.command_state
            if cmd.active and cmd.current_macro in (2, 3, 4):
                branch = cmd.current_macro - 2
                target = self._command_target_value(cmd)
                if target is not None:
                    if branch == 0:
                        duplicate = np.asarray(
                            [abs(self._wrap_angle_deg(float(value - target))) <= self.cfg.runtime.duplicate_heading_tolerance_deg for value in headings]
                        )
                        masks[0, : len(headings)][duplicate] = 0.0
                    elif branch == 1:
                        duplicate = np.abs(altitudes - target) <= self.cfg.runtime.duplicate_altitude_tolerance_ft
                        masks[1, : len(altitudes)][duplicate] = 0.0
                    else:
                        duplicate = np.abs(speeds - target) <= self.cfg.runtime.duplicate_speed_tolerance_kt
                        masks[2, : len(speeds)][duplicate] = 0.0
        return masks

    def get_candidate_parameter_masks(self, acid: str, candidate) -> np.ndarray:
        """Restrict base parameter masks to an LLM candidate's coarse intent."""
        masks = self.get_parameter_masks(acid)
        macro = str(getattr(candidate, "macro_action", ""))
        magnitude = str(getattr(candidate, "magnitude_level", "MEDIUM"))
        idx = bs.traf.id2idx(acid)
        if idx < 0 or macro == "NO_NEW_COMMAND":
            return masks
        headings, altitudes, speeds = self._parameter_values()
        limits = {"SMALL": (0.0, 15.0), "MEDIUM": (15.0, 30.0), "LARGE": (30.0, 45.0)}
        lower, upper = limits.get(magnitude, limits["MEDIUM"])
        if macro in {"TURN_LEFT", "TURN_RIGHT"}:
            current = float(bs.traf.hdg[idx])
            deltas = np.asarray([self._wrap_angle_deg(float(value - current)) for value in headings])
            direction = deltas < 0.0 if macro == "TURN_LEFT" else deltas > 0.0
            masks[0, : len(headings)] *= (
                direction & (np.abs(deltas) > lower + 1e-6) & (np.abs(deltas) <= upper + 1e-6)
            ).astype(np.float32)
        elif macro in {"CLIMB", "DESCEND"}:
            current = float(bs.traf.alt[idx] / aero.ft)
            deltas = altitudes - current
            direction = deltas > 0.0 if macro == "CLIMB" else deltas < 0.0
            masks[1, : len(altitudes)] *= (
                direction & (np.abs(deltas) > lower / 45.0 * 3000.0 + 1e-6) & (np.abs(deltas) <= upper / 45.0 * 3000.0 + 1e-6)
            ).astype(np.float32)
        elif macro in {"ACCELERATE", "DECELERATE"}:
            current = float(bs.traf.tas[idx] / aero.kts)
            deltas = speeds - current
            direction = deltas > 0.0 if macro == "ACCELERATE" else deltas < 0.0
            masks[2, : len(speeds)] *= (
                direction & (np.abs(deltas) > lower / 45.0 * 30.0 + 1e-6) & (np.abs(deltas) <= upper / 45.0 * 30.0 + 1e-6)
            ).astype(np.float32)
        return masks

    def _vertical_group_leaders(
        self, conflict_states: Dict[str, AircraftConflictState]
    ) -> dict[str, str]:
        """Assign one vertical-maneuver authority per active multi-aircraft group."""
        if not self.cfg.runtime.vertical_coordination_enabled:
            return {}
        graph: dict[str, set[str]] = {
            acid: set() for acid, state in conflict_states.items() if state.active
        }
        for acid, state in conflict_states.items():
            if acid not in graph:
                continue
            for contact in state.targets:
                other = contact.callsign
                if other not in graph:
                    continue
                if contact.conflict_flag > 0.0 or contact.temporal_conflict > 0.0 or contact.severity > 0.0:
                    graph[acid].add(other)
                    graph[other].add(acid)

        leaders: dict[str, str] = {}
        seen: set[str] = set()
        for root in sorted(graph):
            if root in seen:
                continue
            stack = [root]
            component: set[str] = set()
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(graph[current] - component)
            seen.update(component)
            if len(component) < self.cfg.runtime.vertical_coordination_min_group_size:
                continue
            ranked: list[tuple[float, str]] = []
            for member in component:
                state = conflict_states[member]
                degree = len(graph[member])
                headroom = self._altitude_headroom_ft(member)
                min_tcpa = min((item.tcpa_s for item in state.targets), default=self.cfg.conflict_lookahead_s)
                # Resolve the most connected and urgent aircraft vertically only
                # when it has enough authority; callsign is a stable tie-breaker.
                score = 2.0 * degree + min(headroom, 3000.0) / 3000.0 - min(min_tcpa, self.cfg.conflict_lookahead_s) / self.cfg.conflict_lookahead_s
                ranked.append((score, member))
            leader = sorted(ranked, key=lambda item: (-item[0], item[1]))[0][1]
            for member in component:
                leaders[member] = leader
        return leaders

    def _altitude_headroom_ft(self, acid: str) -> float:
        idx = bs.traf.id2idx(acid)
        if idx < 0:
            return 0.0
        ranges = self.cfg.command_ranges
        current_alt = float(bs.traf.alt[idx] * 3.28084)
        perf = getattr(bs.traf, "perf", None)
        perf_max_alt = (
            float(getattr(perf, "hmax", [ranges.max_altitude_ft * 0.3048])[idx] / 0.3048)
            if perf is not None
            else ranges.max_altitude_ft
        )
        upper = min(ranges.max_altitude_ft, perf_max_alt)
        return max(0.0, min(current_alt - ranges.min_altitude_ft, upper - current_alt))

    @staticmethod
    def _phase_one_hot(phase: CommandPhase) -> np.ndarray:
        phases = (
            CommandPhase.NORMAL,
            CommandPhase.MANEUVERING,
            CommandPhase.MONITORING,
            CommandPhase.REPLAN_REQUIRED,
            CommandPhase.RECOVERING,
        )
        result = np.zeros(len(phases), dtype=np.float32)
        result[phases.index(phase)] = 1.0
        return result

    def _cooldown_features(self, command_state: CommandState, simt: float) -> np.ndarray:
        scale = max(self.cfg.runtime.replan_cooldown_s, 1e-6)
        return np.asarray(
            [
                np.clip((command_state.cooldown_until_by_macro.get(macro, -1.0) - simt) / scale, 0.0, 1.0)
                for macro in (2, 3, 4)
            ],
            dtype=np.float32,
        )

    def _command_target_value(self, command_state: CommandState) -> float | None:
        if command_state.current_macro == 2:
            value = command_state.target_params.get("heading_deg")
        elif command_state.current_macro == 3:
            value = command_state.target_params.get("altitude_ft")
        elif command_state.current_macro == 4:
            value = command_state.target_params.get("speed_tas_kt", command_state.target_params.get("speed_kt"))
        else:
            return None
        return None if value is None else float(value)

    def _command_target_progress(self, acid: str, command_state: CommandState) -> float:
        if not command_state.active:
            return 0.0
        idx = bs.traf.id2idx(acid)
        target = self._command_target_value(command_state)
        if idx < 0 or target is None:
            return 0.0
        if command_state.current_macro == 2:
            initial = abs(float(command_state.target_params.get("heading_delta_deg", 0.0)))
            remaining = abs(self._wrap_angle_deg(target - float(bs.traf.hdg[idx])))
        elif command_state.current_macro == 3:
            initial = abs(float(command_state.target_params.get("altitude_delta_ft", 0.0)))
            remaining = abs(target - float(bs.traf.alt[idx] / aero.ft))
        elif command_state.current_macro == 4:
            initial = abs(float(command_state.target_params.get("speed_delta_kt", 0.0)))
            remaining = abs(target - float(bs.traf.tas[idx] / aero.kts))
        else:
            return 0.0
        if initial <= 1e-6:
            return 1.0
        return float(np.clip(1.0 - remaining / initial, 0.0, 1.0))

    def _is_duplicate_active_target(self, acid: str, macro_action: int, target: float) -> bool:
        runtime = self.runtime.get(acid)
        if runtime is None:
            return False
        command_state = runtime.command_state
        if not command_state.active or command_state.current_macro != macro_action:
            return False
        previous_target = self._command_target_value(command_state)
        if previous_target is None:
            return False
        if macro_action == 2:
            return abs(self._wrap_angle_deg(target - previous_target)) <= self.cfg.runtime.duplicate_heading_tolerance_deg
        if macro_action == 3:
            return abs(target - previous_target) <= self.cfg.runtime.duplicate_altitude_tolerance_ft
        if macro_action == 4:
            return abs(target - previous_target) <= self.cfg.runtime.duplicate_speed_tolerance_kt
        return False

    def _risk_improved(self, conflict_state: AircraftConflictState, command_state: CommandState) -> bool:
        current = self._risk_snapshot(conflict_state)
        severity_improved = current["severity"] <= command_state.risk_severity_at_issue - self.cfg.runtime.replan_min_severity_drop
        tcpa_improved = (
            np.isfinite(command_state.risk_tcpa_s_at_issue)
            and current["tcpa_s"] >= command_state.risk_tcpa_s_at_issue + self.cfg.runtime.replan_min_tcpa_gain_s
        )
        horizontal_improved = (
            np.isfinite(command_state.risk_horizontal_km_at_issue)
            and current["horizontal_km"] >= command_state.risk_horizontal_km_at_issue + self.cfg.runtime.replan_min_horizontal_gain_km
        )
        vertical_improved = (
            np.isfinite(command_state.risk_vertical_ft_at_issue)
            and current["vertical_ft"] >= command_state.risk_vertical_ft_at_issue + self.cfg.runtime.replan_min_vertical_gain_ft
        )
        targets_reduced = int(current["target_count"]) < command_state.risk_target_count_at_issue
        return bool(severity_improved or tcpa_improved or horizontal_improved or vertical_improved or targets_reduced)

    def _emergency_replan_required(
        self, conflict_state: AircraftConflictState, command_state: CommandState, simt: float
    ) -> bool:
        return bool(
            command_state.active
            and conflict_state.active
            and self.cfg.runtime.emergency_override_enabled
            and simt - command_state.issued_at_s >= self.cfg.runtime.decision_dt - 1e-6
            and (
                conflict_state.collision
                or (
                    conflict_state.loss_of_separation
                    and conflict_state.severity >= self.cfg.runtime.emergency_override_severity
                )
            )
        )

    def _enter_replan(self, command_state: CommandState, simt: float, reason: str) -> None:
        if command_state.phase == CommandPhase.REPLAN_REQUIRED:
            return
        failed_macro = command_state.current_macro
        if failed_macro in (2, 3, 4):
            command_state.cooldown_until_by_macro[failed_macro] = simt + self.cfg.runtime.replan_cooldown_s
        command_state.phase = CommandPhase.REPLAN_REQUIRED
        command_state.replan_reason = reason
        self.reward_stats["replan_count"] += 1.0

    def _refresh_command_phase(self, acid: str, conflict_state: AircraftConflictState, simt: float) -> None:
        runtime = self.runtime.get(acid)
        if runtime is None:
            return
        command_state = runtime.command_state
        command_state.cooldown_until_by_macro = {
            macro: until for macro, until in command_state.cooldown_until_by_macro.items() if until > simt
        }
        if not command_state.active:
            if command_state.phase != CommandPhase.RECOVERING:
                command_state.phase = CommandPhase.NORMAL
            return
        if not conflict_state.active:
            command_state.phase = CommandPhase.RECOVERING
            command_state.last_risk_improved = True
            command_state.stalled_evaluations = 0
            return
        if command_state.phase == CommandPhase.RECOVERING:
            self._enter_replan(command_state, simt, "conflict_reappeared_during_recovery")
            return
        if command_state.phase == CommandPhase.REPLAN_REQUIRED:
            return
        if self._emergency_replan_required(conflict_state, command_state, simt):
            self._enter_replan(command_state, simt, "emergency_risk")
            return
        if simt < command_state.hold_until_s:
            command_state.phase = CommandPhase.MANEUVERING
            return
        if not self.cfg.runtime.replan_enabled:
            command_state.phase = CommandPhase.MONITORING
            return
        if command_state.last_effect_evaluation_s >= simt - self.cfg.runtime.decision_dt + 1e-6:
            return
        improved = self._risk_improved(conflict_state, command_state)
        command_state.last_effect_evaluation_s = simt
        command_state.last_risk_improved = improved
        if improved:
            command_state.stalled_evaluations = 0
            command_state.phase = CommandPhase.MONITORING
            return
        command_state.stalled_evaluations += 1
        command_state.phase = CommandPhase.MONITORING
        if command_state.stalled_evaluations >= self.cfg.runtime.replan_stall_evaluations:
            self._enter_replan(command_state, simt, "risk_not_improving")

    @staticmethod
    def _risk_snapshot(conflict_state: AircraftConflictState | None) -> dict:
        if conflict_state is None:
            return {}
        return {
            "severity": float(conflict_state.severity),
            "tcpa_s": float(min((item.tcpa_s for item in conflict_state.targets), default=float("inf"))),
            "horizontal_km": float(min((item.horiz_km for item in conflict_state.targets), default=float("inf"))),
            "vertical_ft": float(min((item.vert_ft for item in conflict_state.targets), default=float("inf"))),
            "target_count": int(len(conflict_state.targets)),
            "target_ids": tuple(sorted(item.callsign for item in conflict_state.targets)),
        }

    def _altitude_maneuver_available(self, acid: str) -> bool:
        """Return whether the unsigned ALTITUDE macro has useful authority.

        The current five-action policy chooses ALTITUDE before it samples the
        signed continuous parameter.  At an altitude limit, allowing this
        macro causes repeated commands that BlueSky must clip.  A future
        CLIMB/DESCEND action split can mask directions independently; until
        then this conservative guard leaves heading and speed alternatives.
        """
        idx = bs.traf.id2idx(acid)
        if idx < 0:
            return False
        ranges = self.cfg.command_ranges
        current_alt = float(bs.traf.alt[idx] * 3.28084)
        perf = getattr(bs.traf, "perf", None)
        perf_max_alt = (
            float(getattr(perf, "hmax", [ranges.max_altitude_ft * 0.3048])[idx] / 0.3048)
            if perf is not None
            else ranges.max_altitude_ft
        )
        upper = min(ranges.max_altitude_ft, perf_max_alt)
        guard = ranges.altitude_boundary_guard_ft
        return current_alt > ranges.min_altitude_ft + guard and current_alt < upper - guard

    def get_target_aircraft(self, conflict_states: Dict[str, AircraftConflictState]) -> list[str]:
        selected = self.conflict_manager.select_targets(conflict_states, self.cfg.network.max_aircraft)
        active_commands = sorted(
            acid
            for acid, runtime in self.runtime.items()
            if runtime.command_state.current_macro in (2, 3, 4) and acid not in selected
        )
        targets = (selected + active_commands)[: self.cfg.network.max_aircraft]
        self.reward_stats["peak_policy_targets"] = max(
            self.reward_stats.get("peak_policy_targets", 0.0),
            float(len(targets)),
        )
        return targets

    def step_metrics(self) -> dict[str, float]:
        return dict(self.reward_stats)

    def register_command(
        self,
        acid: str,
        macro_action: int,
        parameters: dict,
        snapshot: CommandSnapshot | None = None,
        risk_snapshot: dict[str, float] | None = None,
    ) -> None:
        runtime = self.runtime[acid]
        cmd = runtime.command_state
        cmd.last_macro = cmd.current_macro
        if cmd.current_macro not in (0, macro_action):
            cmd.oscillation_count += 1
        cmd.current_macro = macro_action
        cmd.target_params = dict(parameters)
        cmd.issued_at_s = float(bs.sim.simt)
        cmd.hold_until_s = float(bs.sim.simt + self.cfg.runtime.min_command_hold_s)
        cmd.active = macro_action in (2, 3, 4)
        cmd.command_count += 1
        risk_snapshot = risk_snapshot or {}
        cmd.risk_severity_at_issue = float(risk_snapshot.get("severity", 0.0))
        cmd.risk_tcpa_s_at_issue = float(risk_snapshot.get("tcpa_s", float("inf")))
        cmd.risk_horizontal_km_at_issue = float(risk_snapshot.get("horizontal_km", float("inf")))
        cmd.risk_vertical_ft_at_issue = float(risk_snapshot.get("vertical_ft", float("inf")))
        cmd.risk_target_count_at_issue = int(risk_snapshot.get("target_count", 0))
        cmd.risk_target_ids_at_issue = tuple(risk_snapshot.get("target_ids", ()))
        cmd.phase = CommandPhase.MANEUVERING if macro_action in (2, 3, 4) else CommandPhase.NORMAL
        cmd.last_effect_evaluation_s = -1.0
        cmd.last_risk_improved = False
        cmd.stalled_evaluations = 0
        cmd.replan_reason = ""
        if snapshot is not None:
            cmd.snapshot = snapshot
        runtime.command_history.append(macro_action)
        if len(runtime.command_history) > self.cfg.max_command_history:
            runtime.command_history = runtime.command_history[-self.cfg.max_command_history :]
        runtime.last_command_macro = macro_action
        runtime.last_command_time_s = float(bs.sim.simt)
        command_penalty = self.cfg.reward.command_issue * self.cfg.reward.reward_scale
        runtime.pending_reward += command_penalty
        self.reward_stats["episode_return"] += command_penalty
        self.reward_stats["command_count"] += 1.0
        if macro_action in (2, 3, 4) and self.reward_stats["first_intervention_s"] < 0.0:
            self.reward_stats["first_intervention_s"] = max(
                0.0,
                float(bs.sim.simt) - self.episode_start_time_s,
            )
        if cmd.last_macro not in (0, macro_action):
            oscillation_penalty = self.cfg.reward.oscillation * self.cfg.reward.reward_scale
            runtime.pending_reward += oscillation_penalty
            self.reward_stats["episode_return"] += oscillation_penalty
            self.reward_stats["oscillation_count"] += 1.0

    def mark_release(self, acid: str) -> None:
        runtime = self.runtime.get(acid)
        if runtime is None:
            return
        runtime.command_state.current_macro = 1
        runtime.command_state.active = False
        runtime.command_state.phase = CommandPhase.RECOVERING
        runtime.command_state.target_params.clear()
        runtime.command_state.cooldown_until_by_macro.clear()

    def record_command_failure(self, acid: str) -> None:
        runtime = self.runtime.get(acid)
        if runtime is None:
            return
        penalty = self.cfg.reward.command_failure * self.cfg.reward.reward_scale
        runtime.pending_reward += penalty
        self.reward_stats["episode_return"] += penalty
        self.reward_stats["command_failures"] += 1.0

    def _resume_ready(self, runtime: AircraftRuntime, conf: AircraftConflictState, simt: float) -> bool:
        """Require continuous clear separation before any route recovery."""
        cmd = runtime.command_state
        if conf.active or not conf.allow_resume:
            runtime.command_clear_since_s = None
            return False
        if runtime.command_clear_since_s is None:
            runtime.command_clear_since_s = simt
            return False
        resume_at = max(
            cmd.hold_until_s,
            runtime.command_clear_since_s + self.cfg.runtime.auto_resume_after_clear_s,
        )
        return bool(simt >= resume_at)

    def _mark_terminal_arrival(self, acid: str, runtime: AircraftRuntime) -> None:
        """Complete a one-waypoint route when BlueSky switches LNAV off at its end."""
        if runtime.arrival_rewarded:
            return
        arrival_reward = self.cfg.reward.arrival * self.cfg.reward.reward_scale
        runtime.arrival_rewarded = True
        runtime.pending_reward += arrival_reward
        self.reward_stats["arrival"] += 1.0
        self.reward_stats["episode_return"] += arrival_reward
        self.arrived_aircraft.add(acid)

    def auto_resume_clear_commands(self, conflict_states: Dict[str, AircraftConflictState], simt: float) -> list[str]:
        resumed: list[str] = []
        for acid, runtime in list(self.runtime.items()):
            if acid in self.arrived_aircraft:
                continue
            cmd = runtime.command_state
            conf = conflict_states.get(acid, AircraftConflictState(acid))
            idx = bs.traf.id2idx(acid)
            lnav_lost = (
                idx >= 0
                and cmd.has_snapshot
                and cmd.snapshot.swlnav
                and not bool(bs.traf.swlnav[idx])
            )
            if not cmd.active:
                # Reaching the final waypoint in BlueSky turns LNAV off. It
                # is an arrival event for these single-destination scenarios,
                # not a request to DIRECT back to the same waypoint.
                if lnav_lost:
                    self._mark_terminal_arrival(acid, runtime)
                runtime.command_clear_since_s = None
                continue
            if not self._resume_ready(runtime, conf, simt):
                continue
            result = self.execute_macro(acid, 1, np.zeros(3, dtype=np.float32), np.empty(0, dtype=np.float32))
            if result.success:
                resumed.append(acid)
                runtime.command_clear_since_s = None
            else:
                self.record_command_failure(acid)
                if self.cfg.runtime.fail_on_command_error:
                    raise RuntimeError(f"Automatic RESUME_ROUTE failed for {acid}: {result.message}")
        return resumed

    def execute_macro(
        self,
        acid: str,
        macro_action: int,
        normalized_params: np.ndarray,
        current_obs: np.ndarray,
        conflict_state: AircraftConflictState | None = None,
    ) -> ExecutionResult:
        idx = bs.traf.id2idx(acid)
        if idx < 0:
            return ExecutionResult(False, f"{acid} not found", "NOOP")
        runtime = self.runtime[acid]
        cmd = runtime.command_state
        risk_snapshot = self._risk_snapshot(conflict_state)
        if macro_action in (2, 3, 4) and not cmd.active:
            snapshot = self.executor.snapshot_route(acid)
        else:
            snapshot = cmd.snapshot if cmd.snapshot.has_snapshot else self.executor.snapshot_route(acid)
        if macro_action == 0:
            return ExecutionResult(True, "NO_NEW_COMMAND", "NO_NEW_COMMAND")
        if macro_action == 1:
            result = self.executor.resume_route(acid, snapshot)
            if result.success:
                self.register_command(acid, macro_action, {}, snapshot, risk_snapshot)
                self.mark_release(acid)
            return result
        current_speed = float(bs.traf.cas[idx] * 1.94384)
        current_alt = float(bs.traf.alt[idx] * 3.28084)
        current_hdg = float(bs.traf.hdg[idx])
        if macro_action == 2:
            delta = float(normalized_params[0]) * self.cfg.command_ranges.heading_offset_deg
            heading = (current_hdg + delta) % 360.0
            result = self.executor.apply_heading(acid, heading)
            if result.success:
                self.register_command(acid, macro_action, {"heading_delta_deg": delta, "heading_deg": heading}, snapshot, risk_snapshot)
            return result
        if macro_action == 3:
            delta = float(normalized_params[1]) * self.cfg.command_ranges.altitude_delta_ft
            perf = getattr(bs.traf, "perf", None)
            perf_max_alt = float(getattr(perf, "hmax", [self.cfg.command_ranges.max_altitude_ft * 0.3048])[idx] / 0.3048) if perf is not None else self.cfg.command_ranges.max_altitude_ft
            altitude = float(np.clip(current_alt + delta, self.cfg.command_ranges.min_altitude_ft, min(self.cfg.command_ranges.max_altitude_ft, perf_max_alt)))
            actual_delta = altitude - current_alt
            if abs(actual_delta) < 1.0:
                return ExecutionResult(False, "altitude command has no remaining authority", f"ALT {acid} {altitude:.3f}")
            result = self.executor.apply_altitude(acid, altitude)
            if result.success:
                self.register_command(acid, macro_action, {"altitude_delta_ft": actual_delta, "altitude_ft": altitude}, snapshot, risk_snapshot)
            return result
        if macro_action == 4:
            delta = float(normalized_params[2]) * self.cfg.command_ranges.speed_delta_kt
            perf = getattr(bs.traf, "perf", None)
            perf_min = float(getattr(perf, "vmin", [self.cfg.command_ranges.min_speed_kt * 0.514444])[idx] / 0.514444) if perf is not None else self.cfg.command_ranges.min_speed_kt
            perf_max = float(getattr(perf, "vmax", [self.cfg.command_ranges.max_speed_kt * 0.514444])[idx] / 0.514444) if perf is not None else self.cfg.command_ranges.max_speed_kt
            speed = float(np.clip(current_speed + delta, max(self.cfg.command_ranges.min_speed_kt, perf_min), min(self.cfg.command_ranges.max_speed_kt, perf_max)))
            result = self.executor.apply_speed(acid, speed)
            if result.success:
                self.register_command(acid, macro_action, {"speed_delta_kt": delta, "speed_kt": speed}, snapshot, risk_snapshot)
            return result
        return ExecutionResult(False, f"Unknown macro {macro_action}", "NOOP")

    def execute_discrete_macro(
        self,
        acid: str,
        macro_action: int,
        parameter_action_index: int,
        current_obs: np.ndarray,
        conflict_state: AircraftConflictState | None = None,
    ) -> ExecutionResult:
        idx = bs.traf.id2idx(acid)
        if idx < 0:
            return ExecutionResult(False, f"{acid} not found", "NOOP")
        if macro_action in (0, 1):
            return self.execute_macro(acid, macro_action, np.zeros(3, dtype=np.float32), current_obs, conflict_state)
        headings, altitudes, speeds = self._parameter_values()
        branches = (headings, altitudes, speeds)
        branch = macro_action - 2
        if branch < 0 or branch >= len(branches) or parameter_action_index < 0 or parameter_action_index >= len(branches[branch]):
            return ExecutionResult(False, "invalid discrete parameter action", "NOOP")
        target = float(branches[branch][parameter_action_index])
        if self._is_duplicate_active_target(acid, macro_action, target):
            self.reward_stats["duplicate_commands"] += 1.0
            return ExecutionResult(True, "duplicate_active_target", "", applied=False)
        if self.get_parameter_masks(acid)[branch, parameter_action_index] <= 0.0:
            return ExecutionResult(False, "discrete parameter action is masked", "NOOP")
        runtime = self.runtime[acid]
        cmd = runtime.command_state
        snapshot = self.executor.snapshot_route(acid) if not cmd.active else (cmd.snapshot if cmd.snapshot.has_snapshot else self.executor.snapshot_route(acid))
        risk_snapshot = self._risk_snapshot(conflict_state)
        if macro_action == 2:
            current_hdg = float(bs.traf.hdg[idx])
            delta = self._wrap_angle_deg(target - current_hdg)
            result = self.executor.apply_heading(acid, target)
            if result.success:
                self.register_command(acid, macro_action, {"heading_deg": target, "heading_delta_deg": delta}, snapshot, risk_snapshot)
            return result
        if macro_action == 3:
            current_alt = float(bs.traf.alt[idx] / aero.ft)
            result = self.executor.apply_altitude(acid, target)
            if result.success:
                self.register_command(acid, macro_action, {"altitude_ft": target, "altitude_delta_ft": target - current_alt}, snapshot, risk_snapshot)
            return result
        current_tas = float(bs.traf.tas[idx] / aero.kts)
        result = self.executor.apply_tas_speed(acid, target)
        if result.success:
            self.register_command(acid, macro_action, {"speed_tas_kt": target, "speed_delta_kt": target - current_tas}, snapshot, risk_snapshot)
        return result

    def decision_due(self, simt: float) -> bool:
        return self.last_decision_time_s < 0.0 or simt - self.last_decision_time_s >= self.cfg.runtime.decision_dt - 1e-6

    def state_update_due(self, simt: float) -> bool:
        return self.last_state_update_time_s < 0.0 or simt - self.last_state_update_time_s >= self.cfg.runtime.state_update_dt - 1e-6
