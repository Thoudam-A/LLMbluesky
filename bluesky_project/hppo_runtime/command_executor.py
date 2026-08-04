from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import bluesky as bs
from bluesky.tools import aero

from .config import HPPOConfig


@dataclass(slots=True)
class CommandSnapshot:
    has_snapshot: bool = False
    swlnav: bool = False
    swvnav: bool = False
    swvnavspd: bool = False
    selspd: float = 0.0
    selalt: float = 0.0
    active_waypoint: str = ""
    active_route_index: int = -1
    dest: str = ""
    orig: str = ""


class CommandPhase(str, Enum):
    """Lifecycle of an effective conflict-resolution command."""

    NORMAL = "NORMAL"
    MANEUVERING = "MANEUVERING"
    MONITORING = "MONITORING"
    REPLAN_REQUIRED = "REPLAN_REQUIRED"
    RECOVERING = "RECOVERING"


@dataclass(slots=True)
class CommandState:
    current_macro: int = 0
    target_params: dict = field(default_factory=dict)
    issued_at_s: float = -1.0
    hold_until_s: float = -1.0
    active: bool = False
    snapshot: CommandSnapshot = field(default_factory=CommandSnapshot)
    command_count: int = 0
    oscillation_count: int = 0
    last_macro: int = 0
    risk_severity_at_issue: float = 0.0
    risk_tcpa_s_at_issue: float = float("inf")
    risk_horizontal_km_at_issue: float = float("inf")
    risk_vertical_ft_at_issue: float = float("inf")
    risk_target_ids_at_issue: tuple[str, ...] = ()
    risk_target_count_at_issue: int = 0
    phase: CommandPhase = CommandPhase.NORMAL
    last_effect_evaluation_s: float = -1.0
    last_risk_improved: bool = False
    stalled_evaluations: int = 0
    replan_reason: str = ""
    cooldown_until_by_macro: dict[int, float] = field(default_factory=dict)

    @property
    def has_snapshot(self) -> bool:
        return self.snapshot.has_snapshot


@dataclass(slots=True)
class ExecutionResult:
    success: bool
    message: str = ""
    command_text: str = ""
    # A policy decision may be accepted without issuing a new BlueSky command,
    # for example when it repeats an already active absolute target.
    applied: bool = False


class CommandExecutor:
    def __init__(self, config: HPPOConfig):
        self.cfg = config

    def snapshot_route(self, acid: str) -> CommandSnapshot:
        idx = bs.traf.id2idx(acid)
        if idx < 0:
            return CommandSnapshot()
        snapshot = CommandSnapshot(has_snapshot=True)
        snapshot.swlnav = bool(bs.traf.swlnav[idx])
        snapshot.swvnav = bool(bs.traf.swvnav[idx])
        snapshot.swvnavspd = bool(bs.traf.swvnavspd[idx])
        snapshot.selspd = float(bs.traf.selspd[idx] / aero.kts)
        snapshot.selalt = float(bs.traf.selalt[idx] / aero.ft)
        snapshot.dest = str(getattr(bs.traf.ap, "dest", [""])[idx]) if hasattr(bs.traf.ap, "dest") else ""
        snapshot.orig = str(getattr(bs.traf.ap, "orig", [""])[idx]) if hasattr(bs.traf.ap, "orig") else ""
        route = bs.traf.ap.route[idx] if idx < len(bs.traf.ap.route) else None
        if route is not None and getattr(route, "nwp", 0) > 0 and getattr(route, "iactwp", -1) >= 0:
            snapshot.active_route_index = int(route.iactwp)
            snapshot.active_waypoint = str(route.wpname[route.iactwp])
        return snapshot

    def apply_heading(self, acid: str, heading_deg: float) -> ExecutionResult:
        idx = bs.traf.id2idx(acid)
        if idx < 0:
            return ExecutionResult(False, f"{acid} not found", f"HDG {acid}")
        ret = bs.traf.ap.selhdgcmd(idx, float(heading_deg) % 360.0)
        if ret is False or (isinstance(ret, tuple) and ret and ret[0] is False):
            return ExecutionResult(False, str(ret[1]) if isinstance(ret, tuple) and len(ret) > 1 else "HDG failed", f"HDG {acid} {heading_deg:.3f}")
        return ExecutionResult(True, "heading applied", f"HDG {acid} {heading_deg:.3f}", applied=True)

    def apply_altitude(self, acid: str, altitude_ft: float) -> ExecutionResult:
        idx = bs.traf.id2idx(acid)
        if idx < 0:
            return ExecutionResult(False, f"{acid} not found", f"ALT {acid}")
        ret = bs.traf.ap.selaltcmd(idx, float(altitude_ft) * aero.ft)
        if ret is False or (isinstance(ret, tuple) and ret and ret[0] is False):
            return ExecutionResult(False, str(ret[1]) if isinstance(ret, tuple) and len(ret) > 1 else "ALT failed", f"ALT {acid} {altitude_ft:.3f}")
        return ExecutionResult(True, "altitude applied", f"ALT {acid} {altitude_ft:.3f}", applied=True)

    def apply_speed(self, acid: str, speed_kt: float) -> ExecutionResult:
        """Apply a CAS speed target (legacy path and route restoration)."""
        idx = bs.traf.id2idx(acid)
        if idx < 0:
            return ExecutionResult(False, f"{acid} not found", f"SPD {acid}")
        ret = bs.traf.ap.selspdcmd(idx, float(speed_kt) * aero.kts)
        if ret is False or (isinstance(ret, tuple) and ret and ret[0] is False):
            return ExecutionResult(False, str(ret[1]) if isinstance(ret, tuple) and len(ret) > 1 else "SPD failed", f"SPD {acid} {speed_kt:.3f}")
        return ExecutionResult(True, "speed applied", f"SPD {acid} {speed_kt:.3f}", applied=True)

    def apply_tas_speed(self, acid: str, speed_tas_kt: float) -> ExecutionResult:
        """Apply a TAS target by converting it to BlueSky's CAS command unit."""
        idx = bs.traf.id2idx(acid)
        if idx < 0:
            return ExecutionResult(False, f"{acid} not found", f"SPD {acid}")
        altitude_m = float(bs.traf.alt[idx])
        cas_mps = float(aero.tas2cas(float(speed_tas_kt) * aero.kts, altitude_m))
        ret = bs.traf.ap.selspdcmd(idx, cas_mps)
        cas_kt = cas_mps / aero.kts
        if ret is False or (isinstance(ret, tuple) and ret and ret[0] is False):
            return ExecutionResult(
                False,
                str(ret[1]) if isinstance(ret, tuple) and len(ret) > 1 else "SPD failed",
                f"SPD {acid} TAS {speed_tas_kt:.3f}",
            )
        return ExecutionResult(True, "TAS speed applied", f"SPD {acid} TAS {speed_tas_kt:.3f} (CAS {cas_kt:.3f})", applied=True)

    def resume_route(self, acid: str, snapshot: CommandSnapshot) -> ExecutionResult:
        if not snapshot.has_snapshot:
            return ExecutionResult(False, "No route snapshot available", "RESUME_ROUTE")
        idx = bs.traf.id2idx(acid)
        if idx < 0:
            return ExecutionResult(False, f"{acid} no longer exists", "RESUME_ROUTE")
        route = bs.traf.ap.route[idx]
        if route is None or getattr(route, "nwp", 0) <= 0:
            return ExecutionResult(False, f"{acid} has no route to resume", "RESUME_ROUTE")
        # RESUME_ROUTE returns directly to the scenario destination. The current
        # H-PPO scenarios contain one destination waypoint, but selecting the
        # final waypoint also keeps the behavior correct for longer routes.
        active_idx = route.nwp - 1
        if active_idx is not None and active_idx >= 0 and active_idx < route.nwp:
            try:
                route.direct(idx, route.wpname[active_idx])
            except Exception as exc:
                return ExecutionResult(False, f"route.direct failed: {exc}", "RESUME_ROUTE")
        restore_results = []
        if snapshot.selspd > 0.0:
            restore_results.append(self.apply_speed(acid, snapshot.selspd))
        if snapshot.selalt > 0.0:
            restore_results.append(self.apply_altitude(acid, snapshot.selalt))
        failed = [item for item in restore_results if not item.success]
        if failed:
            return ExecutionResult(False, "; ".join(item.message for item in failed), "RESUME_ROUTE")

        lnav_ret = bs.traf.ap.setLNAV(idx, True)
        if lnav_ret is False or (isinstance(lnav_ret, tuple) and lnav_ret and lnav_ret[0] is False):
            return ExecutionResult(False, str(lnav_ret[1]) if isinstance(lnav_ret, tuple) and len(lnav_ret) > 1 else "LNAV ON failed", "RESUME_ROUTE")
        if snapshot.swvnav:
            vnav_ret = bs.traf.ap.setVNAV(idx, True)
            if vnav_ret is False or (isinstance(vnav_ret, tuple) and vnav_ret and vnav_ret[0] is False):
                return ExecutionResult(False, str(vnav_ret[1]) if isinstance(vnav_ret, tuple) and len(vnav_ret) > 1 else "VNAV ON failed", "RESUME_ROUTE")
        elif not snapshot.swvnav:
            bs.traf.ap.setVNAV(idx, False)
        if hasattr(bs.traf, "swvnavspd") and not snapshot.swvnavspd:
            bs.traf.swvnavspd[idx] = False
        return ExecutionResult(True, "route, altitude and speed restored", "RESUME_ROUTE", applied=True)

    def restore_snapshot(self, acid: str, snapshot: CommandSnapshot) -> list[ExecutionResult]:
        results = []
        if snapshot.selspd > 0.0:
            results.append(self.apply_speed(acid, snapshot.selspd))
        if snapshot.selalt > 0.0:
            results.append(self.apply_altitude(acid, snapshot.selalt))
        if snapshot.swlnav:
            idx = bs.traf.id2idx(acid)
            ret = bs.traf.ap.setLNAV(idx, True) if idx >= 0 else False
            results.append(ExecutionResult(not (ret is False or (isinstance(ret, tuple) and ret and ret[0] is False)), "LNAV restored", f"LNAV {acid} ON"))
        if snapshot.swvnav:
            idx = bs.traf.id2idx(acid)
            ret = bs.traf.ap.setVNAV(idx, True) if idx >= 0 else False
            results.append(ExecutionResult(not (ret is False or (isinstance(ret, tuple) and ret and ret[0] is False)), "VNAV restored", f"VNAV {acid} ON"))
        return results
