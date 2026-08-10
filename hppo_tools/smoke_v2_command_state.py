"""Pure-Python smoke checks for the V2 command lifecycle and replan gate."""
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "bluesky_project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hppo_runtime.command_executor import CommandPhase, CommandState
from hppo_runtime.config import default_config
from hppo_runtime.conflict_manager import AircraftConflictState, ConflictContact
from hppo_runtime.hppo_environment import AircraftRuntime, HPPOEnvironment


def conflict(severity: float, tcpa_s: float, horizontal_km: float, vertical_ft: float) -> AircraftConflictState:
    contact = ConflictContact(
        callsign="KL1",
        horiz_km=horizontal_km,
        vert_ft=vertical_ft,
        dcpa_km=horizontal_km,
        predicted_vert_ft=vertical_ft,
        rel_bearing_deg=0.0,
        rel_bearing_sin=0.0,
        rel_bearing_cos=1.0,
        rel_alt_norm=0.0,
        rel_speed_norm=0.0,
        rel_track_sin=0.0,
        rel_track_cos=1.0,
        tcpa_s=tcpa_s,
        time_gap_s=0.0,
        temporal_conflict=0.0,
        conflict_flag=1.0,
        severity=severity,
    )
    return AircraftConflictState(
        callsign="KL0",
        active=True,
        severity=severity,
        targets=[contact],
        predicted_conflict=True,
    )


def main() -> None:
    env = object.__new__(HPPOEnvironment)
    env.cfg = default_config()
    env.reward_stats = {"replan_count": 0.0}
    cmd = CommandState(
        current_macro=3,
        target_params={"altitude_ft": 38000.0, "altitude_delta_ft": 3000.0},
        issued_at_s=0.0,
        hold_until_s=10.0,
        active=True,
        risk_severity_at_issue=0.9,
        risk_tcpa_s_at_issue=40.0,
        risk_horizontal_km_at_issue=5.0,
        risk_vertical_ft_at_issue=200.0,
        risk_target_count_at_issue=1,
        phase=CommandPhase.MANEUVERING,
    )
    env.runtime = {"KL0": AircraftRuntime(callsign="KL0", command_state=cmd)}

    assert env._is_duplicate_active_target("KL0", 3, 38000.0)
    assert not env._is_duplicate_active_target("KL0", 3, 39000.0)

    stalled = conflict(0.9, 35.0, 5.0, 200.0)
    env._refresh_command_phase("KL0", stalled, 10.0)
    assert cmd.phase == CommandPhase.MONITORING and cmd.stalled_evaluations == 1
    env._refresh_command_phase("KL0", stalled, 15.0)
    assert cmd.phase == CommandPhase.REPLAN_REQUIRED
    assert cmd.cooldown_until_by_macro[3] > 15.0
    assert env.reward_stats["replan_count"] == 1.0

    cmd.phase = CommandPhase.MONITORING
    cmd.stalled_evaluations = 1
    improved = conflict(0.7, 55.0, 6.0, 500.0)
    env._refresh_command_phase("KL0", improved, 20.0)
    assert cmd.phase == CommandPhase.MONITORING
    assert cmd.last_risk_improved and cmd.stalled_evaluations == 0
    print("V2 command-state smoke test passed")


if __name__ == "__main__":
    main()
