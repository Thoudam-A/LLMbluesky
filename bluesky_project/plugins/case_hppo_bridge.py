"""Legacy BlueSky simulation-plugin bridge for the RL+LLM H-PPO runtime."""
from __future__ import annotations

import os

from bluesky import sim

from hppo_runtime import TrainingManager, default_config
from hppo_runtime.legacy_bluesky import configure_simulation, publish_event


manager = None
enabled = True


def _emit(record):
    publish_event(b"HPPO_EVENT", record)


def init_plugin():
    global manager
    cfg = default_config()
    cfg.logging.output_dir.mkdir(parents=True, exist_ok=True)
    manager = TrainingManager(cfg)
    manager.event_sink = _emit
    configure_simulation(cfg.runtime.speed_multiplier)
    # A legacy detached node starts in HOLD. The migrated lifecycle manager
    # needs simulation time to advance before it can load and spawn a route.
    sim.op()

    config = {
        "plugin_name": "CASE_HPPO_BRIDGE",
        "plugin_type": "sim",
        "update_interval": cfg.runtime.state_update_dt,
        "update": update,
        "reset": reset,
    }
    stackfunctions = {
        "HPPO": [
            "HPPO [START|STOP|STATUS]",
            "txt",
            hppo_command,
            "Control the migrated RL+LLM H-PPO runtime.",
        ]
    }
    return config, stackfunctions


def update():
    if manager is None or not enabled:
        return
    manager.update()


def reset():
    if manager is not None:
        manager.reset()


def hppo_command(command="STATUS"):
    global enabled
    action = str(command or "STATUS").strip().upper()
    parts = action.split()
    if action == "START":
        enabled = True
        sim.op()
        return True, "H-PPO runtime started"
    if action == "STOP":
        enabled = False
        return True, "H-PPO runtime stopped; existing commands are retained"
    if action == "STATUS":
        if manager is None:
            return False, "H-PPO runtime is not initialized"
        return True, "H-PPO mode=%s phase=%s episode=%d enabled=%s" % (
            manager.mode,
            manager.phase.value,
            manager.completed_episodes,
            enabled,
        )
    if parts and parts[0] in {"HSEP", "SEPARATION"}:
        if manager is None:
            return False, "H-PPO runtime is not initialized"
        if len(parts) not in {2, 3}:
            return False, "Usage: HPPO HSEP <value> [NM|KM]"
        try:
            value = float(parts[1])
        except ValueError:
            return False, "Horizontal separation must be numeric"
        unit = parts[2] if len(parts) == 3 else "NM"
        if unit == "NM":
            value_km = value * 1.852
        elif unit == "KM":
            value_km = value
        else:
            return False, "Horizontal separation unit must be NM or KM"
        try:
            _, current = manager.set_horizontal_separation_km(value_km, source="stack")
        except ValueError as exc:
            return False, str(exc)
        return True, "H-PPO horizontal separation set to %.2f NM (%.3f km)" % (current / 1.852, current)
    return False, "Usage: HPPO START|STOP|STATUS|HSEP <value> [NM|KM]"
