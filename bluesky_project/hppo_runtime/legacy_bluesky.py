from __future__ import annotations

from typing import Any

import bluesky as bs


def configure_simulation(speed_multiplier: float) -> None:
    """Apply the H-PPO speed setting across legacy and current BlueSky APIs."""
    if speed_multiplier <= 0.0:
        bs.sim.fastforward()
        return
    bs.sim.op()
    setter = getattr(bs.sim, "set_dtmult", None) or getattr(bs.sim, "setDtMultiplier", None)
    if setter is None:
        raise RuntimeError("BlueSky simulation does not expose a speed multiplier API")
    setter(float(speed_multiplier))


def publish_event(name: bytes, payload: dict[str, Any]) -> None:
    """Publish a low-frequency H-PPO record to the QtGL client when connected."""
    sender = getattr(bs, "sim", None)
    if sender is None or not hasattr(sender, "send_event"):
        return
    sender.send_event(name, payload)
