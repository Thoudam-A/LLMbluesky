from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


def save_checkpoint_file(path: Path, agent, episode: int, config, extra: dict | None = None) -> None:
    if is_dataclass(config):
        config_payload = asdict(config)
    else:
        config_payload = config
    payload = {
        "checkpoint_version": 4,
        "episode": episode,
        "config": config_payload,
        "agent": agent.state_dict(),
        "extra": extra or {},
        "seed": agent.seed,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "normalization_state": extra.get("normalization_state") if extra else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _cpu_byte_tensor(state: Any) -> torch.Tensor:
    if isinstance(state, torch.Tensor):
        return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(state, dtype=torch.uint8, device="cpu").contiguous()


def load_checkpoint_file(path: Path, agent, restore_rng: bool = True) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    checkpoint_network = payload.get("config", {}).get("network", {})
    checkpoint_policy_type = checkpoint_network.get("parameter_policy_type", "continuous")
    if checkpoint_policy_type != agent.cfg.parameter_policy_type:
        raise RuntimeError(
            "Checkpoint parameter policy is incompatible "
            f"(checkpoint={checkpoint_policy_type}, current={agent.cfg.parameter_policy_type}). "
            "Continuous and discrete parameter actors require separate checkpoints."
        )
    expected_dimensions = {
        "local_obs_dim": int(agent.cfg.local_obs_dim),
        "global_slot_dim": int(agent.cfg.global_slot_dim),
    }
    mismatches = {
        key: (int(checkpoint_network[key]), expected)
        for key, expected in expected_dimensions.items()
        if key in checkpoint_network and int(checkpoint_network[key]) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={actual}, current={expected}"
            for key, (actual, expected) in mismatches.items()
        )
        raise RuntimeError(
            "Checkpoint observation dimensions are incompatible with the conditioned separation model "
            f"({details}). Train a new checkpoint with the current configuration."
        )
    agent_state = payload.get("agent", payload.get("model"))
    agent.load_state_dict(agent_state)
    rng_state = payload.get("rng_state")
    if restore_rng and rng_state:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(_cpu_byte_tensor(rng_state["torch"]))
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all([_cpu_byte_tensor(state) for state in rng_state["cuda"]])
    return payload
