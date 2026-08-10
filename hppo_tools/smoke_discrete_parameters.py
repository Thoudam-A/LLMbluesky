r"""Fast tensor-level checks for the discrete H-PPO parameter policy.

Run from the repository root:
    python .\hppo_tools\smoke_discrete_parameters.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "bluesky_project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hppo_runtime.core import HPPOAgent, HPPOCoreConfig


def main() -> None:
    cfg = HPPOCoreConfig(local_obs_dim=72, global_slot_dim=26, max_aircraft=4, mini_batch_size=4, epochs=1)
    agent = HPPOAgent(cfg, device="cpu", seed=11)
    observations = np.zeros((4, cfg.local_obs_dim), dtype=np.float32)
    macro_masks = np.tile(np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32), (4, 1))
    parameter_masks = np.zeros((4, 3, cfg.parameter_max_dim), dtype=np.float32)
    parameter_masks[:, 0, [8, 9, 10]] = 1.0
    actions = agent.act(observations, macro_masks, parameter_masks=parameter_masks)

    for index, action in enumerate(actions):
        assert action["macro_action"] == 2
        assert action["parameter_branch"] == 0
        assert action["parameter_action_index"] in {8, 9, 10}
        agent.buffer.add(
            episode_id=1,
            step_index=index,
            trajectory_id=f"KL{index}",
            acid=f"KL{index}",
            local_obs=observations[index],
            global_state=np.zeros((cfg.max_aircraft, cfg.global_slot_dim), dtype=np.float32),
            presence_mask=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
            target_index=0,
            action_mask=macro_masks[index],
            action_prior=np.full(5, 0.2, dtype=np.float32),
            prior_weight=0.0,
            macro_action=action["macro_action"],
            raw_params=action["raw_params"],
            squashed_params=action["squashed_params"],
            selected_param_index=action["selected_param_index"],
            parameter_branch=action["parameter_branch"],
            parameter_action_index=action["parameter_action_index"],
            parameter_mask=action["parameter_mask"],
            parameter_probs=action["parameter_probs"],
            parameter_logp=action["parameter_logp"],
            parameter_entropy=action["parameter_entropy"],
            discrete_logp=action["discrete_logp"],
            discrete_entropy=action["discrete_entropy"],
            discrete_probs=action["discrete_probs"],
            continuous_logp=action["continuous_logp"],
            continuous_entropy=action["continuous_entropy"],
            continuous_mean=action["continuous_mean"],
            continuous_std=action["continuous_std"],
            value=0.0,
            next_value=0.0,
            reward=float(index),
            terminated=True,
            truncated=False,
            policy_version=0,
        )

    metrics = agent.update()
    assert metrics["nan_detected"] == 0.0, metrics
    print("discrete parameter smoke test passed")


if __name__ == "__main__":
    main()
