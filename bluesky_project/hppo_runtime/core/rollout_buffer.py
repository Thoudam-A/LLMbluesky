from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass(slots=True)
class RolloutItem:
    episode_id: int
    step_index: int
    trajectory_id: str
    acid: str
    local_obs: np.ndarray
    global_state: np.ndarray
    presence_mask: np.ndarray
    target_index: int
    action_mask: np.ndarray
    action_prior: np.ndarray
    prior_weight: float
    macro_action: int
    raw_params: np.ndarray
    squashed_params: np.ndarray
    selected_param_index: int
    parameter_branch: int
    parameter_action_index: int
    parameter_mask: np.ndarray
    parameter_probs: np.ndarray
    parameter_logp: float
    parameter_entropy: float
    discrete_logp: float
    discrete_entropy: float
    discrete_probs: np.ndarray
    continuous_logp: float
    continuous_entropy: float
    continuous_mean: np.ndarray
    continuous_std: np.ndarray
    candidate_mode: bool
    candidate_features: np.ndarray
    candidate_mask: np.ndarray
    candidate_macro_actions: np.ndarray
    candidate_index: int
    candidate_logp: float
    candidate_entropy: float
    value: float
    next_value: float
    reward: float
    terminated: bool
    truncated: bool
    policy_version: int
    advantage: float = 0.0
    return_: float = 0.0


class HPPORolloutBuffer:
    def __init__(self, candidate_pool_size: int = 6, candidate_feature_dim: int = 18):
        self.items: list[RolloutItem] = []
        self.candidate_pool_size = candidate_pool_size
        self.candidate_feature_dim = candidate_feature_dim

    def clear(self) -> None:
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)

    def add(self, **kwargs) -> None:
        kwargs.setdefault("action_prior", np.full(5, 0.2, dtype=np.float32))
        kwargs.setdefault("prior_weight", 0.0)
        kwargs.setdefault("candidate_mode", False)
        kwargs.setdefault("candidate_features", np.zeros((self.candidate_pool_size, self.candidate_feature_dim), dtype=np.float32))
        kwargs.setdefault("candidate_mask", np.zeros(self.candidate_pool_size, dtype=np.float32))
        kwargs.setdefault("candidate_macro_actions", np.zeros(self.candidate_pool_size, dtype=np.int64))
        kwargs.setdefault("candidate_index", -1)
        kwargs.setdefault("candidate_logp", 0.0)
        kwargs.setdefault("candidate_entropy", 0.0)
        self.items.append(RolloutItem(**kwargs))

    def compute_gae(self, gamma: float, lam: float, normalize_advantage: bool = True) -> None:
        if not self.items:
            return
        trajectories: Dict[tuple[int, str], list[int]] = {}
        for idx, item in enumerate(self.items):
            trajectories.setdefault((item.episode_id, item.trajectory_id), []).append(idx)
        advantages = np.zeros(len(self.items), dtype=np.float32)
        returns = np.zeros(len(self.items), dtype=np.float32)
        for traj_indices in trajectories.values():
            traj_indices.sort(key=lambda item_idx: self.items[item_idx].step_index)
            gae = 0.0
            for item_idx in reversed(traj_indices):
                item = self.items[item_idx]
                bootstrap_mask = 0.0 if item.terminated else 1.0
                continuation_mask = 0.0 if (item.terminated or item.truncated) else 1.0
                delta = item.reward + gamma * bootstrap_mask * item.next_value - item.value
                gae = delta + gamma * lam * continuation_mask * gae
                advantages[item_idx] = gae
                returns[item_idx] = gae + item.value
        if normalize_advantage and advantages.size > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std
        for idx, item in enumerate(self.items):
            item.advantage = float(advantages[idx])
            item.return_ = float(returns[idx])

    def to_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        if not self.items:
            raise ValueError("Rollout buffer is empty")
        batch = {}
        batch["local_obs"] = torch.tensor(np.stack([x.local_obs for x in self.items]), dtype=torch.float32, device=device)
        batch["global_state"] = torch.tensor(np.stack([x.global_state for x in self.items]), dtype=torch.float32, device=device)
        batch["presence_mask"] = torch.tensor(np.stack([x.presence_mask for x in self.items]), dtype=torch.float32, device=device)
        batch["target_index"] = torch.tensor([x.target_index for x in self.items], dtype=torch.long, device=device)
        batch["action_mask"] = torch.tensor(np.stack([x.action_mask for x in self.items]), dtype=torch.float32, device=device)
        batch["action_prior"] = torch.tensor(np.stack([x.action_prior for x in self.items]), dtype=torch.float32, device=device)
        batch["prior_weight"] = torch.tensor([x.prior_weight for x in self.items], dtype=torch.float32, device=device)
        batch["macro_action"] = torch.tensor([x.macro_action for x in self.items], dtype=torch.long, device=device)
        batch["raw_params"] = torch.tensor(np.stack([x.raw_params for x in self.items]), dtype=torch.float32, device=device)
        batch["squashed_params"] = torch.tensor(np.stack([x.squashed_params for x in self.items]), dtype=torch.float32, device=device)
        batch["selected_param_index"] = torch.tensor([x.selected_param_index for x in self.items], dtype=torch.long, device=device)
        batch["parameter_branch"] = torch.tensor([x.parameter_branch for x in self.items], dtype=torch.long, device=device)
        batch["parameter_action_index"] = torch.tensor([x.parameter_action_index for x in self.items], dtype=torch.long, device=device)
        batch["parameter_mask"] = torch.tensor(np.stack([x.parameter_mask for x in self.items]), dtype=torch.float32, device=device)
        batch["parameter_probs"] = torch.tensor(np.stack([x.parameter_probs for x in self.items]), dtype=torch.float32, device=device)
        batch["parameter_logp"] = torch.tensor([x.parameter_logp for x in self.items], dtype=torch.float32, device=device)
        batch["parameter_entropy"] = torch.tensor([x.parameter_entropy for x in self.items], dtype=torch.float32, device=device)
        batch["discrete_logp"] = torch.tensor([x.discrete_logp for x in self.items], dtype=torch.float32, device=device)
        batch["discrete_entropy"] = torch.tensor([x.discrete_entropy for x in self.items], dtype=torch.float32, device=device)
        batch["discrete_probs"] = torch.tensor(np.stack([x.discrete_probs for x in self.items]), dtype=torch.float32, device=device)
        batch["continuous_logp"] = torch.tensor([x.continuous_logp for x in self.items], dtype=torch.float32, device=device)
        batch["continuous_entropy"] = torch.tensor([x.continuous_entropy for x in self.items], dtype=torch.float32, device=device)
        batch["continuous_mean"] = torch.tensor(np.stack([x.continuous_mean for x in self.items]), dtype=torch.float32, device=device)
        batch["continuous_std"] = torch.tensor(np.stack([x.continuous_std for x in self.items]), dtype=torch.float32, device=device)
        batch["candidate_mode"] = torch.tensor([float(x.candidate_mode) for x in self.items], dtype=torch.float32, device=device)
        batch["candidate_features"] = torch.tensor(np.stack([x.candidate_features for x in self.items]), dtype=torch.float32, device=device)
        batch["candidate_mask"] = torch.tensor(np.stack([x.candidate_mask for x in self.items]), dtype=torch.float32, device=device)
        batch["candidate_macro_actions"] = torch.tensor(np.stack([x.candidate_macro_actions for x in self.items]), dtype=torch.long, device=device)
        batch["candidate_index"] = torch.tensor([x.candidate_index for x in self.items], dtype=torch.long, device=device)
        batch["candidate_logp"] = torch.tensor([x.candidate_logp for x in self.items], dtype=torch.float32, device=device)
        batch["candidate_entropy"] = torch.tensor([x.candidate_entropy for x in self.items], dtype=torch.float32, device=device)
        batch["value"] = torch.tensor([x.value for x in self.items], dtype=torch.float32, device=device)
        batch["next_value"] = torch.tensor([x.next_value for x in self.items], dtype=torch.float32, device=device)
        batch["reward"] = torch.tensor([x.reward for x in self.items], dtype=torch.float32, device=device)
        batch["terminated"] = torch.tensor([float(x.terminated) for x in self.items], dtype=torch.float32, device=device)
        batch["truncated"] = torch.tensor([float(x.truncated) for x in self.items], dtype=torch.float32, device=device)
        batch["advantage"] = torch.tensor([x.advantage for x in self.items], dtype=torch.float32, device=device)
        batch["return"] = torch.tensor([x.return_ for x in self.items], dtype=torch.float32, device=device)
        return batch
