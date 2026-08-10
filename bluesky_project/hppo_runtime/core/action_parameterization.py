from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import HPPOCoreConfig


@dataclass(slots=True)
class ParameterSample:
    raw_params: np.ndarray
    squashed_params: np.ndarray
    log_prob: np.ndarray
    entropy: np.ndarray
    mean: np.ndarray
    std: np.ndarray


@dataclass(slots=True)
class CommandMapping:
    macro_action: int
    selected_param_index: int
    actual_params: dict
    used_parameters: bool


@dataclass(slots=True)
class DiscreteParameterSample:
    branch: np.ndarray
    action_index: np.ndarray
    log_prob: np.ndarray
    entropy: np.ndarray
    probabilities: np.ndarray
    masks: np.ndarray


def _atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -0.999999, 0.999999)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))


class ContinuousParameterPolicy:
    def __init__(self, cfg: HPPOCoreConfig):
        self.cfg = cfg
        self.ranges = cfg.ranges

    def sample_parameters(self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False) -> ParameterSample:
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        raw = mean if deterministic else dist.rsample()
        squashed = torch.tanh(raw)
        log_prob = self.parameter_log_prob(mean, log_std, raw)
        entropy = -self.branch_log_prob(mean, log_std, raw)
        return ParameterSample(
            raw_params=np.asarray(raw.detach().cpu().tolist(), dtype=np.float32),
            squashed_params=np.asarray(squashed.detach().cpu().tolist(), dtype=np.float32),
            log_prob=np.asarray(log_prob.detach().cpu().tolist(), dtype=np.float32),
            entropy=np.asarray(entropy.detach().cpu().tolist(), dtype=np.float32),
            mean=np.asarray(mean.detach().cpu().tolist(), dtype=np.float32),
            std=np.asarray(std.detach().cpu().tolist(), dtype=np.float32),
        )

    def branch_log_prob(self, mean: torch.Tensor, log_std: torch.Tensor, raw_actions: torch.Tensor) -> torch.Tensor:
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        squashed = torch.tanh(raw_actions)
        correction = 2.0 * (np.log(2.0) - raw_actions - torch.nn.functional.softplus(-2.0 * raw_actions))
        log_prob = dist.log_prob(raw_actions) - correction
        return log_prob

    def parameter_log_prob(self, mean: torch.Tensor, log_std: torch.Tensor, raw_actions: torch.Tensor) -> torch.Tensor:
        return self.branch_log_prob(mean, log_std, raw_actions).sum(dim=-1)

    def branch_entropy(self, log_std: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + np.log(2.0 * np.pi)) + log_std

    def evaluate_parameters(self, mean: torch.Tensor, log_std: torch.Tensor, raw_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_prob = self.parameter_log_prob(mean, log_std, raw_actions)
        entropy = -log_prob
        return log_prob, entropy

    def map_to_command(self, macro_action: int, squashed_params: np.ndarray, current_state: dict) -> CommandMapping:
        if macro_action in (0, 1):
            return CommandMapping(macro_action=macro_action, selected_param_index=-1, actual_params={}, used_parameters=False)
        heading_norm, altitude_norm, speed_norm = map(float, squashed_params.tolist())
        if macro_action == 2:
            delta = heading_norm * self.ranges.heading_offset_deg
            return CommandMapping(
                macro_action=macro_action,
                selected_param_index=0,
                actual_params={"heading_delta_deg": delta, "heading_deg": (float(current_state["heading_deg"]) + delta) % 360.0},
                used_parameters=True,
            )
        if macro_action == 3:
            delta = altitude_norm * self.ranges.altitude_delta_ft
            altitude = float(np.clip(float(current_state["altitude_ft"]) + delta, self.ranges.min_altitude_ft, self.ranges.max_altitude_ft))
            return CommandMapping(
                macro_action=macro_action,
                selected_param_index=1,
                actual_params={"altitude_delta_ft": delta, "altitude_ft": altitude},
                used_parameters=True,
            )
        if macro_action == 4:
            delta = speed_norm * self.ranges.speed_delta_kt
            speed = float(np.clip(float(current_state["speed_kt"]) + delta, self.ranges.min_speed_kt, self.ranges.max_speed_kt))
            return CommandMapping(
                macro_action=macro_action,
                selected_param_index=2,
                actual_params={"speed_delta_kt": delta, "speed_kt": speed},
                used_parameters=True,
            )
        raise ValueError(f"Unknown macro action {macro_action}")


class DiscreteParameterPolicy:
    """Categorical policy over absolute command targets.

    Branch 0 is absolute heading, branch 1 is absolute altitude and branch 2
    is the configured absolute speed unit. Only the branch selected by the
    macro action contributes probability, entropy or gradients.
    """

    def __init__(self, cfg: HPPOCoreConfig):
        self.cfg = cfg
        self.ranges = cfg.ranges
        self.branch_values = (
            np.asarray(cfg.heading_bins_deg, dtype=np.float32),
            np.asarray(cfg.altitude_bins_ft, dtype=np.float32),
            np.asarray(cfg.speed_bins, dtype=np.float32),
        )

    @property
    def max_dim(self) -> int:
        return self.cfg.parameter_max_dim

    def _distribution(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        safe_mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        invalid = safe_mask.sum(dim=-1) <= 0.0
        if invalid.any():
            safe_mask = safe_mask.clone()
            # This row is ignored unless its macro branch is active. Keeping a
            # valid categorical distribution avoids NaNs during PPO evaluation.
            safe_mask[invalid, 0] = 1.0
        masked_logits = logits.masked_fill(safe_mask <= 0.0, -1e9)
        return torch.distributions.Categorical(logits=masked_logits), safe_mask

    def sample_parameters(
        self,
        branch_logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        macro_actions: torch.Tensor,
        parameter_masks: torch.Tensor,
        deterministic: bool = False,
    ) -> DiscreteParameterSample:
        batch = macro_actions.shape[0]
        device = macro_actions.device
        selected_branch = macro_actions.to(torch.long) - 2
        action_index = torch.full((batch,), -1, dtype=torch.long, device=device)
        log_prob = torch.zeros(batch, dtype=torch.float32, device=device)
        entropy = torch.zeros(batch, dtype=torch.float32, device=device)
        probabilities = torch.zeros((batch, self.max_dim), dtype=torch.float32, device=device)
        selected_mask = torch.zeros((batch, self.max_dim), dtype=torch.float32, device=device)

        for branch, logits in enumerate(branch_logits):
            dim = logits.shape[-1]
            dist, safe_mask = self._distribution(logits, parameter_masks[:, branch, :dim])
            sample = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            active = selected_branch == branch
            action_index = torch.where(active, sample, action_index)
            log_prob = torch.where(active, dist.log_prob(sample), log_prob)
            entropy = torch.where(active, dist.entropy(), entropy)
            probabilities[:, :dim] = torch.where(active.unsqueeze(-1), dist.probs, probabilities[:, :dim])
            selected_mask[:, :dim] = torch.where(active.unsqueeze(-1), safe_mask, selected_mask[:, :dim])

        return DiscreteParameterSample(
            branch=np.asarray(selected_branch.detach().cpu().tolist(), dtype=np.int64),
            action_index=np.asarray(action_index.detach().cpu().tolist(), dtype=np.int64),
            log_prob=np.asarray(log_prob.detach().cpu().tolist(), dtype=np.float32),
            entropy=np.asarray(entropy.detach().cpu().tolist(), dtype=np.float32),
            probabilities=np.asarray(probabilities.detach().cpu().tolist(), dtype=np.float32),
            masks=np.asarray(selected_mask.detach().cpu().tolist(), dtype=np.float32),
        )

    def evaluate_parameters(
        self,
        branch_logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        branches: torch.Tensor,
        action_indices: torch.Tensor,
        parameter_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = branches.shape[0]
        log_prob = torch.zeros(batch, dtype=torch.float32, device=branches.device)
        entropy = torch.zeros(batch, dtype=torch.float32, device=branches.device)
        for branch, logits in enumerate(branch_logits):
            dim = logits.shape[-1]
            branch_mask = parameter_masks[:, :dim] if parameter_masks.dim() == 2 else parameter_masks[:, branch, :dim]
            dist, _ = self._distribution(logits, branch_mask)
            active = branches == branch
            indices = action_indices.clamp(0, dim - 1)
            log_prob = torch.where(active, dist.log_prob(indices), log_prob)
            entropy = torch.where(active, dist.entropy(), entropy)
        return log_prob, entropy

    def map_to_command(self, macro_action: int, action_index: int, current_state: dict | None = None) -> CommandMapping:
        if macro_action in (0, 1):
            return CommandMapping(macro_action=macro_action, selected_param_index=-1, actual_params={}, used_parameters=False)
        branch = macro_action - 2
        if branch < 0 or branch >= len(self.branch_values):
            raise ValueError(f"Unknown macro action {macro_action}")
        values = self.branch_values[branch]
        if action_index < 0 or action_index >= len(values):
            raise ValueError(f"Invalid parameter action index {action_index} for branch {branch}")
        target = float(values[action_index])
        if macro_action == 2:
            return CommandMapping(
                macro_action=macro_action,
                selected_param_index=0,
                actual_params={"heading_deg": target},
                used_parameters=True,
            )
        if macro_action == 3:
            return CommandMapping(
                macro_action=macro_action,
                selected_param_index=1,
                actual_params={"altitude_ft": target},
                used_parameters=True,
            )
        return CommandMapping(
            macro_action=macro_action,
            selected_param_index=2,
            actual_params={"speed_value": target, "speed_unit": self.ranges.speed_parameterization},
            used_parameters=True,
        )
