from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
import random

import numpy as np
import torch
import torch.nn as nn

from .action_parameterization import ContinuousParameterPolicy, DiscreteParameterPolicy
from .config import HPPOCoreConfig
from .networks import CandidateSelector, ContinuousActor, DiscreteActor, DiscreteParameterActor, GlobalCritic
from .ppo_update import ppo_update
from .rollout_buffer import HPPORolloutBuffer


@dataclass(slots=True)
class ActionResult:
    macro_action: int
    raw_params: np.ndarray
    squashed_params: np.ndarray
    selected_param_index: int
    discrete_logp: float
    discrete_entropy: float
    discrete_probs: np.ndarray
    continuous_logp: float
    continuous_entropy: float
    continuous_mean: np.ndarray
    continuous_std: np.ndarray


class HPPOModel(nn.Module):
    def __init__(self, cfg: HPPOCoreConfig):
        super().__init__()
        self.cfg = cfg
        self.discrete_actor = DiscreteActor(cfg)
        self.candidate_selector = CandidateSelector(cfg)
        if cfg.parameter_policy_type == "discrete":
            self.parameter_actor = DiscreteParameterActor(cfg)
            self.parameter_policy = DiscreteParameterPolicy(cfg)
            self.continuous_actor = None
        else:
            self.parameter_actor = ContinuousActor(cfg)
            self.parameter_policy = ContinuousParameterPolicy(cfg)
            self.continuous_actor = self.parameter_actor
        self.critic = GlobalCritic(cfg)

    def actor_parameters(self):
        return list(self.discrete_actor.parameters()) + list(self.candidate_selector.parameters()) + list(self.parameter_actor.parameters())

    def critic_parameters(self):
        return self.critic.parameters()


class HPPOAgent:
    def __init__(self, cfg: HPPOCoreConfig, device: str | torch.device | None = None, seed: int = 7):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.model = HPPOModel(cfg).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.model.actor_parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.model.critic_parameters(), lr=cfg.critic_lr)
        self.buffer = HPPORolloutBuffer(cfg.candidate_pool_size, cfg.candidate_feature_dim)
        self.parameter_policy = self.model.parameter_policy
        self.policy_version = 0
        self.training = True
        self.update_step = 0
        self.environment_step = 0

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "policy_version": self.policy_version,
            "seed": self.seed,
            "update_step": self.update_step,
            "environment_step": self.environment_step,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        missing, unexpected = self.model.load_state_dict(state_dict["model"], strict=False)
        allowed_missing = {key for key in missing if key.startswith("candidate_selector.")}
        if unexpected or len(allowed_missing) != len(missing):
            raise RuntimeError(f"Checkpoint model parameters are incompatible: missing={missing}, unexpected={unexpected}")
        try:
            self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        except ValueError:
            # Legacy checkpoints predate the selector and cannot restore the
            # expanded optimizer parameter groups. Network weights still load.
            pass
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        for optimizer in (self.actor_optimizer, self.critic_optimizer):
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self.device)
        self.policy_version = int(state_dict.get("policy_version", 0))
        self.seed = int(state_dict.get("seed", self.seed))
        self.update_step = int(state_dict.get("update_step", 0))
        self.environment_step = int(state_dict.get("environment_step", 0))

    def actor_parameters(self):
        return self.model.actor_parameters()

    def critic_parameters(self):
        return self.model.critic_parameters()

    def predict_value(self, global_state: np.ndarray, presence_mask: np.ndarray, target_index: int) -> float:
        self.model.eval()
        with torch.no_grad():
            g_arr = np.nan_to_num(global_state if global_state.ndim == 3 else global_state[None, ...], nan=0.0, posinf=0.0, neginf=0.0)
            p_arr = np.nan_to_num(presence_mask if presence_mask.ndim == 2 else presence_mask[None, ...], nan=0.0, posinf=0.0, neginf=0.0)
            g = torch.tensor(g_arr, dtype=torch.float32, device=self.device)
            p = torch.tensor(p_arr, dtype=torch.float32, device=self.device)
            t = torch.tensor([target_index], dtype=torch.long, device=self.device)
            value = self.model.critic(g, p, t).squeeze(-1)
        self.model.train(self.training)
        return float(value.item())

    def act(
        self,
        local_obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
        action_prior: np.ndarray | None = None,
        prior_weight: float = 0.0,
        parameter_masks: np.ndarray | None = None,
    ) -> list[dict]:
        safe_local = np.nan_to_num(local_obs, nan=0.0, posinf=0.0, neginf=0.0)
        safe_mask = np.nan_to_num(action_mask, nan=0.0, posinf=0.0, neginf=0.0)
        if safe_local.ndim == 1:
            safe_local = safe_local[None, :]
        if safe_mask.ndim == 1:
            safe_mask = safe_mask[None, :]
        if action_prior is None:
            safe_prior = np.ones_like(safe_mask, dtype=np.float32)
        else:
            safe_prior = np.nan_to_num(action_prior, nan=0.0, posinf=0.0, neginf=0.0)
            if safe_prior.ndim == 1:
                safe_prior = safe_prior[None, :]
            if safe_prior.shape != safe_mask.shape:
                raise ValueError(f"Action prior shape {safe_prior.shape} does not match mask {safe_mask.shape}")
            safe_prior = np.maximum(safe_prior, 1e-8)
        invalid_rows = safe_mask.sum(axis=-1) <= 0.0
        if np.any(invalid_rows):
            safe_mask[invalid_rows] = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if self.cfg.parameter_policy_type == "discrete":
            max_dim = self.cfg.parameter_max_dim
            if parameter_masks is None:
                safe_parameter_masks = np.ones((safe_local.shape[0], 3, max_dim), dtype=np.float32)
            else:
                safe_parameter_masks = np.nan_to_num(parameter_masks, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                if safe_parameter_masks.ndim == 2:
                    safe_parameter_masks = safe_parameter_masks[None, ...]
                if safe_parameter_masks.shape != (safe_local.shape[0], 3, max_dim):
                    raise ValueError(
                        f"Parameter mask shape {safe_parameter_masks.shape} does not match "
                        f"({safe_local.shape[0]}, 3, {max_dim})"
                    )
            # A macro whose parameter branch is empty cannot be sampled.
            for branch in range(3):
                safe_mask[:, branch + 2] *= (safe_parameter_masks[:, branch].sum(axis=-1) > 0.0)
            invalid_rows = safe_mask.sum(axis=-1) <= 0.0
            if np.any(invalid_rows):
                safe_mask[invalid_rows] = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            safe_parameter_masks = None
        local = torch.tensor(safe_local, dtype=torch.float32, device=self.device)
        mask = torch.tensor(safe_mask, dtype=torch.float32, device=self.device)
        prior = torch.tensor(safe_prior, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model.discrete_actor(local)
            guided_logits = logits + float(prior_weight) * torch.log(prior.clamp_min(1e-8))
            masked_logits = guided_logits.masked_fill(mask <= 0.0, -1e9)
            discrete_dist = torch.distributions.Categorical(logits=masked_logits)
            macro_action = discrete_dist.probs.argmax(dim=-1) if deterministic else discrete_dist.sample()
            discrete_logp = discrete_dist.log_prob(macro_action)
            discrete_entropy = discrete_dist.entropy()
            discrete_probs = torch.softmax(masked_logits, dim=-1)
            if self.cfg.parameter_policy_type == "discrete":
                parameter_mask_tensor = torch.tensor(safe_parameter_masks, dtype=torch.float32, device=self.device)
                parameter_logits = self.model.parameter_actor(local)
                sampled = self.parameter_policy.sample_parameters(
                    parameter_logits,
                    macro_action,
                    parameter_mask_tensor,
                    deterministic=deterministic,
                )
            else:
                mean, log_std = self.model.continuous_actor(local)
                sampled = self.parameter_policy.sample_parameters(mean, log_std, deterministic=deterministic)
        self.model.train(self.training)
        results = []
        for i in range(local.shape[0]):
            macro = int(macro_action[i].item())
            if mask[i, macro].item() <= 0.0:
                macro = 0
                fallback = torch.tensor(0, dtype=torch.long, device=self.device)
                discrete_logp[i] = discrete_dist.log_prob(fallback.expand_as(macro_action))[i]
            selected_param_index = macro - 2 if macro >= 2 else -1
            if self.cfg.parameter_policy_type == "discrete":
                parameter_action_index = int(sampled.action_index[i])
                parameter_logp = float(sampled.log_prob[i])
                parameter_entropy = float(sampled.entropy[i])
                parameter_probs = sampled.probabilities[i]
                parameter_mask = sampled.masks[i]
                raw_params = np.zeros(self.cfg.continuous_action_dim, dtype=np.float32)
                squashed_params = np.zeros(self.cfg.continuous_action_dim, dtype=np.float32)
                continuous_logp = 0.0
                continuous_entropy = 0.0
                continuous_mean = np.zeros(self.cfg.continuous_action_dim, dtype=np.float32)
                continuous_std = np.zeros(self.cfg.continuous_action_dim, dtype=np.float32)
            elif selected_param_index >= 0:
                raw_t = torch.tensor(sampled.raw_params[i : i + 1, selected_param_index : selected_param_index + 1], dtype=torch.float32, device=self.device)
                cont_logp = float(
                    self.parameter_policy.branch_log_prob(
                        mean[i : i + 1, selected_param_index : selected_param_index + 1],
                        log_std[i : i + 1, selected_param_index : selected_param_index + 1],
                        raw_t,
                    ).item()
                )
                cont_entropy = float(
                    -self.parameter_policy.branch_log_prob(
                        mean[i : i + 1, selected_param_index : selected_param_index + 1],
                        log_std[i : i + 1, selected_param_index : selected_param_index + 1],
                        raw_t,
                    ).item()
                )
            else:
                cont_logp = 0.0
                cont_entropy = 0.0
            if self.cfg.parameter_policy_type != "discrete":
                parameter_action_index = -1
                parameter_logp = 0.0
                parameter_entropy = 0.0
                parameter_probs = np.zeros(self.cfg.parameter_max_dim, dtype=np.float32)
                parameter_mask = np.zeros(self.cfg.parameter_max_dim, dtype=np.float32)
                raw_params = sampled.raw_params[i]
                squashed_params = sampled.squashed_params[i]
                continuous_logp = cont_logp
                continuous_entropy = cont_entropy
                continuous_mean = np.asarray(sampled.mean[i], dtype=np.float32)
                continuous_std = np.asarray(sampled.std[i], dtype=np.float32)
            results.append(
                {
                    "macro_action": macro,
                    "raw_params": raw_params,
                    "squashed_params": squashed_params,
                    "selected_param_index": selected_param_index,
                    "discrete_logp": float(discrete_logp[i].item()),
                    "discrete_entropy": float(discrete_entropy[i].item()),
                    "discrete_probs": np.asarray(discrete_probs[i].detach().cpu().tolist(), dtype=np.float32),
                    "continuous_logp": continuous_logp,
                    "continuous_entropy": continuous_entropy,
                    "continuous_mean": continuous_mean,
                    "continuous_std": continuous_std,
                    "parameter_branch": selected_param_index,
                    "parameter_action_index": parameter_action_index,
                    "parameter_logp": parameter_logp,
                    "parameter_entropy": parameter_entropy,
                    "parameter_probs": parameter_probs,
                    "parameter_mask": parameter_mask,
                }
            )
        return results

    def act_candidates(
        self,
        local_obs: np.ndarray,
        candidate_features: np.ndarray,
        candidate_mask: np.ndarray,
        candidate_macro_actions: np.ndarray,
        deterministic: bool = False,
        prior_weight: float | None = None,
        candidate_parameter_masks: np.ndarray | None = None,
    ) -> list[dict]:
        """Select a valid candidate and sample its conditional parameter branch."""
        local_arr = np.atleast_2d(np.nan_to_num(local_obs, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.float32)
        feature_arr = np.asarray(candidate_features, dtype=np.float32)
        mask_arr = np.asarray(candidate_mask, dtype=np.float32)
        macro_arr = np.asarray(candidate_macro_actions, dtype=np.int64)
        if feature_arr.ndim == 2:
            feature_arr = feature_arr[None, ...]
        if mask_arr.ndim == 1:
            mask_arr = mask_arr[None, ...]
        if macro_arr.ndim == 1:
            macro_arr = macro_arr[None, ...]
        expected = (local_arr.shape[0], self.cfg.candidate_pool_size)
        if feature_arr.shape != (expected[0], self.cfg.candidate_pool_size, self.cfg.candidate_feature_dim):
            raise ValueError("candidate feature tensor has an incompatible shape")
        if mask_arr.shape != expected or macro_arr.shape != expected:
            raise ValueError("candidate masks or macro-action tensor have an incompatible shape")
        if self.cfg.parameter_policy_type == "discrete":
            max_dim = self.cfg.parameter_max_dim
            if candidate_parameter_masks is None:
                parameter_mask_arr = np.ones((expected[0], expected[1], 3, max_dim), dtype=np.float32)
            else:
                parameter_mask_arr = np.asarray(candidate_parameter_masks, dtype=np.float32)
                if parameter_mask_arr.ndim == 3:
                    parameter_mask_arr = parameter_mask_arr[None, ...]
                if parameter_mask_arr.shape != (expected[0], expected[1], 3, max_dim):
                    raise ValueError("candidate parameter masks have an incompatible shape")
            for row in range(expected[0]):
                for candidate in range(expected[1]):
                    macro_value = int(macro_arr[row, candidate])
                    if macro_value >= 2 and parameter_mask_arr[row, candidate, macro_value - 2].sum() <= 0.0:
                        mask_arr[row, candidate] = 0.0
        else:
            parameter_mask_arr = None
        invalid = mask_arr.sum(axis=-1) <= 0.0
        if np.any(invalid):
            raise ValueError("act_candidates requires at least one valid candidate per row")
        local = torch.tensor(local_arr, dtype=torch.float32, device=self.device)
        features = torch.tensor(feature_arr, dtype=torch.float32, device=self.device)
        mask = torch.tensor(mask_arr, dtype=torch.float32, device=self.device)
        macro = torch.tensor(macro_arr, dtype=torch.long, device=self.device)
        parameter_masks = (
            torch.tensor(parameter_mask_arr, dtype=torch.float32, device=self.device)
            if parameter_mask_arr is not None
            else None
        )
        self.model.eval()
        with torch.no_grad():
            selector_prior = features[..., 16].clamp_min(1e-8)
            weight = self.cfg.candidate_prior_weight if prior_weight is None else float(prior_weight)
            logits = (self.model.candidate_selector(local, features) + weight * torch.log(selector_prior)).masked_fill(mask <= 0.0, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            candidate_index = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            selected_macro = torch.gather(macro, 1, candidate_index.unsqueeze(-1)).squeeze(-1)
            if self.cfg.parameter_policy_type == "discrete":
                gather = candidate_index.view(-1, 1, 1, 1).expand(-1, 1, 3, self.cfg.parameter_max_dim)
                selected_parameter_masks = torch.gather(parameter_masks, 1, gather).squeeze(1)
                parameter_logits = self.model.parameter_actor(local)
                sampled = self.parameter_policy.sample_parameters(
                    parameter_logits,
                    selected_macro,
                    selected_parameter_masks,
                    deterministic=deterministic,
                )
            else:
                mean, log_std = self.model.continuous_actor(local)
                sampled = self.parameter_policy.sample_parameters(mean, log_std, deterministic=deterministic)
        self.model.train(self.training)
        results: list[dict] = []
        for index in range(local.shape[0]):
            selected = int(candidate_index[index].item())
            selected_macro_value = int(selected_macro[index].item())
            param_index = selected_macro_value - 2 if selected_macro_value >= 2 else -1
            if self.cfg.parameter_policy_type == "discrete":
                parameter_action_index = int(sampled.action_index[index])
                parameter_logp = float(sampled.log_prob[index])
                parameter_entropy = float(sampled.entropy[index])
                parameter_probs = sampled.probabilities[index]
                parameter_mask = sampled.masks[index]
                raw_params = np.zeros(self.cfg.continuous_action_dim, dtype=np.float32)
                squashed_params = np.zeros(self.cfg.continuous_action_dim, dtype=np.float32)
                continuous_logp = 0.0
                continuous_entropy = 0.0
                continuous_mean = np.zeros(self.cfg.continuous_action_dim, dtype=np.float32)
                continuous_std = np.zeros(self.cfg.continuous_action_dim, dtype=np.float32)
            elif param_index >= 0:
                raw = torch.tensor(sampled.raw_params[index:index + 1, param_index:param_index + 1], dtype=torch.float32, device=self.device)
                continuous_logp = float(self.parameter_policy.branch_log_prob(
                    mean[index:index + 1, param_index:param_index + 1],
                    log_std[index:index + 1, param_index:param_index + 1], raw,
                ).item())
                continuous_entropy = -continuous_logp
            else:
                continuous_logp = 0.0
                continuous_entropy = 0.0
            if self.cfg.parameter_policy_type != "discrete":
                parameter_action_index = -1
                parameter_logp = 0.0
                parameter_entropy = 0.0
                parameter_probs = np.zeros(self.cfg.parameter_max_dim, dtype=np.float32)
                parameter_mask = np.zeros(self.cfg.parameter_max_dim, dtype=np.float32)
                raw_params = sampled.raw_params[index]
                squashed_params = sampled.squashed_params[index]
                continuous_mean = np.asarray(sampled.mean[index], dtype=np.float32)
                continuous_std = np.asarray(sampled.std[index], dtype=np.float32)
            results.append({
                "macro_action": selected_macro_value,
                "raw_params": raw_params,
                "squashed_params": squashed_params,
                "selected_param_index": param_index,
                "discrete_logp": 0.0,
                "discrete_entropy": 0.0,
                "discrete_probs": np.zeros(self.cfg.discrete_action_dim, dtype=np.float32),
                "continuous_logp": continuous_logp,
                "continuous_entropy": continuous_entropy,
                "continuous_mean": continuous_mean,
                "continuous_std": continuous_std,
                "parameter_branch": param_index,
                "parameter_action_index": parameter_action_index,
                "parameter_logp": parameter_logp,
                "parameter_entropy": parameter_entropy,
                "parameter_probs": parameter_probs,
                "parameter_mask": parameter_mask,
                "candidate_mode": True,
                "candidate_index": selected,
                "candidate_logp": float(dist.log_prob(candidate_index)[index].item()),
                "candidate_entropy": float(dist.entropy()[index].item()),
                "candidate_probs": np.asarray(dist.probs[index].detach().cpu().tolist(), dtype=np.float32),
            })
        return results

    def set_training(self, training: bool) -> None:
        self.training = bool(training)
        self.model.train(self.training)

    def update(self) -> dict:
        if len(self.buffer) == 0:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "explained_variance": 0.0,
                "nan_detected": 0.0,
            }
        self.buffer.compute_gae(self.cfg.gamma, self.cfg.gae_lambda, self.cfg.normalize_advantage)
        batch = self.buffer.to_tensors(self.device)
        metrics = ppo_update(self.model, self.actor_optimizer, self.critic_optimizer, batch, self.cfg)
        self.buffer.clear()
        self.policy_version += 1
        self.update_step += 1
        return asdict(metrics)
