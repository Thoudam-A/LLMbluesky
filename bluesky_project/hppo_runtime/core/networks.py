from __future__ import annotations

import torch
import torch.nn as nn

from .config import HPPOCoreConfig


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2, activation=nn.ReLU) -> nn.Sequential:
    modules = []
    last = in_dim
    for _ in range(layers):
        modules.append(nn.Linear(last, hidden_dim))
        modules.append(activation())
        last = hidden_dim
    modules.append(nn.Linear(last, out_dim))
    return nn.Sequential(*modules)


class DiscreteActor(nn.Module):
    def __init__(self, cfg: HPPOCoreConfig):
        super().__init__()
        self.net = build_mlp(cfg.local_obs_dim, cfg.hidden_dim, cfg.discrete_action_dim, layers=2)

    def forward(self, local_obs: torch.Tensor) -> torch.Tensor:
        return self.net(local_obs)


class ContinuousActor(nn.Module):
    def __init__(self, cfg: HPPOCoreConfig):
        super().__init__()
        self.backbone = build_mlp(cfg.local_obs_dim, cfg.hidden_dim, cfg.hidden_dim, layers=2)
        self.mean_head = nn.Linear(cfg.hidden_dim, cfg.continuous_action_dim)
        self.log_std_head = nn.Linear(cfg.hidden_dim, cfg.continuous_action_dim)
        self.log_std_min = cfg.actor_log_std_min
        self.log_std_max = cfg.actor_log_std_max

    def forward(self, local_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(local_obs)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), self.log_std_min, self.log_std_max)
        return mean, log_std


class DiscreteParameterActor(nn.Module):
    """Three conditional categorical parameter heads.

    The heads have fixed global bins. Runtime masks restrict each distribution
    to the absolute heading, flight-level and speed targets that are feasible
    for the selected aircraft at the current decision time.
    """

    def __init__(self, cfg: HPPOCoreConfig):
        super().__init__()
        self.backbone = build_mlp(cfg.local_obs_dim, cfg.hidden_dim, cfg.hidden_dim, layers=2)
        heading_dim, altitude_dim, speed_dim = cfg.parameter_branch_dims
        self.heading_head = nn.Linear(cfg.hidden_dim, heading_dim)
        self.altitude_head = nn.Linear(cfg.hidden_dim, altitude_dim)
        self.speed_head = nn.Linear(cfg.hidden_dim, speed_dim)

    def forward(self, local_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(local_obs)
        return (
            self.heading_head(hidden),
            self.altitude_head(hidden),
            self.speed_head(hidden),
        )


class CandidateSelector(nn.Module):
    """Shared policy over a fixed, masked coarse-candidate pool."""

    def __init__(self, cfg: HPPOCoreConfig):
        super().__init__()
        self.obs_encoder = build_mlp(cfg.local_obs_dim, cfg.hidden_dim, cfg.hidden_dim, layers=2)
        self.candidate_encoder = build_mlp(cfg.candidate_feature_dim, cfg.hidden_dim, cfg.hidden_dim, layers=2)
        self.score = build_mlp(cfg.hidden_dim * 2, cfg.hidden_dim, 1, layers=1)

    def forward(self, local_obs: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        if candidate_features.dim() != 3:
            raise ValueError(f"candidate_features must be [B,K,F], got {candidate_features.shape}")
        if candidate_features.shape[-1] != self.candidate_encoder[0].in_features:
            raise ValueError("candidate feature dimension is incompatible with the selector")
        obs = self.obs_encoder(local_obs).unsqueeze(1)
        candidates = self.candidate_encoder(candidate_features)
        joined = torch.cat([obs.expand_as(candidates), candidates], dim=-1)
        return self.score(joined).squeeze(-1)


class GlobalCritic(nn.Module):
    def __init__(self, cfg: HPPOCoreConfig):
        super().__init__()
        self.slot_encoder = build_mlp(cfg.global_slot_dim, cfg.hidden_dim, cfg.encoder_dim, layers=2)
        self.head = build_mlp(cfg.encoder_dim * 2 + 1, cfg.hidden_dim, 1, layers=2)

    def forward(self, global_state: torch.Tensor, presence_mask: torch.Tensor, target_index: torch.Tensor) -> torch.Tensor:
        if global_state.dim() != 3:
            raise ValueError(f"global_state must be [B,N,F], got {global_state.shape}")
        encoded = self.slot_encoder(global_state)
        mask = presence_mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (encoded * mask).sum(dim=1) / denom
        clamped_target = target_index.clamp(0, encoded.size(1) - 1)
        gather_idx = clamped_target.view(-1, 1, 1).expand(-1, 1, encoded.size(-1))
        target_embed = torch.gather(encoded, 1, gather_idx).squeeze(1)
        target_present = torch.gather(presence_mask, 1, clamped_target.view(-1, 1)).squeeze(1).unsqueeze(-1)
        target_embed = target_embed * target_present
        count = presence_mask.sum(dim=1, keepdim=True) / max(float(presence_mask.shape[1]), 1.0)
        return self.head(torch.cat([target_embed, pooled, count], dim=-1))
