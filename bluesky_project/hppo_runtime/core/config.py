from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CorePolicyRange:
    heading_offset_deg: float = 45.0
    altitude_delta_ft: float = 3000.0
    speed_delta_kt: float = 30.0
    min_altitude_ft: float = 29000.0
    max_altitude_ft: float = 41000.0
    min_speed_kt: float = 420.0
    max_speed_kt: float = 500.0
    speed_parameterization: str = "tas_kt"


@dataclass(slots=True)
class HPPOCoreConfig:
    local_obs_dim: int = 72
    global_slot_dim: int = 26
    max_aircraft: int = 30
    discrete_action_dim: int = 5
    continuous_action_dim: int = 3
    parameter_policy_type: str = "discrete"
    heading_bins_deg: tuple[float, ...] = tuple(float(v) for v in range(0, 360, 10))
    altitude_bins_ft: tuple[float, ...] = tuple(float(v) for v in range(29000, 41001, 1000))
    speed_bins: tuple[float, ...] = tuple(float(v) for v in range(420, 501, 10))
    candidate_pool_size: int = 6
    candidate_feature_dim: int = 18
    candidate_prior_weight: float = 1.0
    hidden_dim: int = 256
    encoder_dim: int = 128
    actor_log_std_min: float = -5.0
    actor_log_std_max: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip_ratio: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    max_grad_norm: float = 1.0
    entropy_coef_discrete: float = 0.01
    entropy_coef_continuous: float = 0.01
    value_loss_coef: float = 0.5
    epochs: int = 6
    mini_batch_size: int = 64
    normalize_advantage: bool = True
    enable_kl_early_stop: bool = True
    target_kl_discrete: float = 0.03
    target_kl_continuous: float = 0.03
    fail_on_nonfinite: bool = True
    ranges: CorePolicyRange = field(default_factory=CorePolicyRange)

    @classmethod
    def from_env(cls, env_cfg) -> "HPPOCoreConfig":
        heading_step = int(round(env_cfg.network.heading_bin_deg))
        altitude_step = int(round(env_cfg.network.altitude_bin_step_ft))
        speed_step = int(round(env_cfg.command_ranges.speed_bin_step_kt))
        return cls(
            local_obs_dim=env_cfg.network.local_obs_dim,
            global_slot_dim=env_cfg.network.global_slot_dim,
            max_aircraft=env_cfg.network.max_aircraft,
            discrete_action_dim=env_cfg.network.discrete_action_dim,
            continuous_action_dim=env_cfg.network.continuous_action_dim,
            parameter_policy_type=env_cfg.network.parameter_policy_type,
            heading_bins_deg=tuple(float(v) for v in range(0, 360, heading_step)),
            altitude_bins_ft=tuple(
                float(v)
                for v in range(
                    int(round(env_cfg.command_ranges.min_altitude_ft)),
                    int(round(env_cfg.command_ranges.max_altitude_ft)) + 1,
                    altitude_step,
                )
            ),
            speed_bins=tuple(
                float(v)
                for v in range(
                    int(round(env_cfg.command_ranges.min_speed_kt)),
                    int(round(env_cfg.command_ranges.max_speed_kt)) + 1,
                    speed_step,
                )
            ),
            candidate_pool_size=env_cfg.llm.candidate_pool_size,
            candidate_feature_dim=18,
            candidate_prior_weight=env_cfg.llm.candidate_prior_weight,
            hidden_dim=env_cfg.network.hidden_dim,
            encoder_dim=env_cfg.network.encoder_dim,
            actor_log_std_min=env_cfg.network.actor_log_std_min,
            actor_log_std_max=env_cfg.network.actor_log_std_max,
            gamma=env_cfg.ppo.gamma,
            gae_lambda=env_cfg.ppo.gae_lambda,
            clip_ratio=env_cfg.ppo.clip_ratio,
            value_clip_ratio=env_cfg.ppo.value_clip_ratio,
            actor_lr=env_cfg.ppo.actor_lr,
            critic_lr=env_cfg.ppo.critic_lr,
            max_grad_norm=env_cfg.ppo.max_grad_norm,
            entropy_coef_discrete=env_cfg.ppo.entropy_coef_discrete,
            entropy_coef_continuous=env_cfg.ppo.entropy_coef_continuous,
            value_loss_coef=env_cfg.ppo.value_loss_coef,
            epochs=env_cfg.ppo.epochs,
            mini_batch_size=env_cfg.ppo.mini_batch_size,
            normalize_advantage=env_cfg.ppo.normalize_advantage,
            enable_kl_early_stop=env_cfg.ppo.enable_kl_early_stop,
            target_kl_discrete=env_cfg.ppo.target_kl_discrete,
            target_kl_continuous=env_cfg.ppo.target_kl_continuous,
            fail_on_nonfinite=env_cfg.ppo.fail_on_nonfinite,
            ranges=CorePolicyRange(
                heading_offset_deg=env_cfg.command_ranges.heading_offset_deg,
                altitude_delta_ft=env_cfg.command_ranges.altitude_delta_ft,
                speed_delta_kt=env_cfg.command_ranges.speed_delta_kt,
                min_altitude_ft=env_cfg.command_ranges.min_altitude_ft,
                max_altitude_ft=env_cfg.command_ranges.max_altitude_ft,
                min_speed_kt=env_cfg.command_ranges.min_speed_kt,
                max_speed_kt=env_cfg.command_ranges.max_speed_kt,
                speed_parameterization=env_cfg.command_ranges.speed_parameterization,
            ),
        )

    @property
    def parameter_branch_dims(self) -> tuple[int, int, int]:
        return (len(self.heading_bins_deg), len(self.altitude_bins_ft), len(self.speed_bins))

    @property
    def parameter_max_dim(self) -> int:
        return max(self.parameter_branch_dims)
