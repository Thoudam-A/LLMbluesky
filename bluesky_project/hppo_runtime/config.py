from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# This package lives directly under bluesky_project in the legacy HMI repo.
# Keep all default paths relative to that project instead of the source repo.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_route_file() -> Path:
    return REPO_ROOT / "routes" / "hppo" / "01_head_on_2ac.npy"


def _default_scenario_files() -> tuple[Path, ...]:
    return tuple(sorted((REPO_ROOT / "routes" / "hppo").glob("0[1-9]_*.npy")))


@dataclass(slots=True)
class CommandRanges:
    heading_offset_deg: float = 45.0
    altitude_delta_ft: float = 3000.0
    speed_delta_kt: float = 30.0
    # Cruise-flight envelope used by the H-PPO scenarios.  Legacy .npy routes
    # carry no altitude field, so the environment starts them at FL350.
    min_altitude_ft: float = 29000.0
    max_altitude_ft: float = 41000.0
    default_initial_altitude_ft: float = 35000.0
    # The five-action baseline has one unsigned ALTITUDE macro.  Near either
    # boundary it cannot mask only the invalid climb/descent direction, so the
    # complete macro is disabled to prevent repeated clipped commands.
    altitude_boundary_guard_ft: float = 250.0
    # Legacy .npy routes also omit speed, so start them at a representative
    # cruise true airspeed inside the configured envelope.
    min_speed_kt: float = 420.0
    max_speed_kt: float = 500.0
    default_initial_speed_kt: float = 450.0
    # The policy represents this range as KTAS. BlueSky receives the matching
    # CAS target after conversion at the aircraft's current altitude.
    speed_parameterization: str = "tas_kt"
    speed_bin_step_kt: float = 10.0


@dataclass(slots=True)
class SeparationConfig:
    horizontal_km: float = 8.0
    vertical_ft: float = 1000.0
    temporal_s: float = 90.0
    mode: str = "combined"
    release_factor: float = 1.2
    max_horizontal_km: float = 20.0
    max_vertical_ft: float = 3000.0
    max_temporal_s: float = 300.0

    def validate(self) -> None:
        if self.mode not in {"horizontal", "spatial", "temporal", "combined"}:
            raise ValueError("separation mode must be horizontal, spatial, temporal, or combined")
        if self.horizontal_km <= 0.0:
            raise ValueError("horizontal separation must be positive")
        if self.vertical_ft <= 0.0:
            raise ValueError("vertical separation must be positive")
        if self.temporal_s <= 0.0:
            raise ValueError("temporal separation must be positive")
        if self.release_factor < 1.0:
            raise ValueError("separation release factor must be at least 1.0")
        if self.horizontal_km > self.max_horizontal_km:
            raise ValueError("horizontal separation exceeds its configured normalization maximum")
        if self.vertical_ft > self.max_vertical_ft:
            raise ValueError("vertical separation exceeds its configured normalization maximum")
        if self.temporal_s > self.max_temporal_s:
            raise ValueError("temporal separation exceeds its configured normalization maximum")


@dataclass(slots=True)
class RewardWeights:
    collision: float = -120.0
    intrusion: float = -30.0
    warning: float = -8.0
    resolution_success: float = 30.0
    secondary_conflict: float = -12.0
    arrival: float = 60.0
    route_deviation: float = -0.02
    extra_distance: float = -0.02
    command_issue: float = -0.25
    oscillation: float = -0.75
    step_cost: float = -0.01
    timeout: float = -8.0
    command_failure: float = -2.0
    progress: float = 0.25
    # Small process reward only for measured conflict-risk reduction. It is kept
    # below terminal safety rewards and can be disabled by setting it to zero.
    risk_improvement: float = 1.0
    reward_scale: float = 0.01


@dataclass(slots=True)
class NetworkConfig:
    local_obs_dim: int = 72
    global_slot_dim: int = 26
    max_aircraft: int = 30
    discrete_action_dim: int = 5
    continuous_action_dim: int = 3
    hidden_dim: int = 256
    encoder_dim: int = 128
    actor_log_std_min: float = -5.0
    actor_log_std_max: float = 1.0
    # "continuous" is retained for baseline ablations. "discrete" is the
    # default operational policy: absolute heading, flight level and speed.
    parameter_policy_type: str = "discrete"
    heading_bin_deg: float = 10.0
    heading_normal_limit_deg: float = 30.0
    heading_emergency_limit_deg: float = 45.0
    heading_min_effective_change_deg: float = 5.0
    altitude_bin_step_ft: float = 1000.0


@dataclass(slots=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip_ratio: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    max_grad_norm: float = 1.0
    # V2.1 keeps macro-action exploration alive longer. The previous value
    # allowed HEADING to collapse to zero probability in complex traffic.
    entropy_coef_discrete: float = 0.03
    entropy_coef_continuous: float = 0.01
    value_loss_coef: float = 0.5
    epochs: int = 6
    mini_batch_size: int = 64
    train_on_buffer_size: int = 128
    normalize_advantage: bool = True
    normalize_observations: bool = False
    observation_clip: float = 10.0
    enable_kl_early_stop: bool = True
    target_kl_discrete: float = 0.03
    target_kl_continuous: float = 0.03
    fail_on_nonfinite: bool = True
    entropy_collapse_threshold: float = 0.25
    action_starvation_min_available: int = 10
    value_loss_alert_threshold: float = 100.0


@dataclass(slots=True)
class LLMConfig:
    enabled: bool = False
    provider: str = "mock"
    mode: str = "online-once"
    experience_db: Path = field(default_factory=lambda: REPO_ROOT / "output" / "H_PPO" / "llm_experience.sqlite3")
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-pro"
    timeout_s: float = 45.0
    max_retries: int = 2
    retry_backoff_s: float = 1.0
    max_tokens: int = 4096
    temperature: float = 0.2
    thinking_enabled: bool = True
    reasoning_effort: str = "high"
    prior_weight: float = 0.25
    prior_floor: float = 0.02
    max_candidates: int = 30
    candidate_pool_size: int = 6
    candidate_selector_enabled: bool = False
    candidate_dataset: Path | None = None
    candidate_min_safe_count: int = 3
    candidate_prior_weight: float = 1.0
    cache_ttl_s: float = 30.0
    prediction_horizon_s: float = 120.0
    prediction_step_s: float = 5.0
    enforce_grounding_mask: bool = False
    max_calls_per_episode: int = 2
    max_calls_per_run: int = 20
    max_daily_tokens: int = 200_000
    collect_cache_misses: bool = True
    safety_gate_enabled: bool = True
    safety_gate_clearance_margin: float = 0.05
    coarse_relative_risk_screen: bool = True
    coarse_min_violation_improvement: float = 0.01
    coarse_hard_clearance_ratio: float = 0.25


@dataclass(slots=True)
class RuntimeConfig:
    sim_dt: float = 0.05
    state_update_dt: float = 1.0
    decision_dt: float = 5.0
    min_command_hold_s: float = 10.0
    emergency_override_enabled: bool = True
    emergency_override_severity: float = 0.9
    urgent_noop_mask_enabled: bool = True
    urgent_noop_threshold_s: float = 60.0
    # V2 evaluates an effective maneuver after its minimum hold. If it does not
    # improve risk for two decision periods, the failed action dimension enters
    # a short cooldown and the shared policy must replan through another branch.
    replan_enabled: bool = True
    replan_stall_evaluations: int = 2
    replan_cooldown_s: float = 15.0
    replan_min_severity_drop: float = 0.05
    replan_min_tcpa_gain_s: float = 5.0
    replan_min_horizontal_gain_km: float = 0.5
    replan_min_vertical_gain_ft: float = 200.0
    duplicate_heading_tolerance_deg: float = 5.0
    duplicate_altitude_tolerance_ft: float = 200.0
    duplicate_speed_tolerance_kt: float = 3.0
    # Keep the experimental group-cap available for targeted ablations, but do
    # not enable it by default until it improves full-suite safety metrics.
    # In a connected conflict group, only one aircraft may initiate a vertical
    # maneuver at a decision instant. This prevents symmetric aircraft from
    # selecting the same altitude target and preserving zero vertical spacing.
    vertical_coordination_enabled: bool = True
    # Empirically, constraining every three-aircraft component overrode useful
    # local maneuvers. Reserve the hard vertical-authority cap for genuinely
    # high-coupling groups; smaller groups remain governed by the replan gate.
    vertical_coordination_min_group_size: int = 2
    group_planner_shadow_enabled: bool = False
    max_episode_steps: int = 12000
    max_episode_time_s: float = 900.0
    max_decision_lag_s: float = 5.0
    min_episode_time_s: float = 30.0
    # Both policy-selected and automatic route recovery must observe this
    # continuous clear window. A short gap in a multi-aircraft encounter is not
    # sufficient evidence that direct-to-route is safe.
    auto_resume_after_clear_s: float = 30.0
    spawn_timeout_s: float = 15.0
    clear_timeout_s: float = 15.0
    auto_start: bool = True
    speed_multiplier: float = 0.0
    max_episodes: int = 100
    fail_on_command_error: bool = False
    auto_frame_gui: bool = True
    gui_enabled: bool = False
    run_id: str = ""

    # 视野边缘预留 25%
    gui_view_margin: float = 1.25
    # 按常见 QtGL 窗口估算
    gui_aspect_ratio: float = 16.0 / 9.0
    # 防止只有很短路线时放得过大
    gui_min_span_deg: float = 0.20
    # 自动缩放限制
    gui_min_zoom: float = 0.05
    gui_max_zoom: float = 10.0
    # PAN/ZOOM 后保持 1 倍速多久，再加速
    gui_settle_time_s: float = 1.0


@dataclass(slots=True)
class LoggingConfig:
    repo_root: Path = REPO_ROOT
    output_dir: Path = field(default_factory=lambda: REPO_ROOT / "output" / "H_PPO" / "training")
    tensorboard_dir: str = "tensorboard"
    checkpoint_dir: str = "checkpoints"
    scenario_dir: str = "scenarios"
    stats_csv: str = "training_stats.csv"
    diagnostics_csv: str = "training_diagnostics.csv"
    events_jsonl: str = "events.jsonl"
    config_json: str = "resolved_config.json"
    save_every_episodes: int = 10
    enable_tensorboard: bool = False


@dataclass(slots=True)
class HPPOConfig:
    command_ranges: CommandRanges = field(default_factory=CommandRanges)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    reward: RewardWeights = field(default_factory=RewardWeights)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 7
    route_file: Path = field(default_factory=_default_route_file)
    scenario_files: tuple[Path, ...] = field(default_factory=_default_scenario_files)
    scenario_selection: str = "cycle"
    scenario_template: Path = field(default_factory=lambda: REPO_ROOT / "routes" / "hppo")
    max_intruders: int = 3
    danger_horizontal_km: float = 4.0
    danger_vertical_ft: float = 800.0
    collision_horizontal_km: float = 2.0
    collision_vertical_ft: float = 500.0
    observation_radius_km: float = 25.0
    control_radius_km: float = 20.0
    resume_clear_radius_km: float = 12.0
    secondary_conflict_window_s: float = 30.0
    conflict_lookahead_s: float = 300.0
    route_deviation_limit_km: float = 25.0
    soft_route_deviation_km: float = 15.0
    arrival_distance_km: float = 2.0
    # A one-waypoint BlueSky route can otherwise command an aircraft to turn
    # back after it has just passed the destination. Treat a close, forward
    # pass through the terminal gate as arrival instead of reacquiring it.
    terminal_capture_distance_km: float = 5.0
    terminal_pass_cross_track_km: float = 3.0
    max_command_history: int = 8
    training: bool = True
    device: str = "cpu"
    checkpoint_path: Path | None = None
    resume_training: bool = False


def default_config() -> HPPOConfig:
    cfg = HPPOConfig()
    mode = os.getenv("HPPO_MODE", "train").strip().lower()
    if mode not in {"train", "eval"}:
        raise ValueError("HPPO_MODE must be 'train' or 'eval'")
    cfg.training = mode == "train"
    cfg.runtime.max_episodes = int(os.getenv("HPPO_EPISODES", cfg.runtime.max_episodes))
    cfg.runtime.speed_multiplier = float(os.getenv("HPPO_SPEED", cfg.runtime.speed_multiplier))
    cfg.runtime.max_episode_time_s = float(os.getenv("HPPO_MAX_EPISODE_TIME", cfg.runtime.max_episode_time_s))
    cfg.runtime.gui_enabled = os.getenv("HPPO_GUI", "none").strip().lower() == "qtgl"
    cfg.runtime.urgent_noop_mask_enabled = os.getenv(
        "HPPO_URGENT_NOOP_MASK", "1" if cfg.runtime.urgent_noop_mask_enabled else "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    cfg.runtime.urgent_noop_threshold_s = float(
        os.getenv("HPPO_URGENT_NOOP_THRESHOLD_S", cfg.runtime.urgent_noop_threshold_s)
    )
    cfg.runtime.auto_resume_after_clear_s = float(
        os.getenv("HPPO_AUTO_RESUME_AFTER_CLEAR_S", cfg.runtime.auto_resume_after_clear_s)
    )
    if cfg.runtime.urgent_noop_threshold_s <= 0.0:
        raise ValueError("urgent NO_NEW_COMMAND threshold must be positive")
    if cfg.runtime.auto_resume_after_clear_s < cfg.runtime.decision_dt:
        raise ValueError("auto-resume clear window must be at least one decision interval")
    cfg.runtime.replan_enabled = os.getenv(
        "HPPO_REPLAN_ENABLED", "1" if cfg.runtime.replan_enabled else "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    cfg.runtime.replan_stall_evaluations = int(
        os.getenv("HPPO_REPLAN_STALL_EVALUATIONS", cfg.runtime.replan_stall_evaluations)
    )
    cfg.runtime.replan_cooldown_s = float(os.getenv("HPPO_REPLAN_COOLDOWN_S", cfg.runtime.replan_cooldown_s))
    cfg.runtime.replan_min_severity_drop = float(
        os.getenv("HPPO_REPLAN_MIN_SEVERITY_DROP", cfg.runtime.replan_min_severity_drop)
    )
    cfg.runtime.replan_min_tcpa_gain_s = float(os.getenv("HPPO_REPLAN_MIN_TCPA_GAIN_S", cfg.runtime.replan_min_tcpa_gain_s))
    cfg.runtime.replan_min_horizontal_gain_km = float(os.getenv("HPPO_REPLAN_MIN_HORIZONTAL_GAIN_KM", cfg.runtime.replan_min_horizontal_gain_km))
    cfg.runtime.replan_min_vertical_gain_ft = float(os.getenv("HPPO_REPLAN_MIN_VERTICAL_GAIN_FT", cfg.runtime.replan_min_vertical_gain_ft))
    cfg.runtime.duplicate_heading_tolerance_deg = float(os.getenv("HPPO_DUPLICATE_HEADING_TOLERANCE_DEG", cfg.runtime.duplicate_heading_tolerance_deg))
    cfg.runtime.duplicate_altitude_tolerance_ft = float(os.getenv("HPPO_DUPLICATE_ALTITUDE_TOLERANCE_FT", cfg.runtime.duplicate_altitude_tolerance_ft))
    cfg.runtime.duplicate_speed_tolerance_kt = float(os.getenv("HPPO_DUPLICATE_SPEED_TOLERANCE_KT", cfg.runtime.duplicate_speed_tolerance_kt))
    cfg.runtime.vertical_coordination_enabled = os.getenv(
        "HPPO_VERTICAL_COORDINATION", "1" if cfg.runtime.vertical_coordination_enabled else "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    cfg.runtime.vertical_coordination_min_group_size = int(
        os.getenv("HPPO_VERTICAL_COORDINATION_MIN_GROUP_SIZE", cfg.runtime.vertical_coordination_min_group_size)
    )
    if cfg.runtime.replan_stall_evaluations < 1:
        raise ValueError("replan stall evaluations must be positive")
    if min(
        cfg.runtime.replan_cooldown_s,
        cfg.runtime.replan_min_severity_drop,
        cfg.runtime.replan_min_tcpa_gain_s,
        cfg.runtime.replan_min_horizontal_gain_km,
        cfg.runtime.replan_min_vertical_gain_ft,
        cfg.runtime.duplicate_heading_tolerance_deg,
        cfg.runtime.duplicate_altitude_tolerance_ft,
        cfg.runtime.duplicate_speed_tolerance_kt,
    ) < 0.0:
        raise ValueError("replan and duplicate-target thresholds must be non-negative")
    if cfg.runtime.vertical_coordination_min_group_size < 2:
        raise ValueError("vertical coordination group size must be at least two")
    cfg.runtime.group_planner_shadow_enabled = os.getenv("HPPO_GROUP_PLANNER_SHADOW", "0").strip().lower() in {"1", "true", "yes", "on"}
    cfg.command_ranges.min_altitude_ft = float(
        os.getenv("HPPO_MIN_ALTITUDE_FT", cfg.command_ranges.min_altitude_ft)
    )
    cfg.command_ranges.max_altitude_ft = float(
        os.getenv("HPPO_MAX_ALTITUDE_FT", cfg.command_ranges.max_altitude_ft)
    )
    cfg.command_ranges.default_initial_altitude_ft = float(
        os.getenv("HPPO_DEFAULT_INITIAL_ALTITUDE_FT", cfg.command_ranges.default_initial_altitude_ft)
    )
    if cfg.command_ranges.min_altitude_ft >= cfg.command_ranges.max_altitude_ft:
        raise ValueError("minimum altitude must be lower than maximum altitude")
    if not (
        cfg.command_ranges.min_altitude_ft
        <= cfg.command_ranges.default_initial_altitude_ft
        <= cfg.command_ranges.max_altitude_ft
    ):
        raise ValueError("default initial altitude must be within the configured altitude envelope")
    cfg.command_ranges.min_speed_kt = float(os.getenv("HPPO_MIN_SPEED_KT", cfg.command_ranges.min_speed_kt))
    cfg.command_ranges.max_speed_kt = float(os.getenv("HPPO_MAX_SPEED_KT", cfg.command_ranges.max_speed_kt))
    cfg.command_ranges.default_initial_speed_kt = float(
        os.getenv("HPPO_DEFAULT_INITIAL_SPEED_KT", cfg.command_ranges.default_initial_speed_kt)
    )
    cfg.command_ranges.speed_parameterization = os.getenv(
        "HPPO_SPEED_PARAMETERIZATION", cfg.command_ranges.speed_parameterization
    ).strip().lower()
    cfg.command_ranges.speed_bin_step_kt = float(
        os.getenv("HPPO_SPEED_BIN_STEP_KT", cfg.command_ranges.speed_bin_step_kt)
    )
    if cfg.command_ranges.min_speed_kt >= cfg.command_ranges.max_speed_kt:
        raise ValueError("minimum speed must be lower than maximum speed")
    if not (
        cfg.command_ranges.min_speed_kt
        <= cfg.command_ranges.default_initial_speed_kt
        <= cfg.command_ranges.max_speed_kt
    ):
        raise ValueError("default initial speed must be within the configured speed envelope")
    if cfg.command_ranges.speed_parameterization != "tas_kt":
        raise ValueError("the current discrete speed policy supports tas_kt only")
    if cfg.command_ranges.speed_bin_step_kt <= 0.0:
        raise ValueError("speed bin step must be positive")
    cfg.network.parameter_policy_type = os.getenv(
        "HPPO_PARAMETER_POLICY", cfg.network.parameter_policy_type
    ).strip().lower()
    if cfg.network.parameter_policy_type not in {"continuous", "discrete"}:
        raise ValueError("parameter policy must be continuous or discrete")
    cfg.network.heading_bin_deg = float(os.getenv("HPPO_HEADING_BIN_DEG", cfg.network.heading_bin_deg))
    cfg.network.heading_normal_limit_deg = float(
        os.getenv("HPPO_HEADING_NORMAL_LIMIT_DEG", cfg.network.heading_normal_limit_deg)
    )
    cfg.network.heading_emergency_limit_deg = float(
        os.getenv("HPPO_HEADING_EMERGENCY_LIMIT_DEG", cfg.network.heading_emergency_limit_deg)
    )
    cfg.network.heading_min_effective_change_deg = float(
        os.getenv("HPPO_HEADING_MIN_EFFECTIVE_CHANGE_DEG", cfg.network.heading_min_effective_change_deg)
    )
    cfg.network.altitude_bin_step_ft = float(
        os.getenv("HPPO_ALTITUDE_BIN_STEP_FT", cfg.network.altitude_bin_step_ft)
    )
    if cfg.network.heading_bin_deg <= 0.0 or 360.0 % cfg.network.heading_bin_deg != 0.0:
        raise ValueError("heading bin must be a positive divisor of 360")
    if not 0.0 < cfg.network.heading_normal_limit_deg <= cfg.network.heading_emergency_limit_deg <= 180.0:
        raise ValueError("heading limits must satisfy 0 < normal <= emergency <= 180")
    if cfg.network.heading_min_effective_change_deg < 0.0:
        raise ValueError("minimum effective heading change must be non-negative")
    if cfg.network.altitude_bin_step_ft <= 0.0:
        raise ValueError("altitude bin step must be positive")
    cfg.command_ranges.altitude_boundary_guard_ft = float(
        os.getenv(
            "HPPO_ALTITUDE_BOUNDARY_GUARD_FT",
            cfg.command_ranges.altitude_boundary_guard_ft,
        )
    )
    if cfg.command_ranges.altitude_boundary_guard_ft < 0.0:
        raise ValueError("altitude boundary guard must be non-negative")
    cfg.runtime.run_id = os.getenv("HPPO_RUN_ID", "").strip()
    cfg.seed = int(os.getenv("HPPO_SEED", cfg.seed))
    cfg.device = os.getenv("HPPO_DEVICE", cfg.device).strip() or "cpu"
    cfg.ppo.entropy_coef_discrete = float(
        os.getenv("HPPO_ENTROPY_COEF_DISCRETE", cfg.ppo.entropy_coef_discrete)
    )
    cfg.ppo.entropy_coef_continuous = float(
        os.getenv("HPPO_ENTROPY_COEF_CONTINUOUS", cfg.ppo.entropy_coef_continuous)
    )
    cfg.ppo.entropy_collapse_threshold = float(
        os.getenv("HPPO_ENTROPY_COLLAPSE_THRESHOLD", cfg.ppo.entropy_collapse_threshold)
    )
    cfg.ppo.action_starvation_min_available = int(
        os.getenv("HPPO_ACTION_STARVATION_MIN_AVAILABLE", cfg.ppo.action_starvation_min_available)
    )
    if min(
        cfg.ppo.entropy_coef_discrete,
        cfg.ppo.entropy_coef_continuous,
        cfg.ppo.entropy_collapse_threshold,
    ) < 0.0:
        raise ValueError("entropy coefficients and threshold must be non-negative")
    if cfg.ppo.action_starvation_min_available < 1:
        raise ValueError("action starvation availability threshold must be positive")
    cfg.scenario_selection = os.getenv("HPPO_SCENARIO_SELECTION", cfg.scenario_selection).strip().lower()
    if cfg.scenario_selection not in {"cycle", "random", "fixed"}:
        raise ValueError("HPPO_SCENARIO_SELECTION must be cycle, random, or fixed")
    cfg.resume_training = os.getenv("HPPO_RESUME", "0").strip().lower() in {"1", "true", "yes", "on"}
    cfg.logging.save_every_episodes = int(os.getenv("HPPO_SAVE_EVERY", cfg.logging.save_every_episodes))
    cfg.logging.enable_tensorboard = os.getenv("HPPO_TENSORBOARD", "0").strip().lower() in {"1", "true", "yes", "on"}
    cfg.llm.enabled = os.getenv("HPPO_LLM_GUIDANCE", "0").strip().lower() in {"1", "true", "yes", "on"}
    cfg.llm.provider = os.getenv("HPPO_LLM_PROVIDER", cfg.llm.provider).strip().lower()
    if cfg.llm.provider not in {"mock", "deepseek"}:
        raise ValueError("HPPO_LLM_PROVIDER must be mock or deepseek")
    cfg.llm.mode = os.getenv("HPPO_LLM_MODE", cfg.llm.mode).strip().lower()
    if cfg.llm.mode not in {"cache-only", "online-once"}:
        raise ValueError("HPPO_LLM_MODE must be cache-only or online-once")
    cfg.llm.experience_db = Path(os.getenv("HPPO_LLM_DB", str(cfg.llm.experience_db))).expanduser().resolve()
    cfg.llm.base_url = os.getenv("HPPO_LLM_BASE_URL", cfg.llm.base_url).strip().rstrip("/")
    cfg.llm.model = os.getenv("HPPO_LLM_MODEL", cfg.llm.model).strip()
    if cfg.llm.model not in {"deepseek-v4-flash", "deepseek-v4-pro"}:
        raise ValueError("HPPO_LLM_MODEL must be deepseek-v4-flash or deepseek-v4-pro")
    cfg.llm.timeout_s = float(os.getenv("HPPO_LLM_TIMEOUT_S", cfg.llm.timeout_s))
    cfg.llm.max_retries = int(os.getenv("HPPO_LLM_MAX_RETRIES", cfg.llm.max_retries))
    cfg.llm.thinking_enabled = os.getenv("HPPO_LLM_THINKING", "1").strip().lower() in {"1", "true", "yes", "on"}
    cfg.llm.max_calls_per_episode = int(os.getenv("HPPO_LLM_MAX_CALLS_EPISODE", cfg.llm.max_calls_per_episode))
    cfg.llm.max_calls_per_run = int(os.getenv("HPPO_LLM_MAX_CALLS_RUN", cfg.llm.max_calls_per_run))
    cfg.llm.max_daily_tokens = int(os.getenv("HPPO_LLM_MAX_DAILY_TOKENS", cfg.llm.max_daily_tokens))
    cfg.llm.candidate_selector_enabled = os.getenv("HPPO_CANDIDATE_SELECTOR", "0").strip().lower() in {"1", "true", "yes", "on"}
    candidate_dataset = os.getenv("HPPO_CANDIDATE_DATASET", "").strip()
    cfg.llm.candidate_dataset = Path(candidate_dataset).expanduser().resolve() if candidate_dataset else None
    cfg.llm.candidate_min_safe_count = int(os.getenv("HPPO_CANDIDATE_MIN_SAFE", cfg.llm.candidate_min_safe_count))
    cfg.llm.candidate_prior_weight = float(os.getenv("HPPO_CANDIDATE_PRIOR_WEIGHT", cfg.llm.candidate_prior_weight))
    cfg.llm.prior_weight = float(os.getenv("HPPO_LLM_PRIOR_WEIGHT", cfg.llm.prior_weight))
    cfg.llm.enforce_grounding_mask = os.getenv("HPPO_LLM_ENFORCE_MASK", "0").strip().lower() in {"1", "true", "yes", "on"}
    cfg.llm.safety_gate_enabled = os.getenv("HPPO_LLM_SAFETY_GATE", "1").strip().lower() in {"1", "true", "yes", "on"}
    cfg.llm.safety_gate_clearance_margin = float(os.getenv("HPPO_LLM_SAFETY_MARGIN", cfg.llm.safety_gate_clearance_margin))
    cfg.llm.coarse_relative_risk_screen = os.getenv("HPPO_LLM_COARSE_RELATIVE_RISK", "1").strip().lower() in {"1", "true", "yes", "on"}
    cfg.llm.coarse_min_violation_improvement = float(
        os.getenv("HPPO_LLM_COARSE_MIN_IMPROVEMENT", cfg.llm.coarse_min_violation_improvement)
    )
    cfg.llm.coarse_hard_clearance_ratio = float(
        os.getenv("HPPO_LLM_COARSE_HARD_CLEARANCE_RATIO", cfg.llm.coarse_hard_clearance_ratio)
    )
    if cfg.llm.prior_weight < 0.0:
        raise ValueError("LLM prior weight must be non-negative")
    if cfg.llm.safety_gate_clearance_margin < 0.0:
        raise ValueError("LLM safety clearance margin must be non-negative")
    if cfg.llm.coarse_min_violation_improvement < 0.0:
        raise ValueError("coarse relative-risk improvement must be non-negative")
    if not 0.0 < cfg.llm.coarse_hard_clearance_ratio <= 1.0:
        raise ValueError("coarse hard-clearance ratio must be in (0, 1]")
    if cfg.llm.timeout_s <= 0.0 or cfg.llm.max_retries < 0:
        raise ValueError("LLM timeout must be positive and retries non-negative")
    if min(cfg.llm.max_calls_per_episode, cfg.llm.max_calls_per_run, cfg.llm.max_daily_tokens) < 0:
        raise ValueError("LLM call and token budgets must be non-negative")
    if cfg.llm.candidate_pool_size <= 0:
        raise ValueError("LLM candidate pool size must be positive")
    if cfg.llm.candidate_min_safe_count <= 0 or cfg.llm.candidate_min_safe_count > cfg.llm.candidate_pool_size:
        raise ValueError("candidate minimum safe count must be within candidate pool size")
    if cfg.llm.candidate_selector_enabled and cfg.llm.candidate_dataset is None:
        raise ValueError("HPPO_CANDIDATE_DATASET is required when the candidate selector is enabled")
    if cfg.llm.candidate_prior_weight < 0.0:
        raise ValueError("candidate prior weight must be non-negative")
    if not 0.0 < cfg.llm.prior_floor < 1.0:
        raise ValueError("LLM prior floor must be between zero and one")
    cfg.separation.horizontal_km = float(os.getenv("HPPO_SEPARATION_HORIZONTAL_KM", cfg.separation.horizontal_km))
    cfg.separation.vertical_ft = float(os.getenv("HPPO_SEPARATION_VERTICAL_FT", cfg.separation.vertical_ft))
    cfg.separation.temporal_s = float(os.getenv("HPPO_SEPARATION_TIME_S", cfg.separation.temporal_s))
    cfg.separation.mode = os.getenv("HPPO_SEPARATION_MODE", cfg.separation.mode).strip().lower()
    cfg.separation.release_factor = float(os.getenv("HPPO_SEPARATION_RELEASE_FACTOR", cfg.separation.release_factor))
    cfg.separation.validate()
    if cfg.observation_radius_km < cfg.separation.horizontal_km * cfg.separation.release_factor:
        raise ValueError("observation radius must cover the horizontal separation release boundary")

    output_value = os.getenv("HPPO_OUTPUT")
    if not cfg.training:
        cfg.logging.stats_csv = "validation_stats.csv"
        cfg.logging.diagnostics_csv = "validation_diagnostics.csv"
    if output_value:
        cfg.logging.output_dir = Path(output_value).expanduser().resolve()
    elif not cfg.training:
        cfg.logging.output_dir = REPO_ROOT / "output" / "H_PPO" / "evaluation"

    checkpoint_value = os.getenv("HPPO_CHECKPOINT")
    if checkpoint_value:
        cfg.checkpoint_path = Path(checkpoint_value).expanduser().resolve()

    testset_value = os.getenv("HPPO_TESTSET_ROOT", "").strip()
    if testset_value:
        testset_root = Path(testset_value).expanduser().resolve()
        if not testset_root.is_dir():
            raise ValueError(f"HPPO_TESTSET_ROOT is not a directory: {testset_root}")
        cfg.scenario_files = tuple(sorted(testset_root.glob("*.npz")))
        if not cfg.scenario_files:
            raise ValueError(f"HPPO_TESTSET_ROOT contains no .npz samples: {testset_root}")
    else:
        scenario_value = os.getenv("HPPO_SCENARIOS", "all").strip()
        available = _default_scenario_files()
        if scenario_value.lower() == "all":
            cfg.scenario_files = available
        else:
            requested = [token.strip() for token in scenario_value.split(",") if token.strip()]
            selected: list[Path] = []
            for token in requested:
                direct = Path(token).expanduser()
                if direct.exists():
                    selected.append(direct.resolve())
                    continue
                selector = token.zfill(2) if token.isdigit() else token
                matches = [path for path in available if path.stem == selector or path.name.startswith(f"{selector}_")]
                if len(matches) != 1:
                    raise ValueError(f"Scenario selector '{token}' matched {len(matches)} files")
                selected.append(matches[0])
            cfg.scenario_files = tuple(selected)
    if not cfg.scenario_files:
        cfg.scenario_files = (cfg.route_file,)
    cfg.route_file = cfg.scenario_files[0]
    return cfg
