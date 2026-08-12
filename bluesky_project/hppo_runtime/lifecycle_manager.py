from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict

import bluesky as bs
import numpy as np

from .core import HPPOAgent, HPPOCoreConfig
from .core.checkpoint import load_checkpoint_file, save_checkpoint_file
from .config import HPPOConfig
from .hppo_environment import HPPOEnvironment
from .conflict_group_planner import ConflictGroupPlanner
from .llm import FrozenCandidateDataset, LLMGuidanceManager, map_candidate_parameters
from .legacy_bluesky import configure_simulation, publish_event


class _NullSummaryWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def _summary_writer(config: HPPOConfig):
    if not config.logging.enable_tensorboard:
        return _NullSummaryWriter()
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(config.logging.output_dir / config.logging.tensorboard_dir))
    except Exception:
        return _NullSummaryWriter()


class EpisodePhase(str, Enum):
    WAITING_CLEAR = "waiting_clear"
    SPAWN_QUEUED = "spawn_queued"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(slots=True)
class PendingDecision:
    episode_id: int
    step_index: int
    trajectory_id: str
    acid: str
    global_state: np.ndarray
    presence_mask: np.ndarray
    target_index: int
    local_obs: np.ndarray
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
    policy_version: int
    reward_accumulator: float = 0.0


class TrainingManager:
    def __init__(self, config: HPPOConfig):
        self.cfg = config
        self.env = HPPOEnvironment(config)
        self.group_planner = ConflictGroupPlanner(config) if config.runtime.group_planner_shadow_enabled else None
        self.agent = HPPOAgent(HPPOCoreConfig.from_env(config), device=config.device, seed=config.seed)
        self.agent.set_training(config.training)
        self.llm_guidance = LLMGuidanceManager(config) if config.llm.enabled else None
        self.candidate_dataset = (
            FrozenCandidateDataset(config.llm.candidate_dataset, config.llm.candidate_pool_size)
            if config.llm.candidate_selector_enabled and config.llm.candidate_dataset is not None
            else None
        )
        self.scene_builder = self.llm_guidance or (LLMGuidanceManager(config) if self.candidate_dataset is not None else None)
        self.writer = _summary_writer(config)
        self.pending: Dict[str, PendingDecision] = {}
        self.trajectory_generation: dict[str, int] = {}
        self.phase = EpisodePhase.WAITING_CLEAR
        self.phase_started_s = float(getattr(bs.sim, "simt", 0.0))
        self.completed_episodes = 0
        self.run_episodes = 0
        self.total_environment_steps = 0
        self.decision_count = 0
        self.actor_calls = 0
        self.command_executions = 0
        self.transition_count = 0
        self.decision_response_samples = 0
        self.decision_response_total_ms = 0.0
        self.decision_response_max_ms = 0.0
        self.action_counts = np.zeros(5, dtype=np.int64)
        self.applied_action_counts = np.zeros(5, dtype=np.int64)
        self.available_action_counts = np.zeros(5, dtype=np.int64)
        self.parameter_mean_sum = np.zeros(3, dtype=np.float64)
        self.parameter_std_sum = np.zeros(3, dtype=np.float64)
        self.parameter_sample_count = 0
        self.had_conflict = False
        self.clear_since_s: float | None = None
        self.last_conflict_time_s: float | None = None
        self.separation_adjustments: dict[str, dict[str, Any]] = {}
        self.separation_adjustment_sequence = 0
        self.finished = False
        self.stats_file = self.cfg.logging.output_dir / self.cfg.logging.stats_csv
        self.diagnostics_file = self.cfg.logging.output_dir / self.cfg.logging.diagnostics_csv
        self.events_file = self.cfg.logging.output_dir / self.cfg.logging.events_jsonl
        self.training_ckpt_file = self.cfg.logging.output_dir / self.cfg.logging.checkpoint_dir / "hppo_checkpoint.pt"
        # The bridge attaches a sink after construction; checkpoint loading may
        # emit an event during construction, so the attribute must exist first.
        self.event_sink = None
        self.cfg.logging.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_resolved_config()
        self._load_checkpoint_if_requested()
        self._event(
            "manager_initialized",
            mode=self.mode,
            device=str(self.agent.device),
            separation=self._jsonable(self.cfg.separation),
        )
        self.speed_applied = False
        self.speed_apply_at_s: float | None = None
    
    @property
    def mode(self) -> str:
        return "train" if self.cfg.training else "eval"

    def set_horizontal_separation_km(self, horizontal_km: float, source: str = "runtime") -> tuple[float, float]:
        """Update the shared standard used by detection, policy input and reward."""
        simt = float(getattr(bs.sim, "simt", 0.0))
        before_states = (
            self.env.conflict_manager.detect(bs.traf, self.cfg.network.max_aircraft)
            if len(bs.traf.id) > 0
            else {}
        )
        previous, current = self.env.set_horizontal_separation_km(horizontal_km)
        conflict_states = (
            self.env.conflict_manager.detect(bs.traf, self.cfg.network.max_aircraft)
            if len(bs.traf.id) > 0
            else {}
        )
        self.env.rebaseline_active_command_risk(conflict_states, simt)
        adjustment = self._begin_separation_adjustment(
            previous, current, source, simt, before_states, conflict_states,
        )
        self._save_resolved_config()
        self._event(
            "separation_updated",
            source=source,
            previous_horizontal_km=previous,
            horizontal_km=current,
            horizontal_nm=current / 1.852,
            vertical_ft=self.cfg.separation.vertical_ft,
            temporal_s=self.cfg.separation.temporal_s,
            separation_mode=self.cfg.separation.mode,
            adjustment_id=adjustment["adjustment_id"],
            config_applied=True,
            detection_recomputed=True,
            affected_pair_count=len(adjustment["pairs"]),
            initial_violation_count=adjustment["initial_violation_count"],
            adjustment_state=adjustment["state"],
        )
        return previous, current

    @staticmethod
    def _risk_pairs(conflict_states: dict) -> dict[tuple[str, str], dict[str, Any]]:
        """Return only pairs that are risky under the currently active standard."""
        pairs: dict[tuple[str, str], dict[str, Any]] = {}
        for acid, state in conflict_states.items():
            for contact in state.targets:
                risky = bool(contact.conflict_flag or contact.temporal_conflict or contact.severity > 0.0)
                if not risky:
                    continue
                key = tuple(sorted((str(acid), str(contact.callsign))))
                current = pairs.get(key)
                item = {
                    "current_loss": bool(contact.conflict_flag),
                    "predicted_conflict": bool(contact.severity > 0.0),
                    "temporal_conflict": bool(contact.temporal_conflict),
                    "horizontal_km": float(contact.horiz_km),
                    "vertical_ft": float(contact.vert_ft),
                    "predicted_horizontal_km": float(contact.dcpa_km),
                    "predicted_vertical_ft": float(contact.predicted_vert_ft),
                    "tcpa_s": float(contact.tcpa_s),
                    "time_gap_s": float(contact.time_gap_s),
                }
                if current is None or item["predicted_horizontal_km"] < current["predicted_horizontal_km"]:
                    pairs[key] = item
        return pairs

    def _begin_separation_adjustment(
        self,
        previous: float,
        current: float,
        source: str,
        simt: float,
        before_states: dict,
        after_states: dict,
    ) -> dict[str, Any]:
        """Start a separate, auditable lifecycle for one HSEP configuration change."""
        for adjustment in list(self.separation_adjustments.values()):
            self._finish_separation_adjustment(adjustment, simt, "superseded_by_new_adjustment")
        self.separation_adjustments.clear()
        self.separation_adjustment_sequence += 1
        before_pairs = self._risk_pairs(before_states)
        after_pairs = self._risk_pairs(after_states)
        pair_keys = sorted(set(before_pairs) | set(after_pairs))
        changed = abs(float(current) - float(previous)) > 1e-9
        if not changed:
            state = "NOT_APPLICABLE"
        elif not pair_keys:
            state = "NOT_APPLICABLE"
        else:
            state = "OBSERVING"
        adjustment = {
            "adjustment_id": f"sep-{self.env.episode_id}-{simt:.3f}-{self.separation_adjustment_sequence}",
            "episode": int(self.env.episode_id),
            "source": source,
            "previous_horizontal_km": float(previous),
            "horizontal_km": float(current),
            "separation_mode": str(self.cfg.separation.mode),
            "pairs": pair_keys,
            "initial_violation_count": len(after_pairs),
            "started_s": float(simt),
            "deadline_s": float(simt + self.cfg.runtime.separation_adjustment_deadline_s),
            "confirmation_window_s": float(self.cfg.runtime.separation_adjustment_confirmation_s),
            "clear_since_s": None,
            "state": state,
        }
        if state == "NOT_APPLICABLE":
            self._finish_separation_adjustment(adjustment, simt, "no_affected_conflict_pair")
        elif self.cfg.runtime.separation_adjustment_enabled:
            self.separation_adjustments[adjustment["adjustment_id"]] = adjustment
        return adjustment

    def _finish_separation_adjustment(self, adjustment: dict[str, Any], simt: float, reason: str) -> None:
        if adjustment.get("state") in {"SUCCESS", "FAILED", "SUPERSEDED"}:
            return
        if reason == "stable_compliance_confirmed":
            state, success = "SUCCESS", True
        elif reason == "superseded_by_new_adjustment":
            state, success = "SUPERSEDED", False
        else:
            state, success = ("NOT_APPLICABLE", False) if adjustment["state"] == "NOT_APPLICABLE" else ("FAILED", False)
        adjustment["state"] = state
        self._event(
            "separation_adjustment_outcome",
            adjustment_id=adjustment["adjustment_id"],
            source=adjustment["source"],
            previous_horizontal_km=adjustment["previous_horizontal_km"],
            horizontal_km=adjustment["horizontal_km"],
            separation_mode=adjustment["separation_mode"],
            affected_pairs=[list(pair) for pair in adjustment["pairs"]],
            initial_violation_count=adjustment["initial_violation_count"],
            secondary_pair_count=int(adjustment.get("secondary_pair_count", 0)),
            config_applied=True,
            state=state,
            success=success,
            reason=reason,
            response_time_s=max(0.0, float(simt) - adjustment["started_s"]),
            confirmation_window_s=adjustment["confirmation_window_s"],
            stable_for_s=(0.0 if adjustment["clear_since_s"] is None else max(0.0, float(simt) - adjustment["clear_since_s"])),
        )

    def _update_separation_adjustments(self, conflict_states: dict, simt: float) -> None:
        if not self.separation_adjustments:
            return
        active_pairs = self._risk_pairs(conflict_states)
        collision = any(state.collision for state in conflict_states.values())
        for adjustment_id, adjustment in list(self.separation_adjustments.items()):
            if collision:
                self._finish_separation_adjustment(adjustment, simt, "collision_after_adjustment")
                self.separation_adjustments.pop(adjustment_id, None)
                continue
            original_pairs = set(adjustment["pairs"])
            secondary_pairs = set(active_pairs).difference(original_pairs)
            adjustment["secondary_pair_count"] = max(
                int(adjustment.get("secondary_pair_count", 0)), len(secondary_pairs)
            )
            remaining = [pair for pair in adjustment["pairs"] if pair in active_pairs]
            if not remaining:
                if adjustment["clear_since_s"] is None:
                    adjustment["clear_since_s"] = float(simt)
                if float(simt) - adjustment["clear_since_s"] >= adjustment["confirmation_window_s"]:
                    outcome_reason = (
                        "secondary_conflict_after_adjustment"
                        if adjustment["secondary_pair_count"] > 0
                        else "stable_compliance_confirmed"
                    )
                    self._finish_separation_adjustment(adjustment, simt, outcome_reason)
                    self.separation_adjustments.pop(adjustment_id, None)
                    continue
            else:
                adjustment["clear_since_s"] = None
            if float(simt) >= adjustment["deadline_s"]:
                self._finish_separation_adjustment(adjustment, simt, "recovery_deadline_exceeded")
                self.separation_adjustments.pop(adjustment_id, None)

    def _finalize_separation_adjustments(self, simt: float, episode_reason: str) -> None:
        for adjustment_id, adjustment in list(self.separation_adjustments.items()):
            self._finish_separation_adjustment(adjustment, simt, f"episode_finished_{episode_reason}")
            self.separation_adjustments.pop(adjustment_id, None)

    @property
    def needs_spawn(self) -> bool:
        return self.phase in {EpisodePhase.WAITING_CLEAR, EpisodePhase.SPAWN_QUEUED}

    def _jsonable(self, value: Any) -> Any:
        if is_dataclass(value):
            return self._jsonable(asdict(value))
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [self._jsonable(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _save_resolved_config(self) -> None:
        path = self.cfg.logging.output_dir / self.cfg.logging.config_json
        path.write_text(json.dumps(self._jsonable(self.cfg), indent=2, ensure_ascii=True), encoding="utf-8")

    def _event(self, event: str, **payload: Any) -> None:
        record = {
            "event": event,
            "mode": self.mode,
            "episode": self.env.episode_id,
            "sim_time_s": float(getattr(bs.sim, "simt", 0.0)),
            **payload,
        }
        self.events_file.parent.mkdir(parents=True, exist_ok=True)
        with self.events_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(self._jsonable(record), ensure_ascii=True) + "\n")
        if self.event_sink is not None:
            self.event_sink(self._jsonable(record))

    def _load_checkpoint_if_requested(self) -> None:
        should_load = (self.cfg.training and self.cfg.resume_training) or not self.cfg.training
        if not should_load:
            return
        checkpoint = self.cfg.checkpoint_path or self.training_ckpt_file
        if not checkpoint.exists():
            raise FileNotFoundError(f"Requested checkpoint does not exist: {checkpoint}")
        payload = load_checkpoint_file(checkpoint, self.agent, restore_rng=self.cfg.training)
        checkpoint_episode = int(payload.get("episode", 0))
        self.completed_episodes = checkpoint_episode if self.cfg.training else 0
        training_state = payload.get("extra", {}).get("training_state", {})
        self.total_environment_steps = int(training_state.get("total_environment_steps", self.agent.environment_step))
        self.env.episode_id = checkpoint_episode if self.cfg.training else 0
        self._event("checkpoint_loaded", path=str(checkpoint), checkpoint_episode=checkpoint_episode)

    def reset(self) -> None:
        if self.cfg.training:
            self._flush_pending(terminated=False, truncated=True)
        self.pending.clear()
        self.env.cancel_pending_spawns()
        self.env.clear_traffic()
        self.env.reset_episode()
        self.phase = EpisodePhase.WAITING_CLEAR
        self.phase_started_s = float(getattr(bs.sim, "simt", 0.0))
        self._reset_episode_tracking()
        self.env.clear_route_overlays()

    def _reset_episode_tracking(self) -> None:
        self.pending.clear()
        self.trajectory_generation.clear()
        self.decision_count = 0
        self.actor_calls = 0
        self.command_executions = 0
        self.transition_count = 0
        self.decision_response_samples = 0
        self.decision_response_total_ms = 0.0
        self.decision_response_max_ms = 0.0
        self.action_counts.fill(0)
        self.applied_action_counts.fill(0)
        self.available_action_counts.fill(0)
        self.parameter_mean_sum.fill(0.0)
        self.parameter_std_sum.fill(0.0)
        self.parameter_sample_count = 0
        self.had_conflict = False
        self.clear_since_s = None
        self.last_conflict_time_s = None
        self.speed_applied = False
        self.speed_apply_at_s = None

    def _zero_global_state(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros((self.cfg.network.max_aircraft, self.cfg.network.global_slot_dim), dtype=np.float32),
            np.zeros((self.cfg.network.max_aircraft,), dtype=np.float32),
        )

    def _close_transition(
        self,
        acid: str,
        next_global_state: np.ndarray,
        next_presence: np.ndarray,
        next_target_index: int,
        terminated: bool,
        truncated: bool,
        reward: float,
    ) -> None:
        pending = self.pending.get(acid)
        if pending is None:
            return
        next_value = 0.0 if terminated else self.agent.predict_value(next_global_state, next_presence, next_target_index)
        self.agent.buffer.add(
            episode_id=pending.episode_id,
            step_index=pending.step_index,
            trajectory_id=pending.trajectory_id,
            acid=acid,
            local_obs=pending.local_obs,
            global_state=pending.global_state,
            presence_mask=pending.presence_mask,
            target_index=pending.target_index,
            action_mask=pending.action_mask,
            action_prior=pending.action_prior,
            prior_weight=pending.prior_weight,
            macro_action=pending.macro_action,
            raw_params=pending.raw_params,
            squashed_params=pending.squashed_params,
            selected_param_index=pending.selected_param_index,
            parameter_branch=pending.parameter_branch,
            parameter_action_index=pending.parameter_action_index,
            parameter_mask=pending.parameter_mask,
            parameter_probs=pending.parameter_probs,
            parameter_logp=pending.parameter_logp,
            parameter_entropy=pending.parameter_entropy,
            discrete_logp=pending.discrete_logp,
            discrete_entropy=pending.discrete_entropy,
            discrete_probs=pending.discrete_probs,
            continuous_logp=pending.continuous_logp,
            continuous_entropy=pending.continuous_entropy,
            continuous_mean=pending.continuous_mean,
            continuous_std=pending.continuous_std,
            candidate_mode=pending.candidate_mode,
            candidate_features=pending.candidate_features,
            candidate_mask=pending.candidate_mask,
            candidate_macro_actions=pending.candidate_macro_actions,
            candidate_index=pending.candidate_index,
            candidate_logp=pending.candidate_logp,
            candidate_entropy=pending.candidate_entropy,
            value=pending.value,
            next_value=next_value,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            policy_version=pending.policy_version,
        )
        self.transition_count += 1
        del self.pending[acid]

    def _flush_pending(self, terminated: bool, truncated: bool) -> None:
        if not self.pending or not self.cfg.training:
            self.pending.clear()
            return
        if len(bs.traf.id) > 0:
            conflict_states = self.env.conflict_manager.detect(bs.traf, self.cfg.network.max_aircraft)
            global_state, presence, id_to_slot = self.env.build_global_state(conflict_states)
        else:
            global_state, presence = self._zero_global_state()
            id_to_slot = {}
        for acid, pending in list(self.pending.items()):
            runtime = self.env.runtime.get(acid)
            reward = runtime.pending_reward if runtime is not None else pending.reward_accumulator
            if runtime is not None:
                runtime.pending_reward = 0.0
            self._close_transition(
                acid,
                global_state,
                presence,
                id_to_slot.get(acid, pending.target_index),
                terminated,
                truncated,
                reward,
            )

    def _begin_episode(self, simt: float) -> None:
        if self.llm_guidance is not None:
            self.llm_guidance.begin_episode()
        if self.cfg.runtime.auto_start:
            bs.sim.op()
            if self.cfg.runtime.speed_multiplier > 0.0:
                configure_simulation(1.0)

        # 清除上一场景的 QtGL 路线
        self.env.clear_route_overlays()
        
        self.env.load_scenario(self.env.episode_id)
        self.env.reset_episode()
        self._reset_episode_tracking()
        count = self.env.spawn_episode_fleet(simt)
        if count != len(self.env.expected_aircraft_ids):
            raise RuntimeError(f"Queued {count} aircraft, expected {len(self.env.expected_aircraft_ids)}")
        self.phase = EpisodePhase.SPAWN_QUEUED
        self.phase_started_s = simt
        self._event("spawn_queued", scenario=self.env.current_scenario_name, aircraft=count)

    def _activate_episode(self, simt: float) -> None:
        self.env.reset_episode()
        self.env.episode_start_time_s = simt
        self.env.sync_runtime()
    
        # 绘制当前 npy 中的起点—终点路线
        self.env.draw_route_overlays()
    
        # 自动定位并缩放到当前场景
        view_info = self.env.frame_gui_to_scenario()
    
        # 给 QtGL 一点时间处理 PAN、ZOOM 和路线图形
        self.speed_applied = False
        self.speed_apply_at_s = (
            simt + self.cfg.runtime.gui_settle_time_s
        )
    
        self.phase = EpisodePhase.RUNNING
        self.phase_started_s = simt
    
        self._event(
            "episode_started",
            scenario=self.env.current_scenario_name,
            aircraft=len(bs.traf.id),
            gui_view=view_info,
        )
    
    def update(self) -> None:
        if self.finished:
            bs.sim.quit()
            return
        simt = float(bs.sim.simt)
        if self.phase == EpisodePhase.WAITING_CLEAR:
            if len(bs.traf.id) > 0:
                self.env.clear_traffic()
            if len(bs.traf.id) == 0:
                self._begin_episode(simt)
            return
        if self.phase == EpisodePhase.SPAWN_QUEUED:
            queued = self.env.spawn_due_aircraft(simt)
            if queued:
                self._event("aircraft_spawn_queued", aircraft=queued)
            if self.env.spawn_complete():
                self._activate_episode(simt)
            elif simt - self.phase_started_s > self.cfg.runtime.spawn_timeout_s + float(np.min(self.env.entry_times_s)):
                self._event("spawn_timeout", observed=list(bs.traf.id), expected=sorted(self.env.expected_aircraft_ids))
                self._finish_episode("spawn_timeout", terminated=False, truncated=True)
            return
        if self.phase == EpisodePhase.RUNNING:
            self._apply_runtime_speed_if_ready(simt)
            self._update_running(simt)
    
    def _apply_runtime_speed_if_ready(self, simt: float) -> None:
        if self.speed_applied:
            return

        if self.speed_apply_at_s is None:
            return

        if simt < self.speed_apply_at_s:
            return

        speed = self.cfg.runtime.speed_multiplier

        configure_simulation(speed)

        self.speed_applied = True

        self._event(
            "simulation_accelerated",
            speed_multiplier=speed,
        )
        
    def _update_running(self, simt: float) -> None:
        queued = self.env.spawn_due_aircraft(simt)
        if queued:
            self._event("aircraft_spawn_queued", aircraft=queued)
        removed = self.env.sync_runtime()
        if removed and self.cfg.training:
            zero_global, zero_presence = self._zero_global_state()
            for acid, runtime in removed.items():
                if acid in self.pending:
                    self._close_transition(acid, zero_global, zero_presence, 0, True, False, runtime.pending_reward)
        if len(bs.traf.id) == 0:
            if not self.env.all_expected_aircraft_spawned():
                elapsed = simt - self.env.episode_start_time_s
                if elapsed >= self.cfg.runtime.max_episode_time_s:
                    self._finish_episode("timeout", terminated=False, truncated=True)
                return
            self._finish_episode("traffic_removed", terminated=False, truncated=True)
            return

        conflict_states = self.env.conflict_manager.detect(bs.traf, self.cfg.network.max_aircraft)
        self._update_separation_adjustments(conflict_states, simt)
        rewards = self.env.compute_rewards(conflict_states)
        for acid, reward in rewards.items():
            runtime = self.env.runtime.get(acid)
            if runtime is not None:
                runtime.pending_reward += reward
                self.env.reward_stats["episode_return"] += reward
                if acid in self.pending:
                    self.pending[acid].reward_accumulator = runtime.pending_reward

        active_conflict = any(state.active for state in conflict_states.values())
        if active_conflict:
            self.had_conflict = True
            self.last_conflict_time_s = simt
            self.clear_since_s = None
        elif self.had_conflict and self.clear_since_s is None:
            self.clear_since_s = simt

        for acid in self.env.auto_resume_clear_commands(conflict_states, simt):
            self._event("automatic_resume", acid=acid)

        elapsed = simt - self.env.episode_start_time_s
        all_arrived = bool(self.env.expected_aircraft_ids) and self.env.expected_aircraft_ids.issubset(
            self.env.arrived_aircraft
        )
        any_collision = any(state.collision for state in conflict_states.values())
        if any_collision:
            self._finish_episode("collision", terminated=True, truncated=False)
            return
        if all_arrived and elapsed >= self.cfg.runtime.min_episode_time_s:
            self._finish_episode("all_arrived", terminated=True, truncated=False)
            return

        arrived_now = self.env.remove_arrived_aircraft()
        if arrived_now:
            self._event("aircraft_arrived", aircraft=arrived_now)
            self.env.last_state_update_time_s = simt
            self.env.episode_step += 1
            self.total_environment_steps += 1
            self.agent.environment_step = self.total_environment_steps
            return
        if self.env.episode_step >= self.cfg.runtime.max_episode_steps or elapsed >= self.cfg.runtime.max_episode_time_s:
            timeout_penalty = self.cfg.reward.timeout * self.cfg.reward.reward_scale
            for runtime in self.env.runtime.values():
                runtime.pending_reward += timeout_penalty
                self.env.reward_stats["episode_return"] += timeout_penalty
            self._finish_episode("timeout", terminated=False, truncated=True)
            return

        if self.env.decision_due(simt):
            self._decision_step(simt, conflict_states)
        self.env.last_state_update_time_s = simt
        self.env.episode_step += 1
        self.total_environment_steps += 1
        self.agent.environment_step = self.total_environment_steps

    @staticmethod
    def _accept_llm_action(guided_risk, base_risk, clearance_margin: float) -> bool:
        if guided_risk.violation_score + 1e-8 < base_risk.violation_score:
            return True
        return bool(
            abs(guided_risk.violation_score - base_risk.violation_score) <= 1e-8
            and guided_risk.minimum_clearance >= base_risk.minimum_clearance + clearance_margin
        )

    def _decision_step(self, simt: float, conflict_states: dict) -> None:
        # This measures actual local response cost: state construction,
        # frozen-candidate lookup, policy inference and BlueSky command call.
        # It deliberately excludes simulated-time scheduling delay and any
        # offline LLM generation that happened before this run.
        response_started = time.perf_counter()
        global_state, presence, id_to_slot = self.env.build_global_state(conflict_states)
        target_ids = self.env.get_target_aircraft(conflict_states)
        if self.group_planner is not None:
            for plan in self.group_planner.evaluate(conflict_states):
                self._event("group_plan_shadow", **plan)

        if self.cfg.training:
            for acid in list(self.pending.keys()):
                pending = self.pending[acid]
                runtime = self.env.runtime.get(acid)
                reward = runtime.pending_reward if runtime is not None else pending.reward_accumulator
                if runtime is not None:
                    runtime.pending_reward = 0.0
                continues = acid in target_ids
                self._close_transition(
                    acid,
                    global_state,
                    presence,
                    id_to_slot.get(acid, pending.target_index),
                    terminated=False,
                    truncated=not continues,
                    reward=reward,
                )
                if not continues:
                    self.trajectory_generation[acid] = self.trajectory_generation.get(acid, 0) + 1

        observations: list[np.ndarray] = []
        payload: list[tuple[str, int, np.ndarray, np.ndarray]] = []
        parameter_masks_by_acid: dict[str, np.ndarray] = {}
        for acid in target_ids:
            if acid not in self.env.runtime:
                continue
            local_obs = self.env.build_local_observation(acid, conflict_states)
            action_mask = self.env.get_action_mask(acid, conflict_states)
            self.available_action_counts += (action_mask > 0.0).astype(np.int64)
            parameter_masks_by_acid[acid] = self.env.get_parameter_masks(acid)
            observations.append(local_obs)
            payload.append((acid, id_to_slot[acid], local_obs, action_mask))

        guidance_by_acid = {}
        llm_scene = None
        if payload and self.scene_builder is not None:
            try:
                scene = self.scene_builder.build_scene(
                    bs.traf,
                    self.env.runtime,
                    conflict_states,
                    [item[0] for item in payload],
                    simt,
                    self.env.current_scenario_name,
                )
                if self.llm_guidance is not None:
                    guidance_by_acid = self.llm_guidance.guidance(
                        scene,
                        {acid: action_mask for acid, _, _, action_mask in payload},
                    )
                llm_scene = scene
                if self.llm_guidance is not None:
                    self._event(
                        "llm_guidance",
                        scene_id=scene.scene_id,
                        provider=self.llm_guidance.provider.name,
                        aircraft={
                            acid: {
                                "candidate_ids": guidance.candidate_ids,
                                "action_prior": guidance.action_prior.tolist(),
                                "grounding_mask": guidance.grounding_mask.tolist(),
                                "cache_hit": guidance.cache_hit,
                                "pending": guidance.pending,
                                "fingerprint": guidance.fingerprint,
                                "source": guidance.source,
                            }
                            for acid, guidance in guidance_by_acid.items()
                        },
                        usage=self.llm_guidance.consume_usage(),
                        store=self.llm_guidance.store.summary(),
                    )
            except Exception as exc:
                guidance_by_acid = {}
                self._event("llm_guidance_failed", error=repr(exc), fallback="hppo")

        guided_payload = []
        masks: list[np.ndarray] = []
        priors: list[np.ndarray] = []
        for acid, target_index, local_obs, action_mask in payload:
            guidance = guidance_by_acid.get(acid)
            if guidance is None:
                prior = np.full(5, 0.2, dtype=np.float32)
                final_mask = action_mask
                prior_weight = 0.0
                metadata = ("none", "none", [])
            else:
                prior = guidance.action_prior
                final_mask = action_mask * guidance.grounding_mask
                prior_weight = float(self.cfg.llm.prior_weight)
                metadata = (guidance.provider, guidance.prompt_version, guidance.candidate_ids)
            masks.append(final_mask)
            priors.append(prior)
            guided_payload.append((acid, target_index, local_obs, final_mask, prior, prior_weight, metadata))

        actions = []
        base_actions: dict[int, dict] = {}
        if observations:
            actions = self.agent.act(
                np.asarray(observations, dtype=np.float32),
                np.asarray(masks, dtype=np.float32),
                deterministic=not self.cfg.training,
                action_prior=np.asarray(priors, dtype=np.float32),
                prior_weight=float(self.cfg.llm.prior_weight) if self.llm_guidance is not None else 0.0,
                parameter_masks=np.asarray(
                    [parameter_masks_by_acid[acid] for acid, _, _, _ in payload], dtype=np.float32
                ),
            )
            self.actor_calls += len(actions)

            if self.candidate_dataset is not None and llm_scene is not None:
                for index, (acid, _, local_obs, base_mask) in enumerate(payload):
                    lookup = self.candidate_dataset.lookup(llm_scene, acid, base_mask)
                    if lookup is None or int(lookup.pool.candidate_mask.sum()) < self.cfg.llm.candidate_min_safe_count:
                        continue
                    candidate_parameter_masks = np.zeros(
                        (self.cfg.llm.candidate_pool_size, 3, self.agent.cfg.parameter_max_dim), dtype=np.float32
                    )
                    for candidate_idx, candidate in enumerate(lookup.pool.candidates):
                        if candidate is not None:
                            candidate_parameter_masks[candidate_idx] = self.env.get_candidate_parameter_masks(acid, candidate)
                    candidate_action = self.agent.act_candidates(
                        local_obs,
                        lookup.pool.features,
                        lookup.pool.candidate_mask,
                        lookup.macro_actions,
                        deterministic=not self.cfg.training,
                        prior_weight=self.cfg.llm.candidate_prior_weight,
                        candidate_parameter_masks=candidate_parameter_masks,
                    )[0]
                    selected_candidate = lookup.pool.candidates[candidate_action["candidate_index"]]
                    if selected_candidate is None:
                        continue
                    if self.cfg.network.parameter_policy_type == "continuous":
                        candidate_action["squashed_params"] = map_candidate_parameters(
                            selected_candidate, candidate_action["squashed_params"]
                        )
                    candidate_action["candidate_features"] = lookup.pool.features
                    candidate_action["candidate_mask"] = lookup.pool.candidate_mask
                    candidate_action["candidate_macro_actions"] = lookup.macro_actions
                    candidate_action["candidate_source"] = lookup.source
                    candidate_action["candidate_id"] = selected_candidate.candidate_id
                    actions[index] = candidate_action
                    self._event(
                        "candidate_selected",
                        acid=acid,
                        scene_id=llm_scene.scene_id,
                        candidate_id=selected_candidate.candidate_id,
                        candidate_index=int(candidate_action["candidate_index"]),
                        candidate_source=lookup.source,
                        macro_action=int(candidate_action["macro_action"]),
                    )

            if (
                self.llm_guidance is not None
                and self.cfg.llm.safety_gate_enabled
                and not self.cfg.training
                and llm_scene is not None
                and self.cfg.network.parameter_policy_type == "continuous"
            ):
                candidate_indices = [
                    index for index, item in enumerate(guided_payload) if item[6][2]
                ]
                if candidate_indices:
                    candidate_base_actions = self.agent.act(
                        np.asarray([guided_payload[index][2] for index in candidate_indices], dtype=np.float32),
                        np.asarray([payload[index][3] for index in candidate_indices], dtype=np.float32),
                        deterministic=True,
                        prior_weight=0.0,
                    )
                    base_actions = dict(zip(candidate_indices, candidate_base_actions))

        for action_index, ((acid, target_index, local_obs, action_mask, action_prior, prior_weight, llm_metadata), action) in enumerate(zip(guided_payload, actions)):
            if action_index in base_actions and int(action["macro_action"]) != int(base_actions[action_index]["macro_action"]):
                guided_action = action
                base_action = base_actions[action_index]
                try:
                    guided_risk = self.llm_guidance.assess_actor_action(
                        llm_scene,
                        acid,
                        int(guided_action["macro_action"]),
                        guided_action["squashed_params"],
                    )
                    base_risk = self.llm_guidance.assess_actor_action(
                        llm_scene,
                        acid,
                        int(base_action["macro_action"]),
                        base_action["squashed_params"],
                    )
                    accepted = self._accept_llm_action(
                        guided_risk,
                        base_risk,
                        self.cfg.llm.safety_gate_clearance_margin,
                    )
                    if not accepted:
                        action = base_action
                    self._event(
                        "llm_safety_gate",
                        acid=acid,
                        accepted=accepted,
                        guided_macro=int(guided_action["macro_action"]),
                        base_macro=int(base_action["macro_action"]),
                        executed_macro=int(action["macro_action"]),
                        guided_violation=float(guided_risk.violation_score),
                        base_violation=float(base_risk.violation_score),
                        guided_clearance=float(guided_risk.minimum_clearance),
                        base_clearance=float(base_risk.minimum_clearance),
                        candidate_ids=llm_metadata[2],
                    )
                except Exception as exc:
                    action = base_action
                    self._event(
                        "llm_safety_gate",
                        acid=acid,
                        accepted=False,
                        guided_macro=int(guided_action["macro_action"]),
                        base_macro=int(base_action["macro_action"]),
                        executed_macro=int(base_action["macro_action"]),
                        candidate_ids=llm_metadata[2],
                        rejection_reason=f"assessment_failed: {exc!r}",
                    )
            macro_action = int(action["macro_action"])
            numeric_fields = (
                action["raw_params"],
                action["squashed_params"],
                action["continuous_mean"],
                action["continuous_std"],
            )
            if not all(np.isfinite(np.asarray(field)).all() for field in numeric_fields):
                raise RuntimeError(f"Non-finite policy output for {acid}")
            self.action_counts[macro_action] += 1
            self.parameter_mean_sum += np.asarray(action["continuous_mean"], dtype=np.float64)
            self.parameter_std_sum += np.asarray(action["continuous_std"], dtype=np.float64)
            self.parameter_sample_count += 1
            phase_before_execution = self.env.runtime[acid].command_state.phase.value
            replan_reason_before_execution = self.env.runtime[acid].command_state.replan_reason
            if self.cfg.network.parameter_policy_type == "discrete":
                result = self.env.execute_discrete_macro(
                    acid,
                    macro_action,
                    int(action["parameter_action_index"]),
                    local_obs,
                    conflict_states.get(acid),
                )
            else:
                result = self.env.execute_macro(
                    acid,
                    macro_action,
                    action["squashed_params"],
                    local_obs,
                    conflict_states.get(acid),
                )
            if macro_action != 0 and result.success and result.applied:
                self.command_executions += 1
            if result.success and result.applied:
                self.applied_action_counts[macro_action] += 1
            if not result.success:
                self.env.record_command_failure(acid)
                self._event(
                    "command_failed",
                    acid=acid,
                    macro_action=macro_action,
                    message=result.message,
                    command=result.command_text,
                    command_applied=bool(result.applied),
                )
                if self.cfg.runtime.fail_on_command_error:
                    raise RuntimeError(f"BlueSky command failed for {acid}: {result.message}")
            else:
                self._event(
                    "action_executed",
                    acid=acid,
                    macro_action=macro_action,
                    command=result.command_text,
                    parameters=np.asarray(action["squashed_params"], dtype=float).tolist(),
                    parameter_branch=int(action["parameter_branch"]),
                    parameter_action_index=int(action["parameter_action_index"]),
                    parameter_mask=np.asarray(action["parameter_mask"], dtype=float).tolist(),
                    parameter_target=dict(self.env.runtime[acid].command_state.target_params),
                    command_applied=bool(result.applied),
                    command_phase=str(self.env.runtime[acid].command_state.phase.value),
                    phase_before_execution=phase_before_execution,
                    replan_triggered=phase_before_execution == "REPLAN_REQUIRED",
                    replan_reason_before_execution=replan_reason_before_execution,
                    target_progress=float(self.env._command_target_progress(acid, self.env.runtime[acid].command_state)),
                    stalled_evaluations=int(self.env.runtime[acid].command_state.stalled_evaluations),
                    replan_reason=self.env.runtime[acid].command_state.replan_reason,
                    branch_cooldowns_s={
                        str(macro): max(0.0, until - float(bs.sim.simt))
                        for macro, until in self.env.runtime[acid].command_state.cooldown_until_by_macro.items()
                    },
                    action_mask=np.asarray(action_mask, dtype=float).tolist(),
                    action_mask_reason=self.env.get_action_mask_reason(acid),
                    action_probabilities=np.asarray(action["discrete_probs"], dtype=float).tolist(),
                    parameter_branch_valid_counts=np.asarray(
                        [mask.sum() for mask in parameter_masks_by_acid[acid]], dtype=float
                    ).tolist(),
                    conflict_severity=float(conflict_states[acid].severity),
                    loss_of_separation=bool(conflict_states[acid].loss_of_separation),
                    llm_provider=llm_metadata[0],
                    llm_prompt_version=llm_metadata[1],
                    llm_candidate_ids=llm_metadata[2],
                    llm_action_prior=np.asarray(action_prior, dtype=float).tolist(),
                    candidate_mode=bool(action.get("candidate_mode", False)),
                    candidate_id=action.get("candidate_id", ""),
                    candidate_index=int(action.get("candidate_index", -1)),
                    candidate_source=action.get("candidate_source", ""),
                    candidate_macro_action=int(action.get("macro_action", -1)) if action.get("candidate_mode", False) else -1,
                    target_ids=[contact.callsign for contact in conflict_states[acid].targets],
                    min_tcpa_s=float(min((contact.tcpa_s for contact in conflict_states[acid].targets), default=float("inf"))),
                    min_horizontal_km=float(min((contact.horiz_km for contact in conflict_states[acid].targets), default=float("inf"))),
                    min_vertical_ft=float(min((contact.vert_ft for contact in conflict_states[acid].targets), default=float("inf"))),
                    min_time_gap_s=float(conflict_states[acid].min_time_gap_s),
                    risk_severity_at_issue=float(self.env.runtime[acid].command_state.risk_severity_at_issue),
                    risk_tcpa_s_at_issue=float(self.env.runtime[acid].command_state.risk_tcpa_s_at_issue),
                )

            if self.cfg.training:
                generation = self.trajectory_generation.get(acid, 0)
                self.pending[acid] = PendingDecision(
                    episode_id=self.env.episode_id,
                    step_index=self.decision_count,
                    trajectory_id=f"{acid}:{generation}",
                    acid=acid,
                    global_state=global_state.copy(),
                    presence_mask=presence.copy(),
                    target_index=target_index,
                    local_obs=local_obs.copy(),
                    action_mask=action_mask.copy(),
                    action_prior=action_prior.copy(),
                    prior_weight=prior_weight,
                    macro_action=macro_action,
                    raw_params=action["raw_params"].copy(),
                    squashed_params=action["squashed_params"].copy(),
                    selected_param_index=int(action["selected_param_index"]),
                    parameter_branch=int(action["parameter_branch"]),
                    parameter_action_index=int(action["parameter_action_index"]),
                    parameter_mask=np.asarray(action["parameter_mask"], dtype=np.float32).copy(),
                    parameter_probs=np.asarray(action["parameter_probs"], dtype=np.float32).copy(),
                    parameter_logp=float(action["parameter_logp"]),
                    parameter_entropy=float(action["parameter_entropy"]),
                    discrete_logp=float(action["discrete_logp"]),
                    discrete_entropy=float(action["discrete_entropy"]),
                    discrete_probs=action["discrete_probs"].copy(),
                    continuous_logp=float(action["continuous_logp"]),
                    continuous_entropy=float(action["continuous_entropy"]),
                    continuous_mean=action["continuous_mean"].copy(),
                    continuous_std=action["continuous_std"].copy(),
                    candidate_mode=bool(action.get("candidate_mode", False)),
                    candidate_features=np.asarray(
                        action.get("candidate_features", np.zeros((self.cfg.llm.candidate_pool_size, 18), dtype=np.float32)),
                        dtype=np.float32,
                    ).copy(),
                    candidate_mask=np.asarray(
                        action.get("candidate_mask", np.zeros(self.cfg.llm.candidate_pool_size, dtype=np.float32)),
                        dtype=np.float32,
                    ).copy(),
                    candidate_macro_actions=np.asarray(
                        action.get("candidate_macro_actions", np.zeros(self.cfg.llm.candidate_pool_size, dtype=np.int64)),
                        dtype=np.int64,
                    ).copy(),
                    candidate_index=int(action.get("candidate_index", -1)),
                    candidate_logp=float(action.get("candidate_logp", 0.0)),
                    candidate_entropy=float(action.get("candidate_entropy", 0.0)),
                    value=self.agent.predict_value(global_state, presence, target_index),
                    policy_version=self.agent.policy_version,
                )

        self.decision_count += 1
        self.env.last_decision_time_s = simt
        if payload:
            response_ms = (time.perf_counter() - response_started) * 1000.0
            self.decision_response_samples += 1
            self.decision_response_total_ms += response_ms
            self.decision_response_max_ms = max(self.decision_response_max_ms, response_ms)

    @staticmethod
    def _empty_update_metrics() -> dict[str, float]:
        return {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "approx_kl_discrete": 0.0,
            "approx_kl_continuous": 0.0,
            "clip_fraction": 0.0,
            "clip_fraction_discrete": 0.0,
            "clip_fraction_continuous": 0.0,
            "explained_variance": 0.0,
            "discrete_entropy": 0.0,
            "continuous_entropy": 0.0,
            "discrete_prob_mean": 0.0,
            "continuous_logp_mean": 0.0,
            "nan_detected": 0.0,
        }

    def _finish_episode(self, reason: str, terminated: bool, truncated: bool) -> None:
        if self.phase != EpisodePhase.RUNNING and reason != "spawn_timeout":
            return
        self._finalize_separation_adjustments(float(getattr(bs.sim, "simt", 0.0)), reason)
        if self.cfg.training:
            self._flush_pending(terminated=terminated, truncated=truncated)
        rollout_size = len(self.agent.buffer)
        metrics = self._empty_update_metrics()
        update_performed = 0
        if self.cfg.training and rollout_size > 0:
            metrics.update(self.agent.update())
            update_performed = 1
        elif not self.cfg.training and len(self.agent.buffer) != 0:
            raise RuntimeError("Evaluation mode unexpectedly collected rollout data")

        self.completed_episodes += 1
        self.run_episodes += 1
        self._log_metrics(metrics, self.completed_episodes)
        self._save_stats(reason, terminated, truncated, rollout_size, update_performed, metrics)
        self._event(
            "episode_finished",
            scenario=self.env.current_scenario_name,
            reason=reason,
            rollout_size=rollout_size,
            update_performed=update_performed,
            episode_return=self.env.reward_stats.get("episode_return", 0.0),
        )
        self.env.episode_id += 1

        if self.cfg.training and (
            self.completed_episodes % self.cfg.logging.save_every_episodes == 0
            or self.run_episodes >= self.cfg.runtime.max_episodes
        ):
            self.save_checkpoint()

        self.env.clear_traffic()
        self.pending.clear()
        if self.run_episodes >= self.cfg.runtime.max_episodes:
            self.phase = EpisodePhase.FINISHED
            self.finished = True
            self.writer.flush()
            self.writer.close()
            self._event(
                "run_finished",
                completed_episodes=self.completed_episodes,
                run_episodes=self.run_episodes,
            )
            self._write_completion_marker()
            self._request_application_shutdown()
            return
        self.phase = EpisodePhase.WAITING_CLEAR
        self.phase_started_s = float(bs.sim.simt)

    def _request_application_shutdown(self) -> None:
        completion = {
            "completed_episodes": self.completed_episodes,
            "run_episodes": self.run_episodes,
            "output_dir": str(self.cfg.logging.output_dir),
        }
        # Legacy detached simulations expose events through sim.send_event;
        # they do not provide the newer bs.net.send / stack.forward contract.
        if self.cfg.runtime.gui_enabled:
            publish_event(b"HPPO_RUN_FINISHED", completion)
        bs.sim.quit()

    def _write_completion_marker(self) -> None:
        marker = self.cfg.logging.output_dir / "run_complete.json"
        marker.write_text(
            json.dumps(
                {
                    "run_id": self.cfg.runtime.run_id,
                    "mode": self.mode,
                    "completed_episodes": self.completed_episodes,
                    "run_episodes": self.run_episodes,
                },
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )

    def _log_metrics(self, info: dict[str, float], step: int) -> None:
        for key, value in info.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"ppo/{key}", value, step)
        for key, value in self.env.reward_stats.items():
            self.writer.add_scalar(f"episode/{key}", value, step)

    def _save_stats(
        self,
        reason: str,
        terminated: bool,
        truncated: bool,
        rollout_size: int,
        update_performed: int,
        metrics: dict[str, float],
    ) -> None:
        diagnostics_row: dict[str, Any] = {
            "episode": self.completed_episodes,
            "run_episode": self.run_episodes,
            "mode": self.mode,
            "scenario": self.env.current_scenario_name,
            "reason": reason,
            "terminated": int(terminated),
            "truncated": int(truncated),
            "duration_s": float(bs.sim.simt - self.env.episode_start_time_s),
            **self.env.reward_stats,
            "had_conflict": int(self.had_conflict),
            "decision_count": self.decision_count,
            "actor_calls": self.actor_calls,
            "command_executions": self.command_executions,
            "decision_response_samples": self.decision_response_samples,
            "mean_decision_response_ms": self.decision_response_total_ms / max(self.decision_response_samples, 1),
            "max_decision_response_ms": self.decision_response_max_ms,
            "rollout_size": rollout_size,
            "transitions": self.transition_count,
            "update_performed": update_performed,
            "policy_version": self.agent.policy_version,
            "separation_horizontal_km": self.cfg.separation.horizontal_km,
            "separation_vertical_ft": self.cfg.separation.vertical_ft,
            "separation_time_s": self.cfg.separation.temporal_s,
            "separation_mode": self.cfg.separation.mode,
            "separation_release_factor": self.cfg.separation.release_factor,
            **{f"action_{index}_count": int(count) for index, count in enumerate(self.action_counts)},
            **{f"applied_action_{index}_count": int(count) for index, count in enumerate(self.applied_action_counts)},
            **{f"available_action_{index}_count": int(count) for index, count in enumerate(self.available_action_counts)},
            "parameter_heading_mean": float(self.parameter_mean_sum[0] / max(self.parameter_sample_count, 1)),
            "parameter_altitude_mean": float(self.parameter_mean_sum[1] / max(self.parameter_sample_count, 1)),
            "parameter_speed_mean": float(self.parameter_mean_sum[2] / max(self.parameter_sample_count, 1)),
            "parameter_heading_std": float(self.parameter_std_sum[0] / max(self.parameter_sample_count, 1)),
            "parameter_altitude_std": float(self.parameter_std_sum[1] / max(self.parameter_sample_count, 1)),
            "parameter_speed_std": float(self.parameter_std_sum[2] / max(self.parameter_sample_count, 1)),
            "entropy_collapse_alert": int(
                bool(update_performed)
                and metrics.get("discrete_entropy", 0.0) < self.cfg.ppo.entropy_collapse_threshold
            ),
            "heading_starvation_alert": int(
                self.available_action_counts[2] >= self.cfg.ppo.action_starvation_min_available
                and self.action_counts[2] == 0
            ),
            "value_loss_alert": int(
                bool(update_performed)
                and metrics.get("critic_loss", 0.0) > self.cfg.ppo.value_loss_alert_threshold
            ),
            **metrics,
        }
        # Arrival alone is not a safety result: an aircraft can reach its goal
        # after an unresolved loss of separation. Keep this explicit in both
        # compact statistics and detailed diagnostics.
        diagnostics_row["safe_success"] = int(
            reason == "all_arrived"
            and float(diagnostics_row["collision_events"]) <= 0.0
            and float(diagnostics_row["safety_violations"]) <= 0.0
            and (
                not bool(diagnostics_row["had_conflict"])
                or float(diagnostics_row["conflict_resolved"])
                >= float(diagnostics_row["conflict_detected"])
            )
        )
        summary_keys = (
            "episode",
            "scenario",
            "reason",
            "duration_s",
            "episode_return",
            "conflict_detected",
            "conflict_resolved",
            "safety_violations",
            "collision_events",
            "arrival",
            "safe_success",
            "command_count",
            "min_contact_horizontal_km",
            "min_predicted_cpa_km",
            "first_intervention_s",
            "rollout_size",
            "actor_loss",
            "critic_loss",
            "entropy",
        )
        summary_row = {key: diagnostics_row[key] for key in summary_keys}
        self._append_csv(self.stats_file, summary_row)
        self._append_csv(self.diagnostics_file, diagnostics_row)

    @staticmethod
    def _append_csv(path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(row.keys())
        reset_file = False
        if path.exists() and path.stat().st_size > 0:
            with path.open("r", newline="", encoding="utf-8") as stream:
                existing_header = next(csv.reader(stream), [])
            if existing_header != fieldnames:
                index = 1
                while True:
                    backup = path.with_name(f"{path.stem}_legacy_{index}{path.suffix}")
                    if not backup.exists():
                        shutil.copyfile(path, backup)
                        reset_file = True
                        break
                    index += 1
        header = reset_file or not path.exists() or path.stat().st_size == 0
        mode = "w" if reset_file else "a"
        with path.open(mode, newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            if header:
                writer.writeheader()
            writer.writerow(row)

    def save_checkpoint(self) -> None:
        if not self.cfg.training:
            raise RuntimeError("Evaluation mode must not save a training checkpoint")
        checkpoint_extra = {
            "training_state": {
                "total_environment_steps": self.total_environment_steps,
                "completed_episodes": self.completed_episodes,
            },
            "normalization_state": {"type": "fixed_feature_scaling"},
        }
        save_checkpoint_file(
            self.training_ckpt_file,
            self.agent,
            episode=self.completed_episodes,
            config=self.cfg,
            extra=checkpoint_extra,
        )
        self._event("checkpoint_saved", path=str(self.training_ckpt_file), checkpoint_episode=self.completed_episodes)
        archive = self.training_ckpt_file.with_name(
            f"{self.training_ckpt_file.stem}_ep{self.completed_episodes:04d}{self.training_ckpt_file.suffix}"
        )
        save_checkpoint_file(
            archive,
            self.agent,
            episode=self.completed_episodes,
            config=self.cfg,
            extra=checkpoint_extra,
        )
        self._event("checkpoint_archived", path=str(archive), checkpoint_episode=self.completed_episodes)
