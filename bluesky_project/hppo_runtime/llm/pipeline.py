from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor
from math import cos, radians, sin
from typing import Any

import numpy as np

from .models import (
    ACTION_TO_MACRO,
    AircraftSnapshot,
    CandidateAction,
    CandidateProposal,
    CoarseCandidateAction,
    CoarseCandidateProposal,
    ConflictEdge,
    ConflictScene,
    parse_coarse_candidate_proposal,
)
from .providers import CandidateProvider, DeepSeekCandidateProvider, MockCandidateProvider
from .experience_store import ExperienceStore, scene_fingerprint
from .candidate_pool import CandidatePool, build_candidate_pool, coarse_from_legacy, to_legacy_candidate


@dataclass(slots=True)
class LLMGuidance:
    action_prior: np.ndarray
    grounding_mask: np.ndarray
    candidate_ids: list[str]
    provider: str
    prompt_version: str
    cache_hit: bool
    pending: bool
    fingerprint: str
    source: str


@dataclass(slots=True)
class ActionRiskAssessment:
    violation_score: float
    minimum_clearance: float
    predicted_min_horizontal_km: float | None
    predicted_min_vertical_ft: float | None


class LLMGuidanceManager:
    def __init__(self, config, provider: CandidateProvider | None = None):
        self.cfg = config
        if provider is not None:
            self.provider = provider
        elif config.llm.mode == "online-once" and config.llm.provider == "deepseek":
            self.provider = DeepSeekCandidateProvider(config)
        else:
            self.provider = MockCandidateProvider()
        self.store = ExperienceStore(config.llm.experience_db)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hppo-llm") if config.llm.mode == "online-once" and config.llm.provider == "deepseek" else None
        self._pending_future: Future | None = None
        self._pending_signature: str | None = None
        self._pending_descriptor = ""
        self._pending_scene: ConflictScene | None = None
        self._requested_fingerprints: set[str] = set()
        self._episode_calls = 0
        self._run_calls = 0
        self._event_usage: dict[str, Any] = {}
        self._last_source = "none"

    def begin_episode(self) -> None:
        self._episode_calls = 0

    def consume_usage(self) -> dict[str, Any]:
        usage = self._event_usage
        self._event_usage = {}
        return usage

    def build_scene(self, traf, runtime: dict, conflict_states: dict, target_ids: list[str], simt: float, scene_name: str) -> ConflictScene:
        aircraft: list[AircraftSnapshot] = []
        for index, acid in enumerate(traf.id):
            command = runtime.get(acid).command_state if acid in runtime else None
            aircraft.append(
                AircraftSnapshot(
                    aircraft_id=acid,
                    latitude_deg=float(traf.lat[index]),
                    longitude_deg=float(traf.lon[index]),
                    altitude_ft=float(traf.alt[index] * 3.28084),
                    speed_kt=float(traf.cas[index] * 1.94384),
                    heading_deg=float(traf.trk[index]) % 360.0,
                    vertical_speed_fpm=float(traf.vs[index] * 196.8504),
                    current_macro=int(command.current_macro) if command is not None else 0,
                    hold_remaining_s=max(0.0, float(command.hold_until_s - simt)) if command is not None else 0.0,
                )
            )
        edges: list[ConflictEdge] = []
        for acid in target_ids:
            state = conflict_states.get(acid)
            if state is None:
                continue
            for contact in state.targets:
                edges.append(
                    ConflictEdge(
                        ownship_id=acid,
                        intruder_id=contact.callsign,
                        horizontal_km=float(contact.horiz_km),
                        vertical_ft=float(contact.vert_ft),
                        tcpa_s=float(contact.tcpa_s),
                        time_gap_s=float(contact.time_gap_s) if np.isfinite(contact.time_gap_s) else None,
                        relative_bearing_deg=float(contact.rel_bearing_deg),
                        severity=float(contact.severity),
                        loss_of_separation=bool(state.loss_of_separation),
                    )
                )
        return ConflictScene(
            schema_version="1.0",
            scene_id=f"{scene_name}:t{simt:.1f}",
            simulation_time_s=float(simt),
            separation_horizontal_km=float(self.cfg.separation.horizontal_km),
            separation_vertical_ft=float(self.cfg.separation.vertical_ft),
            separation_time_s=float(self.cfg.separation.temporal_s),
            aircraft=aircraft,
            conflicts=edges,
            target_ids=list(target_ids),
        )

    def _validate(self, candidate: CandidateAction, scene: ConflictScene) -> None:
        ids = {item.aircraft_id for item in scene.aircraft}
        if candidate.aircraft_id not in ids:
            raise ValueError("unknown aircraft")
        if candidate.action_type not in ACTION_TO_MACRO:
            raise ValueError("unknown action type")
        if not np.isfinite(candidate.prior_score) or candidate.prior_score < 0.0:
            raise ValueError("invalid prior score")
        values = candidate.parameters
        if not all(np.isfinite(float(value)) for value in values.values()):
            raise ValueError("non-finite parameter")
        allowed_parameters = {
            "NO_NEW_COMMAND": set(),
            "RESUME_ROUTE": set(),
            "TURN_LEFT": {"heading_delta_deg", "duration_s"},
            "TURN_RIGHT": {"heading_delta_deg", "duration_s"},
            "CLIMB": {"altitude_delta_ft", "duration_s"},
            "DESCEND": {"altitude_delta_ft", "duration_s"},
            "ACCELERATE": {"speed_delta_kt", "duration_s"},
            "DECELERATE": {"speed_delta_kt", "duration_s"},
        }
        unknown = set(values) - allowed_parameters[candidate.action_type]
        if unknown:
            raise ValueError(f"unsupported parameters for {candidate.action_type}: {sorted(unknown)}")
        if "duration_s" in values and not 5.0 <= float(values["duration_s"]) <= 600.0:
            raise ValueError("duration_s outside supported command duration")
        ranges = self.cfg.command_ranges
        if candidate.action_type in {"TURN_LEFT", "TURN_RIGHT"}:
            delta = float(values.get("heading_delta_deg", 0.0))
            expected_sign = -1.0 if candidate.action_type == "TURN_LEFT" else 1.0
            if delta * expected_sign <= 0.0 or abs(delta) > ranges.heading_offset_deg:
                raise ValueError("heading parameter outside action semantics or range")
        elif candidate.action_type in {"CLIMB", "DESCEND"}:
            delta = float(values.get("altitude_delta_ft", 0.0))
            expected_sign = 1.0 if candidate.action_type == "CLIMB" else -1.0
            if delta * expected_sign <= 0.0 or abs(delta) > ranges.altitude_delta_ft:
                raise ValueError("altitude parameter outside action semantics or range")
        elif candidate.action_type in {"ACCELERATE", "DECELERATE"}:
            delta = float(values.get("speed_delta_kt", 0.0))
            expected_sign = 1.0 if candidate.action_type == "ACCELERATE" else -1.0
            if delta * expected_sign <= 0.0 or abs(delta) > ranges.speed_delta_kt:
                raise ValueError("speed parameter outside action semantics or range")

    @staticmethod
    def _xy_velocity(heading_deg: float, speed_kt: float) -> np.ndarray:
        speed_km_s = speed_kt * 1.852 / 3600.0
        angle = radians(heading_deg)
        return np.array([speed_km_s * sin(angle), speed_km_s * cos(angle)], dtype=np.float64)

    def _predict_action(self, candidate: CandidateAction, scene: ConflictScene) -> ActionRiskAssessment:
        nodes = {item.aircraft_id: item for item in scene.aircraft}
        if candidate.aircraft_id not in nodes:
            raise ValueError("aircraft_missing")
        reference = scene.aircraft[0]
        positions: dict[str, np.ndarray] = {}
        for item in scene.aircraft:
            positions[item.aircraft_id] = np.array(
                [
                    (item.longitude_deg - reference.longitude_deg) * 111.32 * cos(radians(reference.latitude_deg)),
                    (item.latitude_deg - reference.latitude_deg) * 110.57,
                ],
                dtype=np.float64,
            )
        minimum_h = float("inf")
        minimum_v = float("inf")
        unsafe = False
        worst_violation = 0.0
        minimum_clearance = float("inf")
        horizon = self.cfg.llm.prediction_horizon_s
        step = self.cfg.llm.prediction_step_s
        own = nodes[candidate.aircraft_id]
        for elapsed in np.arange(step, horizon + 1e-6, step):
            projected: dict[str, tuple[np.ndarray, float]] = {}
            for acid, item in nodes.items():
                heading = item.heading_deg
                speed = item.speed_kt
                altitude = item.altitude_ft + item.vertical_speed_fpm * elapsed / 60.0
                if acid == candidate.aircraft_id:
                    if candidate.action_type in {"TURN_LEFT", "TURN_RIGHT"}:
                        heading = (heading + candidate.parameters["heading_delta_deg"]) % 360.0
                    elif candidate.action_type in {"ACCELERATE", "DECELERATE"}:
                        speed = float(np.clip(speed + candidate.parameters["speed_delta_kt"], self.cfg.command_ranges.min_speed_kt, self.cfg.command_ranges.max_speed_kt))
                    elif candidate.action_type in {"CLIMB", "DESCEND"}:
                        target = float(np.clip(own.altitude_ft + candidate.parameters["altitude_delta_ft"], self.cfg.command_ranges.min_altitude_ft, self.cfg.command_ranges.max_altitude_ft))
                        max_change = 1500.0 * elapsed / 60.0
                        altitude = own.altitude_ft + float(np.clip(target - own.altitude_ft, -max_change, max_change))
                projected[acid] = (positions[acid] + self._xy_velocity(heading, speed) * elapsed, altitude)
            own_position, own_altitude = projected[candidate.aircraft_id]
            for other_id, (other_position, other_altitude) in projected.items():
                if other_id == candidate.aircraft_id:
                    continue
                horizontal = float(np.linalg.norm(other_position - own_position))
                vertical = abs(float(other_altitude - own_altitude))
                minimum_h = min(minimum_h, horizontal)
                minimum_v = min(minimum_v, vertical)
                horizontal_ratio = horizontal / max(scene.separation_horizontal_km, 1e-6)
                vertical_ratio = vertical / max(scene.separation_vertical_ft, 1e-6)
                minimum_clearance = min(minimum_clearance, max(horizontal_ratio, vertical_ratio))
                worst_violation = max(
                    worst_violation,
                    max(0.0, 1.0 - horizontal_ratio) * max(0.0, 1.0 - vertical_ratio),
                )
                if horizontal < scene.separation_horizontal_km and vertical < scene.separation_vertical_ft:
                    unsafe = True
        result = ActionRiskAssessment(
            violation_score=float(worst_violation),
            minimum_clearance=float(minimum_clearance) if np.isfinite(minimum_clearance) else float("inf"),
            predicted_min_horizontal_km=minimum_h if np.isfinite(minimum_h) else None,
            predicted_min_vertical_ft=minimum_v if np.isfinite(minimum_v) else None,
        )
        candidate.predicted_min_horizontal_km = result.predicted_min_horizontal_km
        candidate.predicted_min_vertical_ft = result.predicted_min_vertical_ft
        candidate.safe = not unsafe
        candidate.rejection_reason = "predicted_separation_loss" if unsafe else ""
        return result

    def _ground(self, candidate: CandidateAction, scene: ConflictScene) -> None:
        self._predict_action(candidate, scene)

    def _validate_coarse(self, candidate: CoarseCandidateAction, scene: ConflictScene) -> None:
        """Validate the v1.1 candidate language, then ground its representative action."""
        ids = {item.aircraft_id for item in scene.aircraft}
        if candidate.aircraft_id not in ids or candidate.aircraft_id not in set(scene.target_ids):
            raise ValueError("coarse_target_not_in_scene")
        if not np.isfinite(candidate.prior_score) or not 0.0 <= candidate.prior_score <= 1.0:
            raise ValueError("invalid_coarse_prior_score")
        if candidate.parameters:
            raise ValueError("coarse_parameters_must_be_empty")
        if candidate.coordination_intent == "PRIORITY_PASS":
            priority, yielding = candidate.priority_aircraft_id, candidate.yield_aircraft_id
            if not priority or not yielding or priority == yielding or priority not in ids or yielding not in ids:
                raise ValueError("invalid_priority_pair")
            if candidate.aircraft_id != yielding:
                raise ValueError("priority_candidate_must_control_yield_aircraft")
            has_edge = any(
                {edge.ownship_id, edge.intruder_id} == {priority, yielding} for edge in scene.conflicts
            )
            if not has_edge:
                raise ValueError("priority_pair_not_in_conflict_graph")
        elif candidate.priority_aircraft_id or candidate.yield_aircraft_id:
            raise ValueError("unexpected_priority_pair")
        snapshot = next(item for item in scene.aircraft if item.aircraft_id == candidate.aircraft_id)
        if (
            candidate.execution_timing == "IMMEDIATE"
            and snapshot.hold_remaining_s > 1e-6
            and candidate.macro_action != "NO_NEW_COMMAND"
        ):
            raise ValueError("hold_period_active")
        legacy = to_legacy_candidate(candidate, self.cfg)
        if legacy is None:
            raise ValueError("macro_not_supported_by_five_action_baseline")
        self._validate(legacy, scene)
        assessment = self._predict_action(legacy, scene)
        candidate.predicted_min_horizontal_km = assessment.predicted_min_horizontal_km
        candidate.predicted_min_vertical_ft = assessment.predicted_min_vertical_ft
        if legacy.safe:
            candidate.safe = True
            candidate.rejection_reason = ""
            return

        # A hard absolute gate makes every difficult, already-predicted conflict
        # yield an empty pool. For coarse candidates only, compare against the
        # no-new-command baseline and retain a non-catastrophic action that
        # measurably reduces predicted violation. Runtime execution still has
        # its normal command-level safety checks.
        baseline = CandidateAction(
            candidate_id="coarse-risk-baseline",
            aircraft_id=candidate.aircraft_id,
            action_type="NO_NEW_COMMAND",
            parameters={},
            prior_score=0.0,
        )
        baseline_assessment = self._predict_action(baseline, scene)
        if candidate.macro_action == "NO_NEW_COMMAND":
            candidate.safe = True
            candidate.rejection_reason = ""
            return
        catastrophic = assessment.minimum_clearance < self.cfg.llm.coarse_hard_clearance_ratio
        improvement = baseline_assessment.violation_score - assessment.violation_score
        if (
            self.cfg.llm.coarse_relative_risk_screen
            and not catastrophic
            and baseline_assessment.violation_score > 0.0
            and improvement >= self.cfg.llm.coarse_min_violation_improvement
        ):
            candidate.safe = True
            candidate.rejection_reason = ""
            return
        candidate.safe = False
        candidate.rejection_reason = (
            "predicted_catastrophic_loss" if catastrophic else "no_relative_risk_improvement"
        )

    def build_coarse_candidate_pool_from_payload(
        self, payload: str | dict[str, Any], scene: ConflictScene, slots: int | None = None
    ) -> tuple[CoarseCandidateProposal, CandidatePool]:
        """Parse a raw v1.1 LLM reply and return a fixed, safety-screened pool."""
        proposal = parse_coarse_candidate_proposal(payload, self.provider.name, "hppo-coarse-candidate-v1")
        if proposal.scene_id != scene.scene_id:
            raise ValueError("coarse_candidate_scene_id_mismatch")
        for candidate in proposal.candidates:
            try:
                self._validate_coarse(candidate, scene)
            except (KeyError, TypeError, ValueError) as exc:
                candidate.safe = False
                candidate.rejection_reason = str(exc)
        return proposal, build_candidate_pool(proposal.candidates, slots or self.cfg.llm.candidate_pool_size)

    def generate_coarse_candidate_pool(self, scene: ConflictScene, slots: int | None = None) -> tuple[CoarseCandidateProposal, CandidatePool]:
        """Generate coarse candidates through a provider and apply all local safety checks.

        This explicit method is for offline collection and controlled experiments;
        the default PPO loop remains cache-only and API-free.
        """
        generator = getattr(self.provider, "generate_coarse_candidates", None)
        if generator is None:
            legacy = self.provider.generate_candidates(scene)
            proposal = CoarseCandidateProposal(
                "1.1", scene.scene_id, legacy.provider, legacy.prompt_version,
                [coarse_from_legacy(item) for item in legacy.candidates],
            )
        else:
            proposal = generator(scene)
        if proposal.scene_id != scene.scene_id:
            raise ValueError("coarse_candidate_scene_id_mismatch")
        for candidate in proposal.candidates:
            try:
                self._validate_coarse(candidate, scene)
            except (KeyError, TypeError, ValueError) as exc:
                candidate.safe = False
                candidate.rejection_reason = str(exc)
        return proposal, build_candidate_pool(proposal.candidates, slots or self.cfg.llm.candidate_pool_size)

    def build_candidate_pool(self, scene: ConflictScene, slots: int | None = None) -> CandidatePool:
        """Expose existing v1.0 cached/provider candidates through the v1.1 pool contract."""
        proposal, _, _ = self._proposal(scene)
        return build_candidate_pool(
            (coarse_from_legacy(candidate) for candidate in proposal.candidates),
            slots or self.cfg.llm.candidate_pool_size,
        )

    def assess_actor_action(
        self,
        scene: ConflictScene,
        aircraft_id: str,
        macro_action: int,
        squashed_params: np.ndarray,
    ) -> ActionRiskAssessment:
        params = np.asarray(squashed_params, dtype=np.float64)
        if params.shape != (3,) or not np.isfinite(params).all():
            raise ValueError("actor parameters must be a finite three-vector")
        ranges = self.cfg.command_ranges
        if macro_action == 0:
            action_type, values = "NO_NEW_COMMAND", {}
        elif macro_action == 1:
            action_type, values = "RESUME_ROUTE", {}
        elif macro_action == 2:
            delta = float(params[0] * ranges.heading_offset_deg)
            action_type = "TURN_RIGHT" if delta >= 0.0 else "TURN_LEFT"
            values = {"heading_delta_deg": delta}
        elif macro_action == 3:
            delta = float(params[1] * ranges.altitude_delta_ft)
            action_type = "CLIMB" if delta >= 0.0 else "DESCEND"
            values = {"altitude_delta_ft": delta}
        elif macro_action == 4:
            delta = float(params[2] * ranges.speed_delta_kt)
            action_type = "ACCELERATE" if delta >= 0.0 else "DECELERATE"
            values = {"speed_delta_kt": delta}
        else:
            raise ValueError(f"unknown macro action {macro_action}")
        candidate = CandidateAction(
            candidate_id="actor-runtime",
            aircraft_id=aircraft_id,
            action_type=action_type,
            parameters=values,
            prior_score=1.0,
        )
        return self._predict_action(candidate, scene)

    def _validated_proposal(self, proposal: CandidateProposal, scene: ConflictScene) -> CandidateProposal:
        valid: list[CandidateAction] = []
        for candidate in proposal.candidates[: self.cfg.llm.max_candidates]:
            try:
                self._validate(candidate, scene)
                self._ground(candidate, scene)
                valid.append(candidate)
            except (KeyError, TypeError, ValueError) as exc:
                candidate.safe = False
                candidate.rejection_reason = str(exc)
        proposal.candidates = valid
        return proposal

    def process_scene_offline(self, scene: ConflictScene) -> tuple[str, CandidateProposal]:
        fingerprint, descriptor = scene_fingerprint(scene)
        proposal = self._validated_proposal(self.provider.generate_candidates(scene), scene)
        proposal.candidates = [candidate for candidate in proposal.candidates if candidate.safe]
        if not proposal.candidates:
            raise ValueError("provider produced no prediction-validated candidates")
        self.store.save_plan(fingerprint, descriptor, scene, proposal)
        usage = getattr(self.provider, "last_usage", {})
        if usage:
            self.store.record_usage(self.provider.name, usage)
        return fingerprint, proposal

    def _within_budget(self) -> bool:
        daily = self.store.daily_usage()
        return (
            self._episode_calls < self.cfg.llm.max_calls_per_episode
            and self._run_calls < self.cfg.llm.max_calls_per_run
            and daily["tokens"] < self.cfg.llm.max_daily_tokens
        )

    def _proposal(self, scene: ConflictScene) -> tuple[CandidateProposal, bool, bool]:
        fingerprint, descriptor = scene_fingerprint(scene)
        self._last_source = "miss"
        cached, match_type = self.store.get_plan_for_scene(scene)
        if cached is not None:
            cached = self._validated_proposal(cached, scene)
            cached.candidates = [candidate for candidate in cached.candidates if candidate.safe]
            if cached.candidates:
                self._last_source = "experience_store" if match_type == "exact" else "experience_store_canonical"
                return cached, True, False

        if self.cfg.llm.collect_cache_misses:
            self.store.queue_scene(fingerprint, descriptor, scene)

        if self.cfg.llm.mode == "cache-only":
            empty = CandidateProposal("1.0", scene.scene_id, "cache-only", "cache-miss", [])
            self._last_source = "queued"
            return empty, False, False

        if self._executor is not None:
            if self._pending_future is not None and self._pending_future.done():
                pending_future = self._pending_future
                pending_scene = self._pending_scene
                pending_signature = self._pending_signature
                pending_descriptor = self._pending_descriptor
                self._pending_future = None
                self._pending_scene = None
                self._pending_signature = None
                self._pending_descriptor = ""
                proposal = self._validated_proposal(pending_future.result(), pending_scene)
                proposal.candidates = [candidate for candidate in proposal.candidates if candidate.safe]
                if proposal.candidates:
                    self.store.save_plan(pending_signature, pending_descriptor, pending_scene, proposal)
                usage = getattr(self.provider, "last_usage", {})
                if usage:
                    self.store.record_usage(self.provider.name, usage)
                    self._event_usage = dict(usage)
                if pending_signature == fingerprint and proposal.candidates:
                    self._last_source = "online_response"
                    return proposal, False, False
            if self._pending_future is None and fingerprint not in self._requested_fingerprints and self._within_budget():
                self._pending_signature = fingerprint
                self._pending_descriptor = descriptor
                self._pending_scene = scene
                self._pending_future = self._executor.submit(self.provider.generate_candidates, scene)
                self._requested_fingerprints.add(fingerprint)
                self._episode_calls += 1
                self._run_calls += 1
            empty = CandidateProposal("1.0", scene.scene_id, self.provider.name, "pending", [])
            is_pending = self._pending_future is not None and self._pending_signature == fingerprint
            self._last_source = "online_pending" if is_pending else "queued"
            return empty, False, is_pending

        if fingerprint in self._requested_fingerprints or not self._within_budget():
            empty = CandidateProposal("1.0", scene.scene_id, self.provider.name, "budget-or-once-limit", [])
            self._last_source = "queued"
            return empty, False, False
        self._requested_fingerprints.add(fingerprint)
        self._episode_calls += 1
        self._run_calls += 1
        proposal = self._validated_proposal(self.provider.generate_candidates(scene), scene)
        proposal.candidates = [candidate for candidate in proposal.candidates if candidate.safe]
        if proposal.candidates:
            self.store.save_plan(fingerprint, descriptor, scene, proposal)
        usage = getattr(self.provider, "last_usage", {})
        if usage:
            self.store.record_usage(self.provider.name, usage)
            self._event_usage = dict(usage)
        self._last_source = "online_response"
        return proposal, False, False

    def guidance(self, scene: ConflictScene, base_masks: dict[str, np.ndarray]) -> dict[str, LLMGuidance]:
        proposal, cache_hit, pending = self._proposal(scene)
        fingerprint, _ = scene_fingerprint(scene)
        output: dict[str, LLMGuidance] = {}
        for acid in scene.target_ids:
            prior = np.full(5, self.cfg.llm.prior_floor, dtype=np.float32)
            mask = np.ones(5, dtype=np.float32)
            relevant = [candidate for candidate in proposal.candidates if candidate.aircraft_id == acid]
            safe_macros: set[int] = set()
            unsafe_macros: set[int] = set()
            for candidate in relevant:
                if candidate.safe:
                    prior[candidate.macro_action] += float(candidate.prior_score)
                    safe_macros.add(candidate.macro_action)
                else:
                    unsafe_macros.add(candidate.macro_action)
            if self.cfg.llm.enforce_grounding_mask:
                for macro in unsafe_macros - safe_macros:
                    mask[macro] = 0.0
            base = np.asarray(base_masks.get(acid, np.ones(5)), dtype=np.float32)
            mask *= (base > 0.0).astype(np.float32)
            if mask.sum() <= 0.0:
                mask = (base > 0.0).astype(np.float32)
            prior *= mask
            if prior.sum() <= 0.0:
                prior = np.where(mask > 0.0, 1.0, 0.0).astype(np.float32)
            prior /= max(float(prior.sum()), 1e-8)
            output[acid] = LLMGuidance(
                action_prior=prior,
                grounding_mask=mask,
                candidate_ids=[candidate.candidate_id for candidate in relevant if candidate.safe],
                provider=proposal.provider,
                prompt_version=proposal.prompt_version,
                cache_hit=cache_hit,
                pending=pending,
                fingerprint=fingerprint,
                source=self._last_source,
            )
        return output
