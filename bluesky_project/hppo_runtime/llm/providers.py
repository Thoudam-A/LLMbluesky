from __future__ import annotations

import os
import time
from typing import Protocol

from .models import CandidateAction, CandidateProposal, CoarseCandidateProposal, ConflictScene, parse_candidate_proposal, parse_coarse_candidate_proposal
from .prompt_builder import COARSE_PROMPT_VERSION, PROMPT_VERSION, build_candidate_prompt, build_coarse_candidate_prompt


class CandidateProvider(Protocol):
    name: str

    def generate_candidates(self, scene: ConflictScene) -> CandidateProposal:
        ...


class MockCandidateProvider:
    """Deterministic provider used to verify the complete integration without an API."""

    name = "mock-v1"

    def generate_candidates(self, scene: ConflictScene) -> CandidateProposal:
        aircraft = {item.aircraft_id: item for item in scene.aircraft}
        candidates: list[CandidateAction] = []
        sequence = 0
        for acid in scene.target_ids:
            own_edges = [edge for edge in scene.conflicts if edge.ownship_id == acid]
            if not own_edges or acid not in aircraft:
                continue
            edge = max(own_edges, key=lambda item: item.severity)
            ownship = aircraft[acid]
            intruder = aircraft.get(edge.intruder_id)
            turn_type = "TURN_LEFT" if edge.relative_bearing_deg >= 0.0 else "TURN_RIGHT"
            heading_delta = -20.0 if turn_type == "TURN_LEFT" else 20.0
            vertical_type = "CLIMB" if intruder is None or ownship.altitude_ft <= intruder.altitude_ft else "DESCEND"
            altitude_delta = 2000.0 if vertical_type == "CLIMB" else -2000.0
            options = (
                (turn_type, {"heading_delta_deg": heading_delta, "duration_s": 20.0}, 0.45, "increase horizontal separation"),
                (vertical_type, {"altitude_delta_ft": altitude_delta, "duration_s": 30.0}, 0.35, "increase vertical separation"),
                ("DECELERATE", {"speed_delta_kt": -20.0, "duration_s": 20.0}, 0.20, "change arrival order at conflict point"),
            )
            for action_type, parameters, score, effect in options:
                sequence += 1
                candidates.append(
                    CandidateAction(
                        candidate_id=f"M{sequence:03d}",
                        aircraft_id=acid,
                        action_type=action_type,
                        parameters=parameters,
                        prior_score=score,
                        expected_effect=effect,
                    )
                )
        return CandidateProposal(
            schema_version="1.0",
            scene_id=scene.scene_id,
            provider=self.name,
            prompt_version="mock-structured-v1",
            candidates=candidates,
        )

    def generate_coarse_candidates(self, scene: ConflictScene) -> CoarseCandidateProposal:
        from .candidate_pool import coarse_from_legacy

        return CoarseCandidateProposal(
            schema_version="1.1",
            scene_id=scene.scene_id,
            provider=self.name,
            prompt_version="mock-coarse-v1",
            candidates=[coarse_from_legacy(item) for item in self.generate_candidates(scene).candidates],
        )


class DeepSeekCandidateProvider:
    def __init__(self, config, client=None):
        self.cfg = config
        self.name = f"deepseek:{config.llm.model}"
        self.last_usage: dict[str, int | float | str] = {}
        if client is not None:
            self.client = client
            return
        api_key = os.getenv("HPPO_LLM_API_KEY", "").strip()
        if not api_key:
            raise ValueError("HPPO_LLM_API_KEY is required for the DeepSeek provider")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is required for the DeepSeek provider") from exc
        self.client = OpenAI(
            api_key=api_key,
            base_url=config.llm.base_url,
            timeout=config.llm.timeout_s,
            max_retries=0,
        )

    def generate_candidates(self, scene: ConflictScene) -> CandidateProposal:
        return self._request(scene, build_candidate_prompt(scene, self.cfg), parse_candidate_proposal, PROMPT_VERSION)

    def generate_coarse_candidates(self, scene: ConflictScene) -> CoarseCandidateProposal:
        return self._request(scene, build_coarse_candidate_prompt(scene, self.cfg), parse_coarse_candidate_proposal, COARSE_PROMPT_VERSION)

    def _request(self, scene, prompt, parser, prompt_version):
        last_error: Exception | None = None
        for attempt in range(self.cfg.llm.max_retries + 1):
            started = time.monotonic()
            try:
                response = self.client.chat.completions.create(
                    model=self.cfg.llm.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Generate air-traffic conflict-resolution candidates. Return json only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=self.cfg.llm.temperature,
                    max_tokens=self.cfg.llm.max_tokens,
                    extra_body={
                        "thinking": {"type": "enabled" if self.cfg.llm.thinking_enabled else "disabled"},
                        "reasoning_effort": self.cfg.llm.reasoning_effort,
                    },
                )
                choice = response.choices[0]
                content = choice.message.content
                if not content or not content.strip():
                    raise ValueError("DeepSeek returned empty JSON content")
                if getattr(choice, "finish_reason", None) == "length":
                    raise ValueError("DeepSeek JSON output was truncated")
                proposal = parser(content, self.name, prompt_version)
                if proposal.scene_id != scene.scene_id:
                    raise ValueError("DeepSeek response scene_id does not match the request")
                usage = getattr(response, "usage", None)
                self.last_usage = {
                    "latency_s": time.monotonic() - started,
                    "attempts": attempt + 1,
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                    "model": str(getattr(response, "model", self.cfg.llm.model)),
                }
                return proposal
            except Exception as exc:
                last_error = exc
                if attempt < self.cfg.llm.max_retries:
                    time.sleep(self.cfg.llm.retry_backoff_s * (attempt + 1))
        raise RuntimeError(f"DeepSeek candidate generation failed after {self.cfg.llm.max_retries + 1} attempts: {last_error}") from last_error
