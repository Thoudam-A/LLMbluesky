#!/usr/bin/env python3
"""Constrained Qwen candidate-reranking helpers for Shanghai ATC replay."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


PROMPT_VERSION = "shanghai_qwen_candidate_prompt_v1.0"


STATE_FIELDS = (
    "callsign",
    "flight_phase",
    "phase",
    "flight_direction",
    "altitude_m",
    "ground_speed_kt",
    "vertical_rate_mps",
    "track_heading_deg",
    "plan_adep",
    "plan_ades",
    "arrival_runway",
    "departure_runway",
    "sid",
    "star",
    "declared_procedure_name",
    "previous_fix",
    "next_fix",
    "distance_to_next_fix_nm",
    "time_to_next_fix_sec",
    "prior_cleared_level_m",
    "plan_cleared_level_m",
    "requested_level_m",
    "next_planned_level_m",
    "prior_speed_constraint_kt",
    "planned_speed_kt",
    "inside_sector_traffic_count",
    "same_sid_traffic_count",
    "same_star_traffic_count",
    "nearest_horizontal_nm",
    "minimum_cpa_horizontal_nm",
    "minimum_vertical_at_cpa_m",
    "predicted_conflict_count",
    "procedure_alignment_status",
    "route_context_confidence",
    "quality_tier",
)


@dataclass(frozen=True)
class QwenChoice:
    candidate_id: str
    reason_codes: tuple[str, ...]
    raw_text: str
    latency_sec: float
    model: str | None = None


def compact_state(state: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in STATE_FIELDS:
        value = state.get(field)
        if value is None or value == "":
            continue
        if isinstance(value, float):
            value = round(value, 3)
        result[field] = value
    return result


def candidate_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "kind": str(row.get("candidate_kind") or ""),
                "target": row.get("candidate_target"),
                "base_score": round(float(row.get("candidate_score") or 0.0), 6),
                "base_margin": round(float(row.get("margin") or 0.0), 6),
            }
        )
    return result


def build_prompt(state: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    payload = {
        "state": compact_state(state),
        "candidates": candidate_payload(candidates),
    }
    return (
        "你是上海进近管制决策候选重排器。只能从给定candidate_id中选择一个，"
        "不得创造高度、速度、航向或候选。优先遵守已有许可、飞行计划、SID/STAR、"
        "飞行阶段和交通调配逻辑；base_score只是参考，不代表最终正确。"
        "只输出一个candidate_id，不要JSON，不要解释，不要任何其他文字。\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _json_objects(text: str) -> list[str]:
    objects: list[str] = []
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None
    return objects


def parse_choice(text: str, allowed_ids: set[str], latency_sec: float) -> QwenChoice:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    if cleaned in allowed_ids:
        return QwenChoice(
            candidate_id=cleaned,
            reason_codes=(),
            raw_text=text,
            latency_sec=latency_sec,
        )
    payload: dict[str, Any] | None = None
    for candidate in [cleaned, *_json_objects(cleaned)]:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "candidate_id" in parsed:
            payload = parsed
            break
    if payload is None:
        raise ValueError("qwen response does not contain a JSON decision")
    candidate_id = str(payload["candidate_id"])
    if candidate_id not in allowed_ids:
        raise ValueError(f"qwen selected unknown candidate: {candidate_id}")
    reasons = payload.get("reason_codes") or []
    if not isinstance(reasons, list):
        reasons = []
    return QwenChoice(
        candidate_id=candidate_id,
        reason_codes=tuple(str(item)[:80] for item in reasons[:3]),
        raw_text=text,
        latency_sec=latency_sec,
        model=str(payload.get("model")) if payload.get("model") else None,
    )


class HttpQwenReranker:
    """Minimal dependency-free client for the isolated local Qwen service."""

    def __init__(self, endpoint: str, timeout_sec: float) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_sec = timeout_sec

    def health(self) -> dict[str, Any]:
        request = urllib.request.Request(self.endpoint + "/health", method="GET")
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))

    def choose(
        self,
        state: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> QwenChoice:
        allowed_ids = {str(row["candidate_id"]) for row in candidates}
        body = json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "prompt": build_prompt(state, candidates),
                "allowed_candidate_ids": sorted(allowed_ids),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint + "/rank",
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"qwen HTTP {exc.code}: {detail[:500]}") from exc
        elapsed = time.perf_counter() - start
        if not isinstance(payload, dict) or "text" not in payload:
            raise ValueError("qwen service returned an invalid envelope")
        choice = parse_choice(str(payload["text"]), allowed_ids, elapsed)
        return QwenChoice(
            candidate_id=choice.candidate_id,
            reason_codes=choice.reason_codes,
            raw_text=choice.raw_text,
            latency_sec=choice.latency_sec,
            model=str(payload.get("model")) if payload.get("model") else None,
        )
