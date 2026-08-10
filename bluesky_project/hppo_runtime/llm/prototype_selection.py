"""Select a small, diverse offline LLM batch from queued conflict scenes."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .experience_store import canonical_scene_fingerprint
from .models import ConflictScene


@dataclass(frozen=True, slots=True)
class QueuedScene:
    fingerprint: str
    seen_count: int
    scene: ConflictScene


def scenario_family(scene_id: str) -> str:
    """Return the two-digit template prefix when it is present."""
    match = re.match(r"^(\d{2})(?:_|$)", scene_id)
    return match.group(1) if match else "other"


def _risk_key(item: QueuedScene) -> tuple[int, float, float, int, str]:
    edges = item.scene.conflicts
    max_severity = max((float(edge.severity) for edge in edges), default=0.0)
    min_tcpa = min((max(0.0, float(edge.tcpa_s)) for edge in edges), default=float("inf"))
    active_holds = sum(float(aircraft.hold_remaining_s) > 1e-6 for aircraft in item.scene.aircraft)
    # A new-conflict snapshot is a cleaner offline LLM input than one already
    # constrained by a prior command; then prefer high-risk representatives.
    return (active_holds, -max_severity, min_tcpa, -int(item.seen_count), item.fingerprint)


def select_prototypes(items: Iterable[QueuedScene], limit: int = 20) -> list[QueuedScene]:
    """Choose at most ``limit`` scene-family-balanced canonical prototypes.

    Every canonical type contributes at most one record. The first pass gives
    each scenario family one representative; subsequent passes round-robin
    through remaining families, preventing dense scenarios from consuming the
    entire API budget.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")

    canonical_best: dict[tuple[str, str], QueuedScene] = {}
    for item in items:
        family = scenario_family(item.scene.scene_id)
        canonical, _ = canonical_scene_fingerprint(item.scene)
        key = (family, canonical)
        previous = canonical_best.get(key)
        if previous is None or _risk_key(item) < _risk_key(previous):
            canonical_best[key] = item

    by_family: dict[str, list[QueuedScene]] = {}
    for (family, _), item in canonical_best.items():
        by_family.setdefault(family, []).append(item)
    for group in by_family.values():
        group.sort(key=_risk_key)

    selected: list[QueuedScene] = []
    families = sorted(by_family)
    positions = {family: 0 for family in families}
    while len(selected) < limit:
        progressed = False
        for family in families:
            position = positions[family]
            group = by_family[family]
            if position >= len(group):
                continue
            selected.append(group[position])
            positions[family] += 1
            progressed = True
            if len(selected) == limit:
                break
        if not progressed:
            break
    return selected
