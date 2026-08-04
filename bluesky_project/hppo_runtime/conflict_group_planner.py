"""Shadow-only coordination-plan evaluation for high-coupling conflict groups."""
from __future__ import annotations

from dataclasses import dataclass

import bluesky as bs
import numpy as np


@dataclass(slots=True)
class GroupPlan:
    members: tuple[str, ...]
    yielder: str
    macro: str
    score: float
    minimum_horizontal_km: float
    minimum_vertical_ft: float
    predicted_losses: int


class ConflictGroupPlanner:
    """Evaluate a small, explainable template set without controlling traffic."""

    _TEMPLATES = ("TURN_LEFT", "TURN_RIGHT", "CLIMB", "DESCEND", "DECELERATE")

    def __init__(self, config):
        self.cfg = config

    def evaluate(self, conflict_states: dict) -> list[dict]:
        groups = self._groups(conflict_states)
        results: list[dict] = []
        for members in groups:
            if len(members) < 2:
                continue
            baseline = self._simulate(members, "", "")
            plans: list[GroupPlan] = []
            for yielder in members:
                for macro in self._TEMPLATES:
                    plans.append(self._simulate(members, yielder, macro))
            best = min(plans, key=lambda item: (item.score, item.yielder, item.macro))
            results.append({
                "members": list(members),
                "baseline": self._record(baseline),
                "recommended": self._record(best),
                "improves_baseline": bool(best.score + 1e-6 < baseline.score),
            })
        return results

    @staticmethod
    def _record(plan: GroupPlan) -> dict:
        return {
            "yielder": plan.yielder,
            "macro": plan.macro,
            "score": round(plan.score, 6),
            "min_horizontal_km": round(plan.minimum_horizontal_km, 4),
            "min_vertical_ft": round(plan.minimum_vertical_ft, 1),
            "predicted_losses": plan.predicted_losses,
        }

    def _groups(self, states: dict) -> list[tuple[str, ...]]:
        graph = {acid: set() for acid, state in states.items() if state.active and bs.traf.id2idx(acid) >= 0}
        for acid, state in states.items():
            if acid not in graph:
                continue
            for contact in state.targets:
                if contact.callsign in graph and (contact.severity > 0.0 or contact.conflict_flag > 0.0 or contact.temporal_conflict > 0.0):
                    graph[acid].add(contact.callsign)
                    graph[contact.callsign].add(acid)
        groups, seen = [], set()
        for root in sorted(graph):
            if root in seen:
                continue
            stack, group = [root], set()
            while stack:
                item = stack.pop()
                if item in group:
                    continue
                group.add(item)
                stack.extend(graph[item] - group)
            seen.update(group)
            groups.append(tuple(sorted(group)))
        return groups

    def _simulate(self, members: tuple[str, ...], yielder: str, macro: str) -> GroupPlan:
        indices = {acid: bs.traf.id2idx(acid) for acid in members}
        lat0 = float(np.mean([bs.traf.lat[idx] for idx in indices.values()]))
        cos_lat = max(np.cos(np.radians(lat0)), 1e-6)
        position, velocity, altitude = {}, {}, {}
        for acid, idx in indices.items():
            position[acid] = np.array([
                float(bs.traf.lon[idx]) * 111.0 * cos_lat,
                float(bs.traf.lat[idx]) * 111.0,
            ])
            speed = float(bs.traf.cas[idx] * 1.94384) * 0.000514444
            heading = float(bs.traf.trk[idx])
            if acid == yielder:
                if macro == "TURN_LEFT":
                    heading -= 30.0
                elif macro == "TURN_RIGHT":
                    heading += 30.0
                elif macro == "DECELERATE":
                    speed = max(0.0, speed - 20.0 * 0.000514444)
            velocity[acid] = speed * np.array([np.sin(np.radians(heading)), np.cos(np.radians(heading))])
            altitude[acid] = float(bs.traf.alt[idx] * 3.28084)

        min_h, min_v, losses = float("inf"), float("inf"), 0
        horizon, step = 120.0, 5.0
        for elapsed in np.arange(0.0, horizon + 1e-6, step):
            projected_alt = dict(altitude)
            if yielder:
                if macro == "CLIMB":
                    projected_alt[yielder] += min(2000.0, elapsed * 1500.0 / 60.0)
                elif macro == "DESCEND":
                    projected_alt[yielder] -= min(2000.0, elapsed * 1500.0 / 60.0)
            for index, own in enumerate(members):
                for other in members[index + 1:]:
                    horizontal = float(np.linalg.norm((position[own] + velocity[own] * elapsed) - (position[other] + velocity[other] * elapsed)))
                    vertical = abs(projected_alt[own] - projected_alt[other])
                    min_h, min_v = min(min_h, horizontal), min(min_v, vertical)
                    if horizontal < self.cfg.separation.horizontal_km and vertical < self.cfg.separation.vertical_ft:
                        losses += 1
        # Collisions dominate; residual distance terms only rank non-collision templates.
        score = 1000.0 * losses + max(0.0, self.cfg.separation.horizontal_km - min_h) + max(0.0, self.cfg.separation.vertical_ft - min_v) / 1000.0
        return GroupPlan(members, yielder, macro, float(score), float(min_h), float(min_v), losses)
