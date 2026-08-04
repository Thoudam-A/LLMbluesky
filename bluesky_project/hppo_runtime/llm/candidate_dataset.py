"""Read-only frozen coarse-candidate datasets for runtime candidate selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .candidate_pool import CandidatePool, build_candidate_pool
from .experience_store import canonical_scene_fingerprint, role_order, scene_fingerprint
from .models import CoarseCandidateAction, ConflictScene, conflict_scene_from_dict


COARSE_TO_MACRO = {
    "NO_NEW_COMMAND": 0,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 2,
    "CLIMB": 3,
    "DESCEND": 3,
    "ACCELERATE": 4,
    "DECELERATE": 4,
}


@dataclass(slots=True)
class CandidateLookup:
    pool: CandidatePool
    macro_actions: np.ndarray
    source: str


@dataclass(slots=True)
class _DatasetEntry:
    scene: ConflictScene
    candidates: list[CoarseCandidateAction]


class FrozenCandidateDataset:
    def __init__(self, path: Path, pool_size: int = 6):
        self.path = Path(path).expanduser().resolve()
        self.pool_size = int(pool_size)
        if not self.path.is_file():
            raise FileNotFoundError(f"Frozen candidate dataset does not exist: {self.path}")
        self._exact: dict[str, _DatasetEntry] = {}
        self._canonical: dict[str, list[_DatasetEntry]] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if "error" in record or "scene" not in record or "proposal" not in record:
                continue
            scene = conflict_scene_from_dict(record["scene"])
            candidates = [CoarseCandidateAction(**item) for item in record["proposal"]["candidates"] if item.get("safe", False)]
            entry = _DatasetEntry(scene, candidates)
            fingerprint = str(record["fingerprint"])
            self._exact[fingerprint] = entry
            canonical, _ = canonical_scene_fingerprint(scene)
            self._canonical.setdefault(canonical, []).append(entry)

    @staticmethod
    def _map_candidates(entry: _DatasetEntry, scene: ConflictScene) -> list[CoarseCandidateAction]:
        stored_roles = role_order(entry.scene)
        current_roles = role_order(scene)
        if len(stored_roles) != len(current_roles):
            return []
        id_map = dict(zip(stored_roles, current_roles))
        mapped: list[CoarseCandidateAction] = []
        for candidate in entry.candidates:
            aircraft_id = id_map.get(candidate.aircraft_id)
            if aircraft_id is None:
                continue
            data = asdict(candidate)
            data["aircraft_id"] = aircraft_id
            if data["priority_aircraft_id"]:
                data["priority_aircraft_id"] = id_map.get(data["priority_aircraft_id"], "")
            if data["yield_aircraft_id"]:
                data["yield_aircraft_id"] = id_map.get(data["yield_aircraft_id"], "")
            mapped.append(CoarseCandidateAction(**data))
        return mapped

    def lookup(self, scene: ConflictScene, acid: str, action_mask: np.ndarray) -> CandidateLookup | None:
        exact, _ = scene_fingerprint(scene)
        entry = self._exact.get(exact)
        source = "exact"
        if entry is None:
            canonical, _ = canonical_scene_fingerprint(scene)
            matches = self._canonical.get(canonical, [])
            if not matches:
                return None
            entry = matches[0]
            source = "canonical"
        allowed = np.asarray(action_mask, dtype=np.float32)
        candidates = []
        for candidate in self._map_candidates(entry, scene):
            macro = COARSE_TO_MACRO.get(candidate.macro_action)
            if candidate.aircraft_id == acid and macro is not None and macro < allowed.size and allowed[macro] > 0.0:
                candidates.append(candidate)
        pool = build_candidate_pool(candidates, self.pool_size)
        if pool.candidate_mask.sum() <= 0.0:
            return None
        macros = np.zeros(self.pool_size, dtype=np.int64)
        for index, candidate in enumerate(pool.candidates):
            if candidate is not None:
                macros[index] = COARSE_TO_MACRO[candidate.macro_action]
        return CandidateLookup(pool, macros, source)


def map_candidate_parameters(candidate: CoarseCandidateAction, squashed_params: np.ndarray) -> np.ndarray:
    """Map one tanh output to the candidate direction and magnitude band."""
    macro = COARSE_TO_MACRO.get(candidate.macro_action)
    params = np.asarray(squashed_params, dtype=np.float32).copy()
    if macro is None or macro < 2:
        return params
    branch = macro - 2
    lower, upper = {"SMALL": (0.0, 1.0 / 3.0), "MEDIUM": (1.0 / 3.0, 2.0 / 3.0), "LARGE": (2.0 / 3.0, 1.0)}[candidate.magnitude_level]
    magnitude = lower + (float(np.clip(params[branch], -1.0, 1.0)) + 1.0) * 0.5 * (upper - lower)
    sign = -1.0 if candidate.macro_action in {"TURN_LEFT", "DESCEND", "DECELERATE"} else 1.0
    params[branch] = sign * magnitude
    return params
