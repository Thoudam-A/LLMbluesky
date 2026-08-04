from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any
from contextlib import closing

from .models import CandidateProposal, ConflictScene, candidate_proposal_from_dict, conflict_scene_from_dict


def _bucket(value: float, width: float) -> int:
    return int(round(float(value) / width))


def scene_fingerprint(scene: ConflictScene) -> tuple[str, str]:
    nodes = {item.aircraft_id: item for item in scene.aircraft}
    degrees = {acid: 0 for acid in scene.target_ids}
    edges = []
    for edge in scene.conflicts:
        if edge.ownship_id not in nodes or edge.intruder_id not in nodes:
            continue
        own = nodes[edge.ownship_id]
        intruder = nodes[edge.intruder_id]
        relative_heading = abs((own.heading_deg - intruder.heading_deg + 180.0) % 360.0 - 180.0)
        degrees[edge.ownship_id] = degrees.get(edge.ownship_id, 0) + 1
        edges.append(
            (
                _bucket(relative_heading, 15.0),
                _bucket(abs(own.altitude_ft - intruder.altitude_ft), 500.0),
                _bucket(abs(own.speed_kt - intruder.speed_kt), 10.0),
                _bucket(edge.tcpa_s, 30.0),
                int(edge.loss_of_separation),
            )
        )
    payload = {
        "aircraft_count": len(scene.target_ids),
        "degree_sequence": sorted(degrees.values(), reverse=True),
        "edges": sorted(edges),
        "separation": [
            _bucket(scene.separation_horizontal_km, 1.0),
            _bucket(scene.separation_vertical_ft, 500.0),
            _bucket(scene.separation_time_s, 30.0),
        ],
    }
    descriptor = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()[:24]
    return digest, descriptor


def role_order(scene: ConflictScene) -> list[str]:
    """Assign stable graph roles without depending on callsigns."""
    nodes = {item.aircraft_id: item for item in scene.aircraft}
    stats: dict[str, dict[str, float]] = {
        acid: {"degree": 0.0, "severity": 0.0, "tcpa": 1e9}
        for acid in scene.target_ids
        if acid in nodes
    }
    for edge in scene.conflicts:
        if edge.ownship_id not in stats:
            continue
        item = stats[edge.ownship_id]
        item["degree"] += 1.0
        item["severity"] = max(item["severity"], float(edge.severity))
        item["tcpa"] = min(item["tcpa"], max(0.0, float(edge.tcpa_s)))
    return sorted(
        stats,
        key=lambda acid: (
            -stats[acid]["degree"],
            -_bucket(stats[acid]["severity"], 0.25),
            _bucket(stats[acid]["tcpa"], 60.0),
            -_bucket(nodes[acid].altitude_ft, 1000.0),
            -_bucket(nodes[acid].speed_kt, 20.0),
            _bucket(nodes[acid].heading_deg % 360.0, 30.0),
            acid,
        ),
    )


def canonical_scene_fingerprint(scene: ConflictScene) -> tuple[str, str]:
    """Coarse, callsign-independent key used only as a cache fallback."""
    nodes = {item.aircraft_id: item for item in scene.aircraft}
    roles = role_order(scene)
    role_by_id = {acid: index for index, acid in enumerate(roles)}
    undirected: dict[tuple[int, int], tuple[int, ...]] = {}
    for edge in scene.conflicts:
        if edge.ownship_id not in role_by_id or edge.intruder_id not in role_by_id:
            continue
        own = nodes[edge.ownship_id]
        intruder = nodes[edge.intruder_id]
        pair = tuple(sorted((role_by_id[edge.ownship_id], role_by_id[edge.intruder_id])))
        relative_heading = abs((own.heading_deg - intruder.heading_deg + 180.0) % 360.0 - 180.0)
        value = (
            pair[0],
            pair[1],
            _bucket(relative_heading, 30.0),
            _bucket(abs(own.altitude_ft - intruder.altitude_ft), 1000.0),
            _bucket(abs(own.speed_kt - intruder.speed_kt), 20.0),
            _bucket(max(0.0, edge.tcpa_s), 60.0),
            _bucket(edge.severity, 0.25),
        )
        previous = undirected.get(pair)
        if previous is None or value[-2:] < previous[-2:]:
            undirected[pair] = value
    payload = {
        "aircraft_count": len(roles),
        "degree_sequence": sorted(
            [sum(1 for pair in undirected if role in pair) for role in range(len(roles))],
            reverse=True,
        ),
        "edges": sorted(undirected.values()),
        "separation": [
            _bucket(scene.separation_horizontal_km, 2.0),
            _bucket(scene.separation_vertical_ft, 1000.0),
            _bucket(scene.separation_time_s, 60.0),
        ],
    }
    descriptor = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()[:24]
    return digest, descriptor


def _remap_proposal(proposal: CandidateProposal, stored: ConflictScene, current: ConflictScene) -> CandidateProposal | None:
    stored_roles = role_order(stored)
    current_roles = role_order(current)
    if len(stored_roles) != len(current_roles):
        return None
    role_map = dict(zip(stored_roles, current_roles))
    data = proposal.to_dict()
    for candidate in data["candidates"]:
        mapped = role_map.get(candidate["aircraft_id"])
        if mapped is None:
            return None
        candidate["aircraft_id"] = mapped
    data["scene_id"] = current.scene_id
    return candidate_proposal_from_dict(data)


class ExperienceStore:
    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=TRUNCATE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS plans (
                    fingerprint TEXT PRIMARY KEY,
                    descriptor TEXT NOT NULL,
                    scene_json TEXT NOT NULL,
                    proposal_json TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    validation_status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS novel_scenes (
                    fingerprint TEXT PRIMARY KEY,
                    descriptor TEXT NOT NULL,
                    scene_json TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    seen_count INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'pending'
                );
                CREATE TABLE IF NOT EXISTS api_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    occurred_at TEXT NOT NULL,
                    usage_date TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    latency_s REAL NOT NULL
                );
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(novel_scenes)")}
            if "failure_reason" not in columns:
                connection.execute("ALTER TABLE novel_scenes ADD COLUMN failure_reason TEXT NOT NULL DEFAULT ''")
            connection.execute(
                "UPDATE novel_scenes SET failure_reason = 'legacy failure: detail was not recorded' "
                "WHERE status = 'failed' AND failure_reason = ''"
            )
            plan_columns = {row["name"] for row in connection.execute("PRAGMA table_info(plans)")}
            if "canonical_fingerprint" not in plan_columns:
                connection.execute("ALTER TABLE plans ADD COLUMN canonical_fingerprint TEXT NOT NULL DEFAULT ''")
            stale = connection.execute(
                "SELECT fingerprint, scene_json FROM plans WHERE canonical_fingerprint = ''"
            ).fetchall()
            for row in stale:
                stored_scene = conflict_scene_from_dict(json.loads(row["scene_json"]))
                canonical = canonical_scene_fingerprint(stored_scene)[0]
                connection.execute(
                    "UPDATE plans SET canonical_fingerprint = ? WHERE fingerprint = ?",
                    (canonical, row["fingerprint"]),
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_plans_canonical ON plans(canonical_fingerprint, validation_status)"
            )

    def get_plan(self, fingerprint: str) -> CandidateProposal | None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT proposal_json FROM plans WHERE fingerprint = ? AND validation_status = 'prediction_validated'",
                (fingerprint,),
            ).fetchone()
        return candidate_proposal_from_dict(json.loads(row["proposal_json"])) if row else None

    def get_plan_for_scene(self, scene: ConflictScene) -> tuple[CandidateProposal | None, str]:
        exact, _ = scene_fingerprint(scene)
        canonical, _ = canonical_scene_fingerprint(scene)
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT fingerprint, scene_json, proposal_json FROM plans "
                "WHERE validation_status = 'prediction_validated' "
                "AND (fingerprint = ? OR canonical_fingerprint = ?) "
                "ORDER BY CASE WHEN fingerprint = ? THEN 0 ELSE 1 END",
                (exact, canonical, exact),
            ).fetchall()
        fallback: tuple[CandidateProposal, ConflictScene] | None = None
        for row in rows:
            stored_scene = conflict_scene_from_dict(json.loads(row["scene_json"]))
            proposal = candidate_proposal_from_dict(json.loads(row["proposal_json"]))
            if row["fingerprint"] == exact:
                mapped = _remap_proposal(proposal, stored_scene, scene)
                return (mapped, "exact") if mapped is not None else (None, "miss")
            if fallback is None and canonical_scene_fingerprint(stored_scene)[0] == canonical:
                fallback = (proposal, stored_scene)
        if fallback is None:
            return None, "miss"
        mapped = _remap_proposal(fallback[0], fallback[1], scene)
        return (mapped, "canonical") if mapped is not None else (None, "miss")

    def save_plan(self, fingerprint: str, descriptor: str, scene: ConflictScene, proposal: CandidateProposal) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO plans(
                    fingerprint, descriptor, scene_json, proposal_json, provider,
                    prompt_version, validation_status, created_at, canonical_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, 'prediction_validated', ?, ?)
                ON CONFLICT(fingerprint) DO UPDATE SET
                    descriptor=excluded.descriptor,
                    scene_json=excluded.scene_json,
                    proposal_json=excluded.proposal_json,
                    provider=excluded.provider,
                    prompt_version=excluded.prompt_version,
                    validation_status=excluded.validation_status,
                    created_at=excluded.created_at,
                    canonical_fingerprint=excluded.canonical_fingerprint
                """,
                (
                    fingerprint,
                    descriptor,
                    json.dumps(scene.to_dict(), sort_keys=True, allow_nan=False),
                    json.dumps(proposal.to_dict(), sort_keys=True, allow_nan=False),
                    proposal.provider,
                    proposal.prompt_version,
                    now,
                    canonical_scene_fingerprint(scene)[0],
                ),
            )
            connection.execute("UPDATE novel_scenes SET status = 'resolved' WHERE fingerprint = ?", (fingerprint,))

    def queue_scene(self, fingerprint: str, descriptor: str, scene: ConflictScene) -> None:
        now = datetime.now(timezone.utc).isoformat()
        scene_json = json.dumps(scene.to_dict(), sort_keys=True, allow_nan=False)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO novel_scenes(fingerprint, descriptor, scene_json, first_seen, last_seen, seen_count, status)
                VALUES (?, ?, ?, ?, ?, 1, 'pending')
                ON CONFLICT(fingerprint) DO UPDATE SET
                    scene_json=excluded.scene_json,
                    last_seen=excluded.last_seen,
                    seen_count=novel_scenes.seen_count + 1
                """,
                (fingerprint, descriptor, scene_json, now, now),
            )

    def pending_scenes(self, limit: int | None = None) -> list[tuple[str, str, ConflictScene]]:
        sql = "SELECT fingerprint, descriptor, scene_json FROM novel_scenes WHERE status = 'pending' ORDER BY seen_count DESC, first_seen"
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (int(limit),)
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(sql, params).fetchall()
        return [(row["fingerprint"], row["descriptor"], conflict_scene_from_dict(json.loads(row["scene_json"]))) for row in rows]

    def mark_scene(self, fingerprint: str, status: str, failure_reason: str = "") -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE novel_scenes SET status = ?, failure_reason = ? WHERE fingerprint = ?",
                (status, failure_reason[:2000], fingerprint),
            )

    def record_usage(self, provider: str, usage: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """INSERT INTO api_usage(
                    occurred_at, usage_date, provider, model, prompt_tokens,
                    completion_tokens, total_tokens, latency_s
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now.isoformat(),
                    now.date().isoformat(),
                    provider,
                    str(usage.get("model", "")),
                    int(usage.get("prompt_tokens", 0)),
                    int(usage.get("completion_tokens", 0)),
                    int(usage.get("total_tokens", 0)),
                    float(usage.get("latency_s", 0.0)),
                ),
            )

    def daily_usage(self) -> dict[str, int]:
        today = datetime.now(timezone.utc).date().isoformat()
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT COUNT(*) calls, COALESCE(SUM(total_tokens), 0) tokens FROM api_usage WHERE usage_date = ?",
                (today,),
            ).fetchone()
        return {"calls": int(row["calls"]), "tokens": int(row["tokens"])}

    def summary(self) -> dict[str, int]:
        with closing(self._connect()) as connection, connection:
            plans = connection.execute("SELECT COUNT(*) count FROM plans").fetchone()["count"]
            pending = connection.execute("SELECT COUNT(*) count FROM novel_scenes WHERE status = 'pending'").fetchone()["count"]
        return {"plans": int(plans), "pending_scenes": int(pending)}
