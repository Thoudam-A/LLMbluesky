"""Portable, user-configured dataset and model catalog.

Large datasets and model weights are intentionally not stored in Git.  A local
``catalog.json`` maps stable public IDs to files on each contributor's machine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = Path(os.environ.get(
    "ATC_EVAL_CATALOG",
    REPO_ROOT / "evaluation_platform/config/catalog.json",
))


def _resolve(value: str, base: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_catalog(path: Path = DEFAULT_CATALOG) -> tuple[dict[str, dict], dict[str, dict], dict[str, Any]]:
    if not path.exists():
        return {}, {}, {
            "configured": False,
            "catalog_path": str(path),
            "message": "Copy catalog.example.json to catalog.json and configure local artifact paths.",
        }
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("catalog root must be a JSON object")
    base = path.parent
    datasets: dict[str, dict] = {}
    models: dict[str, dict] = {}
    for item in raw.get("datasets", []):
        dataset_id = str(item.get("id", "")).strip()
        if not dataset_id or dataset_id in datasets:
            raise ValueError(f"invalid or duplicate dataset id: {dataset_id!r}")
        row = dict(item)
        for key in (
            "references", "replay", "trajectory_parquet",
            "controller_instructions", "intent_references",
        ):
            if row.get(key):
                row[key] = _resolve(str(row[key]), base)
        datasets[dataset_id] = row
    for item in raw.get("models", []):
        model_id = str(item.get("id", "")).strip()
        if not model_id or model_id in models:
            raise ValueError(f"invalid or duplicate model id: {model_id!r}")
        row = dict(item)
        for key in (
            "model", "config", "system_outputs", "archived_metric",
            "intent_predictions",
        ):
            if row.get(key):
                row[key] = _resolve(str(row[key]), base)
        models[model_id] = row
    return datasets, models, {
        "configured": True,
        "catalog_path": str(path.resolve()),
        "message": "catalog loaded",
    }


DATASETS, MODELS, CATALOG_STATUS = load_catalog()


def _exists(item: dict, keys: tuple[str, ...]) -> tuple[bool, list[str]]:
    missing = [key for key in keys if not item.get(key) or not Path(item[key]).exists()]
    return not missing, missing


def public_catalog() -> dict:
    datasets = []
    for item in DATASETS.values():
        available, missing = _exists(item, ("references", "replay"))
        full_available, full_missing = _exists(item, ("references", "replay", "trajectory_parquet"))
        datasets.append({
            "id": item["id"],
            "name": item.get("name", item["id"]),
            "description": item.get("description", ""),
            "reference_count": item.get("reference_count"),
            "available": available,
            "full_replay_available": full_available,
            "missing_fields": sorted(set(missing + full_missing)),
            "intent_available": _exists(item, ("controller_instructions", "intent_references"))[0],
            "intent_missing_fields": _exists(item, ("controller_instructions", "intent_references"))[1],
            "controller_instruction_file": (
                Path(item["controller_instructions"]).name if item.get("controller_instructions") else None
            ),
            "intent_reference_file": (
                Path(item["intent_references"]).name if item.get("intent_references") else None
            ),
        })
    models = []
    for item in MODELS.values():
        available, missing = _exists(item, ("model", "config", "system_outputs"))
        models.append({
            "id": item["id"],
            "name": item.get("name", item["id"]),
            "available": available,
            "missing_fields": missing,
            "intent_available": _exists(item, ("intent_predictions",))[0],
            "intent_missing_fields": _exists(item, ("intent_predictions",))[1],
            "intent_prediction_file": (
                Path(item["intent_predictions"]).name if item.get("intent_predictions") else None
            ),
        })
    return {"datasets": datasets, "models": models, "configuration": CATALOG_STATUS}
