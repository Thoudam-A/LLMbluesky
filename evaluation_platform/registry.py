"""Metric plug-in discovery and validation."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_FIELDS = {
    "metric_id", "name", "group", "version", "status", "primary_metric",
    "runner", "scorer", "required_inputs", "views",
}


class MetricRegistry:
    def __init__(self, root: Path):
        self.root = root
        self._metrics: dict[str, dict] = {}

    def load(self) -> None:
        metrics: dict[str, dict] = {}
        for manifest_path in sorted(self.root.glob("*/manifest.json")):
            if manifest_path.parent.name.startswith("_"):
                continue
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            missing = REQUIRED_FIELDS - set(data)
            if missing:
                raise ValueError(f"{manifest_path}: missing fields {sorted(missing)}")
            metric_id = str(data["metric_id"])
            if metric_id in metrics:
                raise ValueError(f"duplicate metric_id: {metric_id}")
            data["directory"] = str(manifest_path.parent)
            metrics[metric_id] = data
        self._metrics = metrics

    def get(self, metric_id: str) -> dict:
        if metric_id not in self._metrics:
            raise KeyError(f"unknown metric_id: {metric_id}")
        return dict(self._metrics[metric_id])

    def public_items(self) -> list[dict]:
        hidden = {"directory", "runner", "scorer"}
        return [{k: v for k, v in item.items() if k not in hidden} for item in self._metrics.values()]
