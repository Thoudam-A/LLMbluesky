from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_weekly_platform_snapshot_matches_report_bundle() -> None:
    report = json.loads(
        (ROOT / "docs/weekly_metrics_20260810/metrics_summary.json").read_text(encoding="utf-8")
    )
    platform = json.loads(
        (ROOT / "evaluation_platform/static/weekly_metrics_20260810.json").read_text(
            encoding="utf-8"
        )
    )
    assert platform["controller_imitation"]["metrics"]["system_command_precision"] == report[
        "controller_imitation"
    ]["metrics"]["system_command_precision"]
    assert platform["intent_understanding"]["metrics"]["action_hit_rate"] == report[
        "intent_understanding"
    ]["metrics"]["action_hit_rate"]


def test_report_embeds_nonempty_platform_screenshots() -> None:
    markdown = (ROOT / "docs/weekly_metrics_20260810/WEEKLY_METRICS_REPORT.md").read_text(
        encoding="utf-8"
    )
    for name in ("platform_imitation_precision.jpg", "platform_intent_action_accuracy.jpg"):
        path = ROOT / "docs/weekly_metrics_20260810/screenshots" / name
        assert path.stat().st_size > 50_000
        assert f"screenshots/{name}" in markdown
