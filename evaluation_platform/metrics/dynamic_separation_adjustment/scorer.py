"""Score H-PPO runtime horizontal-separation changes from the event log."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if isinstance(item, dict):
            rows.append(item)
    return rows


def read_csv(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {int(row["episode"]): row for row in csv.DictReader(stream) if row.get("episode")}


def truthy(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes"}


def score(events_path: Path, diagnostics_path: Path) -> dict[str, Any]:
    diagnostics = read_csv(diagnostics_path)
    changes = [row for row in read_jsonl(events_path) if row.get("event") == "separation_updated"]
    details = []
    category_counts = Counter()
    for row in changes:
        episode = int(row.get("episode", -1))
        old_nm = float(row.get("previous_horizontal_km", 0.0)) / 1.852
        new_nm = float(row.get("horizontal_km", 0.0)) / 1.852
        kind = "收紧" if new_nm > old_nm else "放宽" if new_nm < old_nm else "未变化"
        summary = diagnostics.get(episode, {})
        eligible = kind != "未变化" and bool(summary)
        success = eligible and truthy(summary, "safe_success")
        reason = ""
        if not summary:
            reason = "缺少同回合诊断文件"
        elif not eligible:
            reason = "间隔值未发生有效变化"
        elif not success:
            reason = str(summary.get("reason", "未安全完成"))
        category_counts[kind] += 1
        details.append({
            "episode": episode,
            "sim_time_s": row.get("sim_time_s"),
            "source": row.get("source", "runtime"),
            "old_horizontal_nm": old_nm,
            "new_horizontal_nm": new_nm,
            "change_type": kind,
            "eligible": eligible,
            "success": success,
            "reason": reason,
            "safety_violations": int(float(summary.get("safety_violations", 0) or 0)),
            "collision_events": int(float(summary.get("collision_events", 0) or 0)),
        })
    eligible_rows = [row for row in details if row["eligible"]]
    success_rows = [row for row in eligible_rows if row["success"]]
    by_change_type = {}
    for kind in ("收紧", "放宽"):
        rows = [row for row in eligible_rows if row["change_type"] == kind]
        by_change_type[kind] = {
            "events": len(rows),
            "successes": sum(row["success"] for row in rows),
            "success_rate": None if not rows else sum(row["success"] for row in rows) / len(rows),
        }
    value = None if not eligible_rows else len(success_rows) / len(eligible_rows)
    return {
        "metric_id": "dynamic_separation_adjustment",
        "status": "complete" if eligible_rows else "not_evaluable",
        "primary": {"name": "动态间隔调整成功率", "value": value, "unit": "%"},
        "metrics": {"dynamic_separation_adjustment_success_rate": value},
        "counts": {
            "change_events": len(details),
            "eligible_events": len(eligible_rows),
            "successful_events": len(success_rows),
        },
        "by_change_type": by_change_type,
        "details": details,
        "claim_boundary": "初版按一次运行内的水平间隔变更事件评分。成功表示变更后该回合安全结束，未验证整个恢复窗口内的连续稳定达标。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = score(args.events, args.diagnostics)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
