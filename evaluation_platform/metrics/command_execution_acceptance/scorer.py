"""Score BlueSky acceptance of H-PPO commands from an immutable event log."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MACRO_NAMES = {1: "恢复航路", 2: "航向", 3: "高度", 4: "速度"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _macro(event: dict[str, Any]) -> int:
    try:
        return int(event.get("macro_action", -1))
    except (TypeError, ValueError):
        return -1


def score(events_path: Path) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    ignored = Counter()

    for event in read_jsonl(events_path):
        macro = _macro(event)
        if macro not in MACRO_NAMES:
            continue
        event_name = str(event.get("event", ""))
        applied = bool(event.get("command_applied", False))
        command = str(event.get("command", ""))
        if event_name == "action_executed":
            if applied:
                accepted.append(event)
            else:
                ignored["accepted_without_new_command"] += 1
            continue
        if event_name == "command_failed":
            # A NOOP or an empty command was rejected by a guard before it was
            # sent to BlueSky. It is useful diagnostic data but not part of the
            # interface-acceptance denominator.
            if not command or command == "NOOP":
                ignored["precheck_rejected"] += 1
            else:
                rejected.append(event)

    by_macro: dict[int, dict[str, Any]] = {}
    for macro, name in MACRO_NAMES.items():
        ok = [row for row in accepted if _macro(row) == macro]
        failed = [row for row in rejected if _macro(row) == macro]
        submitted = len(ok) + len(failed)
        by_macro[str(macro)] = {
            "name": name,
            "submitted": submitted,
            "accepted": len(ok),
            "rejected": len(failed),
            "acceptance_rate": None if submitted == 0 else len(ok) / submitted,
        }

    submitted = len(accepted) + len(rejected)
    failure_reasons = Counter(str(row.get("message", "unknown")) for row in rejected)
    details = [
        {
            "episode": row.get("episode"),
            "sim_time_s": row.get("sim_time_s"),
            "aircraft_id": row.get("acid", ""),
            "macro_action": _macro(row),
            "macro_name": MACRO_NAMES[_macro(row)],
            "command": row.get("command", ""),
            "accepted": True,
            "reason": "",
        }
        for row in accepted
    ] + [
        {
            "episode": row.get("episode"),
            "sim_time_s": row.get("sim_time_s"),
            "aircraft_id": row.get("acid", ""),
            "macro_action": _macro(row),
            "macro_name": MACRO_NAMES[_macro(row)],
            "command": row.get("command", ""),
            "accepted": False,
            "reason": row.get("message", "unknown"),
        }
        for row in rejected
    ]
    details.sort(key=lambda row: (row["episode"] or -1, row["sim_time_s"] or -1))
    value = None if submitted == 0 else len(accepted) / submitted
    return {
        "metric_id": "command_execution_acceptance",
        "status": "complete" if submitted else "not_evaluable",
        "primary": {"name": "管制指令执行接受度", "value": value, "unit": "%"},
        "metrics": {"command_execution_acceptance_rate": value},
        "counts": {
            "submitted": submitted,
            "accepted": len(accepted),
            "rejected": len(rejected),
            "ignored": sum(ignored.values()),
        },
        "by_macro": by_macro,
        "failure_reasons": [{"reason": key, "count": value} for key, value in failure_reasons.most_common()],
        "details": details,
        "claim_boundary": "仅衡量 BlueSky 对实际提交指令的接口接受情况；不表示真实飞行员、机组或管制员的人因接受度。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = score(args.events)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
