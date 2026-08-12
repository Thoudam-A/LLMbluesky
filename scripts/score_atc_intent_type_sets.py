#!/usr/bin/env python3
"""Score ATC intent-type recognition with false-positive penalties."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def type_counts(row: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in row.get("intents", []) or []:
        if isinstance(item, dict):
            intent_type = item.get("intent_type")
            if isinstance(intent_type, str) and intent_type:
                counts[intent_type] += 1
    return counts


def f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def score(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pred_by_utt = {row.get("utterance_id"): row for row in pred_rows}
    tp = fp = fn = 0
    exact = 0
    total_eval_rows = 0
    skipped_gold_rows = 0
    pending_gold_rows = 0
    per_type: dict[str, Counter[str]] = defaultdict(Counter)
    errors: list[dict[str, Any]] = []

    for gold_row in gold_rows:
        status = gold_row.get("annotation_status")
        if status == "pending":
            pending_gold_rows += 1
            continue
        if status == "skip":
            skipped_gold_rows += 1
            gold_counts: Counter[str] = Counter()
        elif status == "done":
            gold_counts = type_counts(gold_row)
        else:
            pending_gold_rows += 1
            continue

        pred_counts = type_counts(pred_by_utt.get(gold_row.get("utterance_id"), {}))
        total_eval_rows += 1
        exact += int(gold_counts == pred_counts)

        all_types = set(gold_counts) | set(pred_counts)
        row_tp = row_fp = row_fn = 0
        for intent_type in all_types:
            matched = min(gold_counts[intent_type], pred_counts[intent_type])
            extra = max(0, pred_counts[intent_type] - gold_counts[intent_type])
            missing = max(0, gold_counts[intent_type] - pred_counts[intent_type])
            tp += matched
            fp += extra
            fn += missing
            row_tp += matched
            row_fp += extra
            row_fn += missing
            per_type[intent_type]["tp"] += matched
            per_type[intent_type]["fp"] += extra
            per_type[intent_type]["fn"] += missing

        if row_fp or row_fn:
            errors.append(
                {
                    "utterance_id": gold_row.get("utterance_id"),
                    "annotation_id": gold_row.get("annotation_id"),
                    "text": gold_row.get("text"),
                    "gold_types": dict(gold_counts),
                    "pred_types": dict(pred_counts),
                    "tp": row_tp,
                    "fp": row_fp,
                    "fn": row_fn,
                }
            )

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    intent_type_accuracy = f1(precision, recall)
    metrics = {
        "gold_rows": len(gold_rows),
        "prediction_rows": len(pred_rows),
        "evaluated_rows_including_skip": total_eval_rows,
        "pending_gold_rows": pending_gold_rows,
        "skipped_gold_rows": skipped_gold_rows,
        "intent_type_tp": tp,
        "intent_type_fp": fp,
        "intent_type_fn": fn,
        "intent_type_precision": precision,
        "intent_type_recall": recall,
        "intent_type_f1": intent_type_accuracy,
        "intent_type_accuracy": intent_type_accuracy,
        "intent_type_accuracy_definition": (
            "per-utterance intent_type multiset F1; extra predicted intent types are false positives, "
            "missing gold intent types are false negatives"
        ),
        "utterance_exact_type_set_accuracy": ratio(exact, total_eval_rows),
        "per_intent_type": {},
        "errors": errors,
    }
    for intent_type, counts in sorted(per_type.items()):
        p = ratio(counts["tp"], counts["tp"] + counts["fp"])
        r = ratio(counts["tp"], counts["tp"] + counts["fn"])
        metrics["per_intent_type"][intent_type] = {
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "precision": p,
            "recall": r,
            "f1": f1(p, r),
        }
    return metrics


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# ATC Intent-Type Set Scoring Report",
        "",
        "## Definition",
        "",
        "Intent types are scored as a per-utterance multiset. Extra predicted intent types count as false positives.",
        "The headline intent_type_accuracy is this multiset F1, not the old gold-only recall-like score.",
        "",
        "## Summary",
        "",
        f"- Gold rows: {metrics['gold_rows']}",
        f"- Prediction rows: {metrics['prediction_rows']}",
        f"- Evaluated rows including skip: {metrics['evaluated_rows_including_skip']}",
        f"- TP: {metrics['intent_type_tp']}",
        f"- FP: {metrics['intent_type_fp']}",
        f"- FN: {metrics['intent_type_fn']}",
        f"- Intent type precision: {fmt(metrics['intent_type_precision'])}",
        f"- Intent type recall: {fmt(metrics['intent_type_recall'])}",
        f"- Intent type F1: {fmt(metrics['intent_type_f1'])}",
        f"- Intent type accuracy: {fmt(metrics['intent_type_accuracy'])}",
        f"- Utterance exact type-set accuracy: {fmt(metrics['utterance_exact_type_set_accuracy'])}",
        "",
        "## Per Intent Type",
        "",
    ]
    for intent_type, item in metrics["per_intent_type"].items():
        lines.append(
            f"- {intent_type}: P={fmt(item['precision'])} R={fmt(item['recall'])} F1={fmt(item['f1'])} "
            f"TP={item['tp']} FP={item['fp']} FN={item['fn']}"
        )
    lines.extend(["", "## First Errors", ""])
    for err in metrics["errors"][:50]:
        lines.append(
            f"- {err.get('utterance_id')} TP={err.get('tp')} FP={err.get('fp')} FN={err.get('fn')} "
            f"gold={err.get('gold_types')} pred={err.get('pred_types')}"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    metrics = score(read_jsonl(Path(args.gold)), read_jsonl(Path(args.pred)))
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(Path(args.report), metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "errors"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
