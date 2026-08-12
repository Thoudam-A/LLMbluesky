"""Check multi-metric implementation integrity and local runtime readiness."""

from __future__ import annotations

import importlib.util
import argparse
import json
import os
import sys
from pathlib import Path

from .catalog import CATALOG_STATUS, DATASETS, MODELS, REPO_ROOT, public_catalog
from .registry import MetricRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation-only",
        action="store_true",
        help="check committed source and plug-ins without requiring private data or model files",
    )
    args = parser.parse_args()
    checks: list[dict] = []
    notes: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    def note(name: str, ok: bool, detail: str) -> None:
        notes.append({"name": name, "ok": bool(ok), "detail": detail})

    static = REPO_ROOT / "evaluation_platform/static/index.html"
    add("web_static", static.exists(), str(static))
    scripts = [
        REPO_ROOT / "evaluation_scripts/run_shanghai_program_policy_replay.py",
        REPO_ROOT / "evaluation_scripts/convert_decision_log_for_imitation.py",
        REPO_ROOT / "evaluation_scripts/score_seu_imitation.py",
    ]
    add("evaluation_scripts", all(path.exists() for path in scripts), ", ".join(str(path.name) for path in scripts))
    try:
        registry = MetricRegistry(REPO_ROOT / "evaluation_platform/metrics")
        registry.load()
        expected_metrics = {
            "controller_imitation",
            "dynamic_separation_adjustment",
            "command_execution_acceptance",
            "autonomous_command_response_time",
            "controller_intent_understanding_accuracy",
        }
        registered = {x["metric_id"] for x in registry.public_items()}
        add("metric_registry", registered == expected_metrics, ", ".join(sorted(registered)))
        scorer_paths = []
        for metric_id in sorted(expected_metrics & registered):
            scorer = registry.get(metric_id).get("scorer")
            if scorer:
                scorer_paths.append(REPO_ROOT / str(scorer))
        add(
            "metric_scorers",
            bool(scorer_paths) and all(path.is_file() for path in scorer_paths),
            ", ".join(path.name for path in scorer_paths),
        )
    except Exception as exc:
        add("metric_registry", False, str(exc))

    hppo_required = [
        REPO_ROOT / "bluesky_project/hppo_runtime/lifecycle_manager.py",
        REPO_ROOT / "bluesky_project/plugins/case_hppo_bridge.py",
        REPO_ROOT / "bluesky_project/config/settings_hppo.cfg",
        REPO_ROOT / "hppo_tools/run_hppo_target.py",
    ]
    add("hppo_integration", all(path.is_file() for path in hppo_required), ", ".join(path.name for path in hppo_required))
    checkpoint = REPO_ROOT / "artifacts/hppo/checkpoints/hppo_candidate_selector_v2_curriculum.pt"
    note(
        "hppo_checkpoint",
        checkpoint.is_file(),
        str(checkpoint) if checkpoint.is_file() else "not tracked; provide --checkpoint or train locally",
    )

    catalog = public_catalog()
    if not args.implementation_only:
        add("catalog_configured", bool(CATALOG_STATUS["configured"]), CATALOG_STATUS["catalog_path"])
        add("dataset_available", bool(catalog["datasets"]) and any(x["available"] or x.get("intent_available") for x in catalog["datasets"]), json.dumps(catalog["datasets"], ensure_ascii=False))
        add("model_available", bool(catalog["models"]) and any(x["available"] or x.get("intent_available") for x in catalog["models"]), json.dumps(catalog["models"], ensure_ascii=False))

    full_requested = any(x.get("full_replay_available") for x in catalog["datasets"])
    if not args.implementation_only and full_requested:
        missing_deps = [name for name in ("joblib", "pandas", "pyarrow", "sklearn") if importlib.util.find_spec(name) is None]
        add("full_replay_dependencies", not missing_deps, "missing: " + ", ".join(missing_deps) if missing_deps else "available")
        compat_candidates = [
            Path(os.environ["ATC_PYARROW_COMPAT_PATH"]) if os.environ.get("ATC_PYARROW_COMPAT_PATH") else None,
            REPO_ROOT / ".deps/pyarrow20",
            REPO_ROOT.parent / ".deps/pyarrow20",
        ]
        compat = next((path for path in compat_candidates if path and path.exists()), None)
        if compat is not None:
            add("pyarrow_parquet_compatibility", True, str(compat))
        else:
            try:
                import pyarrow
                major = int(pyarrow.__version__.split(".", 1)[0])
                add("pyarrow_parquet_compatibility", major >= 20, f"pyarrow {pyarrow.__version__}; require >=20,<21")
            except Exception as exc:
                add("pyarrow_parquet_compatibility", False, str(exc))

    for row in checks:
        print(("PASS" if row["ok"] else "FAIL") + f"  {row['name']}: {row['detail']}")
    for row in notes:
        print(("PASS" if row["ok"] else "NOTE") + f"  {row['name']}: {row['detail']}")
    failed = [row for row in checks if not row["ok"]]
    if failed:
        print("\nSetup is incomplete. See evaluation_platform/README.md.")
        return 2
    if args.implementation_only:
        print("\nCommitted multi-metric implementation is internally complete.")
    else:
        print("\nEvaluation platform local inputs are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
