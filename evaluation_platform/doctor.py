"""Check whether a fresh clone is ready to run imitation evaluation."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from .catalog import CATALOG_STATUS, DATASETS, MODELS, REPO_ROOT, public_catalog
from .registry import MetricRegistry


def main() -> int:
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

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
        add("metric_registry", "controller_imitation" in {x["metric_id"] for x in registry.public_items()}, "controller_imitation")
    except Exception as exc:
        add("metric_registry", False, str(exc))

    add("catalog_configured", bool(CATALOG_STATUS["configured"]), CATALOG_STATUS["catalog_path"])
    catalog = public_catalog()
    add("dataset_available", bool(catalog["datasets"]) and all(x["available"] for x in catalog["datasets"]), json.dumps(catalog["datasets"], ensure_ascii=False))
    add("model_available", bool(catalog["models"]) and all(x["available"] for x in catalog["models"]), json.dumps(catalog["models"], ensure_ascii=False))
    full_requested = any(x.get("full_replay_available") for x in catalog["datasets"])
    if full_requested:
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
    failed = [row for row in checks if not row["ok"]]
    if failed:
        print("\nSetup is incomplete. See evaluation_platform/README.md.")
        return 2
    print("\nEvaluation platform setup is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
