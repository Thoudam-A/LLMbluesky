"""Background execution and artifact management for metric runs."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from .catalog import DATASETS, MODELS, REPO_ROOT


VALID_MODES = {"archived_result", "rescore_existing", "full_replay"}
IMITATION_METRIC_IDS = {"controller_imitation"}
INTENT_METRIC_IDS = {"controller_intent_understanding_accuracy"}
HPPO_METRIC_IDS = {
    "dynamic_separation_adjustment",
    "command_execution_acceptance",
    "command_acceptability_proxy",
    "autonomous_command_response_time",
}
HPPO_OUTPUT_ROOT = REPO_ROOT / "output" / "H_PPO"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class RunManager:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._runs: dict[str, dict[str, Any]] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._lock = threading.RLock()

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        metric_id = str(request.get("metric_id", "controller_imitation"))
        if metric_id in HPPO_METRIC_IDS:
            return self._create_hppo_run(metric_id, request)
        if metric_id in INTENT_METRIC_IDS:
            return self._create_intent_run(metric_id, request)
        dataset_id = str(request.get("dataset_id", "shanghai_approach_2025_07_02"))
        model_id = str(request.get("model_id", "program_policy_v12_recalibrated"))
        mode = str(request.get("mode", "rescore_existing"))
        if metric_id not in IMITATION_METRIC_IDS:
            raise ValueError(f"unsupported metric_id: {metric_id}")
        if dataset_id not in DATASETS:
            raise ValueError(f"unknown dataset_id: {dataset_id}")
        if model_id not in MODELS:
            raise ValueError(f"unknown model_id: {model_id}")
        if mode not in VALID_MODES:
            raise ValueError(f"unknown mode: {mode}")

        scorer = request.get("scorer") or {}
        window = float(scorer.get("match_window_sec", 60.0))
        if not 1 <= window <= 600:
            raise ValueError("match_window_sec must be between 1 and 600")
        max_ticks = request.get("max_ticks")
        if max_ticks is not None:
            max_ticks = int(max_ticks)
            if not 1 <= max_ticks <= 21600:
                raise ValueError("max_ticks must be between 1 and 21600")
        run_id = f"EV-{dt.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6].upper()}"
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True)
        config = {
            "run_id": run_id,
            "metric_id": metric_id,
            "dataset_id": dataset_id,
            "model_id": model_id,
            "mode": mode,
            "max_ticks": max_ticks,
            "scorer": {
                "match_window_sec": window,
                "families": str(scorer.get("families", "altitude,speed")),
                "reference_scope": str(scorer.get("reference_scope", "inside_sector")),
                "min_state_altitude_m": float(scorer.get("min_state_altitude_m", 1400)),
                "max_state_altitude_m": float(scorer.get("max_state_altitude_m", 6200)),
            },
            "created_at": utc_now(),
        }
        (run_dir / "run_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        state = {
            "run_id": run_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "message": "任务已创建",
            "created_at": config["created_at"],
            "updated_at": config["created_at"],
            "config": config,
            "result_available": False,
            "error": None,
        }
        with self._lock:
            self._runs[run_id] = state
        self._persist_state(run_id)
        threading.Thread(target=self._execute, args=(run_id,), daemon=True).start()
        return self.get(run_id)

    def _create_intent_run(self, metric_id: str, request: dict[str, Any]) -> dict[str, Any]:
        dataset_id = str(request.get("dataset_id", ""))
        model_id = str(request.get("model_id", ""))
        if dataset_id not in DATASETS:
            raise ValueError(f"unknown dataset_id: {dataset_id}")
        if model_id not in MODELS:
            raise ValueError(f"unknown model_id: {model_id}")
        run_id = f"EV-{dt.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6].upper()}"
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True)
        config = {
            "run_id": run_id,
            "metric_id": metric_id,
            "dataset_id": dataset_id,
            "model_id": model_id,
            "mode": "rescore_intent_predictions",
            "created_at": utc_now(),
        }
        (run_dir / "run_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        state = {
            "run_id": run_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "message": "意图分类评估任务已创建",
            "created_at": config["created_at"],
            "updated_at": config["created_at"],
            "config": config,
            "result_available": False,
            "error": None,
        }
        with self._lock:
            self._runs[run_id] = state
        self._persist_state(run_id)
        threading.Thread(target=self._execute, args=(run_id,), daemon=True).start()
        return self.get(run_id)

    def _create_hppo_run(self, metric_id: str, request: dict[str, Any]) -> dict[str, Any]:
        source_value = request.get("source_run")
        if not source_value:
            raise ValueError("source_run is required for H-PPO metrics")
        source_run = Path(str(source_value)).expanduser().resolve()
        try:
            source_run.relative_to(HPPO_OUTPUT_ROOT.resolve())
        except ValueError as exc:
            raise ValueError("source_run must be inside output/H_PPO") from exc
        if not source_run.is_dir():
            raise FileNotFoundError(f"H-PPO run directory does not exist: {source_run}")
        events = source_run / "events.jsonl"
        if not events.exists():
            raise FileNotFoundError(f"events.jsonl is required: {events}")
        run_id = f"EV-{dt.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6].upper()}"
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True)
        config = {
            "run_id": run_id,
            "metric_id": metric_id,
            "mode": "hppo_archived_run",
            "source_run": str(source_run),
            "created_at": utc_now(),
        }
        (run_dir / "run_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        state = {
            "run_id": run_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "message": "H-PPO 离线评估任务已创建",
            "created_at": config["created_at"],
            "updated_at": config["created_at"],
            "config": config,
            "result_available": False,
            "error": None,
        }
        with self._lock:
            self._runs[run_id] = state
        self._persist_state(run_id)
        threading.Thread(target=self._execute, args=(run_id,), daemon=True).start()
        return self.get(run_id)

    def get(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            if run_id not in self._runs:
                state_path = self.output_root / run_id / "run_state.json"
                if not state_path.exists():
                    raise KeyError(run_id)
                self._runs[run_id] = json.loads(state_path.read_text(encoding="utf-8"))
            return dict(self._runs[run_id])

    def list_runs(self) -> list[dict[str, Any]]:
        for state_path in self.output_root.glob("*/run_state.json"):
            run_id = state_path.parent.name
            if run_id not in self._runs:
                try:
                    self._runs[run_id] = json.loads(state_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
        return sorted((dict(item) for item in self._runs.values()), key=lambda x: x["created_at"], reverse=True)

    def result(self, run_id: str) -> dict[str, Any]:
        self.get(run_id)
        path = self.output_root / run_id / "metric_result.json"
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def artifacts(self, run_id: str) -> list[dict[str, Any]]:
        self.get(run_id)
        run_dir = self.output_root / run_id
        return [
            {"name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(run_dir.iterdir()) if path.is_file()
        ]

    def cancel(self, run_id: str) -> dict[str, Any]:
        state = self.get(run_id)
        if state["status"] in {"complete", "failed", "cancelled"}:
            return state
        with self._lock:
            process = self._processes.get(run_id)
            if process and process.poll() is None:
                process.terminate()
            self._set(run_id, status="cancelled", stage="cancelled", message="任务已取消")
        return self.get(run_id)

    def _set(self, run_id: str, **updates: Any) -> None:
        with self._lock:
            self._runs[run_id].update(updates)
            self._runs[run_id]["updated_at"] = utc_now()
        self._persist_state(run_id)

    def _persist_state(self, run_id: str) -> None:
        path = self.output_root / run_id / "run_state.json"
        path.write_text(json.dumps(self._runs[run_id], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _command(self, run_id: str, stage: str, progress: int, args: list[str], extra_env: dict[str, str] | None = None) -> None:
        if self.get(run_id)["status"] == "cancelled":
            raise InterruptedError("run cancelled")
        self._set(run_id, status="running", stage=stage, progress=progress, message=f"正在执行：{stage}")
        log_path = self.output_root / run_id / f"{stage}.log"
        with log_path.open("w", encoding="utf-8") as log:
            child_env = os.environ.copy()
            if extra_env:
                child_env.update(extra_env)
            process = subprocess.Popen(
                args, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", env=child_env,
            )
            with self._lock:
                self._processes[run_id] = process
            code = process.wait()
        with self._lock:
            self._processes.pop(run_id, None)
        if self.get(run_id)["status"] == "cancelled":
            raise InterruptedError("run cancelled")
        if code != 0:
            raise RuntimeError(f"{stage} failed with exit code {code}; see {log_path.name}")

    def _execute(self, run_id: str) -> None:
        try:
            config = self.get(run_id)["config"]
            if config["metric_id"] in HPPO_METRIC_IDS:
                self._execute_hppo_metric(run_id, config)
                return
            if config["metric_id"] in INTENT_METRIC_IDS:
                self._execute_intent_metric(run_id, config)
                return
            dataset = DATASETS[config["dataset_id"]]
            model = MODELS[config["model_id"]]
            run_dir = self.output_root / run_id
            result_path = run_dir / "metric_result.json"
            mode = config["mode"]
            self._set(run_id, status="running", stage="validating", progress=5, message="正在校验输入")
            required = [dataset.get("references")]
            if mode == "archived_result":
                required.append(model.get("archived_metric"))
            if mode == "rescore_existing":
                required.append(model.get("system_outputs"))
            if mode == "full_replay":
                required.extend([dataset.get("replay"), dataset.get("trajectory_parquet"), model.get("model"), model.get("config")])
            missing = [str(path) for path in required if not path or not Path(path).exists()]
            if missing:
                raise FileNotFoundError("missing required inputs: " + ", ".join(missing))

            input_manifest = {
                "dataset_id": config["dataset_id"], "model_id": config["model_id"],
                "inputs": [
                    {"role": role, "path": str(Path(path).resolve()), "size": Path(path).stat().st_size}
                    for role, path in {
                        "references": dataset.get("references"), "replay": dataset.get("replay"),
                        "trajectory_parquet": dataset.get("trajectory_parquet"), "model": model.get("model"),
                        "config": model.get("config"), "existing_outputs": model.get("system_outputs"),
                    }.items() if path and Path(path).exists()
                ],
                "note": "Large immutable source hashes are retained in their upstream manifests; run artifacts are hashed locally."
            }
            (run_dir / "input_manifest.json").write_text(
                json.dumps(input_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

            if mode == "archived_result":
                self._set(run_id, stage="loading_archived", progress=55, message="正在加载已固化结果")
                shutil.copy2(model["archived_metric"], result_path)
            else:
                outputs = Path(model["system_outputs"])
                if mode == "full_replay":
                    decisions = run_dir / "decision_events.jsonl"
                    outputs = run_dir / "system_outputs.jsonl"
                    replay_args = [
                        sys.executable, str(REPO_ROOT / "evaluation_scripts/run_shanghai_program_policy_replay.py"),
                        "--replay", str(dataset["replay"]), "--trajectory-parquet", str(dataset["trajectory_parquet"]),
                        "--model", str(model["model"]), "--config", str(model["config"]),
                        "--output-log", str(decisions), "--local-date", str(dataset["local_date"]),
                        "--decision-threshold", str(model["decision_threshold"]),
                    ]
                    if config.get("max_ticks") is not None:
                        replay_args.extend(["--max-ticks", str(config["max_ticks"])])
                    compat_candidates = [
                        Path(os.environ["ATC_PYARROW_COMPAT_PATH"]) if os.environ.get("ATC_PYARROW_COMPAT_PATH") else None,
                        REPO_ROOT / ".deps/pyarrow20",
                        REPO_ROOT.parent / ".deps/pyarrow20",
                    ]
                    compat_path = next((path for path in compat_candidates if path and path.exists()), None)
                    replay_env = None
                    if compat_path is not None:
                        prior = os.environ.get("PYTHONPATH", "")
                        replay_env = {"PYTHONPATH": str(compat_path) + (os.pathsep + prior if prior else "")}
                    self._command(run_id, "frozen_replay", 15, replay_args, replay_env)
                    self._command(run_id, "normalizing", 70, [
                        sys.executable, str(REPO_ROOT / "evaluation_scripts/convert_decision_log_for_imitation.py"),
                        "--input", str(decisions), "--output", str(outputs),
                    ])
                scorer = config["scorer"]
                self._command(run_id, "scoring", 82, [
                    sys.executable, str(REPO_ROOT / "evaluation_scripts/score_seu_imitation.py"),
                    "--references", str(dataset["references"]), "--system-outputs", str(outputs),
                    "--output", str(result_path), "--match-window-sec", str(scorer["match_window_sec"]),
                    "--system-name", config["model_id"], "--families", scorer["families"],
                    "--reference-scope", scorer["reference_scope"],
                    "--min-state-altitude-m", str(scorer["min_state_altitude_m"]),
                    "--max-state-altitude-m", str(scorer["max_state_altitude_m"]),
                ])

            manifest = {"run_id": run_id, "generated_at": utc_now(), "artifacts": []}
            for path in sorted(run_dir.iterdir()):
                if path.is_file() and path.name != "run_state.json":
                    manifest["artifacts"].append({"name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)})
            (run_dir / "hashes.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            self._set(run_id, status="complete", stage="complete", progress=100, message="评估完成", result_available=True)
        except InterruptedError:
            self._set(run_id, status="cancelled", stage="cancelled", message="任务已取消")
        except Exception as exc:  # keep worker failure visible to the UI
            self._set(run_id, status="failed", stage="failed", message="评估失败", error=str(exc))

    def _execute_intent_metric(self, run_id: str, config: dict[str, Any]) -> None:
        dataset = DATASETS[config["dataset_id"]]
        model = MODELS[config["model_id"]]
        run_dir = self.output_root / run_id
        inputs = {
            "controller_instructions": dataset.get("controller_instructions"),
            "intent_reference_events": dataset.get("intent_references"),
            "intent_predictions": model.get("intent_predictions"),
        }
        self._set(run_id, status="running", stage="validating", progress=10, message="正在校验意图分类输入")
        missing = [role for role, path in inputs.items() if not path or not Path(path).exists()]
        if missing:
            raise FileNotFoundError("missing required intent inputs: " + ", ".join(missing))
        manifest = {
            "dataset_id": config["dataset_id"],
            "model_id": config["model_id"],
            "label_isolation": "intent_reference_events are read only by the scorer",
            "inputs": [
                {"role": role, "path": str(Path(path).resolve()), "size": Path(path).stat().st_size}
                for role, path in inputs.items()
            ],
        }
        (run_dir / "input_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        scorer = REPO_ROOT / "evaluation_platform" / "metrics" / "controller_intent_understanding" / "scorer.py"
        self._command(run_id, "scoring", 75, [
            sys.executable,
            str(scorer),
            "--instructions", str(inputs["controller_instructions"]),
            "--references", str(inputs["intent_reference_events"]),
            "--predictions", str(inputs["intent_predictions"]),
            "--output", str(run_dir / "metric_result.json"),
        ])
        hashes = {"run_id": run_id, "generated_at": utc_now(), "artifacts": []}
        for path in sorted(run_dir.iterdir()):
            if path.is_file() and path.name != "run_state.json":
                hashes["artifacts"].append({
                    "name": path.name,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
        (run_dir / "hashes.json").write_text(
            json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        self._set(
            run_id,
            status="complete",
            stage="complete",
            progress=100,
            message="意图分类评估完成",
            result_available=True,
        )

    def _execute_hppo_metric(self, run_id: str, config: dict[str, Any]) -> None:
        source_run = Path(config["source_run"])
        run_dir = self.output_root / run_id
        events = source_run / "events.jsonl"
        diagnostics = next(
            (source_run / name for name in ("validation_diagnostics.csv", "training_diagnostics.csv")
             if (source_run / name).exists()),
            None,
        )
        self._set(run_id, status="running", stage="validating", progress=10, message="正在校验 H-PPO 运行日志")
        manifest = {
            "source_run": str(source_run),
            "inputs": [{"role": "events", "path": str(events), "size": events.stat().st_size}],
        }
        if diagnostics is not None:
            manifest["inputs"].append({"role": "diagnostics", "path": str(diagnostics), "size": diagnostics.stat().st_size})
        (run_dir / "input_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        metric_id = config["metric_id"]
        scorer = REPO_ROOT / "evaluation_platform" / "metrics" / metric_id / "scorer.py"
        args = [sys.executable, str(scorer), "--output", str(run_dir / "metric_result.json")]
        if metric_id == "autonomous_command_response_time":
            if diagnostics is None:
                raise FileNotFoundError("response-time scoring requires validation_diagnostics.csv or training_diagnostics.csv")
            args.extend(["--diagnostics", str(diagnostics)])
        else:
            args.extend(["--events", str(events)])
        if metric_id in {"dynamic_separation_adjustment", "command_acceptability_proxy"} and diagnostics is not None:
            # Formal interval-adjustment outcomes are self-contained runtime
            # events. Diagnostics remain useful supporting evidence and are
            # still consumed when available, but are not a hard prerequisite.
            args.extend(["--diagnostics", str(diagnostics)])
        self._command(run_id, "scoring", 75, args)
        hashes = {"run_id": run_id, "generated_at": utc_now(), "artifacts": []}
        for path in sorted(run_dir.iterdir()):
            if path.is_file() and path.name != "run_state.json":
                hashes["artifacts"].append({"name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)})
        (run_dir / "hashes.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._set(run_id, status="complete", stage="complete", progress=100, message="评估完成", result_available=True)

    @staticmethod
    def available_hppo_runs() -> list[dict[str, str]]:
        if not HPPO_OUTPUT_ROOT.exists():
            return []
        runs = []
        for events in HPPO_OUTPUT_ROOT.rglob("events.jsonl"):
            run_dir = events.parent
            diagnostics = next((run_dir / name for name in ("validation_diagnostics.csv", "training_diagnostics.csv") if (run_dir / name).exists()), None)
            runs.append({
                "path": str(run_dir),
                "name": str(run_dir.relative_to(HPPO_OUTPUT_ROOT)),
                "has_diagnostics": "true" if diagnostics else "false",
            })
        return sorted(runs, key=lambda row: row["path"], reverse=True)
