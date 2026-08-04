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
        dataset_id = str(request.get("dataset_id", "shanghai_approach_2025_07_02"))
        model_id = str(request.get("model_id", "program_policy_v12_recalibrated"))
        mode = str(request.get("mode", "rescore_existing"))
        if metric_id != "controller_imitation":
            raise ValueError("controller_imitation is the only runnable metric in v0.1")
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
