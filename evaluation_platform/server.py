"""Dependency-free local HTTP service for the ATC evaluation platform."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import uuid
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .catalog import CATALOG_STATUS, MODELS, REPO_ROOT, public_catalog
from .registry import MetricRegistry
from .run_manager import RunManager


PACKAGE_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = PACKAGE_ROOT / "static"
OUTPUT_ROOT = REPO_ROOT / "output/evaluation_runs"
SIM_ROOT = REPO_ROOT / "output/simulation_sessions"
MAX_BODY = 1024 * 1024


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class SimulationStore:
    """Append-only bridge for selected BlueSky/PyQt events."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}

    def create(self, payload: dict) -> dict:
        session_id = f"SIM-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6].upper()}"
        directory = self.root / session_id
        directory.mkdir()
        meta = {
            "session_id": session_id,
            "created_at": now_utc(),
            "scenario": payload.get("scenario", "unknown"),
            "source": payload.get("source", "bluesky_pyqt"),
            "event_count": 0,
        }
        (directory / "session.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._locks[session_id] = threading.Lock()
        return meta

    def append(self, session_id: str, event: dict) -> dict:
        directory = self.root / session_id
        meta_path = directory / "session.json"
        if not meta_path.exists():
            raise KeyError(session_id)
        lock = self._locks.setdefault(session_id, threading.Lock())
        with lock:
            row = dict(event)
            row.setdefault("received_at", now_utc())
            with (directory / "simulation_events.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["event_count"] = int(meta.get("event_count", 0)) + 1
            meta["updated_at"] = now_utc()
            meta["last_event"] = row.get("event")
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return meta

    def get(self, session_id: str) -> dict:
        path = self.root / session_id / "session.json"
        if not path.exists():
            raise KeyError(session_id)
        return json.loads(path.read_text(encoding="utf-8"))


class EvaluationHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler):
        super().__init__(address, handler)
        self.registry = MetricRegistry(PACKAGE_ROOT / "metrics")
        self.registry.load()
        self.runs = RunManager(OUTPUT_ROOT)
        self.simulations = SimulationStore(SIM_ROOT)


class Handler(BaseHTTPRequestHandler):
    server_version = "AtcEvaluation/0.2"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _json(self, payload, status=HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_BODY:
            raise ValueError("request body is too large")
        if length == 0:
            return {}
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _error(self, exc: Exception, status=HTTPStatus.BAD_REQUEST) -> None:
        self._json({"error": type(exc).__name__, "message": str(exc)}, status)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/health":
                return self._json({"status": "ok", "service": "atc-evaluation", "version": "0.2.0", "time": now_utc(), "catalog": CATALOG_STATUS})
            if path == "/api/catalog":
                return self._json(public_catalog())
            if path == "/api/metrics":
                return self._json({"items": self.server.registry.public_items()})
            if path == "/api/hppo/runs":
                return self._json({"items": self.server.runs.available_hppo_runs()})
            if path == "/api/runs":
                return self._json({"items": self.server.runs.list_runs()})
            if path == "/api/snapshot":
                source = next((item.get("archived_metric") for item in MODELS.values() if item.get("archived_metric")), None)
                if source is None or not Path(source).exists():
                    raise FileNotFoundError("no archived metric is configured")
                return self._json(json.loads(source.read_text(encoding="utf-8")))
            parts = [unquote(item) for item in path.strip("/").split("/") if item]
            if len(parts) >= 3 and parts[:2] == ["api", "runs"]:
                run_id = parts[2]
                if len(parts) == 3:
                    return self._json(self.server.runs.get(run_id))
                if len(parts) == 4 and parts[3] == "result":
                    return self._json(self.server.runs.result(run_id))
                if len(parts) == 4 and parts[3] == "artifacts":
                    return self._json({"items": self.server.runs.artifacts(run_id)})
            if len(parts) == 4 and parts[:3] == ["api", "simulation", "sessions"]:
                return self._json(self.server.simulations.get(parts[3]))
            return self._static(path)
        except KeyError as exc:
            self._error(exc, HTTPStatus.NOT_FOUND)
        except FileNotFoundError as exc:
            self._error(exc, HTTPStatus.CONFLICT)
        except Exception as exc:
            self._error(exc, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._body()
            if path == "/api/runs":
                return self._json(self.server.runs.create(payload), HTTPStatus.ACCEPTED)
            if path == "/api/simulation/sessions":
                return self._json(self.server.simulations.create(payload), HTTPStatus.CREATED)
            parts = [unquote(item) for item in path.strip("/").split("/") if item]
            if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "cancel":
                return self._json(self.server.runs.cancel(parts[2]))
            if len(parts) == 5 and parts[:3] == ["api", "simulation", "sessions"] and parts[4] == "events":
                return self._json(self.server.simulations.append(parts[3], payload), HTTPStatus.ACCEPTED)
            self._error(ValueError("unknown endpoint"), HTTPStatus.NOT_FOUND)
        except KeyError as exc:
            self._error(exc, HTTPStatus.NOT_FOUND)
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(exc, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._error(exc, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _static(self, path: str) -> None:
        relative = "index.html" if path in {"", "/"} else path.lstrip("/")
        target = (STATIC_ROOT / relative).resolve()
        try:
            target.relative_to(STATIC_ROOT.resolve())
        except ValueError:
            return self._error(ValueError("invalid static path"), HTTPStatus.FORBIDDEN)
        if not target.is_file():
            return self._error(FileNotFoundError(relative), HTTPStatus.NOT_FOUND)
        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the dashboard in the default browser.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost"} and os.environ.get("ATC_EVAL_ALLOW_REMOTE") != "1":
        raise SystemExit("Remote binding is disabled; set ATC_EVAL_ALLOW_REMOTE=1 explicitly.")
    server = EvaluationHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"ATC evaluation platform listening on {url}")
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
