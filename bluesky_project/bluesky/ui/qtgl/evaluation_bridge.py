"""Non-blocking bridge from the PyQt simulator to the local evaluation service."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from urllib import error, request


class EvaluationEventBridge:
    """Send selected simulator events without blocking the Qt main thread."""

    def __init__(self, scenario: str):
        self.base_url = os.environ.get("ATC_EVAL_URL", "").rstrip("/")
        self.scenario = scenario
        self.session_id = None
        self.pending = queue.Queue(maxsize=1000)
        self.retry_after = 0.0
        self.worker = None
        if self.base_url:
            self.worker = threading.Thread(target=self._work, name="atc-evaluation-bridge", daemon=True)
            self.worker.start()

    @property
    def enabled(self):
        return bool(self.base_url)

    def publish(self, record):
        if not self.enabled:
            return
        try:
            self.pending.put_nowait(dict(record))
        except queue.Full:
            # Simulator responsiveness is more important than telemetry completeness.
            pass

    def _post(self, path, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=0.8) as response:
            return json.loads(response.read().decode("utf-8"))

    def _ensure_session(self):
        if self.session_id:
            return True
        if time.monotonic() < self.retry_after:
            return False
        try:
            created = self._post(
                "/api/simulation/sessions",
                {"scenario": self.scenario, "source": "bluesky_pyqt"},
            )
            self.session_id = created["session_id"]
            return True
        except (OSError, ValueError, KeyError, error.URLError):
            self.retry_after = time.monotonic() + 5.0
            return False

    def _work(self):
        while True:
            record = self.pending.get()
            if not self._ensure_session():
                continue
            try:
                self._post(
                    "/api/simulation/sessions/%s/events" % self.session_id,
                    record,
                )
            except (OSError, ValueError, error.URLError):
                self.session_id = None
                self.retry_after = time.monotonic() + 5.0
