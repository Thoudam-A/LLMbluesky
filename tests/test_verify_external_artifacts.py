from __future__ import annotations

import hashlib
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "evaluation_scripts"
sys.path.insert(0, str(SCRIPTS))

from verify_external_artifacts import verify  # noqa: E402


def test_verify_accepts_matching_artifact_and_reports_missing(tmp_path: Path) -> None:
    payload = b"portable ATC handoff fixture\n"
    artifact = tmp_path / "data" / "fixture.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(payload)
    manifest = {
        "external_artifacts": [
            {
                "logical_name": "fixture",
                "relative_path": "data/fixture.bin",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            {
                "logical_name": "missing",
                "relative_path": "data/missing.bin",
                "bytes": 1,
                "sha256": "0" * 64,
            },
        ]
    }
    results = verify(tmp_path, manifest)
    assert results[0]["status"] == "ok"
    assert results[1]["status"] == "missing"
