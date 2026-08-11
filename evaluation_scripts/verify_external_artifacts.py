#!/usr/bin/env python3
"""Verify a team ATC metric handoff directory against its committed manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="ATC_SHARED_ROOT directory")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/handoff/external_artifact_manifest_20260811.json"),
    )
    return parser.parse_args()


def verify(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for artifact in manifest.get("external_artifacts", []):
        relative = Path(str(artifact["relative_path"]))
        path = root / relative
        if not path.is_file():
            results.append({"name": artifact["logical_name"], "status": "missing"})
            continue
        size_ok = path.stat().st_size == int(artifact["bytes"])
        actual_hash = sha256_file(path)
        hash_ok = actual_hash == str(artifact["sha256"]).lower()
        results.append(
            {
                "name": artifact["logical_name"],
                "status": "ok" if size_ok and hash_ok else "mismatch",
                "size_ok": size_ok,
                "sha256_ok": hash_ok,
            }
        )
    return results


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = verify(args.root, manifest)
    print(json.dumps({"root": str(args.root.resolve()), "results": results}, ensure_ascii=False, indent=2))
    return 0 if results and all(row["status"] == "ok" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
