"""Content-hash manifest for a finished job's artifact directory.

Used by worker.py at the end of a run to record what was produced (adapter,
merged model, logs) and let a caller verify nothing was altered afterward.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact_manifest(root: Path) -> Path:
    output = root / "artifact-manifest.json"
    entries = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != output
    ]
    # Write-then-rename so a crash mid-write never leaves a truncated manifest at the
    # final path; `Path.replace` is an atomic rename on both POSIX and Windows.
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps({"files": entries}, indent=2), encoding="utf-8")
    temporary.replace(output)
    return output
