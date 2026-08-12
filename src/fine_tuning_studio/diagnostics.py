from __future__ import annotations

import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from fine_tuning_studio.jobs import list_jobs
from fine_tuning_studio.storage import SCHEMA_VERSION, integrity_check
from fine_tuning_studio.system_info import scan_machine

SECRET = re.compile(r"(?i)(token|password|secret|authorization)([=: ]+)([^\s,]+)")


def redact(value: str) -> str:
    return SECRET.sub(r"\1\2[REDACTED]", value)


def build_diagnostics(home: Path) -> bytes:
    jobs = list_jobs(limit=1_000)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "database": integrity_check(home)[1],
        "job_status_counts": dict(Counter(str(job["status"]) for job in jobs)),
    }
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("system.json", scan_machine(home).to_json())
        archive.writestr("storage.json", json.dumps(summary, indent=2))
        for job in jobs[:5]:
            directory = home / "jobs" / str(job["id"])
            for name in ("stdout.log", "stderr.log", "events.jsonl"):
                path = directory / name
                if path.is_file():
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                    text = "\n".join(lines[-200:])
                    archive.writestr(f"logs/{job['id']}/{name}", redact(text))
    return memory.getvalue()
