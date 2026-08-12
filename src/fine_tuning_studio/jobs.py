from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fine_tuning_studio.domain import JobStatus, RunManifest, ensure_within


def studio_home() -> Path:
    configured = os.environ.get("FINE_TUNING_STUDIO_HOME")
    root = Path(configured).expanduser() if configured else Path.home() / ".fine-tuning-studio"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _database() -> sqlite3.Connection:
    connection = sqlite3.connect(studio_home() / "studio.db")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            stage TEXT NOT NULL,
            progress REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            pid INTEGER,
            error TEXT,
            manifest_path TEXT NOT NULL
        )
        """
    )
    return connection


def job_directory(job_id: str) -> Path:
    root = studio_home() / "jobs"
    root.mkdir(exist_ok=True)
    path = ensure_within(root, root / job_id)
    path.mkdir(exist_ok=True)
    return path


def create_job(
    manifest: RunManifest, upload_name: str | None = None, upload: bytes | None = None
) -> Path:
    directory = job_directory(manifest.id)
    if upload_name and upload is not None:
        safe_name = Path(upload_name).name
        input_path = ensure_within(directory, directory / "inputs" / safe_name)
        input_path.parent.mkdir(exist_ok=True)
        input_path.write_bytes(upload)
        raw = manifest.to_dict()
        raw["dataset"]["location"] = str(input_path)
        manifest = RunManifest.from_dict(raw)
    if manifest.training.reward_module:
        from fine_tuning_studio.recipes import copy_trusted_reward

        copied, digest = copy_trusted_reward(Path(manifest.training.reward_module), directory)
        manifest = replace(
            manifest,
            training=replace(manifest.training, reward_module=str(copied)),
            provenance=replace(
                manifest.provenance,
                package_versions={
                    **manifest.provenance.package_versions,
                    "trusted_reward_sha256": digest,
                },
            ),
        )
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    now = datetime.now(UTC).isoformat()
    with _database() as connection:
        connection.execute(
            "INSERT INTO jobs(id,status,stage,created_at,updated_at,manifest_path) "
            "VALUES(?,?,?,?,?,?)",
            (manifest.id, JobStatus.QUEUED, "Queued", now, now, str(manifest_path)),
        )
    return manifest_path


def resume_job(parent_job_id: str, checkpoint: Path) -> RunManifest:
    parent = get_job(parent_job_id)
    if not parent:
        raise ValueError(f"Unknown parent job: {parent_job_id}")
    parent_manifest = RunManifest.from_dict(
        json.loads(Path(parent["manifest_path"]).read_text(encoding="utf-8"))
    )
    checkpoint = checkpoint.resolve()
    checkpoints_root = (job_directory(parent_job_id) / "checkpoints").resolve()
    if checkpoint != checkpoints_root and checkpoints_root not in checkpoint.parents:
        raise ValueError("Resume checkpoint must belong to the parent job.")
    if not checkpoint.is_dir():
        raise ValueError("Resume checkpoint does not exist.")
    child = replace(
        parent_manifest,
        id=str(uuid4()),
        schema_version=2,
        created_at=datetime.now(UTC).isoformat(),
        parent_job_id=parent_job_id,
        training=replace(parent_manifest.training, resume_checkpoint=str(checkpoint)),
    )
    create_job(child)
    return child


def update_job(job_id: str, **fields: Any) -> None:
    allowed = {"status", "stage", "progress", "pid", "error"}
    values = {key: value for key, value in fields.items() if key in allowed}
    values["updated_at"] = datetime.now(UTC).isoformat()
    assignments = ",".join(f"{key}=?" for key in values)
    with _database() as connection:
        connection.execute(
            f"UPDATE jobs SET {assignments} WHERE id=?",
            (*values.values(), job_id),
        )


def get_job(job_id: str) -> dict[str, Any] | None:
    with _database() as connection:
        row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with _database() as connection:
        rows = connection.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def launch_job(job_id: str) -> int:
    active = {
        JobStatus.PREPARING,
        JobStatus.EVALUATING_BEFORE,
        JobStatus.TRAINING,
        JobStatus.EVALUATING_AFTER,
        JobStatus.EXPORTING,
    }
    if any(row["status"] in active for row in list_jobs() if row["id"] != job_id):
        raise RuntimeError("Another training job is already active.")
    job = get_job(job_id)
    if not job:
        raise ValueError(f"Unknown job: {job_id}")
    manifest = RunManifest.from_dict(
        json.loads(Path(job["manifest_path"]).read_text(encoding="utf-8"))
    )
    interpreter = Path(sys.executable)
    if manifest.training.runtime_profile != "current":
        from fine_tuning_studio.runtimes import profile_python

        managed = profile_python(studio_home(), manifest.training.runtime_profile)
        if managed.exists():
            interpreter = managed
    process = subprocess.Popen(
        [str(interpreter), "-m", "fine_tuning_studio.worker", job_id],
        cwd=Path.cwd(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        start_new_session=os.name != "nt",
    )
    update_job(job_id, pid=process.pid)
    return process.pid


def request_cancel(job_id: str) -> None:
    (job_directory(job_id) / "cancel.requested").touch()
    update_job(job_id, status=JobStatus.CANCELLING, stage="Cancellation requested")
