from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
from collections.abc import Mapping
from contextlib import closing, suppress
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fine_tuning_studio.domain import JobStatus, RunManifest, ensure_within
from fine_tuning_studio.storage import connect


def studio_home() -> Path:
    configured = os.environ.get("FINE_TUNING_STUDIO_HOME")
    root = Path(configured).expanduser() if configured else Path.home() / ".fine-tuning-studio"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _database() -> sqlite3.Connection:
    return connect(studio_home())


def job_directory(job_id: str) -> Path:
    root = studio_home() / "jobs"
    root.mkdir(exist_ok=True)
    path = ensure_within(root, root / job_id)
    path.mkdir(exist_ok=True)
    return path


def create_job(
    manifest: RunManifest,
    uploads: Mapping[int, tuple[str, bytes]] | None = None,
) -> Path:
    directory = job_directory(manifest.id)
    staged_sources = list(manifest.dataset.sources)
    for index, (upload_name, upload) in (uploads or {}).items():
        if index < 0 or index >= len(staged_sources):
            raise ValueError(f"Unknown dataset upload index: {index}")
        if staged_sources[index].source != "local":
            raise ValueError(f"Dataset {index + 1} is not a local upload.")
        safe_name = Path(upload_name).name
        input_path = ensure_within(
            directory, directory / "inputs" / f"dataset-{index + 1}" / safe_name
        )
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_bytes(upload)
        staged_sources[index] = replace(staged_sources[index], location=str(input_path))
    manifest = replace(
        manifest,
        dataset=replace(manifest.dataset, sources=staged_sources),
    )
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
    with closing(_database()) as connection, connection:
        connection.execute(
            "INSERT INTO jobs(id,status,stage,created_at,updated_at,manifest_path,kind) "
            "VALUES(?,?,?,?,?,?,?)",
            (manifest.id, JobStatus.QUEUED, "Queued", now, now, str(manifest_path), "training"),
        )
    return manifest_path


def resume_job(parent_job_id: str, checkpoint: Path) -> RunManifest:
    parent = get_job(parent_job_id)
    if not parent:
        raise ValueError(f"Unknown parent job: {parent_job_id}")
    parent_manifest = RunManifest.from_dict(
        json.loads(Path(parent["manifest_path"]).read_text(encoding="utf-8"))
    )
    if parent_manifest.training.objective == "ppo":
        raise ValueError("PPO checkpoint resume is unavailable with the experimental trainer.")
    checkpoint = checkpoint.resolve()
    checkpoints_root = (job_directory(parent_job_id) / "checkpoints").resolve()
    if checkpoint != checkpoints_root and checkpoints_root not in checkpoint.parents:
        raise ValueError("Resume checkpoint must belong to the parent job.")
    if not checkpoint.is_dir():
        raise ValueError("Resume checkpoint does not exist.")
    child = replace(
        parent_manifest,
        id=str(uuid4()),
        schema_version=5,
        created_at=datetime.now(UTC).isoformat(),
        parent_job_id=parent_job_id,
        training=replace(parent_manifest.training, resume_checkpoint=str(checkpoint)),
    )
    create_job(child)
    return child


def update_job(job_id: str, **fields: Any) -> None:
    allowed = {"status", "stage", "progress", "pid", "error", "exit_code"}
    values = {key: value for key, value in fields.items() if key in allowed}
    values["updated_at"] = datetime.now(UTC).isoformat()
    assignments = ",".join(f"{key}=?" for key in values)
    with closing(_database()) as connection, connection:
        connection.execute(
            f"UPDATE jobs SET {assignments} WHERE id=?",
            (*values.values(), job_id),
        )


def get_job(job_id: str) -> dict[str, Any] | None:
    with closing(_database()) as connection:
        row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with closing(_database()) as connection:
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
    directory = job_directory(job_id)
    _rotate_logs(directory)
    command = [str(interpreter), "-m", "fine_tuning_studio.worker", job_id]
    environment = os.environ.copy()
    if manifest.training.device_indices:
        selected = ",".join(str(index) for index in manifest.training.device_indices)
        if manifest.training.runtime_profile == "xpu":
            environment["ZE_AFFINITY_MASK"] = selected
        else:
            environment["CUDA_VISIBLE_DEVICES"] = selected
    if manifest.training.world_size > 1:
        strategy = manifest.training.distributed_strategy
        if strategy == "auto":
            strategy = "ddp"
        config: dict[str, Any] = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "FSDP" if strategy == "fsdp2" else "MULTI_GPU",
            "machine_rank": 0,
            "main_training_function": "main",
            "num_machines": 1,
            "num_processes": manifest.training.world_size,
            "mixed_precision": (
                manifest.training.precision if manifest.training.precision != "auto" else "no"
            ),
        }
        if strategy == "fsdp2":
            config["fsdp_config"] = {
                "fsdp_version": 2,
                "fsdp_state_dict_type": "SHARDED_STATE_DICT",
                "fsdp_cpu_offload": False,
                "fsdp_activation_checkpointing": manifest.training.gradient_checkpointing,
            }
        config_path = directory / "accelerate-config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        command = [
            str(interpreter),
            "-m",
            "accelerate.commands.launch",
            "--config_file",
            str(config_path),
            "-m",
            "fine_tuning_studio.worker",
            job_id,
        ]
    stdout = (directory / "stdout.log").open("a", encoding="utf-8")
    stderr = (directory / "stderr.log").open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            stdout=stdout,
            stderr=stderr,
            env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
    finally:
        stdout.close()
        stderr.close()
    update_job(job_id, pid=process.pid)
    return process.pid


def _rotate_logs(directory: Path, limit: int = 10 * 1024 * 1024) -> None:
    for name in ("stdout.log", "stderr.log"):
        path = directory / name
        if path.is_file() and path.stat().st_size >= limit:
            older = path.with_name(name + ".1")
            older.unlink(missing_ok=True)
            path.replace(older)


def request_cancel(job_id: str) -> None:
    (job_directory(job_id) / "cancel.requested").touch()
    update_job(job_id, status=JobStatus.CANCELLING, stage="Cancellation requested")


def terminate_job(job_id: str) -> None:
    job = get_job(job_id)
    if not job or not job.get("pid"):
        request_cancel(job_id)
        return
    request_cancel(job_id)
    pid = int(job["pid"])
    try:
        import psutil

        process = psutil.Process(pid)
        for child in process.children(recursive=True):
            child.terminate()
        process.terminate()
    except (psutil.Error, OSError):
        if os.name != "nt":
            with suppress(OSError):
                os.killpg(pid, signal.SIGTERM)


def reconcile_jobs() -> int:
    active = {
        JobStatus.PREPARING,
        JobStatus.EVALUATING_BEFORE,
        JobStatus.TRAINING,
        JobStatus.EVALUATING_AFTER,
        JobStatus.EXPORTING,
        JobStatus.CANCELLING,
    }
    changed = 0
    for job in list_jobs(limit=1000):
        if job["status"] not in active:
            continue
        pid = job.get("pid")
        alive = False
        if pid:
            try:
                import psutil

                alive = psutil.pid_exists(int(pid))
            except (ValueError, TypeError):
                alive = False
        if not alive:
            update_job(job["id"], status=JobStatus.INTERRUPTED, stage="Worker interrupted")
            changed += 1
    return changed
