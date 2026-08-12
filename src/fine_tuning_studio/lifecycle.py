from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import psutil

from fine_tuning_studio.domain import JobStatus
from fine_tuning_studio.jobs import get_job, list_jobs, request_cancel, update_job

ACTIVE_JOB_STATUSES = {
    JobStatus.PREPARING,
    JobStatus.EVALUATING_BEFORE,
    JobStatus.TRAINING,
    JobStatus.EVALUATING_AFTER,
    JobStatus.EXPORTING,
    JobStatus.CANCELLING,
}


@dataclass(frozen=True)
class ShutdownReport:
    requested_jobs: tuple[str, ...]
    stopped_pids: tuple[int, ...]
    interrupted_jobs: tuple[str, ...]
    errors: tuple[str, ...]


def _process_is_running(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.Error:
        return True


def _registered_worker(process: psutil.Process, job_id: str, application_pid: int) -> bool:
    try:
        if any(parent.pid == application_pid for parent in process.parents()):
            return True
        command = " ".join(process.cmdline()).lower()
    except psutil.Error:
        return False
    return job_id.lower() in command and (
        "fine_tuning_studio.worker" in command or "accelerate.commands.launch" in command
    )


def _terminate_tree(process: psutil.Process, timeout: float) -> tuple[set[int], set[int]]:
    try:
        descendants = process.children(recursive=True)
    except psutil.Error as exc:
        raise RuntimeError(f"cannot inspect PID {process.pid}: {type(exc).__name__}") from exc
    targets = [*reversed(descendants), process]
    for target in targets:
        try:
            target.terminate()
        except psutil.NoSuchProcess:
            pass
        except psutil.Error:
            continue
    _, alive = psutil.wait_procs(targets, timeout=timeout)
    for target in alive:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.Error:
            continue
    _, survivors = psutil.wait_procs(alive, timeout=timeout)
    survivor_pids = {target.pid for target in survivors}
    stopped_pids = {target.pid for target in targets} - survivor_pids
    return stopped_pids, survivor_pids


def stop_application_processes(
    *, grace_period: float = 10.0, terminate_timeout: float = 3.0
) -> ShutdownReport:
    application = psutil.Process(os.getpid())
    active_jobs = [job for job in list_jobs(limit=1000) if job["status"] in ACTIVE_JOB_STATUSES]
    requested = tuple(str(job["id"]) for job in active_jobs)
    pending: dict[str, int] = {}
    interrupted: set[str] = set()
    stopped_pids: set[int] = set()
    errors: list[str] = []

    for job in active_jobs:
        job_id = str(job["id"])
        request_cancel(job_id)
        if job.get("pid"):
            pending[job_id] = int(job["pid"])
        else:
            update_job(
                job_id,
                status=JobStatus.INTERRUPTED,
                stage="Stopped during application shutdown",
                pid=None,
            )
            interrupted.add(job_id)

    deadline = time.monotonic() + max(0.0, grace_period)
    while pending and time.monotonic() < deadline:
        for job_id, pid in list(pending.items()):
            if _process_is_running(pid):
                continue
            pending.pop(job_id)
            current = get_job(job_id)
            if current and current["status"] == JobStatus.CANCELLING:
                update_job(
                    job_id,
                    status=JobStatus.INTERRUPTED,
                    stage="Worker exited during application shutdown",
                    pid=None,
                )
                interrupted.add(job_id)
        if pending:
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))

    for job_id, pid in pending.items():
        try:
            process = psutil.Process(pid)
            if not _registered_worker(process, job_id, application.pid):
                errors.append(f"Job {job_id}: PID {pid} could not be verified as an app worker.")
                continue
            stopped, survivors = _terminate_tree(process, terminate_timeout)
            stopped_pids.update(stopped)
            if survivors:
                errors.append(
                    f"Job {job_id}: worker PIDs still running: "
                    + ", ".join(str(value) for value in sorted(survivors))
                )
                continue
            update_job(
                job_id,
                status=JobStatus.INTERRUPTED,
                stage="Stopped during application shutdown",
                pid=None,
            )
            interrupted.add(job_id)
        except psutil.NoSuchProcess:
            update_job(
                job_id,
                status=JobStatus.INTERRUPTED,
                stage="Worker exited during application shutdown",
                pid=None,
            )
            interrupted.add(job_id)
        except (OSError, psutil.Error, RuntimeError) as exc:
            errors.append(f"Job {job_id}: {exc}")

    try:
        descendants = application.children(recursive=True)
    except psutil.Error as exc:
        errors.append(f"App child processes could not be inspected: {type(exc).__name__}.")
        descendants = []
    if descendants:
        roots: list[psutil.Process] = []
        for process in descendants:
            try:
                if process.ppid() == application.pid:
                    roots.append(process)
            except psutil.NoSuchProcess:
                continue
            except psutil.Error as exc:
                errors.append(
                    f"App child PID {process.pid} could not be verified: {type(exc).__name__}."
                )
        for process in roots:
            try:
                stopped, survivors = _terminate_tree(process, terminate_timeout)
                stopped_pids.update(stopped)
                if survivors:
                    errors.append(
                        "App-owned PIDs still running: "
                        + ", ".join(str(value) for value in sorted(survivors))
                    )
            except (OSError, psutil.Error, RuntimeError) as exc:
                errors.append(f"App child PID {process.pid}: {exc}")

    return ShutdownReport(
        requested_jobs=requested,
        stopped_pids=tuple(sorted(stopped_pids)),
        interrupted_jobs=tuple(sorted(interrupted)),
        errors=tuple(errors),
    )


def schedule_clean_exit(delay: float = 1.0) -> None:
    timer = threading.Timer(delay, os._exit, args=(0,))
    timer.daemon = True
    timer.start()
