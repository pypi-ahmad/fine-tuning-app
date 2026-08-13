from __future__ import annotations

from pathlib import Path
from typing import cast

import psutil
import pytest

from fine_tuning_studio.domain import (
    DatasetSourceSpec,
    DatasetSpec,
    EvaluationSpec,
    ExportSpec,
    ModelSpec,
    RunManifest,
    TrainingSpec,
)
from fine_tuning_studio.jobs import create_job, get_job, update_job
from fine_tuning_studio.lifecycle import (
    _registered_worker,
    schedule_clean_exit,
    stop_application_processes,
)


class FakeProcess:
    def __init__(
        self,
        pid: int,
        *,
        parents: tuple[FakeProcess, ...] = (),
        command: tuple[str, ...] = (),
        children: tuple[FakeProcess, ...] = (),
        parent_pid: int = 0,
    ) -> None:
        self.pid = pid
        self._parents = parents
        self._command = command
        self._children = children
        self._parent_pid = parent_pid

    def parents(self) -> list[FakeProcess]:
        return list(self._parents)

    def cmdline(self) -> list[str]:
        return list(self._command)

    def children(self, recursive: bool = False) -> list[FakeProcess]:
        return list(self._children)

    def ppid(self) -> int:
        return self._parent_pid


def _manifest() -> RunManifest:
    return RunManifest(
        dataset=DatasetSpec(sources=[DatasetSourceSpec(source="hub", location="org/data")]),
        model=ModelSpec(source="hub", location="org/model"),
        training=TrainingSpec(),
        evaluation=EvaluationSpec(),
        export=ExportSpec(),
    )


def test_registered_worker_requires_ancestry_or_matching_command() -> None:
    application = FakeProcess(10)
    assert _registered_worker(
        cast(psutil.Process, FakeProcess(20, parents=(application,))), "job-1", 10
    )
    assert _registered_worker(
        cast(
            psutil.Process,
            FakeProcess(30, command=("python", "-m", "fine_tuning_studio.worker", "job-1")),
        ),
        "job-1",
        10,
    )
    assert not _registered_worker(
        cast(psutil.Process, FakeProcess(40, command=("python", "unrelated.py"))),
        "job-1",
        10,
    )


def test_shutdown_marks_active_job_without_pid_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FINE_TUNING_STUDIO_HOME", str(tmp_path))
    manifest = _manifest()
    create_job(manifest)
    update_job(manifest.id, status="training")
    monkeypatch.setattr("fine_tuning_studio.lifecycle.psutil.Process.children", lambda *_, **__: [])

    report = stop_application_processes(grace_period=0)

    job = get_job(manifest.id)
    assert job is not None
    assert job["status"] == "interrupted"
    assert report.requested_jobs == (manifest.id,)
    assert report.interrupted_jobs == (manifest.id,)
    assert not report.errors


def test_shutdown_leaves_completed_and_queued_jobs_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FINE_TUNING_STUDIO_HOME", str(tmp_path))
    queued = _manifest()
    completed = _manifest()
    create_job(queued)
    create_job(completed)
    update_job(completed.id, status="completed")
    monkeypatch.setattr("fine_tuning_studio.lifecycle.psutil.Process.children", lambda *_, **__: [])

    report = stop_application_processes(grace_period=0)

    assert report.requested_jobs == ()
    queued_job = get_job(queued.id)
    completed_job = get_job(completed.id)
    assert queued_job is not None and queued_job["status"] == "queued"
    assert completed_job is not None and completed_job["status"] == "completed"


def test_shutdown_forces_verified_worker_and_marks_it_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FINE_TUNING_STUDIO_HOME", str(tmp_path))
    manifest = _manifest()
    create_job(manifest)
    update_job(manifest.id, status="training", pid=200)
    application = FakeProcess(100)
    worker = FakeProcess(200, parents=(application,))
    monkeypatch.setattr("fine_tuning_studio.lifecycle.os.getpid", lambda: 100)
    monkeypatch.setattr(
        "fine_tuning_studio.lifecycle.psutil.Process",
        lambda pid: {100: application, 200: worker}[pid],
    )
    monkeypatch.setattr("fine_tuning_studio.lifecycle._terminate_tree", lambda *_: ({200}, set()))

    report = stop_application_processes(grace_period=0)

    job = get_job(manifest.id)
    assert job is not None
    assert job["status"] == "interrupted"
    assert job["pid"] is None
    assert report.stopped_pids == (200,)
    assert not report.errors


def test_shutdown_refuses_unverified_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FINE_TUNING_STUDIO_HOME", str(tmp_path))
    manifest = _manifest()
    create_job(manifest)
    update_job(manifest.id, status="training", pid=200)
    application = FakeProcess(100)
    unrelated = FakeProcess(200, command=("python", "unrelated.py"))
    monkeypatch.setattr("fine_tuning_studio.lifecycle.os.getpid", lambda: 100)
    monkeypatch.setattr(
        "fine_tuning_studio.lifecycle.psutil.Process",
        lambda pid: {100: application, 200: unrelated}[pid],
    )

    report = stop_application_processes(grace_period=0)

    assert report.errors == (f"Job {manifest.id}: PID 200 could not be verified as an app worker.",)
    job = get_job(manifest.id)
    assert job is not None and job["status"] == "cancelling"


def test_shutdown_stops_other_direct_app_descendants(monkeypatch: pytest.MonkeyPatch) -> None:
    child = FakeProcess(101, parent_pid=100)
    application = FakeProcess(100, children=(child,))
    monkeypatch.setattr("fine_tuning_studio.lifecycle.os.getpid", lambda: 100)
    monkeypatch.setattr("fine_tuning_studio.lifecycle.list_jobs", lambda **_: [])
    monkeypatch.setattr("fine_tuning_studio.lifecycle.psutil.Process", lambda _: application)
    monkeypatch.setattr("fine_tuning_studio.lifecycle._terminate_tree", lambda *_: ({101}, set()))

    report = stop_application_processes(grace_period=0)

    assert report.stopped_pids == (101,)
    assert not report.errors


def test_schedule_clean_exit_uses_success_status(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeTimer:
        daemon = False

        def __init__(self, delay: float, function: object, args: tuple[int, ...]) -> None:
            captured.update(delay=delay, function=function, args=args)

        def start(self) -> None:
            captured["started"] = True

    monkeypatch.setattr("fine_tuning_studio.lifecycle.threading.Timer", FakeTimer)

    schedule_clean_exit(0.5)

    assert captured["delay"] == 0.5
    assert captured["args"] == (0,)
    assert captured["started"] is True
