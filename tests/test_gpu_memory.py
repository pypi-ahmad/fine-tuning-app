from dataclasses import replace
from types import SimpleNamespace

import pytest

from fine_tuning_studio.gpu_memory import (
    GpuProcess,
    inspect_gpu_memory,
    process_protection_reason,
    release_reclaimable_memory,
    terminate_gpu_process,
)
from fine_tuning_studio.ollama import OllamaRunningModel


def completed(stdout: str, returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)


def test_inspects_nvidia_memory_and_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_which(command: str) -> str | None:
        return command if command == "nvidia-smi" else None

    def fake_run(command: list[str], timeout: float = 4) -> SimpleNamespace:
        query = next(part for part in command if part.startswith("--query-"))
        if query.startswith("--query-gpu"):
            return completed("0, GPU-abc, RTX 4060, 8188, 6543, 1645\n")
        return completed(
            "GPU-abc, 123, C:\\Python\\python.exe, 2048\n"
            "GPU-abc, 4, [Insufficient Permissions], [N/A]\n"
        )

    monkeypatch.setattr("fine_tuning_studio.gpu_memory.shutil.which", fake_which)
    monkeypatch.setattr("fine_tuning_studio.gpu_memory._run", fake_run)
    monkeypatch.setattr(
        "fine_tuning_studio.gpu_memory._process_identity",
        lambda pid, fallback: (fallback, fallback, "user", float(pid)),
    )

    report = inspect_gpu_memory(current_pid=999, current_username="user")

    assert report.devices[0].free_mb == 1645
    assert report.processes[0].pid == 123
    assert report.processes[0].memory_mb == 2048
    assert report.processes[0].protected_reason is None
    assert report.processes[1].protected_reason == "System process"


@pytest.mark.parametrize(
    ("pid", "name", "username", "expected"),
    [
        (4, "System", "SYSTEM", "System process"),
        (10, "python.exe", "user", "Fine-Tuning Studio process"),
        (11, "python.exe", "user", "Fine-Tuning Studio process"),
        (12, "ollama.exe", "user", "Managed through the Ollama API"),
        (13, "explorer.exe", "user", "Protected operating-system process"),
        (16, "avpui.exe", "user", "Protected operating-system process"),
        (14, "python.exe", "other", "Owned by another user"),
        (15, "python.exe", "user", None),
    ],
)
def test_process_protection_policy(
    pid: int, name: str, username: str, expected: str | None
) -> None:
    assert (
        process_protection_reason(
            pid,
            name,
            username,
            current_pid=10,
            current_username="user",
            ancestor_pids={11},
            studio_worker_pids=set(),
        )
        == expected
    )


def test_active_studio_worker_is_protected() -> None:
    assert (
        process_protection_reason(
            22,
            "python.exe",
            "user",
            current_pid=10,
            current_username="user",
            ancestor_pids=set(),
            studio_worker_pids={22},
        )
        == "Active Fine-Tuning Studio worker"
    )


def test_inspects_amd_json_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_which(command: str) -> str | None:
        return command if command == "amd-smi" else None

    def fake_run(command: list[str], timeout: float = 4) -> SimpleNamespace:
        if "monitor" in command:
            return completed(
                '[{"gpu":0,"vram_total":{"value":16,"unit":"GB"},'
                '"vram_used":{"value":4,"unit":"GB"},'
                '"vram_free":{"value":12,"unit":"GB"}}]'
            )
        return completed(
            '[{"gpu":0,"process_info":[{"pid":321,"name":"python",'
            '"vram_mem":{"value":4,"unit":"GB"}}]}]'
        )

    monkeypatch.setattr("fine_tuning_studio.gpu_memory.shutil.which", fake_which)
    monkeypatch.setattr("fine_tuning_studio.gpu_memory._run", fake_run)
    monkeypatch.setattr(
        "fine_tuning_studio.gpu_memory._process_identity",
        lambda pid, fallback: (fallback, fallback, "user", float(pid)),
    )

    report = inspect_gpu_memory(current_pid=999, current_username="user")

    assert report.devices[0].total_mb == 16 * 1024
    assert report.processes[0].memory_mb == 4 * 1024


def test_inspects_intel_json_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fine_tuning_studio.gpu_memory.shutil.which",
        lambda command: command if command == "xpu-smi" else None,
    )
    monkeypatch.setattr(
        "fine_tuning_studio.gpu_memory._run",
        lambda *_args, **_kwargs: completed(
            '{"device_list":[{"device_id":0,"process_list":'
            '[{"pid":456,"command":"python","mem":2097152}]}]}'
        ),
    )
    monkeypatch.setattr(
        "fine_tuning_studio.gpu_memory._process_identity",
        lambda pid, fallback: (fallback, fallback, "user", float(pid)),
    )

    report = inspect_gpu_memory(current_pid=999, current_username="user")

    assert report.processes[0].vendor == "Intel"
    assert report.processes[0].memory_mb == 2048


def test_safe_cleanup_unloads_ollama_and_refreshes_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = SimpleNamespace(devices=[], processes=[], warnings=[])
    after = SimpleNamespace(devices=[], processes=[], warnings=[])
    reports = iter([before, after])
    actions: list[str] = []

    class FakeClient:
        def list_running_models(self) -> list[OllamaRunningModel]:
            return [
                OllamaRunningModel("gemma3:latest", 1, 4096, ""),
                OllamaRunningModel("qwen:latest", 1, 4096, ""),
            ]

        def unload_model(self, model: str) -> None:
            actions.append(model)

    monkeypatch.setattr(
        "fine_tuning_studio.gpu_memory.inspect_gpu_memory", lambda **_: next(reports)
    )
    monkeypatch.setattr("fine_tuning_studio.gpu_memory._clear_torch_caches", lambda: ["CUDA cache"])

    result = release_reclaimable_memory(FakeClient())

    assert actions == ["gemma3:latest", "qwen:latest"]
    assert result.unloaded_models == ("gemma3:latest", "qwen:latest")
    assert result.cache_actions == ("CUDA cache",)
    assert result.before is before
    assert result.after is after


def test_termination_revalidates_process_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = GpuProcess(
        vendor="NVIDIA",
        gpu_index=0,
        pid=123,
        name="python.exe",
        executable="C:\\Python\\python.exe",
        username="user",
        create_time=100.0,
        memory_mb=2048,
        protected_reason=None,
    )
    fake = SimpleNamespace(
        create_time=lambda: 101.0,
        username=lambda: "user",
        exe=lambda: "C:\\Python\\python.exe",
    )
    monkeypatch.setattr("fine_tuning_studio.gpu_memory.psutil.Process", lambda _: fake)

    with pytest.raises(RuntimeError, match="identity changed"):
        terminate_gpu_process(candidate)


def test_protected_process_cannot_be_terminated() -> None:
    candidate = GpuProcess(
        vendor="NVIDIA",
        gpu_index=0,
        pid=4,
        name="System",
        executable="",
        username="SYSTEM",
        create_time=1.0,
        memory_mb=None,
        protected_reason="System process",
    )
    with pytest.raises(PermissionError, match="System process"):
        terminate_gpu_process(candidate)

    assert replace(candidate, protected_reason=None).pid == 4
