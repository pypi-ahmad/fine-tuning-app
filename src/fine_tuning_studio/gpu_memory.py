from __future__ import annotations

import csv
import gc
import json
import os
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import psutil

from fine_tuning_studio.ollama import OllamaClient, OllamaError, OllamaRunningModel


@dataclass(frozen=True)
class GpuDeviceMemory:
    vendor: str
    gpu_index: int
    name: str
    total_mb: int | None
    used_mb: int | None
    free_mb: int | None
    uuid: str = ""


@dataclass(frozen=True)
class GpuProcess:
    vendor: str
    gpu_index: int
    pid: int
    name: str
    executable: str
    username: str
    create_time: float
    memory_mb: int | None
    protected_reason: str | None


@dataclass(frozen=True)
class GpuMemoryReport:
    devices: tuple[GpuDeviceMemory, ...]
    processes: tuple[GpuProcess, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class GpuCleanupResult:
    before: GpuMemoryReport
    after: GpuMemoryReport
    cache_actions: tuple[str, ...]
    unloaded_models: tuple[str, ...]
    warnings: tuple[str, ...]


class OllamaMemoryClient(Protocol):
    def list_running_models(self) -> list[OllamaRunningModel]: ...

    def unload_model(self, model: str) -> None: ...


_SYSTEM_PROCESSES = {
    "avp.exe",
    "avpui.exe",
    "csrss.exe",
    "dwm.exe",
    "explorer.exe",
    "lsass.exe",
    "msmpeng.exe",
    "registry",
    "services.exe",
    "securityhealthservice.exe",
    "securityhealthsystray.exe",
    "smss.exe",
    "system",
    "wininit.exe",
    "winlogon.exe",
}


def inspect_gpu_memory(
    *,
    studio_worker_pids: set[int] | None = None,
    current_pid: int | None = None,
    current_username: str | None = None,
) -> GpuMemoryReport:
    pid = current_pid if current_pid is not None else os.getpid()
    username = current_username or _current_username()
    ancestors = _ancestor_pids(pid)
    protected = studio_worker_pids or set()
    devices: list[GpuDeviceMemory] = []
    raw_processes: list[tuple[str, int, int, str, int | None]] = []
    warnings: list[str] = []

    if shutil.which("nvidia-smi"):
        nvidia_devices, nvidia_processes, nvidia_warnings = _inspect_nvidia()
        devices.extend(nvidia_devices)
        raw_processes.extend(nvidia_processes)
        warnings.extend(nvidia_warnings)
    if shutil.which("amd-smi"):
        amd_devices, amd_processes, amd_warnings = _inspect_amd()
        devices.extend(amd_devices)
        raw_processes.extend(amd_processes)
        warnings.extend(amd_warnings)
    if shutil.which("xpu-smi"):
        intel_processes, intel_warnings = _inspect_intel()
        raw_processes.extend(intel_processes)
        warnings.extend(intel_warnings)
    if not any(shutil.which(tool) for tool in ("nvidia-smi", "amd-smi", "xpu-smi")):
        warnings.append("No supported GPU process-inspection tool is installed.")

    processes = []
    seen: set[tuple[str, int, int]] = set()
    for vendor, gpu_index, process_pid, fallback_name, memory_mb in raw_processes:
        key = (vendor, gpu_index, process_pid)
        if key in seen:
            continue
        seen.add(key)
        name, executable, owner, created = _process_identity(process_pid, fallback_name)
        reason = process_protection_reason(
            process_pid,
            name,
            owner,
            current_pid=pid,
            current_username=username,
            ancestor_pids=ancestors,
            studio_worker_pids=protected,
        )
        processes.append(
            GpuProcess(
                vendor=vendor,
                gpu_index=gpu_index,
                pid=process_pid,
                name=name,
                executable=executable,
                username=owner,
                create_time=created,
                memory_mb=memory_mb,
                protected_reason=reason,
            )
        )
    return GpuMemoryReport(tuple(devices), tuple(processes), tuple(warnings))


def process_protection_reason(
    pid: int,
    name: str,
    username: str,
    *,
    current_pid: int,
    current_username: str,
    ancestor_pids: set[int],
    studio_worker_pids: set[int],
) -> str | None:
    normalized = _basename(name).casefold()
    if pid <= 4:
        return "System process"
    if pid == current_pid or pid in ancestor_pids:
        return "Fine-Tuning Studio process"
    if pid in studio_worker_pids:
        return "Active Fine-Tuning Studio worker"
    if normalized.startswith("ollama") or "ollama runner" in normalized:
        return "Managed through the Ollama API"
    if normalized in _SYSTEM_PROCESSES:
        return "Protected operating-system process"
    if not username:
        return "Process ownership unavailable"
    if username.casefold() != current_username.casefold():
        return "Owned by another user"
    return None


def release_reclaimable_memory(
    ollama: OllamaMemoryClient | None = None,
    *,
    studio_worker_pids: set[int] | None = None,
) -> GpuCleanupResult:
    before = inspect_gpu_memory(studio_worker_pids=studio_worker_pids)
    gc.collect()
    cache_actions = _clear_torch_caches()
    warnings: list[str] = []
    unloaded: list[str] = []
    client = ollama or OllamaClient()
    try:
        for model in client.list_running_models():
            try:
                client.unload_model(model.name)
                unloaded.append(model.name)
            except OllamaError:
                warnings.append(f"Could not unload Ollama model {model.name}.")
    except (OllamaError, ValueError):
        warnings.append("Ollama was unavailable; no Ollama models were unloaded.")
    after = inspect_gpu_memory(studio_worker_pids=studio_worker_pids)
    return GpuCleanupResult(
        before=before,
        after=after,
        cache_actions=tuple(cache_actions),
        unloaded_models=tuple(unloaded),
        warnings=tuple(warnings),
    )


def terminate_gpu_process(candidate: GpuProcess) -> tuple[int, ...]:
    if candidate.protected_reason:
        raise PermissionError(candidate.protected_reason)
    try:
        process = psutil.Process(candidate.pid)
        if abs(process.create_time() - candidate.create_time) > 0.01:
            raise RuntimeError("Process identity changed; refresh GPU processes and try again.")
        if process.username().casefold() != candidate.username.casefold():
            raise RuntimeError("Process identity changed; refresh GPU processes and try again.")
        if os.path.normcase(process.exe()) != os.path.normcase(candidate.executable):
            raise RuntimeError("Process identity changed; refresh GPU processes and try again.")
        targets = [*process.children(recursive=True), process]
        for target in targets:
            target.terminate()
        _, alive = psutil.wait_procs(targets, timeout=3)
        for target in alive:
            target.kill()
        if alive:
            psutil.wait_procs(alive, timeout=3)
        return tuple(target.pid for target in targets)
    except (psutil.Error, OSError) as exc:
        raise RuntimeError("The selected process could not be terminated safely.") from exc


def _clear_torch_caches() -> list[str]:
    try:
        import torch
    except (ImportError, OSError):
        return []
    actions: list[str] = []
    cuda = getattr(torch, "cuda", None)
    try:
        if cuda is not None and cuda.is_initialized():
            cuda.empty_cache()
            if hasattr(cuda, "ipc_collect"):
                cuda.ipc_collect()
            actions.append("CUDA/ROCm allocator cache")
    except (RuntimeError, OSError):
        pass
    xpu = getattr(torch, "xpu", None)
    try:
        if xpu is not None and xpu.is_initialized():
            xpu.empty_cache()
            actions.append("Intel XPU allocator cache")
    except (RuntimeError, OSError):
        pass
    return actions


def _inspect_nvidia() -> tuple[
    list[GpuDeviceMemory], list[tuple[str, int, int, str, int | None]], list[str]
]:
    gpu_result = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    process_result = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if not gpu_result or gpu_result.returncode:
        return [], [], ["nvidia-smi could not report GPU memory."]
    devices: list[GpuDeviceMemory] = []
    uuid_to_index: dict[str, int] = {}
    for row in csv.reader(gpu_result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 6:
            continue
        index = _integer(row[0])
        if index is None:
            continue
        uuid_to_index[row[1].strip()] = index
        devices.append(
            GpuDeviceMemory(
                "NVIDIA",
                index,
                row[2].strip(),
                _integer(row[3]),
                _integer(row[4]),
                _integer(row[5]),
                row[1].strip(),
            )
        )
    processes: list[tuple[str, int, int, str, int | None]] = []
    if process_result and not process_result.returncode:
        for row in csv.reader(process_result.stdout.splitlines(), skipinitialspace=True):
            if len(row) != 4:
                continue
            process_pid = _integer(row[1])
            if process_pid is None:
                continue
            processes.append(
                (
                    "NVIDIA",
                    uuid_to_index.get(row[0].strip(), 0),
                    process_pid,
                    row[2].strip(),
                    _integer(row[3]),
                )
            )
    return devices, processes, []


def _inspect_amd() -> tuple[
    list[GpuDeviceMemory], list[tuple[str, int, int, str, int | None]], list[str]
]:
    metric_result = _run(["amd-smi", "monitor", "--vram-usage", "--json"])
    process_result = _run(["amd-smi", "process", "--general", "--json"])
    devices: list[GpuDeviceMemory] = []
    processes: list[tuple[str, int, int, str, int | None]] = []
    warnings: list[str] = []
    try:
        metrics = (
            json.loads(metric_result.stdout)
            if metric_result and not metric_result.returncode
            else []
        )
        for record in _records(metrics):
            index = _integer(_find_value(record, "gpu", "gpu_id"))
            total = _megabytes(_find_value(record, "vram_total", "total_vram", "total"))
            used = _megabytes(_find_value(record, "vram_used", "used_vram", "used"))
            free = _megabytes(_find_value(record, "vram_free", "free_vram", "free"))
            if index is not None and any(value is not None for value in (total, used, free)):
                devices.append(GpuDeviceMemory("AMD", index, f"AMD GPU {index}", total, used, free))
    except (json.JSONDecodeError, TypeError):
        warnings.append("amd-smi returned unreadable memory data.")
    try:
        payload = (
            json.loads(process_result.stdout)
            if process_result and not process_result.returncode
            else []
        )
        for record in _records(payload):
            process_pid = _integer(_find_value(record, "pid"))
            if process_pid is None:
                continue
            processes.append(
                (
                    "AMD",
                    _integer(_find_value(record, "gpu", "gpu_id")) or 0,
                    process_pid,
                    str(_find_value(record, "name", "process_name") or "AMD GPU process"),
                    _megabytes(_find_value(record, "vram_mem", "vram_memory", "memory_usage")),
                )
            )
    except (json.JSONDecodeError, TypeError):
        warnings.append("amd-smi returned unreadable process data.")
    return devices, processes, warnings


def _inspect_intel() -> tuple[list[tuple[str, int, int, str, int | None]], list[str]]:
    result = _run(["xpu-smi", "ps", "-j"])
    if not result or result.returncode:
        return [], ["xpu-smi process reporting is unavailable on this platform."]
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return [], ["xpu-smi returned unreadable process data."]
    processes = []
    for record in _records(payload):
        process_pid = _integer(_find_value(record, "pid"))
        if process_pid is None:
            continue
        memory_kb = _integer(_find_value(record, "mem", "memory"))
        processes.append(
            (
                "Intel",
                _integer(_find_value(record, "device_id", "deviceid")) or 0,
                process_pid,
                str(_find_value(record, "command", "name") or "Intel GPU process"),
                memory_kb // 1024 if memory_kb is not None else None,
            )
        )
    return processes, []


def _run(command: list[str], timeout: float = 4) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _process_identity(pid: int, fallback: str) -> tuple[str, str, str, float]:
    try:
        process = psutil.Process(pid)
        name = process.name() or _basename(fallback)
        executable = process.exe() or fallback
        return name, executable, process.username(), process.create_time()
    except (psutil.Error, OSError):
        return _basename(fallback), fallback, "", 0.0


def _current_username() -> str:
    try:
        return psutil.Process().username()
    except (psutil.Error, OSError):
        return ""


def _ancestor_pids(pid: int) -> set[int]:
    try:
        return {parent.pid for parent in psutil.Process(pid).parents()}
    except (psutil.Error, OSError):
        return set()


def _integer(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _megabytes(value: object) -> int | None:
    if isinstance(value, dict):
        amount = value.get("value")
        unit = str(value.get("unit") or "MB").upper()
        number = _integer(amount)
        if number is None:
            return None
        if unit in {"B", "BYTES"}:
            return number // 1024**2
        if unit in {"KB", "KIB"}:
            return number // 1024
        if unit in {"GB", "GIB"}:
            return number * 1024
        return number
    return _integer(value)


def _records(value: object) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _records(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _records(nested)


def _find_value(record: dict[str, Any], *names: str) -> object | None:
    wanted = {name.casefold() for name in names}
    for key, value in record.items():
        if str(key).casefold() in wanted:
            return value
    return None


def _basename(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1]
