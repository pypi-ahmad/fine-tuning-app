from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psutil

PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "huggingface-hub",
    "peft",
    "trl",
    "accelerate",
    "bitsandbytes",
    "triton",
    "unsloth",
    "streamlit",
)


@dataclass(frozen=True)
class GPUInfo:
    vendor: str
    name: str
    memory_total_mb: int | None = None
    memory_free_mb: int | None = None
    driver: str | None = None
    compute_capability: str | None = None
    device_index: int | None = None


@dataclass(frozen=True)
class MachineReport:
    os: dict[str, Any]
    cpu: dict[str, Any]
    memory: dict[str, Any]
    disk: dict[str, Any]
    gpus: list[GPUInfo] = field(default_factory=list)
    runtimes: dict[str, Any] = field(default_factory=dict)
    packages: dict[str, str | None] = field(default_factory=dict)
    integrations: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def sanitized_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.sanitized_dict(), indent=2, sort_keys=True)


def _run(command: list[str], timeout: float = 4) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _nvidia_gpus() -> list[GPUInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if not result or result.returncode:
        return []
    gpus: list[GPUInfo] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            gpus.append(
                GPUInfo(
                    vendor="NVIDIA",
                    device_index=int(parts[0]),
                    name=parts[1],
                    memory_total_mb=int(parts[2]),
                    memory_free_mb=int(parts[3]),
                    driver=parts[4],
                    compute_capability=parts[5],
                )
            )
        except ValueError:
            continue
    return gpus


def _windows_display_adapters() -> list[GPUInfo]:
    if platform.system() != "Windows" or not shutil.which("powershell"):
        return []
    script = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress"
    )
    result = _run(["powershell", "-NoProfile", "-Command", script])
    if not result or result.returncode or not result.stdout.strip():
        return []
    try:
        rows = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    if isinstance(rows, dict):
        rows = [rows]
    adapters: list[GPUInfo] = []
    for row in rows:
        name = str(row.get("Name") or "Unknown adapter")
        vendor = next((v for v in ("NVIDIA", "AMD", "Intel") if v.lower() in name.lower()), "Other")
        if vendor == "NVIDIA":
            continue
        memory = row.get("AdapterRAM")
        adapters.append(
            GPUInfo(
                vendor=vendor,
                name=name,
                memory_total_mb=int(memory) // 1024**2 if isinstance(memory, int) else None,
                driver=row.get("DriverVersion"),
            )
        )
    return adapters


def _torch_runtime() -> dict[str, Any]:
    try:
        import torch
    except (ImportError, OSError) as exc:
        return {"available": False, "error": type(exc).__name__}
    xpu = getattr(torch, "xpu", None)
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "hip_version": torch.version.hip,
        "xpu_available": bool(xpu and xpu.is_available()),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }


def _ollama_status() -> dict[str, Any]:
    import urllib.error
    import urllib.request

    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=1.5) as response:
            payload = json.load(response)
        return {
            "reachable": True,
            "host": host,
            "models": [item.get("name", "") for item in payload.get("models", [])],
        }
    except (OSError, ValueError, urllib.error.URLError):
        return {"reachable": False, "host": host, "models": []}


def scan_machine(workspace: Path | None = None) -> MachineReport:
    workspace = (workspace or Path.cwd()).resolve()
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(workspace)
    gpus = _nvidia_gpus()
    gpus.extend(_windows_display_adapters())
    warnings: list[str] = []
    if not gpus:
        warnings.append(
            "No supported GPU was detected. Training is disabled; inspection remains available."
        )
    runtime = _torch_runtime()
    if not runtime.get("available"):
        warnings.append("PyTorch is unavailable or failed to load.")
    return MachineReport(
        os={
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.machine(),
            "python": sys.version.split()[0],
            "wsl": "microsoft" in platform.release().lower(),
        },
        cpu={
            "model": platform.processor() or "Unknown",
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
        },
        memory={
            "total_gb": round(memory.total / 1024**3, 2),
            "available_gb": round(memory.available / 1024**3, 2),
        },
        disk={"total_gb": round(disk.total / 1024**3, 2), "free_gb": round(disk.free / 1024**3, 2)},
        gpus=gpus,
        runtimes={
            "torch": runtime,
            "rocm_smi": bool(shutil.which("rocm-smi")),
            "xpu_smi": bool(shutil.which("xpu-smi")),
        },
        packages=_package_versions(),
        integrations={"hf_token_present": "HF_TOKEN" in os.environ, "ollama": _ollama_status()},
        warnings=warnings,
    )


def qlora_capability(report: MachineReport) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    nvidia = [gpu for gpu in report.gpus if gpu.vendor == "NVIDIA"]
    if not nvidia:
        reasons.append("V0.1 QLoRA requires an NVIDIA GPU.")
    if not report.runtimes.get("torch", {}).get("cuda_available"):
        reasons.append("The installed PyTorch build cannot access CUDA.")
    for package in ("transformers", "datasets", "peft", "trl", "accelerate", "bitsandbytes"):
        if report.packages.get(package) is None:
            reasons.append(f"Missing package: {package}.")
    if nvidia and max((gpu.memory_free_mb or 0) for gpu in nvidia) < 3500:
        reasons.append(
            "At least 3.5 GB of currently free VRAM is required for the smallest "
            "supported QLoRA jobs."
        )
    return not reasons, reasons
