from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fine_tuning_studio.system_info import MachineReport


@dataclass(frozen=True)
class RuntimeProfile:
    id: str
    vendor: str
    operating_systems: tuple[str, ...]
    status: str
    torch_index: str
    notes: str


PROFILES = {
    profile.id: profile
    for profile in (
        RuntimeProfile("cuda", "NVIDIA", ("Windows", "Linux"), "stable", "cu130", "CUDA 13"),
        RuntimeProfile("rocm", "AMD", ("Windows", "Linux"), "experimental", "rocm7.1", "ROCm"),
        RuntimeProfile("xpu", "Intel", ("Windows", "Linux"), "experimental", "xpu", "Intel XPU"),
    )
}


@dataclass(frozen=True)
class RuntimeVerdict:
    profile: str
    state: str
    reasons: list[str]


def profile_directory(root: Path, profile: str) -> Path:
    if profile not in PROFILES:
        raise ValueError(f"Unknown runtime profile: {profile}")
    return root / "runtimes" / profile


def profile_python(root: Path, profile: str) -> Path:
    directory = profile_directory(root, profile) / ".venv"
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def runtime_verdict(report: MachineReport, profile: str, method: str = "qlora") -> RuntimeVerdict:
    definition = PROFILES[profile]
    reasons: list[str] = []
    if report.os["system"] not in definition.operating_systems:
        return RuntimeVerdict(profile, "unsupported", ["Operating system is unsupported."])
    if not any(gpu.vendor == definition.vendor for gpu in report.gpus):
        return RuntimeVerdict(profile, "unsupported", [f"No {definition.vendor} GPU detected."])
    torch = report.runtimes.get("torch", {})
    available = {
        "cuda": torch.get("cuda_available"),
        "rocm": bool(torch.get("hip_version")),
        "xpu": torch.get("xpu_available"),
    }[profile]
    state = "ready" if available else "install-required"
    if definition.status == "experimental":
        reasons.append("This runtime is experimental until validated on this machine.")
    if method == "qlora" and report.packages.get("bitsandbytes") is None:
        reasons.append("QLoRA requires bitsandbytes in the selected runtime.")
    return RuntimeVerdict(profile, state, reasons)


def smoke_test(profile: str) -> dict[str, Any]:
    import torch

    device = {"cuda": "cuda", "rocm": "cuda", "xpu": "xpu"}[profile]
    module = torch.cuda if device == "cuda" else torch.xpu
    if not module.is_available():
        return {"ok": False, "error": f"{profile} device is unavailable"}
    left = torch.ones((16, 16), device=device)
    result = left @ left
    return {
        "ok": bool(result.sum().item() == 4096),
        "device": str(result.device),
        "dtype": str(result.dtype),
    }


def save_smoke_result(root: Path, profile: str, result: dict[str, Any]) -> Path:
    output = profile_directory(root, profile) / "smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output


def provision_profile(root: Path, profile: str) -> Path:
    definition = PROFILES[profile]
    if not shutil.which("uv"):
        raise RuntimeError("uv is required to provision managed runtimes.")
    directory = profile_directory(root, profile)
    directory.mkdir(parents=True, exist_ok=True)
    python = profile_python(root, profile)
    subprocess.run(
        ["uv", "venv", str(python.parent.parent), "--python", sys.executable], check=True
    )
    packages = [
        "transformers",
        "datasets",
        "peft",
        "trl",
        "accelerate",
        "safetensors",
        "bitsandbytes",
    ]
    torch_source = f"https://download.pytorch.org/whl/{definition.torch_index}"
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "torch", "--index-url", torch_source],
        check=True,
    )
    subprocess.run(["uv", "pip", "install", "--python", str(python), *packages], check=True)
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), ".", "--no-deps"], check=True
    )
    (directory / "profile.json").write_text(
        json.dumps(asdict(definition), indent=2), encoding="utf-8"
    )
    return python
