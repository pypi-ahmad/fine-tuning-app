from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FullTrainingGate:
    allowed: bool
    estimated_vram_gb: float
    required_disk_gb: float
    reasons: list[str]


def full_training_gate(
    parameter_count: int,
    free_vram_gb: float,
    workspace: Path,
    bytes_per_parameter: float = 12.0,
) -> FullTrainingGate:
    estimated = parameter_count * bytes_per_parameter / 1024**3
    model_gb = parameter_count * 2 / 1024**3
    required_disk = model_gb * 3
    free_disk = shutil.disk_usage(workspace).free / 1024**3
    reasons: list[str] = []
    if estimated > free_vram_gb * 0.85:
        reasons.append("Estimated training memory exceeds 85% of free accelerator memory.")
    if required_disk > free_disk:
        reasons.append("Disk cannot hold two checkpoints and the final model.")
    return FullTrainingGate(not reasons, round(estimated, 2), round(required_disk, 2), reasons)
