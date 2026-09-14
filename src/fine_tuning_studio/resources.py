"""Pre-flight safety gate for full-parameter (non-adapter) fine-tuning.

Full fine-tuning has no LoRA/QLoRA-style memory ceiling, so worker.py calls
`full_training_gate` before starting one to refuse jobs that would very
likely OOM or fill the disk partway through a run. See preflight.py for the
lighter per-method VRAM estimate used earlier, at configuration time.
"""

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
    # Rough heuristics, not exact accounting: 12 bytes/parameter approximates fp32
    # weights + gradients + Adam optimizer state (the dominant full fine-tuning
    # cost); model_gb assumes fp16 weights on disk, and required_disk budgets for
    # two on-disk checkpoints plus the final saved model.
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
