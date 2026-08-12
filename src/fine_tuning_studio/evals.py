from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

TASKS = ("mmlu", "gsm8k", "hellaswag", "arc", "truthfulqa", "winogrande")


def inspect_command(model: Path, tasks: list[str], limit: int = 20) -> list[str]:
    unknown = set(tasks) - set(TASKS)
    if unknown:
        raise ValueError(f"Unknown Inspect task: {', '.join(sorted(unknown))}")
    if limit < 1:
        raise ValueError("Evaluation limit must be positive.")
    return ["inspect", "eval", *tasks, "--model", f"hf/{model}", "--limit", str(limit)]


def run_inspect(model: Path, tasks: list[str], limit: int, output: Path) -> None:
    if not shutil.which("inspect"):
        raise RuntimeError("Install the optional Inspect AI evaluation profile first.")
    result = subprocess.run(
        inspect_command(model, tasks, limit), capture_output=True, text=True, timeout=86400
    )
    output.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("Inspect AI evaluation failed; see the evaluation log.")
