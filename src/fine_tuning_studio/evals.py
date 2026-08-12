from __future__ import annotations

import csv
import html
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TASKS = ("mmlu", "gsm8k", "hellaswag", "arc", "truthfulqa", "winogrande")


@dataclass(frozen=True)
class EvaluationManifest:
    model: str
    backend: str = "transformers"
    tasks: list[str] = field(default_factory=list)
    limit: int = 20
    schema_version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def write_evaluation_report(
    output: Path, manifest: EvaluationManifest, scores: dict[str, float]
) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"manifest": asdict(manifest), "scores": scores}
    json_path = output / "evaluation.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = output / "evaluation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", "score"])
        writer.writerows(sorted(scores.items()))
    rows = "".join(
        f"<tr><td>{html.escape(task)}</td><td>{score:.6g}</td></tr>"
        for task, score in sorted(scores.items())
    )
    html_path = output / "evaluation.html"
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Evaluation report</title>"
        f"<h1>Evaluation report</h1><p>{html.escape(manifest.model)}</p>"
        f"<table><thead><tr><th>Task</th><th>Score</th></tr></thead><tbody>{rows}</tbody></table>",
        encoding="utf-8",
    )
    return {"json": json_path, "csv": csv_path, "html": html_path}


def compare_reports(reports: list[dict[str, float]]) -> dict[str, list[float]]:
    if not 2 <= len(reports) <= 4:
        raise ValueError("Compare between two and four evaluation reports.")
    tasks = sorted(set().union(*(report.keys() for report in reports)))
    return {task: [report.get(task, float("nan")) for report in reports] for task in tasks}


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
