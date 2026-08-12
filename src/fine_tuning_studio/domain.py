from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4


class JobStatus(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    EVALUATING_BEFORE = "evaluating_before"
    TRAINING = "training"
    EVALUATING_AFTER = "evaluating_after"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class DatasetSpec:
    source: str
    location: str
    revision: str = "main"
    split: str = "train"
    format: str = "auto"
    mapping: str = "text"
    text_column: str = "text"
    prompt_column: str = "prompt"
    response_column: str = "response"
    chat_template: str = "tokenizer"
    validation_fraction: float = 0.1
    seed: int = 42


@dataclass(frozen=True)
class ModelSpec:
    source: str
    location: str
    revision: str = "main"
    trust_remote_code: bool = False


@dataclass(frozen=True)
class TrainingSpec:
    objective: str = "sft"
    method: str = "qlora"
    backend: str = "transformers"
    epochs: float = 1.0
    learning_rate: float = 2e-4
    max_sequence_length: int = 1024
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    packing: bool = False
    gradient_checkpointing: bool = True
    save_steps: int = 100
    logging_steps: int = 10
    resume_checkpoint: str | None = None


@dataclass(frozen=True)
class EvaluationSpec:
    before: bool = True
    after: bool = True
    benchmarks: list[str] = field(default_factory=list)
    benchmark_limit: int = 100


@dataclass(frozen=True)
class ExportSpec:
    adapter: bool = True
    merged_model: bool = False
    push_to_hub: bool = False
    hub_repo_id: str = ""
    hub_private: bool = True
    import_to_ollama: bool = False
    ollama_model_name: str = ""


@dataclass(frozen=True)
class RunManifest:
    dataset: DatasetSpec
    model: ModelSpec
    training: TrainingSpec
    evaluation: EvaluationSpec
    export: ExportSpec
    id: str = field(default_factory=lambda: str(uuid4()))
    schema_version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    parent_job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunManifest:
        return cls(
            id=value["id"],
            schema_version=value.get("schema_version", 1),
            created_at=value["created_at"],
            parent_job_id=value.get("parent_job_id"),
            dataset=DatasetSpec(**value["dataset"]),
            model=ModelSpec(**value["model"]),
            training=TrainingSpec(**value["training"]),
            evaluation=EvaluationSpec(**value["evaluation"]),
            export=ExportSpec(**value["export"]),
        )


def validate_manifest(manifest: RunManifest) -> list[str]:
    errors: list[str] = []
    if not manifest.dataset.location.strip():
        errors.append("Choose a dataset.")
    if not manifest.model.location.strip():
        errors.append("Choose a model.")
    if not 0 < manifest.dataset.validation_fraction < 0.5:
        errors.append("Validation fraction must be greater than 0 and less than 0.5.")
    if manifest.training.max_sequence_length < 64:
        errors.append("Maximum sequence length must be at least 64 tokens.")
    if manifest.training.method != "qlora" or manifest.training.objective != "sft":
        errors.append("V0.1 supports QLoRA supervised fine-tuning only.")
    if manifest.export.push_to_hub and not manifest.export.hub_repo_id.strip():
        errors.append("Enter a Hugging Face repository ID before enabling Hub upload.")
    if manifest.export.import_to_ollama and not manifest.export.ollama_model_name.strip():
        errors.append("Enter an Ollama model name before enabling Ollama import.")
    if manifest.export.import_to_ollama and not manifest.export.merged_model:
        errors.append("Ollama import requires merged-model export for QLoRA jobs.")
    return errors


def ensure_within(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    if resolved_candidate != resolved_root and resolved_root not in resolved_candidate.parents:
        raise ValueError("Path escapes the Fine-Tuning Studio workspace.")
    return resolved_candidate
