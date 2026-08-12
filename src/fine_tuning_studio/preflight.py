from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fine_tuning_studio.domain import DatasetSpec, ModelSpec


@dataclass(frozen=True)
class DatasetReport:
    rows: int
    columns: list[str]
    null_counts: dict[str, int]
    fingerprint: str
    approximate_tokens: int
    errors: list[str]


@dataclass(frozen=True)
class ModelReport:
    resolved_revision: str
    architecture: str
    parameters: int | None
    context_length: int | None
    gated: bool | str
    license: str | None
    weight_bytes: int
    estimated_training_vram_gb: float | None


def _required_columns(spec: DatasetSpec) -> set[str]:
    canonical = {
        "preference": {"prompt", "chosen", "rejected"},
        "kto": {"prompt", "completion", "label"},
        "grpo": {"prompt"},
    }
    if spec.mapping in canonical:
        return canonical[spec.mapping]
    if spec.mapping == "prompt_response":
        return {spec.prompt_column, spec.response_column}
    return {spec.text_column}


def inspect_dataset(dataset: Any, spec: DatasetSpec, sample_limit: int = 1_000) -> DatasetReport:
    columns = list(dataset.column_names)
    missing = _required_columns(spec) - set(columns)
    errors = [f"Missing column: {name}" for name in sorted(missing)]
    nulls = {name: 0 for name in columns}
    characters = 0
    digest = hashlib.sha256()
    sample = dataset.select(range(min(len(dataset), sample_limit)))
    for row in sample:
        encoded = json.dumps(row, sort_keys=True, default=str).encode()
        digest.update(encoded)
        for name in columns:
            if row.get(name) is None:
                nulls[name] += 1
        for name in _required_columns(spec) - missing:
            characters += len(str(row.get(name, "")))
    if len(dataset) < 2:
        errors.append("Dataset must contain at least two rows.")
    return DatasetReport(
        rows=len(dataset),
        columns=columns,
        null_counts=nulls,
        fingerprint=getattr(dataset, "_fingerprint", None) or digest.hexdigest(),
        approximate_tokens=max(1, characters // 4),
        errors=errors,
    )


def inspect_model(spec: ModelSpec, method: str = "qlora") -> ModelReport:
    from huggingface_hub import HfApi
    from transformers import AutoConfig

    info = HfApi().model_info(spec.location, revision=spec.revision)
    config = AutoConfig.from_pretrained(
        spec.location,
        revision=spec.revision,
        trust_remote_code=spec.trust_remote_code,
    )
    weight_bytes = sum(sibling.size or 0 for sibling in (info.siblings or []))
    parameters = getattr(info, "safetensors", None)
    parameter_count = sum(parameters.parameters.values()) if parameters else None
    multiplier = {"qlora": 0.7, "lora": 2.2, "full": 12.0}.get(method, 2.2)
    estimate = round(parameter_count * multiplier / 1024**3, 2) if parameter_count else None
    return ModelReport(
        resolved_revision=info.sha or spec.revision,
        architecture=(getattr(config, "architectures", None) or [type(config).__name__])[0],
        parameters=parameter_count,
        context_length=getattr(config, "max_position_embeddings", None),
        gated=info.gated or False,
        license=(info.card_data or {}).get("license") if info.card_data else None,
        weight_bytes=weight_bytes,
        estimated_training_vram_gb=estimate,
    )


def write_report(path: Path, report: DatasetReport | ModelReport) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    temporary.replace(path)
