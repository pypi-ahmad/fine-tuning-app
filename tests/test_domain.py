from pathlib import Path

import pytest

from fine_tuning_studio.domain import (
    DatasetSpec,
    EvaluationSpec,
    ExportSpec,
    ModelSpec,
    RunManifest,
    TrainingSpec,
    ensure_within,
    validate_manifest,
)


def manifest(
    *,
    import_to_ollama: bool = False,
    ollama_model_name: str = "",
    merged_model: bool = False,
) -> RunManifest:
    export = ExportSpec(
        import_to_ollama=import_to_ollama,
        ollama_model_name=ollama_model_name,
        merged_model=merged_model,
    )
    return RunManifest(
        dataset=DatasetSpec(source="hub", location="org/data"),
        model=ModelSpec(source="hub", location="org/model"),
        training=TrainingSpec(),
        evaluation=EvaluationSpec(),
        export=export,
    )


def test_manifest_round_trip() -> None:
    original = manifest()
    assert RunManifest.from_dict(original.to_dict()) == original


def test_ollama_requires_merged_model() -> None:
    errors = validate_manifest(manifest(import_to_ollama=True, ollama_model_name="demo"))
    assert "Ollama import requires merged-model export for adapter jobs." in errors


def test_schema_one_manifest_is_readable_and_upgrades_on_new_runs() -> None:
    raw = manifest().to_dict()
    raw["schema_version"] = 1
    raw.pop("provenance")
    assert RunManifest.from_dict(raw).schema_version == 1
    assert manifest().schema_version == 2


def test_path_must_remain_in_workspace(tmp_path: Path) -> None:
    assert ensure_within(tmp_path, tmp_path / "job") == (tmp_path / "job").resolve()
    with pytest.raises(ValueError, match="escapes"):
        ensure_within(tmp_path, tmp_path.parent / "outside")
