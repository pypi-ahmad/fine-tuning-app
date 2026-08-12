import json
from pathlib import Path

from pytest import MonkeyPatch

from fine_tuning_studio.domain import (
    DatasetSpec,
    EvaluationSpec,
    ExportSpec,
    ModelSpec,
    RunManifest,
    TrainingSpec,
)
from fine_tuning_studio.jobs import create_job, get_job


def test_create_job_persists_manifest_and_sanitized_upload(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setenv("FINE_TUNING_STUDIO_HOME", str(tmp_path))
    manifest = RunManifest(
        dataset=DatasetSpec(source="local", location="data.jsonl"),
        model=ModelSpec(source="hub", location="org/model"),
        training=TrainingSpec(),
        evaluation=EvaluationSpec(),
        export=ExportSpec(),
    )
    path = create_job(manifest, "../data.jsonl", b'{"text":"hello"}\n')
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert Path(stored["dataset"]["location"]).name == "data.jsonl"
    job = get_job(manifest.id)
    assert job is not None
    assert job["status"] == "queued"
