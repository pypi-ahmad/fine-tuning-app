import json
from pathlib import Path

from pytest import MonkeyPatch

from fine_tuning_studio.domain import (
    DatasetSourceSpec,
    DatasetSpec,
    EvaluationSpec,
    ExportSpec,
    ModelSpec,
    RunManifest,
    TrainingSpec,
)
from fine_tuning_studio.jobs import create_job, get_job, reconcile_jobs, resume_job, update_job


def test_create_job_persists_manifest_and_sanitized_upload(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setenv("FINE_TUNING_STUDIO_HOME", str(tmp_path))
    manifest = RunManifest(
        dataset=DatasetSpec(sources=[DatasetSourceSpec(source="local", location="data.jsonl")]),
        model=ModelSpec(source="hub", location="org/model"),
        training=TrainingSpec(),
        evaluation=EvaluationSpec(),
        export=ExportSpec(),
    )
    path = create_job(manifest, {0: ("../data.jsonl", b'{"text":"hello"}\n')})
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert Path(stored["dataset"]["sources"][0]["location"]).name == "data.jsonl"
    job = get_job(manifest.id)
    assert job is not None
    assert job["status"] == "queued"


def test_create_job_stages_multiple_local_datasets(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setenv("FINE_TUNING_STUDIO_HOME", str(tmp_path))
    manifest = RunManifest(
        dataset=DatasetSpec(
            sources=[
                DatasetSourceSpec(source="local", location="same.jsonl"),
                DatasetSourceSpec(source="local", location="same.jsonl"),
            ]
        ),
        model=ModelSpec(source="hub", location="org/model"),
        training=TrainingSpec(),
        evaluation=EvaluationSpec(),
        export=ExportSpec(),
    )
    path = create_job(
        manifest,
        {0: ("same.jsonl", b'{"text":"one"}\n'), 1: ("same.jsonl", b'{"text":"two"}\n')},
    )
    sources = json.loads(path.read_text(encoding="utf-8"))["dataset"]["sources"]
    assert Path(sources[0]["location"]).parent.name == "dataset-1"
    assert Path(sources[1]["location"]).parent.name == "dataset-2"


def test_resume_creates_v5_child_job(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("FINE_TUNING_STUDIO_HOME", str(tmp_path))
    parent = RunManifest(
        dataset=DatasetSpec(sources=[DatasetSourceSpec(source="hub", location="org/data")]),
        model=ModelSpec(source="hub", location="org/model"),
        training=TrainingSpec(),
        evaluation=EvaluationSpec(),
        export=ExportSpec(),
    )
    create_job(parent)
    checkpoint = tmp_path / "jobs" / parent.id / "checkpoints" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    child = resume_job(parent.id, checkpoint)
    assert child.parent_job_id == parent.id
    assert child.schema_version == 5
    assert child.training.resume_checkpoint == str(checkpoint.resolve())


def test_reconcile_marks_missing_worker_interrupted(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setenv("FINE_TUNING_STUDIO_HOME", str(tmp_path))
    run = RunManifest(
        dataset=DatasetSpec(sources=[DatasetSourceSpec(source="hub", location="org/data")]),
        model=ModelSpec(source="hub", location="org/model"),
        training=TrainingSpec(),
        evaluation=EvaluationSpec(),
        export=ExportSpec(),
    )
    create_job(run)
    update_job(run.id, status="training", pid=999_999_999)
    assert reconcile_jobs() == 1
    job = get_job(run.id)
    assert job is not None
    assert job["status"] == "interrupted"
