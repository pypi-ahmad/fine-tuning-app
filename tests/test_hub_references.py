from pathlib import Path
from types import SimpleNamespace

import pytest

from fine_tuning_studio.hub_references import (
    RepoType,
    download_hub_repository,
    normalize_hub_reference,
    plan_hub_download,
)


@pytest.mark.parametrize(
    ("value", "repo_type", "expected"),
    [
        ("LiquidAI/LFM2.5-2.6B", "model", "LiquidAI/LFM2.5-2.6B"),
        (
            "https://huggingface.co/LiquidAI/LFM2.5-2.6B",
            "model",
            "LiquidAI/LFM2.5-2.6B",
        ),
        (
            "https://huggingface.co/datasets/MatrAIx2026/MatrAIx_Persona_1M",
            "dataset",
            "MatrAIx2026/MatrAIx_Persona_1M",
        ),
        (
            "https://huggingface.co/LiquidAI/LFM2.5-2.6B/tree/main?download=true#files",
            "model",
            "LiquidAI/LFM2.5-2.6B",
        ),
        (
            "https://huggingface.co/datasets/MatrAIx2026/MatrAIx_Persona_1M/viewer/default/train",
            "dataset",
            "MatrAIx2026/MatrAIx_Persona_1M",
        ),
        (
            "https://huggingface.co/datasets/MatrAIx2026/MatrAIx_Persona_1M/blob/main/data.jsonl",
            "dataset",
            "MatrAIx2026/MatrAIx_Persona_1M",
        ),
        ("https://huggingface.co/org/tree", "model", "org/tree"),
        ("https://huggingface.co/org/tree/tree/main", "model", "org/tree"),
    ],
)
def test_normalize_hub_reference(value: str, repo_type: RepoType, expected: str) -> None:
    assert normalize_hub_reference(value, repo_type) == expected


@pytest.mark.parametrize(
    ("value", "repo_type", "message"),
    [
        ("http://huggingface.co/org/model", "model", "HTTPS"),
        ("https://example.com/org/model", "model", "huggingface.co"),
        ("https://user:secret@huggingface.co/org/model", "model", "credentials"),
        ("https://huggingface.co/datasets/org/data", "model", "dataset"),
        ("https://huggingface.co/org/model", "dataset", "model"),
        ("https://huggingface.co/spaces/org/demo", "model", "Spaces"),
        ("org/repo/extra", "model", "repository ID"),
    ],
)
def test_normalize_hub_reference_rejects_unsafe_or_mismatched_values(
    value: str, repo_type: RepoType, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_hub_reference(value, repo_type)


def test_plan_and_download_use_the_hugging_face_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    dry_run = [
        SimpleNamespace(
            commit_hash="abc123",
            file_size=100,
            filename="config.json",
            is_cached=True,
            will_download=False,
        ),
        SimpleNamespace(
            commit_hash="abc123",
            file_size=900,
            filename="model.safetensors",
            is_cached=False,
            will_download=True,
        ),
    ]

    def fake_snapshot_download(**kwargs: object) -> object:
        calls.append(kwargs)
        if kwargs.get("dry_run"):
            return dry_run
        return str(tmp_path / "models--org--model" / "snapshots" / "abc123")

    monkeypatch.setattr(
        "fine_tuning_studio.hub_references.snapshot_download", fake_snapshot_download
    )
    monkeypatch.setattr(
        "fine_tuning_studio.hub_references.shutil.disk_usage",
        lambda _: SimpleNamespace(free=10_000),
    )

    plan = plan_hub_download("org/model", "model", "v1")
    cache_path = download_hub_repository(plan)

    assert plan.commit_hash == "abc123"
    assert plan.file_count == 2
    assert plan.total_bytes == 1_000
    assert plan.download_bytes == 900
    assert plan.cached_files == 1
    assert cache_path.name == "abc123"
    assert calls == [
        {
            "repo_id": "org/model",
            "repo_type": "model",
            "revision": "v1",
            "dry_run": True,
        },
        {
            "repo_id": "org/model",
            "repo_type": "model",
            "revision": "v1",
        },
    ]
    assert all("token" not in call for call in calls)


def test_plan_rejects_download_larger_than_available_cache_disk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fine_tuning_studio.hub_references.snapshot_download",
        lambda **_: [
            SimpleNamespace(
                commit_hash="abc123",
                file_size=1_001,
                filename="model.safetensors",
                is_cached=False,
                will_download=True,
            )
        ],
    )
    monkeypatch.setattr(
        "fine_tuning_studio.hub_references.shutil.disk_usage",
        lambda _: SimpleNamespace(free=1_000),
    )

    with pytest.raises(OSError, match="requires 1,001 bytes"):
        plan_hub_download("org/model", "model", "main")
