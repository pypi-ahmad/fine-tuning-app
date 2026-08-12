from pathlib import Path

from fine_tuning_studio.artifacts import write_artifact_manifest


def test_artifact_manifest_hashes_outputs(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
    output = write_artifact_manifest(tmp_path)
    assert '"sha256"' in output.read_text(encoding="utf-8")
