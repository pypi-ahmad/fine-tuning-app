from pathlib import Path

from pytest import MonkeyPatch

from fine_tuning_studio.cli import main


def test_cli_version_and_doctor(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("FINE_TUNING_STUDIO_HOME", str(tmp_path))
    assert main(["version"]) == 0
    assert main(["doctor", "--json"]) == 0
