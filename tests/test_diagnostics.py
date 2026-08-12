import io
import zipfile
from contextlib import closing
from pathlib import Path

from pytest import MonkeyPatch

from fine_tuning_studio.diagnostics import build_diagnostics, redact
from fine_tuning_studio.storage import connect


def test_diagnostics_redacts_secrets(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("FINE_TUNING_STUDIO_HOME", str(tmp_path))
    with closing(connect(tmp_path)):
        pass
    payload = build_diagnostics(tmp_path)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert {"system.json", "storage.json"} <= set(archive.namelist())
    assert "abc" not in redact("token=abc")
