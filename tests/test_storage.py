import json
from contextlib import closing
from pathlib import Path

from fine_tuning_studio.storage import SCHEMA_VERSION, backup, connect, integrity_check, restore


def test_migrate_backup_and_restore(tmp_path: Path) -> None:
    with closing(connect(tmp_path)) as connection:
        version = connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
    assert version == SCHEMA_VERSION
    manifest = tmp_path / "jobs" / "one" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"id": "one"}), encoding="utf-8")
    saved = backup(tmp_path)
    manifest.write_text("changed", encoding="utf-8")
    safety = restore(tmp_path, saved)
    assert safety.is_dir()
    assert json.loads(manifest.read_text(encoding="utf-8"))["id"] == "one"
    assert integrity_check(tmp_path) == (True, "ok")
