"""The jobs SQLite database: connection setup, schema migration, backup, and restore.

studio.db (under studio_home(), see jobs.py) is the single source of truth
for job status; per-job manifest.json files hold the larger configuration.
Next: jobs.py, which is the only module that reads/writes job rows.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

# Bump this and add a new entry to the `migrations` dict in migrate() below when the
# schema changes; never edit or renumber an existing entry, since already-migrated
# databases have that version permanently recorded in schema_migrations.
SCHEMA_VERSION = 3


def connect(home: Path) -> sqlite3.Connection:
    home.mkdir(parents=True, exist_ok=True)
    database = home / "studio.db"
    # Take a safety backup before an irreversible schema migration runs, so a bad
    # migration (or an interrupted one) can be recovered from manually.
    if database.exists() and _schema_version(database) < SCHEMA_VERSION:
        backup(home, "pre-migration")
    connection = sqlite3.connect(database, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    migrate(connection)
    return connection


def _schema_version(database: Path) -> int:
    try:
        with closing(sqlite3.connect(database)) as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
            ).fetchone()
            if not exists:
                return 0
            return int(
                connection.execute(
                    "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
                ).fetchone()[0]
            )
    except sqlite3.DatabaseError:
        # Treat an unreadable file as "already current" rather than 0: returning 0
        # here would make connect() run every migration's ALTER TABLE against a
        # corrupt file, which is more likely to make things worse than leave it be.
        return SCHEMA_VERSION


def migrate(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    current = connection.execute(
        "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
    ).fetchone()[0]
    migrations = {
        1: """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, stage TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, pid INTEGER, error TEXT,
                manifest_path TEXT NOT NULL
            )
        """,
        2: "ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'training'",
        3: "ALTER TABLE jobs ADD COLUMN exit_code INTEGER",
    }
    # Each version applies in its own transaction and records itself immediately, so
    # a crash partway through leaves schema_migrations accurately reflecting what
    # was actually applied (safe to resume from `current + 1` on the next connect).
    for version in range(current + 1, SCHEMA_VERSION + 1):
        with connection:
            connection.execute(migrations[version])
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES(?, ?)",
                (version, datetime.now(UTC).isoformat()),
            )


def backup(home: Path, label: str = "manual") -> Path:
    home = home.resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    target = home / "backups" / f"{timestamp}-{label}"
    target.mkdir(parents=True, exist_ok=False)
    source = sqlite3.connect(home / "studio.db")
    destination = sqlite3.connect(target / "studio.db")
    try:
        # sqlite3's online backup API, not a file copy: safe to run against a live
        # database that another connection (e.g. a running job) may be writing to.
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    manifests = target / "manifests"
    hashes: dict[str, str] = {}
    for path in sorted((home / "jobs").glob("*/manifest.json")):
        relative = path.relative_to(home / "jobs")
        output = manifests / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output)
        hashes[str(relative)] = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = {"created_at": datetime.now(UTC).isoformat(), "label": label, "sha256": hashes}
    (target / "backup.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target


def integrity_check(home: Path) -> tuple[bool, str]:
    with closing(sqlite3.connect(home / "studio.db")) as connection:
        result = str(connection.execute("PRAGMA quick_check").fetchone()[0])
    return result == "ok", result


def restore(home: Path, source: Path) -> Path:
    home = home.resolve()
    source = source.resolve()
    if not (source / "studio.db").is_file() or not (source / "backup.json").is_file():
        raise ValueError("Backup must contain studio.db and backup.json.")
    # Always take a fresh backup of the current state before overwriting it, so a
    # restore is itself undoable.
    safety = backup(home, "pre-restore")
    restored = sqlite3.connect(source / "studio.db")
    destination = sqlite3.connect(home / "studio.db")
    try:
        restored.backup(destination)
    finally:
        destination.close()
        restored.close()
    for manifest in (source / "manifests").glob("*/manifest.json"):
        output = home / "jobs" / manifest.relative_to(source / "manifests")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest, output)
    return safety
