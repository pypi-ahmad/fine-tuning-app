from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from fine_tuning_studio import __version__
from fine_tuning_studio.jobs import list_jobs, reconcile_jobs, studio_home
from fine_tuning_studio.storage import backup, integrity_check, restore
from fine_tuning_studio.system_info import scan_machine


def doctor(as_json: bool = False) -> int:
    home = studio_home()
    database_ok, database_result = integrity_check(home)
    interrupted = reconcile_jobs()
    report = scan_machine(home).sanitized_dict()
    result = {
        "version": __version__,
        "home": str(home),
        "database": {"ok": database_ok, "result": database_result},
        "interrupted_jobs_reconciled": interrupted,
        "machine": report,
    }
    print(json.dumps(result, indent=2) if as_json else _doctor_text(result))
    return 0 if database_ok else 1


def _doctor_text(result: Mapping[str, object]) -> str:
    database = result["database"]
    assert isinstance(database, dict)
    return "\n".join(
        [
            f"Fine-Tuning Studio {result['version']}",
            f"Home: {result['home']}",
            f"Database: {'ok' if database['ok'] else database['result']}",
            f"Interrupted jobs reconciled: {result['interrupted_jobs_reconciled']}",
        ]
    )


def run_app() -> int:
    entry = Path(__file__).with_name("app.py")
    environment = os.environ.copy()
    environment.setdefault("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")
    environment.setdefault("STREAMLIT_SERVER_PORT", "8503")
    environment.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(entry)], env=environment)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fine-tuning-studio")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", help="Run the loopback-only Streamlit app")
    doctor_parser = commands.add_parser("doctor", help="Check storage, hardware, and integrations")
    doctor_parser.add_argument("--json", action="store_true")
    commands.add_parser("backup", help="Create a consistent local backup")
    restore_parser = commands.add_parser("restore", help="Restore a local backup")
    restore_parser.add_argument("path", type=Path)
    commands.add_parser("version", help="Print the installed version")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_app()
    if args.command == "doctor":
        return doctor(args.json)
    if args.command == "backup":
        print(backup(studio_home()))
        return 0
    if args.command == "restore":
        if any(job["status"] in {"preparing", "training", "exporting"} for job in list_jobs()):
            parser.error("Stop active jobs before restoring a backup.")
        print(f"Safety backup: {restore(studio_home(), args.path)}")
        return 0
    print(__version__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
