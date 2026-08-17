# Development

## Setup

```powershell
uv sync --locked
uv run streamlit run src/fine_tuning_studio/app.py
```

Python 3.13 and 3.14 are supported. Runtime and development dependencies are declared
in `pyproject.toml` and locked in `uv.lock`.

## Project structure

- `app.py`: Streamlit pages and run-manifest construction.
- `domain.py`: immutable specifications, job states, and validation.
- `system_info.py`: machine, accelerator, package, and integration inspection.
- `jobs.py`: SQLite job registry, workspace management, worker launch, cancellation.
- `worker.py`: multi-dataset preparation, recipe dispatch, evaluation, and export.
- `tests/`: domain, job, system-inspection, and Streamlit smoke tests.

## Verification

Run the same checks used by CI:

```powershell
uv run ruff check .
uv run ty check
uv run pytest -q
uv lock --check
uv run pip-audit --local --progress-spinner off
```

`uv run ruff format --check .` is not run in CI but is a good idea to run locally before
committing, since it catches formatting drift `ruff check` alone does not.

CI runs these checks on Windows and Ubuntu with Python 3.13 and 3.14.

## Design constraints

- Keep credentials in the user environment and never log secret values.
- Ollama remains an export/inference integration rather than a training backend.
- Persist complete run manifests so jobs remain reproducible and inspectable.
- Keep unsupported hardware and training techniques capability-gated in the UI.
- Treat uploaded names and configured paths as untrusted input.

## Releases

The project version is defined in `pyproject.toml`. Before publishing a GitHub tag or
release, run the complete verification suite, update user-facing documentation and
release notes, and ensure no credentials or generated training artifacts are tracked.
