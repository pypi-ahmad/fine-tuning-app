# Contributing to Fine-Tuning Studio

Thanks for contributing. Fine-Tuning Studio is a local-first application, so changes
must remain safe for users' machines, models, datasets, and credentials.

## Before you start

- Read the [Code of Conduct](CODE_OF_CONDUCT.md), [Security Policy](SECURITY.md), and
  [Technical architecture](TECHNICAL.md).
- Do not open a public issue for a security vulnerability; use the process in
  `SECURITY.md` instead.
- Discuss substantial changes in an issue before investing in a large pull request.

## Development setup

1. Fork the repository and create a focused branch from `main`.
2. Install [uv](https://docs.astral.sh/uv/).
3. Run `uv sync --locked`.
4. Start the application with `uv run streamlit run src/fine_tuning_studio/app.py`.

Python 3.13 and 3.14 are supported. Do not add or expose real Hugging Face tokens,
model credentials, user datasets, generated checkpoints, or job artifacts in commits.

## Pull requests

- Keep each pull request focused on one user-visible behavior or tightly related fix.
- Explain the problem, the proposed behavior, and verification performed.
- Add or update tests for changed behavior.
- Update user, technical, or release documentation when the public behavior changes.
- Preserve local-first operation and do not introduce network services or telemetry
  without prior discussion.
- Treat file paths, uploaded filenames, model repositories, datasets, and environment
  variables as untrusted input.

Run the checks used by CI before opening a pull request:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run ty check src tests
uv run pytest -q
uv lock --check
uv audit --frozen
```

CI runs this suite on Windows and Ubuntu with Python 3.13 and 3.14.

## Style and compatibility

Use the existing Python style and keep changes small. Avoid speculative configuration,
unrequested abstractions, and unrelated refactors. New training backends or hardware
paths must be capability-gated, documented, and validated on the claimed platform.

## Contributor agreement

By submitting a contribution, you agree that it is licensed under the repository's
[Apache License 2.0](LICENSE), unless you explicitly state otherwise before submission.
