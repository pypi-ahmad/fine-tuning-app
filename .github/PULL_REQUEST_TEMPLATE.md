## What does this PR do?

A short description of the change and why it's needed. Link the issue it addresses, if any (`Closes #___`).

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactor / internal cleanup
- [ ] Other (describe above)

## How was this tested?

- Commands run locally (see [CONTRIBUTING.md](../CONTRIBUTING.md#pull-requests)):
  - [ ] `uv run ruff format --check .`
  - [ ] `uv run ruff check .`
  - [ ] `uv run ty check src tests`
  - [ ] `uv run pytest -q`
  - [ ] `uv lock --check`
- OS/GPU/runtime profile exercised, if relevant:

## Checklist

- [ ] I read [CONTRIBUTING.md](../CONTRIBUTING.md)
- [ ] I updated user, technical, or release documentation if this changes public behavior
- [ ] I did not commit any Hugging Face tokens, credentials, datasets, or model artifacts
- [ ] New training backends or hardware paths are capability-gated and documented
- [ ] This PR is focused on one change (not several unrelated things bundled together)
