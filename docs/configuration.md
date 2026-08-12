# Configuration

Fine-Tuning Studio reads configuration from user environment variables. Never place
tokens in source files, `.env` files intended for Git, screenshots, or issue reports.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `HF_TOKEN` | Only for gated/private resources or Hub uploads | unset | Hugging Face authentication |
| `OLLAMA_HOST` | No | `http://127.0.0.1:11434` | Ollama API used for status and `/api/tags` |
| `FINE_TUNING_STUDIO_HOME` | No | `~/.fine-tuning-studio` | Job database, manifests, uploads, logs, checkpoints, and artifacts |

## Windows PowerShell

Persist variables for the current user, then reopen the terminal or launcher:

```powershell
[Environment]::SetEnvironmentVariable("HF_TOKEN", "your-token", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_HOST", "http://127.0.0.1:11434", "User")
```

## Linux

Add exports to your shell profile and start a new shell:

```bash
export HF_TOKEN="your-token"
export OLLAMA_HOST="http://127.0.0.1:11434"
```

The system report exposes only a boolean indicating whether `HF_TOKEN` is present.

## Storage layout

The application home contains `studio.db` and `jobs/<job-id>/`. A job directory can
contain `manifest.json`, uploaded inputs, `events.jsonl`, cancellation state,
checkpoints, and exported artifacts. Changing `FINE_TUNING_STUDIO_HOME` does not move
existing jobs.
