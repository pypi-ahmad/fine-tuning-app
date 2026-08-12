# Technical Architecture

Fine-Tuning Studio is a local-first Python application. The Streamlit process builds a
validated, immutable run manifest; a separate worker process performs model work. No
remote training service, hosted database, or telemetry backend is used.

## Components

| Component | Responsibility |
|---|---|
| Streamlit UI | System inspection, configuration, manifest review, job monitoring, and artifact/Ollama views. |
| Domain model | Dataclasses for dataset, model, training, evaluation, export, run manifest, validation, and job states. |
| System inspection | Detects OS, CPU, memory, disk, GPU, PyTorch/CUDA, package availability, Hugging Face token presence, and Ollama status. |
| Job registry | SQLite database and application workspace for job metadata, manifests, inputs, events, checkpoints, and artifacts. |
| Training worker | Dispatches SFT, CPT, preference, reward, and GRPO recipes and exports results. |
| Ollama integration | Imports a completed merged model through `ollama create` and lists local models through `/api/tags`. |

## Run lifecycle

```text
UI configuration
  -> RunManifest validation
  -> SQLite job record + job workspace
  -> background worker
  -> preparing -> optional baseline evaluation -> training
  -> optional final evaluation -> export -> completed
```

The worker writes structured events to `events.jsonl` and updates job progress in
SQLite. The UI shows the most recent 100 events. Only one active training worker is
permitted at a time.

## Data and workspace

`FINE_TUNING_STUDIO_HOME` selects the application home; the default is
`~/.fine-tuning-studio`. It contains `studio.db` and `jobs/<job-id>/` directories.
Each job may contain a manifest, sanitized local uploads, cancellation marker,
checkpoints, event log, adapter, merged model, tokenizer, and metrics.

The worker accepts Hugging Face datasets or local JSON, JSONL, CSV, and Parquet files.
It converts the configured `text`, prompt/response, or messages mapping to a `text`
field and creates a seeded train/validation split.

## Training and exports

V0.5 supports LoRA, NF4 QLoRA, and safety-gated full training. Recipes define dataset
contracts and trainer dispatch. Managed CUDA, ROCm, and XPU interpreters are separate
from the lightweight UI process; SQLite is only a status projection, while immutable
manifests, JSONL events, checkpoints, and hashed artifacts form the durable job record.

Successful jobs always export an adapter and tokenizer. A merged model is optional;
Ollama import requires it. Hub upload pushes the adapter through the authenticated
Hugging Face client.

## Constraints

- CUDA is stable; ROCm and XPU are experimental until validated on the target machine.
- Full training is single-accelerator only and refuses unsafe memory or disk estimates.
- ORPO remains blocked because the installed TRL release has no ORPO trainer.
- Custom reward modules are trusted local Python and are not sandboxed.
- Cancellation is cooperative and takes effect at the next trainer log callback.
- Ollama is post-training inference/export integration, never a training backend.
