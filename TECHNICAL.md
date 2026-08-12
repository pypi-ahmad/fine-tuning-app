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
| Training worker | Loads data/model, renders examples, creates a validation split, runs QLoRA SFT, evaluates, and exports. |
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

V0.1 supports causal-language-model supervised fine-tuning using 4-bit NF4 QLoRA.
The backend is Transformers + PEFT + TRL + bitsandbytes. It selects bfloat16 when CUDA
supports it and float16 otherwise. LoRA targets all linear layers.

Successful jobs always export an adapter and tokenizer. A merged model is optional;
Ollama import requires it. Hub upload pushes the adapter through the authenticated
Hugging Face client.

## Constraints

- Training requires a compatible NVIDIA GPU, CUDA-enabled PyTorch, required packages,
  and at least 3.5 GB free VRAM.
- AMD/Intel hardware may be detected but is not a training backend.
- Standard LoRA, full fine-tuning, Unsloth, preference optimization, and benchmark
  execution are not implemented in v0.1.
- Cancellation is cooperative and takes effect at the next trainer log callback.
- Ollama is post-training inference/export integration, never a training backend.
