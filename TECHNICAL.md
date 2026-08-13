# Technical Architecture

Fine-Tuning Studio is a local-first Python application. The Streamlit process builds a
validated, immutable run manifest; a separate worker process performs model work. No
remote training service, hosted database, or telemetry backend is used.

## Components

| Component | Responsibility |
|---|---|
| Streamlit UI | System inspection, configuration, manifest review, job monitoring, and artifact/Ollama views. |
| Domain model | Dataclasses for dataset, model, training, evaluation, export, run manifest, validation, and job states. |
| System inspection | Detects OS, CPU, memory, disk, GPU processes/VRAM, PyTorch/CUDA, package availability, Hugging Face token presence, and Ollama status. |
| Job registry | Migrated SQLite database plus immutable manifests, inputs, events, checkpoints, artifacts, and verified backups. |
| Training worker | Dispatches SFT, CPT, preference, reward, PPO, ORPO, SimPO, and GRPO recipes and exports results. |
| Lifecycle control | Cooperatively cancels jobs, verifies app-owned process trees, escalates when necessary, and cleanly exits the local server. |
| Ollama integration | Keeps explicit merged-model import separate from an installed-model playground using `/api/tags`, `/api/show`, `/api/ps`, and streamed `/api/chat`. |

GPU cleanup uses vendor tools when available: `nvidia-smi`, `amd-smi`, or `xpu-smi`.
It can clear only unused allocator cache in the UI process, unload Ollama models through
their API, or terminate a user-selected process. It cannot clear live allocations owned
by another process without ending that process and never performs a driver-level reset.

## Run lifecycle

```text
UI configuration
  -> RunManifest validation
  -> SQLite job record + job workspace
  -> preflight -> background worker or Accelerate launcher
  -> preparing -> optional baseline evaluation -> training
  -> optional final evaluation -> export -> completed
```

The worker writes structured events to `events.jsonl` and updates job progress in
SQLite. The UI shows the most recent 100 events. Only one active training worker is
permitted at a time.

The global stop control first writes cancellation markers and waits up to ten seconds.
Remaining worker PIDs must match the current app ancestry or registered worker command
before their process trees are terminated. If ownership cannot be verified or a process
survives termination, the UI remains available and reports the error instead of hiding
an orphaned job. Ollama and unrelated processes remain outside this lifecycle.

## Data and workspace

`FINE_TUNING_STUDIO_HOME` selects the application home; the default is
`~/.fine-tuning-studio`. It contains `studio.db`, `jobs/<job-id>/`, managed runtimes,
and timestamped backups. SQLite migrations are transactional and a consistent database
plus manifest backup is made before an upgrade.
Each job may contain a manifest, sanitized local uploads, cancellation marker,
checkpoints, event log, adapter, merged model, tokenizer, and metrics.

The worker accepts one or more Hugging Face datasets or local JSON, JSONL, CSV, and
Parquet files. Each source is mapped independently. Text, prompt/response, and messages
shapes become a `text` field; preference, KTO, PPO, and GRPO shapes keep their recipe
columns. Prepared sources are concatenated, then a seeded train/validation split is
applied once. Schema-v5 manifests store `dataset.sources`; older single-source
manifests still load.

## Training and exports

V1.1 supports LoRA, NF4 QLoRA, OFT, QOFT, and safety-gated full training. Recipes define
dataset contracts and trainer dispatch, including PPO through TRL's experimental trainer
and ORPO/SimPO through TRL DPO loss settings. Managed CUDA, ROCm, and XPU interpreters
are separate from the lightweight UI process; SQLite is only a status projection, while
immutable manifests, JSONL events, checkpoints, and hashed artifacts form the durable
job record.

Successful adapter jobs export a PEFT adapter and tokenizer. Full tuning exports a
full model. A merged model is optional; Ollama import requires it. Hub upload pushes
the adapter through the authenticated Hugging Face client.

## Constraints

- CUDA single-GPU is stable; ROCm, XPU, Unsloth, DDP, and FSDP2 are beta until their
  on-machine probes pass. Native Windows ROCm is not claimed; use a supported Linux or
  optional WSL2 environment.
- Full training is single-accelerator only and refuses unsafe memory or disk estimates.
- PPO cannot resume from a checkpoint. Unsloth is limited to single-GPU SFT with LoRA
  or QLoRA and does not support OFT or QOFT.
- Custom reward modules are trusted local Python and are not sandboxed.
- Cancellation writes a cooperative marker and signals the complete worker process tree.
- Ollama is post-training inference/export integration, never a training backend.
