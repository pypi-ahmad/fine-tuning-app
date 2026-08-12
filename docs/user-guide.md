# User guide

Fine-Tuning Studio presents the workflow as eight pages. Complete them in order for a
new run.

**Stop app** is always available at the bottom of the sidebar. Its confirmation dialog
requests safe cancellation for active jobs, waits briefly, stops remaining verified
Fine-Tuning Studio process trees, and then closes the local server. Checkpoints, logs,
and job records remain available after restart. Ollama and unrelated processes are not
stopped.

## 1. System

The initial scan reports the operating system, CPU threads, available memory, free disk
space, accelerators, installed training packages, CUDA state, Hugging Face token state,
and Ollama state. It does not install drivers or reveal credentials.

Training is enabled only when an NVIDIA GPU, CUDA-enabled PyTorch, the required Python
packages, and at least 3.5 GB of free VRAM are detected. The sanitized report can be
downloaded for troubleshooting.

The GPU cleanup panel reports free VRAM and GPU processes. **Release reclaimable VRAM**
clears unused allocator cache and unloads Ollama models. Ending another user-owned GPU
process requires selecting it and typing `TERMINATE`; protected processes cannot be
selected.

## 2. Dataset

Choose a Hugging Face dataset ID or repository URL and revision/split, or upload a JSON,
JSONL, CSV, or Parquet file. **Validate & download dataset** places Hub files in the
shared Hugging Face cache. Select the column mapping and chat template described in
[Datasets](datasets.md). Local files are copied into the job workspace when the run is
created.

The worker requires at least two rows. It creates a shuffled train/validation split
using the configured fraction and seed.

## 3. Model

Choose a Hugging Face causal language-model ID, repository URL, or local model directory.
Hub models default to revision `main`. **Validate & download model** caches the selected
repository and then displays basic Hub metadata.

Leave **Allow model repository code** disabled unless the model requires custom code
and you have reviewed the repository.

## 4. Training

Configure epochs, learning rate, sequence length, batch size, gradient accumulation,
LoRA parameters, packing, gradient checkpointing, and logging/checkpoint intervals.
See [Training](training.md) for defaults and behavior.

## 5. Review and run

Choose baseline/final evaluation and export options. Review the generated run manifest.
The start button remains disabled until the configuration and machine capability checks
pass. The optional GPU cleanup expander refreshes memory immediately before launch. Only
one training worker can be active at a time.

## 6. Monitor

The monitor lists recent jobs, current stage, progress, and the last 100 structured
events. Cancellation is cooperative: the worker stops at the next training callback and
saves the current adapter as `adapter-cancelled`.

## 7. Export

Completed-job artifacts are listed here. Explicit merged-model import remains part of
the export workflow. See [Export and Ollama](export-and-ollama.md).

## 8. Ollama playground

Select a model already installed in Ollama and start a streamed, multi-turn text chat.
Generation settings include a system prompt, temperature, top-p, seed, output-token
limit, and unload-after-response. Conversations remain only in the current browser
session. The playground never pulls, creates, deletes, merges, or imports a model.
