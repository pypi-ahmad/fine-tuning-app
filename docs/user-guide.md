# User guide

Fine-Tuning Studio presents the workflow as seven pages. Complete them in order for a
new run.

## 1. System

The initial scan reports the operating system, CPU threads, available memory, free disk
space, accelerators, installed training packages, CUDA state, Hugging Face token state,
and Ollama state. It does not install drivers or reveal credentials.

Training is enabled only when an NVIDIA GPU, CUDA-enabled PyTorch, the required Python
packages, and at least 3.5 GB of free VRAM are detected. The sanitized report can be
downloaded for troubleshooting.

## 2. Dataset

Choose a Hugging Face dataset ID and revision/split, or upload a JSON, JSONL, CSV, or
Parquet file. Select the column mapping and chat template described in
[Datasets](datasets.md). Local files are copied into the job workspace when the run is
created.

The worker requires at least two rows. It creates a shuffled train/validation split
using the configured fraction and seed.

## 3. Model

Choose a Hugging Face causal language-model ID or a local model directory. Hub models
default to revision `main`. The metadata inspection button displays basic Hub metadata.

Leave **Allow model repository code** disabled unless the model requires custom code
and you have reviewed the repository.

## 4. Training

Configure epochs, learning rate, sequence length, batch size, gradient accumulation,
LoRA parameters, packing, gradient checkpointing, and logging/checkpoint intervals.
See [Training](training.md) for defaults and behavior.

## 5. Review and run

Choose baseline/final evaluation and export options. Review the generated run manifest.
The start button remains disabled until the configuration and machine capability checks
pass. Only one training worker can be active at a time.

## 6. Monitor

The monitor lists recent jobs, current stage, progress, and the last 100 structured
events. Cancellation is cooperative: the worker stops at the next training callback and
saves the current adapter as `adapter-cancelled`.

## 7. Export

Completed-job artifacts are listed here. The page can also query Ollama's `/api/tags`
endpoint to show installed local models. See [Export and Ollama](export-and-ollama.md).
