# Fine-Tuning Studio

Fine-Tuning Studio is a local-first Streamlit application for preparing datasets,
running QLoRA supervised fine-tuning, evaluating results, exporting adapters or merged
models, and testing exported models with Ollama.

The training process runs on your machine. Hugging Face credentials are read from the
user environment and are never displayed or written into the project.

## Current capabilities

- Detect the operating system, CPU, memory, disk, GPUs, CUDA runtime, Python packages,
  Hugging Face authentication state, and Ollama availability.
- Load datasets from the Hugging Face Hub or upload JSON, JSONL, CSV, and Parquet files.
- Map plain text, prompt/response, or chat-message datasets into training text.
- Fine-tune causal language models using QLoRA SFT with Transformers, PEFT, TRL, and
  bitsandbytes.
- Track jobs, checkpoints, progress, logs, cancellation, and evaluation metrics.
- Export a LoRA adapter, optionally merge it into the base model, push the adapter to
  the Hugging Face Hub, or import a merged model into Ollama.

Version 0.1 supports QLoRA SFT on NVIDIA CUDA. AMD and Intel accelerators are detected
but are not enabled as training backends. Unsloth, standard LoRA, full fine-tuning, and
preference-optimization techniques are planned rather than implemented.

## Requirements

- Windows 11 or Linux
- Python 3.13 or 3.14
- [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU with a working CUDA runtime
- At least 3.5 GB of currently free VRAM for the smallest supported jobs
- Enough RAM and disk space for the selected model, dataset, checkpoints, and exports
- Optional: [Ollama](https://ollama.com/) for post-training local inference

## Quick start

1. Clone or download the repository.
2. Install `uv`.
3. Set `HF_TOKEN` in your user environment if you use gated or private Hub resources.
4. Double-click `launch.cmd` on Windows, or run `./launch.sh` on Linux.
5. Open the Streamlit URL printed by the launcher.

The launchers run `uv run --locked`, so dependencies come from the committed lockfile.

## Documentation

- [User guide](docs/user-guide.md)
- [Datasets](docs/datasets.md)
- [Training](docs/training.md)
- [Configuration](docs/configuration.md)
- [Export and Ollama](docs/export-and-ollama.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Development](docs/development.md)
- [Usage](USAGE.md)
- [Technical architecture](TECHNICAL.md)

## Community and project policies

- [Contributing](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security](SECURITY.md)
- [Changelog](CHANGELOG.md)

## Security

Fine-Tuning Studio reports only whether `HF_TOKEN` exists; it does not show or store the
token value. Uploaded filenames are sanitized and job paths are constrained to the app
workspace. Enabling `trust_remote_code` can execute code from a model repository, so use
it only after reviewing and trusting that repository.

Standard benchmark execution is currently disabled because the stable Lighteval
dependency chain includes unresolved security advisories. Validation loss and
perplexity-style evaluation before and after training remain available.

## License

Apache License 2.0. See [LICENSE](LICENSE).
