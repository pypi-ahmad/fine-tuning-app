# Fine-Tuning Studio

Fine-Tuning Studio is a local-first Streamlit application for preparing datasets,
running post-training recipes, evaluating results, exporting adapters or merged
models, and testing exported models with Ollama.

The training process runs on your machine. Hugging Face credentials are read from the
user environment and are never displayed or written into the project.

## Current capabilities

- Detect the operating system, CPU, memory, disk, GPUs, CUDA runtime, Python packages,
  Hugging Face authentication state, and Ollama availability.
- Inspect GPU processes, release reclaimable allocator/Ollama VRAM, and selectively
  terminate an eligible user-owned GPU process after typed confirmation.
- Load datasets and models from a Hugging Face ID or repository URL, with validation,
  cache-aware downloads, or upload JSON, JSONL, CSV, and Parquet datasets.
- Map plain text, prompt/response, or chat-message datasets into training text.
- Run SFT or continued pretraining with LoRA, QLoRA, or guarded full fine-tuning.
- Run DPO, KTO, reward-model, and GRPO recipes; ORPO is visible but blocked while the
  installed TRL release does not provide an ORPO trainer.
- Use Transformers/PEFT/TRL or compatible installed Unsloth builds for SFT.
- Select native Windows/Linux CUDA, ROCm, or XPU runtime profiles. ROCm and XPU remain
  experimental until their on-machine smoke test passes.
- Track jobs, checkpoints, progress, logs, cancellation, and evaluation metrics.
- Export a LoRA adapter, optionally merge it into the base model, push the adapter to
  the Hugging Face Hub, or import a merged model into Ollama.
- Stream text chats with models already installed in Ollama through a separate playground.

Version 1.0 adds durable migrations and backups, schema-v4 manifests, bounded preflight,
evaluation reports, physical GPU selection, probe-gated DDP/FSDP2, a packaged CLI, and
attested GitHub release artifacts.

## Requirements

- Windows 11 or Linux
- Python 3.12.10, 3.13, or 3.14
- [uv](https://docs.astral.sh/uv/)
- Supported NVIDIA, AMD, or Intel GPU with a working selected runtime
- At least 3.5 GB of currently free VRAM for the smallest supported jobs
- Enough RAM and disk space for the selected model, dataset, checkpoints, and exports
- Optional: [Ollama](https://ollama.com/) for post-training local inference

## Install and run locally

Install [Git](https://git-scm.com/downloads) and
[uv](https://docs.astral.sh/uv/getting-started/installation/) first. Ollama is optional
and is used only to test trained or already-installed models, not as a training backend.

### 1. Clone the repository

```text
git clone https://github.com/pypi-ahmad/fine-tuning-app.git
cd fine-tuning-app
```

### 2. Install Python and dependencies

```text
uv python install 3.13
uv sync --locked
```

`uv` creates an isolated project environment and installs the exact dependency versions
from `uv.lock`.

### 3. Configure Hugging Face access (optional)

Public models and datasets do not require a token. For gated/private resources or Hub
uploads, create a Hugging Face token and save it as the `HF_TOKEN` user environment
variable. Never place the token in this repository.

Windows PowerShell (reopen the terminal afterward):

```powershell
[Environment]::SetEnvironmentVariable("HF_TOKEN", "your-token", "User")
```

Linux (add this to your shell profile for future sessions):

```bash
export HF_TOKEN="your-token"
```

See [Configuration](docs/configuration.md) for storage and optional environment settings.

### 4. Set up Ollama (optional)

Install [Ollama](https://ollama.com/download), start its local service, and verify it:

```text
ollama list
```

### 5. Launch Fine-Tuning Studio

On Windows 11, double-click `launch.cmd` or run:

```powershell
.\launch.cmd
```

On Linux, run:

```bash
./launch.sh
```

You can also use `uv run --locked fine-tuning-studio run` on either platform. Open
[http://127.0.0.1:8503](http://127.0.0.1:8503) after the server starts.

The web server is loopback-only. Multi-user and LAN deployment are unsupported.

Useful maintenance commands:

```text
fine-tuning-studio doctor --json
fine-tuning-studio backup
fine-tuning-studio restore PATH
fine-tuning-studio version
```

## Documentation

- [User guide](docs/user-guide.md)
- [Datasets](docs/datasets.md)
- [Training](docs/training.md)
- [Configuration](docs/configuration.md)
- [Export and Ollama](docs/export-and-ollama.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Development](docs/development.md)
- [Zero-to-hero handbook](docs/handbook.html)
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

Optional benchmark execution uses an isolated Inspect AI installation. Custom Python
reward modules are explicitly trusted, copied and hashed into the job, and execute
unsandboxed.

## License

Apache License 2.0. See [LICENSE](LICENSE).
