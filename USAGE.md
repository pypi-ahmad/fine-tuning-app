# Usage

## Start the application

1. Install [uv](https://docs.astral.sh/uv/).
2. Set `HF_TOKEN` in your user environment if gated/private Hugging Face access or Hub
   uploads are needed.
3. Run `launch.cmd` on Windows or `./launch.sh` on Linux.
4. Open the Streamlit address shown by the launcher.

The equivalent command is `uv run --locked fine-tuning-studio run`. The server binds
only to `127.0.0.1`.

Use `OLLAMA_HOST` to point to a non-default Ollama API and
`FINE_TUNING_STUDIO_HOME` to choose where local application data is stored. See
[Configuration](docs/configuration.md) for details.

## Fine-tune a model

1. Open **System**, choose a runtime profile, resolve its capability warnings, and use
   **Release reclaimable VRAM** if another cache or Ollama model is occupying the GPU.
2. In **Dataset**, enter a Hugging Face dataset ID or repository URL, or upload JSON,
   JSONL, CSV, or Parquet. For Hub data, select **Validate & download dataset** to cache
   the selected revision, then configure the mapping and validation split.
3. In **Model**, enter a Hugging Face model ID or repository URL, or a local model
   directory. Select **Validate & download model** to cache Hub files. Review model code
   before enabling **Allow model repository code**.
4. In **Training**, set epochs, learning rate, sequence length, and optional advanced
   recipe, strategy, runtime, and optional advanced parameters.
5. In **Review & run**, choose evaluation and exports, review the generated manifest,
   optionally refresh/release GPU memory immediately before launch, then start training.
6. In **Monitor**, follow status/events or request cancellation.
7. In **Export**, inspect artifacts and any explicit merged-model import result.
8. In **Ollama playground**, test an already-installed Ollama model with streamed text
   chat. This page does not import a training artifact.

## Export options

- Adapter jobs write an adapter and tokenizer; full tuning writes a full model.
- Enable **Create merged model** to combine the adapter with its base model.
- Enable **Push adapter to Hugging Face Hub** and enter `username/repository` to upload
  the adapter with the environment-provided Hugging Face credential.
- Enable **Import merged model into Ollama** to run `ollama create` after export. This
  requires Ollama on `PATH`, a running service, a model name, and compatible output.

## Important limits

V1.0 supports stable single-GPU CUDA plus probe-gated ROCm/XPU and distributed beta
profiles. Native Windows ROCm is not claimed; WSL2 is optional for supported AMD
hardware. Ollama is an inference/export target, not a training backend. For dataset
requirements, training defaults, and error recovery, see the detailed
[documentation](README.md#documentation).

## Maintenance

Use `fine-tuning-studio doctor --json`, `fine-tuning-studio backup`, and
`fine-tuning-studio restore PATH`. Stop all jobs before restore.
