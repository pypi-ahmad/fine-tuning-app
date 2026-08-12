# Usage

## Start the application

1. Install [uv](https://docs.astral.sh/uv/).
2. Set `HF_TOKEN` in your user environment if gated/private Hugging Face access or Hub
   uploads are needed.
3. Run `launch.cmd` on Windows or `./launch.sh` on Linux.
4. Open the Streamlit address shown by the launcher.

Use `OLLAMA_HOST` to point to a non-default Ollama API and
`FINE_TUNING_STUDIO_HOME` to choose where local application data is stored. See
[Configuration](docs/configuration.md) for details.

## Fine-tune a model

1. Open **System** and resolve all QLoRA capability warnings.
2. In **Dataset**, choose a Hugging Face dataset or upload JSON, JSONL, CSV, or
   Parquet, then configure the dataset mapping and validation split.
3. In **Model**, enter a Hugging Face model ID or local model directory. Review model
   code before enabling **Allow model repository code**.
4. In **Training**, set epochs, learning rate, sequence length, and optional advanced
   QLoRA parameters.
5. In **Review & run**, choose evaluation and exports, review the generated manifest,
   then start training.
6. In **Monitor**, follow status/events or request cancellation.
7. In **Export**, inspect artifacts and refresh the installed Ollama model list.

## Export options

- Every completed job writes an adapter and tokenizer.
- Enable **Create merged model** to combine the adapter with its base model.
- Enable **Push adapter to Hugging Face Hub** and enter `username/repository` to upload
  the adapter with the environment-provided Hugging Face credential.
- Enable **Import merged model into Ollama** to run `ollama create` after export. This
  requires Ollama on `PATH`, a running service, a model name, and compatible output.

## Important limits

V0.1 runs QLoRA supervised fine-tuning only on compatible NVIDIA CUDA systems. Ollama
is an inference/export target, not a training backend. For dataset requirements,
training defaults, and error recovery, see the detailed
[documentation](README.md#documentation).
