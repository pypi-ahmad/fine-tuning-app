# Export and Ollama

## Adapter export

Every successful job saves the PEFT adapter and tokenizer under
`artifacts/adapter/`. Evaluation output is stored in `artifacts/metrics.json`.

Cancelled training saves the current adapter under `artifacts/adapter-cancelled/`.

## Merged model

Enable **Create merged model** to merge the trained adapter into the base model. The
result is saved with safe serialization under `artifacts/merged-model/`. Merging needs
additional memory and disk space beyond training requirements.

## Hugging Face Hub

Enable Hub upload, enter `username/repository`, and choose repository visibility. The
worker pushes the adapter through the authenticated Hugging Face client. `HF_TOKEN`
(preferred) or `HUGGING_FACE_HUB_TOKEN` must have suitable write permission.

## Ollama

Ollama is a post-training inference target, not a training backend.

1. Install Ollama and ensure the `ollama` command is on `PATH`.
2. Start the Ollama service.
3. Enable **Import merged model into Ollama** and choose a model name.
4. Fine-Tuning Studio automatically enables merged-model export.
5. After training, it writes a minimal `Modelfile` and runs
   `ollama create <name> -f <Modelfile>`.
6. Open **Ollama playground**, refresh installed models, and select the imported name.

Import compatibility ultimately depends on whether the exported model architecture and
format are supported by the installed Ollama version. A successful Transformers merge
does not guarantee Ollama compatibility.

## Ollama playground

The playground calls the local API configured by `OLLAMA_HOST`. It lists models already
installed through `/api/tags`, inspects the selected model through `/api/show`, and
streams text chat through `/api/chat`. It can unload the selected model from memory but
does not pull, create, delete, merge, or import models. Trained-model import occurs only
through the explicit export option above.
