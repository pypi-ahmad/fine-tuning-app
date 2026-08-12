# Troubleshooting

## Training button is disabled

Open **System** and read every capability warning. CUDA requires compatible NVIDIA
hardware; experimental ROCm/XPU profiles require compatible AMD/Intel hardware and an
on-machine smoke test. Also verify that the recipe, dataset, model, and strategy agree.

## CUDA is unavailable

Check `nvidia-smi`, the NVIDIA driver, and the PyTorch runtime shown in the system
report. Restart the app after driver or environment changes. The project lockfile uses
the configured PyTorch CUDA package index; do not mix unrelated PyTorch builds into the
managed environment.

## Out of memory

Open **System** or the cleanup panel in **Review & run**, refresh GPU usage, and select
**Release reclaimable VRAM**. This clears only unused app caches and unloads running
Ollama models. Memory owned by another application cannot be reclaimed without closing
it; the UI can terminate an eligible same-user process only after explicit selection and
typed confirmation. System/display processes and active training workers are protected.

If memory remains insufficient, select a smaller base model, reduce maximum sequence
length or device batch size, keep gradient checkpointing enabled, and increase gradient
accumulation. Model merging may require more memory than adapter training.

## Gated model or dataset fails to load

Accept the resource license on Hugging Face, set `HF_TOKEN` (preferred) or
`HUGGING_FACE_HUB_TOKEN` in the user environment, confirm the token has read access, and
restart the launcher. The UI reports presence, not the token value.

## Dataset fails after preview

Confirm the file has at least two rows and the configured text, prompt/response, or
messages columns exist. For messages data, each row should contain a list of role/content
objects.

## A job appears stuck

Inspect the Monitor page events and free disk/GPU memory. Cancellation takes effect at
the next trainer logging callback and may wait for model loading, evaluation, or the
current training step.

## Ollama is unavailable

Confirm the service is running, `ollama` is on `PATH`, and `OLLAMA_HOST` points to its
HTTP API. Test the configured endpoint's `/api/tags` route. Ollama import also requires
a merged model and an architecture Ollama supports.

## Resetting local state

Jobs live under `FINE_TUNING_STUDIO_HOME` or `~/.fine-tuning-studio`. Back up anything
important before manually removing job data. Fine-Tuning Studio has no built-in delete
or reset operation.

## Restarting after Stop app

Run `launch.cmd` on Windows or `./launch.sh` on Linux. Jobs that exited cooperatively
remain cancelled; jobs that required process termination are marked interrupted and can
be resumed from an existing checkpoint. If shutdown reports an unverified or surviving
PID, resolve that process manually before retrying. The stop control never ends Ollama.
