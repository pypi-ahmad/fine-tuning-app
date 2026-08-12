# Training

## Supported method

Version 0.1 supports supervised fine-tuning of causal language models using QLoRA. The
backend is Transformers + PEFT + TRL, with bitsandbytes 4-bit NF4 quantization and
double quantization. LoRA targets all linear layers.

The worker uses bfloat16 when the GPU supports it and float16 otherwise.

## Defaults

| Setting | Default |
|---|---:|
| Epochs | 1.0 |
| Learning rate | 0.0002 |
| Maximum sequence length | 1024 |
| Device batch size | 1 |
| Gradient accumulation | 8 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Sequence packing | Off |
| Gradient checkpointing | On |
| Checkpoint interval | 100 steps |
| Logging interval | 10 steps |

The trainer retains at most two periodic checkpoints. Effective batch size is device
batch size multiplied by gradient accumulation steps.

## Evaluation

Evaluation before and after training is enabled by default and uses the validation
split. Metrics are written to `artifacts/metrics.json`. Standard benchmark execution is
disabled in v0.1 because of unresolved advisories in the stable Lighteval dependency
chain.

## Jobs and cancellation

Each run is executed in a background Python process. Job metadata is stored in SQLite,
events in JSONL, and checkpoints/artifacts in the job directory. Only one worker may be
active at a time.

Cancellation is checked during trainer log callbacks. The current adapter is saved
before the job becomes `cancelled`; cancellation is not instantaneous during model
loading, evaluation, or a long training step.

## Not currently supported

- Standard LoRA without 4-bit base-model loading
- Full-parameter fine-tuning
- Unsloth backend
- DPO, ORPO, KTO, PPO, reward modeling, or continued pretraining
- AMD ROCm, Intel XPU, Apple Silicon, or CPU training
