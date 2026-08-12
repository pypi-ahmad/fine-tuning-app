# Training

## Supported methods

Version 0.5 supports SFT and continued pretraining with LoRA, QLoRA, or guarded full
tuning. DPO, KTO, reward modeling, and GRPO use their objective-specific dataset
contracts. ORPO remains visible but blocked because TRL 1.9 has no ORPO trainer.

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
split. Metrics are written to `artifacts/metrics.json`. Optional MMLU, GSM8K,
HellaSwag, ARC, TruthfulQA, and Winogrande tasks run through isolated Inspect AI.

## Jobs and cancellation

Each run is executed in a background Python process. Job metadata is stored in SQLite,
events in JSONL, and checkpoints/artifacts in the job directory. Only one worker may be
active at a time.

Cancellation is checked during trainer log callbacks. The current adapter is saved
before the job becomes `cancelled`; cancellation is not instantaneous during model
loading, evaluation, or a long training step.

## Limits

- ORPO with the installed TRL release
- PPO, multi-GPU, FSDP, DeepSpeed, CPU offload, Apple Silicon, or CPU training
- vLLM or tool environments for GRPO
