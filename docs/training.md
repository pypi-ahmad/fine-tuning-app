# Training

## Supported methods

| Approach | LoRA | QLoRA | OFT | QOFT |
| --- | :---: | :---: | :---: | :---: |
| Supervised Fine-Tuning | ✅ | ✅ | ✅ | ✅ |
| Reward Modeling | ✅ | ✅ | ✅ | ✅ |
| PPO Training | ✅ | ✅ | ✅ | ✅ |
| DPO Training | ✅ | ✅ | ✅ | ✅ |
| KTO Training | ✅ | ✅ | ✅ | ✅ |
| ORPO Training | ✅ | ✅ | ✅ | ✅ |
| SimPO Training | ✅ | ✅ | ✅ | ✅ |

The dataset page accepts multiple Hub datasets and local uploads. Each source has its own
split, mapping, and template. Sources are normalized, concatenated, shuffled, and divided
once using the global validation fraction and seed. Rows are not weighted or deduplicated.

PPO requires a scalar reward-model ID or local path. Its TRL trainer is experimental and
does not support checkpoint resume.

Continued pretraining supports LoRA, QLoRA, and guarded full tuning. Reward modeling also
supports guarded full tuning. GRPO supports LoRA and QLoRA. Unsloth is limited to
single-GPU SFT with LoRA or QLoRA and does not support OFT or QOFT.

| Approach | Required columns |
| --- | --- |
| Supervised Fine-Tuning / Continued Pre-Training | `text`, or prompt/response or messages mapped to `text` |
| Reward Modeling | `chosen`, `rejected` |
| PPO / GRPO | `prompt` |
| DPO / ORPO / SimPO | `prompt`, `chosen`, `rejected` |
| KTO | `prompt`, `completion`, boolean `label` |

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
| OFT block size | 32 |
| OFT module dropout | 0.0 |
| Preference beta | 0.1 |
| SimPO gamma | 0.5 |
| PPO epochs | 4 |
| PPO response length | 53 |
| PPO KL coefficient | 0.05 |
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

- PPO checkpoint resume
- DeepSpeed, CPU offload, Apple Silicon, or CPU training
- vLLM or tool environments for GRPO
