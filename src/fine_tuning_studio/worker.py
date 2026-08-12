from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from fine_tuning_studio.artifacts import write_artifact_manifest
from fine_tuning_studio.domain import JobStatus, RunManifest
from fine_tuning_studio.evals import run_inspect
from fine_tuning_studio.jobs import get_job, job_directory, update_job
from fine_tuning_studio.recipes import REWARDS, load_trusted_reward, validate_recipe
from fine_tuning_studio.resources import full_training_gate


class EventWriter:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.path = job_directory(job_id) / "events.jsonl"

    def write(self, stage: str, message: str, progress: float | None = None, **data: Any) -> None:
        event = {
            "time": datetime.now(UTC).isoformat(),
            "stage": stage,
            "message": message,
            "data": data,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, default=str) + "\n")
        fields: dict[str, Any] = {"stage": message}
        if progress is not None:
            fields["progress"] = max(0.0, min(1.0, progress))
        update_job(self.job_id, **fields)


def load_manifest(job_id: str) -> RunManifest:
    job = get_job(job_id)
    if not job:
        raise ValueError(f"Unknown job: {job_id}")
    value = json.loads(Path(job["manifest_path"]).read_text(encoding="utf-8"))
    return RunManifest.from_dict(value)


def load_data(manifest: RunManifest, events: EventWriter) -> Any:
    from datasets import load_dataset

    spec = manifest.dataset
    events.write("preparing", "Loading dataset", 0.05)
    if spec.source == "hub":
        dataset = load_dataset(spec.location, revision=spec.revision, split=spec.split)
    else:
        suffix = Path(spec.location).suffix.lower()
        loader = {".csv": "csv", ".json": "json", ".jsonl": "json", ".parquet": "parquet"}.get(
            suffix
        )
        if not loader:
            raise ValueError(f"Unsupported dataset format: {suffix}")
        dataset = load_dataset(loader, data_files=spec.location, split="train")
    if len(dataset) < 2:
        raise ValueError("The dataset must contain at least two rows.")
    return dataset


def render_dataset(dataset: Any, manifest: RunManifest, tokenizer: Any) -> Any:
    spec = manifest.dataset
    columns = set(dataset.column_names)
    if manifest.training.objective in {"dpo", "kto", "reward", "orpo", "grpo"}:
        errors = validate_recipe(manifest.training.objective, manifest.training.method, columns)
        if errors:
            raise ValueError(" ".join(errors))
        if manifest.training.objective == "kto" and not all(
            isinstance(value, bool) for value in dataset["label"]
        ):
            raise ValueError("KTO label values must be booleans.")
        return dataset
    if spec.mapping == "text":
        if spec.text_column not in columns:
            raise ValueError(f"Missing text column: {spec.text_column}")
        if spec.text_column != "text":
            dataset = dataset.rename_column(spec.text_column, "text")
        if manifest.training.objective == "continued_pretraining":
            eos = tokenizer.eos_token or ""

            def append_eos(row: dict[str, Any]) -> dict[str, str]:
                return {"text": str(row[spec.text_column]) + eos}

            return dataset.map(append_eos)
        return dataset
    if spec.mapping == "prompt_response":
        missing = {spec.prompt_column, spec.response_column} - columns
        if missing:
            raise ValueError(f"Missing prompt/response columns: {', '.join(sorted(missing))}")

        def combine(row: dict[str, Any]) -> dict[str, str]:
            prompt = str(row[spec.prompt_column]).strip()
            response = str(row[spec.response_column]).strip()
            if spec.chat_template == "alpaca":
                text = f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
            elif spec.chat_template == "chatml":
                text = (
                    f"<|im_start|>user\n{prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n{response}<|im_end|>"
                )
            else:
                text = f"{prompt}\n{response}"
            return {"text": text}

        return dataset.map(combine)
    if spec.text_column not in columns:
        raise ValueError(f"Missing messages column: {spec.text_column}")

    def render_messages(row: dict[str, Any]) -> dict[str, str]:
        messages = row[spec.text_column]
        if spec.chat_template == "tokenizer" and tokenizer.chat_template:
            return {
                "text": tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            }
        parts = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            if spec.chat_template == "chatml":
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            else:
                parts.append(f"{role.title()}: {content}")
        return {"text": "\n".join(parts)}

    return dataset.map(render_messages)


def import_into_ollama(manifest: RunManifest, merged_path: Path, events: EventWriter) -> None:
    if not shutil.which("ollama"):
        raise RuntimeError("Ollama CLI is not installed or not on PATH.")
    modelfile = merged_path.parent / "Modelfile"
    modelfile.write_text(f"FROM {merged_path.as_posix()}\n", encoding="utf-8")
    events.write("exporting", "Importing merged model into Ollama", 0.95)
    result = subprocess.run(
        ["ollama", "create", manifest.export.ollama_model_name, "-f", str(modelfile)],
        check=False,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if result.returncode:
        raise RuntimeError(f"Ollama import failed: {result.stderr[-1000:]}")


def run(job_id: str) -> None:
    import torch
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
    )
    from trl import SFTConfig, SFTTrainer

    manifest = load_manifest(job_id)
    events = EventWriter(job_id)
    directory = job_directory(job_id)
    cancel_path = directory / "cancel.requested"
    artifacts = directory / "artifacts"
    artifacts.mkdir(exist_ok=True)

    class ProgressCallback(TrainerCallback):
        def on_log(
            self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **_: Any
        ) -> Any:
            total = max(1, state.max_steps)
            progress = 0.25 + (state.global_step / total) * 0.55
            events.write(
                "training", "Training", progress, step=state.global_step, metrics=logs or {}
            )
            if cancel_path.exists():
                control.should_training_stop = True
            return control

    update_job(job_id, status=JobStatus.PREPARING, stage="Preparing", progress=0.01)
    events.write("preparing", "Loading tokenizer", 0.02)
    model_ref = manifest.model.location
    revision = manifest.model.revision if manifest.model.source == "hub" else None
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        revision=revision,
        trust_remote_code=manifest.model.trust_remote_code,
    )
    tokenizer = cast(Any, tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = render_dataset(load_data(manifest, events), manifest, tokenizer)
    split = dataset.train_test_split(
        test_size=manifest.dataset.validation_fraction,
        seed=manifest.dataset.seed,
        shuffle=True,
    )
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = None
    if manifest.training.method == "qlora":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    events.write("preparing", f"Loading base model for {manifest.training.method}", 0.12)
    if manifest.training.backend == "unsloth":
        FastLanguageModel = importlib.import_module("unsloth").FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_ref,
            max_seq_length=manifest.training.max_sequence_length,
            load_in_4bit=manifest.training.method == "qlora",
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=manifest.training.lora_rank,
            lora_alpha=manifest.training.lora_alpha,
            lora_dropout=manifest.training.lora_dropout,
            target_modules="all-linear",
            use_gradient_checkpointing=manifest.training.gradient_checkpointing,
        )
    else:
        model_class = (
            AutoModelForSequenceClassification
            if manifest.training.objective == "reward"
            else AutoModelForCausalLM
        )
        model = model_class.from_pretrained(
            model_ref,
            revision=revision,
            trust_remote_code=manifest.model.trust_remote_code,
            quantization_config=quantization,
            device_map="auto",
            dtype=compute_dtype,
        )
    if manifest.training.method == "full":
        free_bytes, _ = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
        gate = full_training_gate(
            model.num_parameters(), free_bytes / 1024**3, directory
        )
        if not gate.allowed:
            raise RuntimeError("Full-training safety gate: " + " ".join(gate.reasons))
    if manifest.training.method == "qlora" and manifest.training.backend != "unsloth":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=manifest.training.gradient_checkpointing,
        )
    peft_config = None
    if manifest.training.method != "full" and manifest.training.backend != "unsloth":
        peft_config = LoraConfig(
            r=manifest.training.lora_rank,
            lora_alpha=manifest.training.lora_alpha,
            lora_dropout=manifest.training.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
    common_args: dict[str, Any] = dict(
        output_dir=str(directory / "checkpoints"),
        num_train_epochs=manifest.training.epochs,
        learning_rate=manifest.training.learning_rate,
        max_length=manifest.training.max_sequence_length,
        per_device_train_batch_size=manifest.training.batch_size,
        per_device_eval_batch_size=manifest.training.batch_size,
        gradient_accumulation_steps=manifest.training.gradient_accumulation_steps,
        gradient_checkpointing=manifest.training.gradient_checkpointing,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        logging_steps=manifest.training.logging_steps,
        save_steps=manifest.training.save_steps,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=manifest.training.save_steps,
        report_to="none",
        seed=manifest.dataset.seed,
    )
    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[ProgressCallback()],
    )
    objective = manifest.training.objective
    if objective in {"sft", "continued_pretraining"}:
        args = SFTConfig(
            **common_args,
            packing=manifest.training.packing or objective == "continued_pretraining",
            eval_packing=False,
            dataset_text_field="text",
        )
        trainer = SFTTrainer(args=args, **trainer_kwargs)
    elif objective == "dpo":
        from trl import DPOConfig, DPOTrainer

        trainer = DPOTrainer(args=DPOConfig(**common_args), **trainer_kwargs)
    elif objective == "kto":
        from trl import KTOConfig, KTOTrainer

        trainer = KTOTrainer(args=KTOConfig(**common_args), **trainer_kwargs)
    elif objective == "reward":
        from trl import RewardConfig, RewardTrainer

        trainer = RewardTrainer(args=RewardConfig(**common_args), **trainer_kwargs)
    elif objective == "grpo":
        from trl import GRPOConfig, GRPOTrainer

        selected = manifest.training.reward_functions or ["length"]
        reward_functions = [REWARDS[name] for name in selected]
        if manifest.training.reward_module:
            reward_functions.append(load_trusted_reward(Path(manifest.training.reward_module)))
        trainer = GRPOTrainer(
            args=GRPOConfig(**common_args), reward_funcs=reward_functions, **trainer_kwargs
        )
    else:
        raise ValueError("ORPO is not provided by the installed TRL release.")
    if manifest.training.method == "full":
        events.write("preparing", "Running one-microbatch memory probe", 0.18)
        sample = split["train"][0]["text"]
        probe = tokenizer(
            sample,
            return_tensors="pt",
            truncation=True,
            max_length=manifest.training.max_sequence_length,
        ).to(model.device)
        loss = model(**probe, labels=probe["input_ids"]).loss
        loss.backward()
        model.zero_grad(set_to_none=True)
    metrics: dict[str, Any] = {}
    if manifest.evaluation.before:
        update_job(
            job_id, status=JobStatus.EVALUATING_BEFORE, stage="Evaluating baseline", progress=0.2
        )
        metrics["before"] = trainer.evaluate(metric_key_prefix="before")
    update_job(job_id, status=JobStatus.TRAINING, stage="Training", progress=0.25)
    trainer.train(resume_from_checkpoint=manifest.training.resume_checkpoint)
    if cancel_path.exists():
        trainer.save_model(str(artifacts / "adapter-cancelled"))
        update_job(job_id, status=JobStatus.CANCELLED, stage="Cancelled", progress=1.0)
        events.write("cancelled", "Cancelled safely; adapter checkpoint saved", 1.0)
        return
    if manifest.evaluation.after:
        update_job(
            job_id,
            status=JobStatus.EVALUATING_AFTER,
            stage="Evaluating trained adapter",
            progress=0.82,
        )
        metrics["after"] = trainer.evaluate(metric_key_prefix="after")
    if manifest.evaluation.benchmarks:
        events.write(
            "evaluating_after",
            "Benchmark selections saved; Lighteval execution is isolated from the training worker",
            0.86,
            benchmarks=manifest.evaluation.benchmarks,
            limit=manifest.evaluation.benchmark_limit,
        )
    update_job(job_id, status=JobStatus.EXPORTING, stage="Saving adapter", progress=0.88)
    output_path = artifacts / ("full-model" if manifest.training.method == "full" else "adapter")
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(output_path)
    if manifest.evaluation.benchmarks:
        events.write("evaluating_after", "Running isolated Inspect AI benchmarks", 0.9)
        run_inspect(
            output_path,
            manifest.evaluation.benchmarks,
            manifest.evaluation.benchmark_limit,
            artifacts / "inspect-eval.log",
        )
    (artifacts / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    merged_path: Path | None = output_path if manifest.training.method == "full" else None
    if manifest.export.merged_model and manifest.training.method != "full":
        events.write("exporting", "Merging adapter into base model", 0.92)
        merged_path = artifacts / "merged-model"
        peft_model = cast(Any, trainer.model)
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(merged_path, safe_serialization=True)
        tokenizer.save_pretrained(merged_path)
    if manifest.export.push_to_hub:
        events.write("exporting", "Pushing adapter to Hugging Face Hub", 0.94)
        trainer.push_to_hub(
            repo_id=manifest.export.hub_repo_id,
            private=manifest.export.hub_private,
        )
    if manifest.export.import_to_ollama and merged_path:
        import_into_ollama(manifest, merged_path, events)
    write_artifact_manifest(artifacts)
    update_job(job_id, status=JobStatus.COMPLETED, stage="Completed", progress=1.0)
    events.write("completed", "Training and export completed", 1.0)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m fine_tuning_studio.worker JOB_ID", file=sys.stderr)
        return 2
    job_id = sys.argv[1]
    try:
        run(job_id)
    except Exception as exc:
        EventWriter(job_id).write("failed", f"{type(exc).__name__}: {exc}", 1.0)
        update_job(
            job_id,
            status=JobStatus.FAILED,
            stage="Failed",
            progress=1.0,
            error=f"{type(exc).__name__}: {exc}",
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
