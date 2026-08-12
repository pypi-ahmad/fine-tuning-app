from __future__ import annotations

import importlib.util
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.csv as arrow_csv
import pyarrow.json as arrow_json
import pyarrow.parquet as arrow_parquet
import streamlit as st
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from fine_tuning_studio.diagnostics import build_diagnostics
from fine_tuning_studio.domain import (
    DatasetSpec,
    EvaluationSpec,
    ExportSpec,
    ModelSpec,
    RunManifest,
    TrainingSpec,
    validate_manifest,
)
from fine_tuning_studio.gpu_memory import (
    GpuMemoryReport,
    GpuProcess,
    inspect_gpu_memory,
    release_reclaimable_memory,
    terminate_gpu_process,
)
from fine_tuning_studio.hub_references import (
    RepoType,
    download_hub_repository,
    normalize_hub_reference,
    plan_hub_download,
)
from fine_tuning_studio.jobs import (
    create_job,
    get_job,
    job_directory,
    launch_job,
    list_jobs,
    reconcile_jobs,
    resume_job,
    studio_home,
    terminate_job,
)
from fine_tuning_studio.ollama import OllamaClient, OllamaError, OllamaModel
from fine_tuning_studio.runtimes import (
    PROFILES,
    profile_python,
    provision_profile,
    runtime_verdict,
)
from fine_tuning_studio.system_info import MachineReport, qlora_capability, scan_machine

st.set_page_config(page_title="Fine-Tuning Studio", page_icon="🧪", layout="wide")


def initialize_state() -> None:
    defaults: dict[str, Any] = {
        "machine_report": None,
        "dataset": {},
        "model": {},
        "training": {},
        "upload": None,
        "gpu_memory_report": None,
        "ollama_models": None,
        "ollama_model_details": {},
        "ollama_chats": {},
        "ollama_metrics": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def machine_report() -> MachineReport:
    if st.session_state.machine_report is None:
        st.session_state.machine_report = scan_machine()
    return st.session_state.machine_report


def compatibility_rail() -> None:
    selected = st.session_state.training.get("runtime_profile", "cuda")
    verdict = runtime_verdict(machine_report(), selected)
    supported = verdict.state == "ready" or profile_python(studio_home(), selected).exists()
    labels = [
        f"Machine {'ready' if supported else 'attention'}",
        f"Dataset {'ready' if st.session_state.dataset.get('location') else 'pending'}",
        f"Model {'ready' if st.session_state.model.get('location') else 'pending'}",
        "Transformers backend",
    ]
    with st.container(border=True):
        st.caption("Compatibility · " + " → ".join(labels))


def active_worker_pids() -> set[int]:
    active = {
        "preparing",
        "evaluating_before",
        "training",
        "evaluating_after",
        "exporting",
        "cancelling",
    }
    return {int(row["pid"]) for row in list_jobs() if row["status"] in active and row.get("pid")}


def format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


def process_label(process: GpuProcess) -> str:
    memory = f" · {process.memory_mb:,} MiB" if process.memory_mb is not None else ""
    return f"PID {process.pid} · {process.name}{memory}"


def gpu_memory_rows(report: GpuMemoryReport) -> list[dict[str, Any]]:
    return [
        {
            "vendor": process.vendor,
            "gpu": process.gpu_index,
            "pid": process.pid,
            "process": process.name,
            "executable": process.executable,
            "VRAM MiB": process.memory_mb,
            "action": process.protected_reason or "Eligible after confirmation",
        }
        for process in report.processes
    ]


def render_gpu_cleanup(key_prefix: str) -> None:
    st.subheader("GPU memory cleanup")
    st.caption(
        "Release reclaimable caches and Ollama models before training. Memory owned by "
        "another application requires ending that application."
    )
    workers = active_worker_pids()
    report = st.session_state.gpu_memory_report
    if report is None:
        report = inspect_gpu_memory(studio_worker_pids=workers)
        st.session_state.gpu_memory_report = report

    with st.container(horizontal=True):
        refresh = st.button(
            "Refresh GPU usage", icon=":material/refresh:", key=f"{key_prefix}_gpu_refresh"
        )
        release = st.button(
            "Release reclaimable VRAM",
            icon=":material/memory:",
            type="primary",
            key=f"{key_prefix}_gpu_release",
        )
    if refresh:
        report = inspect_gpu_memory(studio_worker_pids=workers)
        st.session_state.gpu_memory_report = report
    if release:
        with st.status("Releasing reclaimable GPU memory…", expanded=True) as status:
            result = release_reclaimable_memory(studio_worker_pids=workers)
            report = result.after
            st.session_state.gpu_memory_report = report
            st.session_state.machine_report = scan_machine()
            for action in result.cache_actions:
                status.write(f"Released {action}.")
            for model in result.unloaded_models:
                status.write(f"Unloaded Ollama model `{model}`.")
            for warning in result.warnings:
                status.warning(warning)
            status.update(label="GPU cleanup complete", state="complete", expanded=False)

    if report.devices:
        columns = st.columns(min(4, len(report.devices)))
        for column, device in zip(columns, report.devices, strict=False):
            free = f"{device.free_mb:,} MiB" if device.free_mb is not None else "Unavailable"
            used = f"{device.used_mb:,} MiB used" if device.used_mb is not None else None
            column.metric(f"{device.vendor} GPU {device.gpu_index} free", free, used)
    else:
        st.info("No vendor tool reported dedicated GPU memory.")
    for warning in report.warnings:
        st.warning(warning)

    if report.processes:
        st.dataframe(gpu_memory_rows(report), hide_index=True, width="stretch")
    candidates: dict[int, GpuProcess] = {}
    for process in report.processes:
        if process.protected_reason is None:
            candidates.setdefault(process.pid, process)
    if not candidates:
        st.caption("No user-owned GPU processes are eligible for termination.")
        return

    with st.form(f"{key_prefix}_gpu_termination", border=True):
        selected = st.multiselect(
            "Other GPU processes to terminate",
            list(candidates),
            format_func=lambda pid: process_label(candidates[pid]),
        )
        st.warning("Save your work first. Terminating a process can discard unsaved data.")
        confirmation = st.text_input("Type TERMINATE to confirm")
        submitted = st.form_submit_button(
            "Terminate selected processes",
            icon=":material/stop_circle:",
            disabled=not selected,
        )
    if not submitted:
        return
    if confirmation != "TERMINATE":
        st.error("Type TERMINATE exactly before ending another process.")
        return
    fresh_report = inspect_gpu_memory(studio_worker_pids=active_worker_pids())
    fresh_candidates = {
        process.pid: process
        for process in fresh_report.processes
        if process.protected_reason is None
    }
    terminated: list[int] = []
    for process_pid in selected:
        candidate = fresh_candidates.get(process_pid)
        if candidate is None:
            st.error(f"PID {process_pid} is no longer eligible. Refresh GPU usage and review it.")
            continue
        try:
            terminated.extend(terminate_gpu_process(candidate))
        except (PermissionError, RuntimeError) as exc:
            st.error(f"PID {process_pid}: {exc}")
    if terminated:
        st.success("Terminated PIDs: " + ", ".join(str(pid) for pid in sorted(set(terminated))))
        result = release_reclaimable_memory(studio_worker_pids=workers)
        st.session_state.gpu_memory_report = result.after
        st.session_state.machine_report = scan_machine()
        st.rerun()


def system_page() -> None:
    st.caption("01 · Inspect")
    st.title("Know the machine before allocating it")
    st.caption("The scan never installs drivers or displays secret values.")
    if st.button("Refresh system scan", type="primary"):
        st.session_state.machine_report = scan_machine()
    report = machine_report()
    supported, reasons = qlora_capability(report)
    (st.success if supported else st.warning)(
        "QLoRA training is ready." if supported else "QLoRA training needs attention."
    )
    for reason in reasons:
        st.write(f"- {reason}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Operating system", report.os["system"], report.os["release"])
    c2.metric("CPU threads", report.cpu["logical_cores"] or "Unknown")
    c3.metric("Available RAM", f"{report.memory['available_gb']} GB")
    c4.metric("Free disk", f"{report.disk['free_gb']} GB")
    st.subheader("Accelerators")
    if report.gpus:
        st.dataframe([gpu.__dict__ for gpu in report.gpus], width="stretch", hide_index=True)
    else:
        st.info("No GPU was detected.")
    render_gpu_cleanup("system")
    st.subheader("Managed training runtimes")
    st.dataframe(
        [
            {
                "profile": name,
                "support": profile.status,
                "verdict": runtime_verdict(report, name).state,
                "platforms": ", ".join(profile.operating_systems),
            }
            for name, profile in PROFILES.items()
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption("WSL is an optional Linux host. Native Windows is supported directly.")
    with st.expander("Install a managed runtime"):
        profile = st.selectbox("Profile to install", list(PROFILES), key="install_profile")
        st.warning("Installation downloads a separate PyTorch environment and can use several GB.")
        if st.button("Install selected runtime"):
            with st.spinner(f"Installing {profile} runtime…"):
                try:
                    python = provision_profile(studio_home(), profile)
                    st.success(f"Installed managed interpreter: {python}")
                except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                    st.error(f"Runtime installation failed: {exc}")
    with st.expander("Software and integrations"):
        st.json(
            {
                "packages": report.packages,
                "runtimes": report.runtimes,
                "integrations": report.integrations,
            }
        )
    st.download_button(
        "Download sanitized report",
        report.to_json(),
        "fine-tuning-studio-system.json",
        "application/json",
    )
    if st.button("Generate sanitized diagnostics"):
        with st.spinner("Collecting local diagnostics…"):
            st.session_state.diagnostics = build_diagnostics(studio_home())
    if payload := st.session_state.get("diagnostics"):
        st.download_button(
            "Download sanitized diagnostics",
            payload,
            "fine-tuning-studio-diagnostics.zip",
            "application/zip",
        )


def preview_upload(upload: Any) -> pa.Table:
    payload = pa.BufferReader(upload.getvalue())
    suffix = Path(upload.name).suffix.lower()
    if suffix == ".csv":
        return arrow_csv.read_csv(payload)
    if suffix in {".json", ".jsonl"}:
        return arrow_json.read_json(payload)
    if suffix == ".parquet":
        return arrow_parquet.read_table(payload)
    raise ValueError("Upload JSONL, CSV, or Parquet.")


def hub_error_message(exc: Exception) -> str:
    if isinstance(exc, GatedRepoError):
        return "This repository is gated. Accept its terms on Hugging Face and set HF_TOKEN."
    if isinstance(exc, RepositoryNotFoundError):
        return "Repository not found or inaccessible. Check the ID and your HF_TOKEN access."
    if isinstance(exc, RevisionNotFoundError):
        return "Revision not found. Check the branch, tag, or commit in the Revision field."
    if isinstance(exc, HfHubHTTPError):
        return (
            "Hugging Face rejected the request. Check your network, repository access, "
            "and HF_TOKEN."
        )
    if isinstance(exc, OSError) and str(exc).startswith("Download requires"):
        return str(exc)
    return f"Download failed ({type(exc).__name__}). Check your network and try again."


def render_hub_download(repo_id: str, repo_type: RepoType, revision: str) -> bool:
    label = f"Validate & download {repo_type}"
    if not st.button(label, disabled=not repo_id, key=f"download_{repo_type}"):
        return False
    status = st.status(f"Validating {repo_id}…", expanded=True)
    try:
        plan = plan_hub_download(repo_id, repo_type, revision)
        status.write(
            f"{plan.file_count:,} files · {plan.total_bytes:,} bytes total · "
            f"{plan.download_bytes:,} bytes to download"
        )
        if plan.download_bytes:
            status.update(label=f"Downloading {repo_id}…", state="running")
        else:
            status.update(label=f"Verifying cached {repo_id}…", state="running")
        cache_path = download_hub_repository(plan)
    except Exception as exc:
        status.update(label="Download failed", state="error", expanded=True)
        st.error(hub_error_message(exc))
        return False
    status.write(f"Commit: `{plan.commit_hash}`")
    status.write(f"Cache: `{cache_path}`")
    status.update(label=f"{repo_id} is ready", state="complete", expanded=False)
    return True


def dataset_page() -> None:
    st.caption("02 · Prepare")
    st.title("Shape the training material")
    source = st.selectbox("Dataset source", ["Hugging Face", "Local upload"])
    data: dict[str, Any] = {"source": "hub" if source == "Hugging Face" else "local"}
    if source == "Hugging Face":
        reference = st.text_input(
            "Dataset ID or Hugging Face URL",
            value=st.session_state.dataset.get("location", ""),
            placeholder="org/dataset or https://huggingface.co/datasets/org/dataset",
        )
        try:
            data["location"] = normalize_hub_reference(reference, "dataset")
        except ValueError as exc:
            data["location"] = ""
            st.error(str(exc))
        if reference and data["location"] and reference.strip() != data["location"]:
            st.caption(f"Resolved dataset ID: `{data['location']}`")
        c1, c2 = st.columns(2)
        data["revision"] = c1.text_input("Revision", value="main")
        data["split"] = c2.text_input("Split", value="train")
        render_hub_download(data["location"], "dataset", data["revision"])
    else:
        upload = st.file_uploader("Upload data", type=["jsonl", "json", "csv", "parquet"])
        if upload:
            st.session_state.upload = upload
            data["location"] = upload.name
            data["format"] = Path(upload.name).suffix.lstrip(".")
            try:
                preview = preview_upload(upload)
                st.dataframe(preview.slice(0, 100), width="stretch")
                st.caption(f"{preview.num_rows:,} rows · {preview.num_columns} columns")
            except (ValueError, pa.ArrowException) as exc:
                st.error(str(exc))
        else:
            data["location"] = st.session_state.dataset.get("location", "")
    mapping = st.selectbox(
        "Data shape",
        ["text", "prompt_response", "messages", "preference", "kto", "grpo"],
        format_func=lambda value: value.replace("_", " / ").title(),
    )
    data["mapping"] = mapping
    if mapping == "text":
        data["text_column"] = st.text_input("Text column", value="text")
    elif mapping == "prompt_response":
        c1, c2 = st.columns(2)
        data["prompt_column"] = c1.text_input("Prompt column", value="prompt")
        data["response_column"] = c2.text_input("Response column", value="response")
    elif mapping == "messages":
        data["text_column"] = st.text_input("Messages column", value="messages")
    else:
        contracts = {
            "preference": "prompt, chosen, rejected",
            "kto": "prompt, completion, label (boolean)",
            "grpo": "prompt, with optional ground_truth",
        }
        st.info(f"Required canonical columns: {contracts[mapping]}")
    data["chat_template"] = st.selectbox(
        "Chat template", ["tokenizer", "chatml", "alpaca", "plain"]
    )
    data["validation_fraction"] = st.slider("Validation fraction", 0.05, 0.4, 0.1, 0.05)
    data["seed"] = int(st.number_input("Split seed", min_value=0, value=42))
    st.session_state.dataset = data


def model_page() -> None:
    st.caption("03 · Select")
    st.title("Choose the base model")
    source = st.selectbox("Model source", ["Hugging Face", "Local directory"])
    model: dict[str, Any] = {"source": "hub" if source == "Hugging Face" else "local"}
    if source == "Hugging Face":
        reference = st.text_input(
            "Model ID or Hugging Face URL",
            value=st.session_state.model.get("location", ""),
            placeholder="org/model or https://huggingface.co/org/model",
        )
        try:
            model["location"] = normalize_hub_reference(reference, "model")
        except ValueError as exc:
            model["location"] = ""
            st.error(str(exc))
        if reference and model["location"] and reference.strip() != model["location"]:
            st.caption(f"Resolved model ID: `{model['location']}`")
    else:
        model["location"] = st.text_input(
            "Directory", value=st.session_state.model.get("location", "")
        )
    model["revision"] = st.text_input("Revision", value="main", disabled=source != "Hugging Face")
    with st.expander("Advanced trust settings"):
        model["trust_remote_code"] = st.toggle(
            "Allow model repository code",
            value=False,
            help="This can execute code from the selected repository.",
        )
        if model["trust_remote_code"]:
            st.warning("Enable this only after reviewing the model repository.")
            confirmation = st.text_input("Type I UNDERSTAND to enable repository code")
            model["trust_remote_code_acknowledged"] = confirmation == "I UNDERSTAND"
    if source == "Hugging Face" and render_hub_download(
        model["location"], "model", model["revision"]
    ):
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(model["location"], revision=model["revision"])
            st.json(
                {
                    "id": info.id,
                    "sha": info.sha,
                    "gated": info.gated,
                    "downloads": info.downloads,
                    "tags": (info.tags or [])[:20],
                }
            )
        except Exception as exc:
            st.error(f"Metadata inspection failed: {hub_error_message(exc)}")
    st.session_state.model = model


def training_page() -> None:
    st.caption("04 · Configure")
    st.title("Set the training run")
    c1, c2, c3 = st.columns(3)
    objective = c1.selectbox(
        "Objective",
        ["sft", "continued_pretraining", "dpo", "kto", "reward", "orpo", "grpo"],
        format_func=lambda value: value.replace("_", " ").upper(),
    )
    method = c2.selectbox("Method", ["qlora", "lora", "full"], format_func=str.upper)
    backends = ["transformers"]
    if importlib.util.find_spec("unsloth") and objective == "sft" and method != "full":
        backends.append("unsloth")
    backend = c3.selectbox(
        "Backend",
        backends,
        format_func=lambda value: "Unsloth" if value == "unsloth" else "Transformers + PEFT + TRL",
    )
    detected_vendors = {gpu.vendor for gpu in machine_report().gpus}
    suggested = next(
        (name for name, profile in PROFILES.items() if profile.vendor in detected_vendors),
        "cpu",
    )
    runtime_profile = st.selectbox(
        "Runtime profile", list(PROFILES), index=list(PROFILES).index(suggested)
    )
    training: dict[str, Any] = {
        "objective": objective,
        "method": method,
        "backend": backend,
        "runtime_profile": runtime_profile,
    }
    preset = st.selectbox("Training preset", ["safe", "balanced", "custom"], index=1)
    training["preset"] = preset
    preset_values = {
        "safe": {
            "learning_rate": 1e-4,
            "max_sequence_length": 512,
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
        },
        "balanced": {
            "learning_rate": 2e-4,
            "max_sequence_length": 1024,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
        },
        "custom": {
            "learning_rate": 2e-4,
            "max_sequence_length": 1024,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
        },
    }[preset]
    if PROFILES[runtime_profile].status == "experimental":
        st.warning("AMD ROCm and Intel XPU profiles are experimental until the smoke test passes.")
        training["experimental_acknowledged"] = st.toggle(
            "I understand this runtime is experimental", value=False
        )
    c1, c2 = st.columns(2)
    training["epochs"] = c1.number_input("Epochs", 0.1, 100.0, 1.0, 0.5)
    training["learning_rate"] = c2.number_input(
        "Learning rate", 1e-7, 1.0, preset_values["learning_rate"], format="%.7f"
    )
    training["max_sequence_length"] = st.select_slider(
        "Maximum sequence length",
        [256, 512, 1024, 2048, 4096, 8192],
        value=preset_values["max_sequence_length"],
    )
    with st.expander("Advanced training settings"):
        c1, c2 = st.columns(2)
        training["batch_size"] = c1.number_input(
            "Device batch size", 1, 64, preset_values["batch_size"]
        )
        training["gradient_accumulation_steps"] = c2.number_input(
            "Gradient accumulation", 1, 256, preset_values["gradient_accumulation_steps"]
        )
        training["lora_rank"] = c1.selectbox("LoRA rank", [4, 8, 16, 32, 64], index=2)
        training["lora_alpha"] = c2.number_input("LoRA alpha", 1, 256, 32)
        training["lora_dropout"] = c1.number_input("LoRA dropout", 0.0, 0.5, 0.05, 0.01)
        training["packing"] = c2.toggle("Sequence packing", value=False)
        training["gradient_checkpointing"] = c1.toggle("Gradient checkpointing", value=True)
        training["save_steps"] = c1.number_input("Checkpoint interval", 1, 10000, 100)
        training["logging_steps"] = c2.number_input("Logging interval", 1, 1000, 10)
        if objective == "grpo":
            training["reward_functions"] = st.multiselect(
                "Built-in rewards", ["exact", "numeric", "regex", "length"], default=["length"]
            )
            st.warning("Custom Python rewards execute unsandboxed. Use only code you trust.")
            reward_path = st.text_input("Trusted local reward module (optional)")
            training["reward_module"] = reward_path or None
            if reward_path:
                confirmation = st.text_input("Type I UNDERSTAND to enable custom reward code")
                training["custom_code_acknowledged"] = confirmation == "I UNDERSTAND"
        devices = [
            gpu.device_index for gpu in machine_report().gpus if gpu.device_index is not None
        ]
        selected_devices = st.multiselect("Physical devices", devices, default=devices[:1])
        training["device_indices"] = selected_devices
        training["world_size"] = len(selected_devices) or 1
        strategies = ["single", "auto"] + (["ddp", "fsdp2"] if len(selected_devices) > 1 else [])
        training["distributed_strategy"] = st.selectbox("Distributed strategy", strategies)
    st.session_state.training = training


def build_manifest(evaluation: dict[str, Any], export: dict[str, Any]) -> RunManifest:
    return RunManifest(
        dataset=DatasetSpec(**st.session_state.dataset),
        model=ModelSpec(**st.session_state.model),
        training=TrainingSpec(**st.session_state.training),
        evaluation=EvaluationSpec(**evaluation),
        export=ExportSpec(**export),
    )


def review_page() -> None:
    st.caption("05 · Commit")
    st.title("Review and launch")
    evaluation = {
        "before": st.toggle("Evaluate before training", value=True),
        "after": st.toggle("Evaluate after training", value=True),
        "benchmarks": st.multiselect(
            "Optional Inspect AI benchmarks",
            ["mmlu", "gsm8k", "hellaswag", "arc", "truthfulqa", "winogrande"],
        ),
        "benchmark_limit": int(st.number_input("Examples per benchmark", 1, 10000, 20)),
    }
    st.caption("Benchmarks run through the optional isolated Inspect AI environment.")
    st.subheader("Exports")
    merged = st.toggle("Create merged model", value=False)
    import_ollama = st.toggle("Import merged model into Ollama", value=False)
    export = {
        "adapter": st.session_state.training.get("method") != "full",
        "merged_model": import_ollama
        or (merged and st.session_state.training.get("method") != "full"),
        "push_to_hub": st.toggle("Push adapter to Hugging Face Hub", value=False),
        "hub_repo_id": "",
        "hub_private": True,
        "import_to_ollama": import_ollama,
        "ollama_model_name": "",
    }
    if export["push_to_hub"]:
        export["hub_repo_id"] = st.text_input(
            "Hub repository ID", placeholder="username/model-name"
        )
        export["hub_private"] = st.toggle("Private Hub repository", value=True)
    if import_ollama:
        export["ollama_model_name"] = st.text_input("Ollama model name", placeholder="my-fine-tune")
    try:
        manifest = build_manifest(evaluation, export)
    except TypeError as exc:
        st.error(f"Configuration is incomplete: {exc}")
        return
    errors = validate_manifest(manifest)
    selected_profile = manifest.training.runtime_profile
    verdict = runtime_verdict(machine_report(), selected_profile, manifest.training.method)
    supported = verdict.state == "ready" or profile_python(studio_home(), selected_profile).exists()
    reasons = verdict.reasons
    if verdict.state == "install-required" and not supported:
        reasons = ["Install the selected managed runtime before launching.", *reasons]
    with st.expander("Run manifest", expanded=True):
        st.json(manifest.to_dict())
    for message in errors:
        st.error(message)
    for reason in reasons:
        st.warning(reason)
    cleanup = st.expander("Free GPU memory before training", on_change="rerun")
    if cleanup.open:
        with cleanup:
            render_gpu_cleanup("review")
    if st.button("Start training", type="primary", disabled=bool(errors) or not supported):
        upload = st.session_state.upload if manifest.dataset.source == "local" else None
        create_job(manifest, getattr(upload, "name", None), upload.getvalue() if upload else None)
        launch_job(manifest.id)
        st.success(f"Job {manifest.id} started.")


def monitor_page() -> None:
    st.caption("06 · Observe")
    st.title("Training jobs")
    rows = list_jobs()
    if not rows:
        st.info("No jobs have been submitted.")
        return
    st.dataframe(rows, width="stretch", hide_index=True)
    job_id = st.selectbox("Job", [row["id"] for row in rows])
    job = get_job(job_id)
    if not job:
        return
    st.progress(float(job["progress"]), text=job["stage"])
    active = {"preparing", "evaluating_before", "training", "evaluating_after", "exporting"}
    if st.button("Request cancellation", disabled=job["status"] not in active):
        terminate_job(job_id)
        st.warning("Cancellation requested and the worker process tree was signalled.")
    checkpoints = sorted((job_directory(job_id) / "checkpoints").glob("checkpoint-*"))
    if checkpoints:
        checkpoint = st.selectbox("Resume checkpoint", checkpoints, format_func=lambda p: p.name)
        if st.button("Resume as a new job"):
            child = resume_job(job_id, checkpoint)
            launch_job(child.id)
            st.success(f"Resumed as job {child.id}.")
    events = job_directory(job_id) / "events.jsonl"
    if events.exists():
        lines = events.read_text(encoding="utf-8", errors="replace").splitlines()[-100:]
        st.code("\n".join(lines), language="json")


def export_page() -> None:
    st.caption("07 · Use")
    st.title("Artifacts and Ollama export")
    completed = [row for row in list_jobs() if row["status"] == "completed"]
    if completed:
        selected = st.selectbox("Completed job", [row["id"] for row in completed])
        artifacts = job_directory(selected) / "artifacts"
        if artifacts.exists():
            st.write(
                [
                    str(path.relative_to(artifacts))
                    for path in artifacts.rglob("*")
                    if path.is_file()
                ]
            )
    else:
        st.info("Complete a job to see its artifacts.")
    st.info(
        "Use Ollama playground to test models already installed in Ollama. The playground "
        "does not import training artifacts."
    )


def render_ollama_metrics(metrics: dict[str, Any]) -> None:
    if not metrics:
        return
    evaluated = int(metrics.get("eval_count") or 0)
    duration_ns = int(metrics.get("eval_duration") or 0)
    rate = evaluated / (duration_ns / 1_000_000_000) if duration_ns else 0
    prompt_count = int(metrics.get("prompt_eval_count") or 0)
    total_seconds = int(metrics.get("total_duration") or 0) / 1_000_000_000
    st.caption(
        f"Prompt {prompt_count:,} tokens · Output {evaluated:,} tokens · "
        f"{rate:.1f} tokens/s · {total_seconds:.2f}s total"
    )


def ollama_playground_page() -> None:
    st.caption("08 · Test")
    st.title("Ollama playground")
    st.caption(
        "Chat with models already installed in local Ollama. This page never pulls, creates, "
        "deletes, or imports a model."
    )
    try:
        client = OllamaClient()
    except ValueError as exc:
        st.error(str(exc))
        return

    refresh = st.button("Refresh installed models", icon=":material/refresh:")
    if refresh or st.session_state.ollama_models is None:
        try:
            st.session_state.ollama_models = client.list_models()
            if refresh:
                st.session_state.ollama_model_details = {}
        except OllamaError as exc:
            st.error(str(exc))
            return
    models: list[OllamaModel] = st.session_state.ollama_models
    if not models:
        st.info("Ollama is running, but it has no installed models.")
        return

    by_name = {model.name: model for model in models if model.name}
    selected = st.selectbox("Installed model", list(by_name), key="ollama_selected_model")
    details_cache = st.session_state.ollama_model_details
    if selected not in details_cache:
        try:
            details_cache[selected] = client.show_model(selected)
        except OllamaError as exc:
            st.error(str(exc))
            return
    details = details_cache[selected]
    model = by_name[selected]
    st.caption(
        f"{details.family or model.family or 'Unknown family'} · "
        f"{details.parameter_size or model.parameter_size or 'Unknown size'} · "
        f"{details.quantization_level or model.quantization_level or 'Unknown quantization'} · "
        f"{format_bytes(model.size)} on disk"
    )
    if details.capabilities:
        st.caption("Capabilities: " + ", ".join(details.capabilities))

    chats: dict[str, list[dict[str, str]]] = st.session_state.ollama_chats
    history = chats.setdefault(selected, [])
    with st.container(horizontal=True):
        if st.button("Clear conversation", icon=":material/delete_sweep:"):
            chats[selected] = []
            st.session_state.ollama_metrics.pop(selected, None)
            st.rerun()
        if st.button("Unload model", icon=":material/eject:"):
            try:
                client.unload_model(selected)
                st.success(f"Unloaded `{selected}` from memory.")
                st.session_state.gpu_memory_report = None
            except OllamaError as exc:
                st.error(str(exc))

    with st.expander("Generation settings"):
        system_prompt = st.text_area(
            "System prompt", key="ollama_system_prompt", placeholder="Optional behavior guidance"
        )
        temperature = st.slider("Temperature", 0.0, 2.0, 0.8, 0.1, key="ollama_temperature")
        top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.05, key="ollama_top_p")
        num_predict = int(
            st.number_input("Maximum output tokens", 16, 4096, 512, 16, key="ollama_num_predict")
        )
        seed = int(st.number_input("Seed", 0, 2**31 - 1, 0, key="ollama_seed"))
        unload_after = st.toggle(
            "Unload model after each response", value=False, key="ollama_unload_after"
        )

    for message in history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    render_ollama_metrics(st.session_state.ollama_metrics.get(selected, {}))

    chat_capable = not details.capabilities or "completion" in details.capabilities
    if not chat_capable:
        st.info("This installed model does not advertise text-completion capability.")
        return
    prompt = st.chat_input(f"Message {selected}", submit_mode="disable")
    if not prompt:
        return
    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    request_messages = list(history)
    if system_prompt.strip():
        request_messages.insert(0, {"role": "system", "content": system_prompt.strip()})
    final_chunk: dict[str, Any] = {}

    def response_stream() -> Any:
        for chunk in client.stream_chat(
            selected,
            request_messages,
            options={
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": num_predict,
                "seed": seed,
            },
            keep_alive=0 if unload_after else "5m",
        ):
            if chunk.get("done"):
                final_chunk.update(chunk)
            message = chunk.get("message") or {}
            if content := message.get("content"):
                yield str(content)

    with st.chat_message("assistant"):
        try:
            response = st.write_stream(response_stream())
        except OllamaError as exc:
            st.error(str(exc))
            return
    response_text = response if isinstance(response, str) else "".join(map(str, response))
    history.append({"role": "assistant", "content": response_text})
    st.session_state.ollama_metrics[selected] = final_chunk
    if unload_after:
        st.session_state.gpu_memory_report = None
    render_ollama_metrics(final_chunk)


initialize_state()
if "jobs_reconciled" not in st.session_state:
    try:
        reconcile_jobs()
    except sqlite3.DatabaseError as exc:
        st.error(f"Storage recovery required: {exc}")
        st.info("Run `fine-tuning-studio doctor` and restore a verified backup before training.")
        st.stop()
    st.session_state.jobs_reconciled = True
compatibility_rail()
page = st.navigation(
    [
        st.Page(system_page, title="System", icon=":material/memory:", default=True),
        st.Page(dataset_page, title="Dataset", icon=":material/dataset:"),
        st.Page(model_page, title="Model", icon=":material/deployed_code:"),
        st.Page(training_page, title="Training", icon=":material/tune:"),
        st.Page(review_page, title="Review & run", icon=":material/play_circle:"),
        st.Page(monitor_page, title="Monitor", icon=":material/monitoring:"),
        st.Page(export_page, title="Export", icon=":material/output:"),
        st.Page(ollama_playground_page, title="Ollama playground", icon=":material/chat:"),
    ],
    expanded=True,
)
page.run()
