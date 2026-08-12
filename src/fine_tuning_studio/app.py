from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.csv as arrow_csv
import pyarrow.json as arrow_json
import pyarrow.parquet as arrow_parquet
import streamlit as st

from fine_tuning_studio.domain import (
    DatasetSpec,
    EvaluationSpec,
    ExportSpec,
    ModelSpec,
    RunManifest,
    TrainingSpec,
    validate_manifest,
)
from fine_tuning_studio.jobs import (
    create_job,
    get_job,
    job_directory,
    launch_job,
    list_jobs,
    request_cancel,
    studio_home,
)
from fine_tuning_studio.runtimes import PROFILES, provision_profile, runtime_verdict
from fine_tuning_studio.system_info import MachineReport, qlora_capability, scan_machine

st.set_page_config(page_title="Fine-Tuning Studio", page_icon="🧪", layout="wide")
st.markdown(
    """<style>
    [data-testid="stSidebar"] {border-right:1px solid #cdd7eb}
    .fts-rail {padding:.75rem 1rem;border:1px solid #cdd7eb;border-radius:.65rem;
      background:linear-gradient(90deg,#eef3ff,#f8fbff);margin-bottom:1rem}
    .fts-kicker {color:#2855d9;font-weight:700;letter-spacing:.08em;
      text-transform:uppercase;font-size:.72rem}
    </style>""",
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    defaults: dict[str, Any] = {
        "machine_report": None,
        "dataset": {},
        "model": {},
        "training": {},
        "upload": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def machine_report() -> MachineReport:
    if st.session_state.machine_report is None:
        st.session_state.machine_report = scan_machine()
    return st.session_state.machine_report


def compatibility_rail() -> None:
    supported, _ = qlora_capability(machine_report())
    labels = [
        f"Machine {'ready' if supported else 'attention'}",
        f"Dataset {'ready' if st.session_state.dataset.get('location') else 'pending'}",
        f"Model {'ready' if st.session_state.model.get('location') else 'pending'}",
        "Transformers backend",
    ]
    st.markdown(
        f'<div class="fts-rail"><b>Compatibility</b> · {" &nbsp;→&nbsp; ".join(labels)}</div>',
        unsafe_allow_html=True,
    )


def system_page() -> None:
    st.markdown('<div class="fts-kicker">01 · Inspect</div>', unsafe_allow_html=True)
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


def dataset_page() -> None:
    st.markdown('<div class="fts-kicker">02 · Prepare</div>', unsafe_allow_html=True)
    st.title("Shape the training material")
    source = st.selectbox("Dataset source", ["Hugging Face", "Local upload"])
    data: dict[str, Any] = {"source": "hub" if source == "Hugging Face" else "local"}
    if source == "Hugging Face":
        data["location"] = st.text_input(
            "Dataset ID",
            value=st.session_state.dataset.get("location", ""),
            placeholder="org/dataset",
        )
        c1, c2 = st.columns(2)
        data["revision"] = c1.text_input("Revision", value="main")
        data["split"] = c2.text_input("Split", value="train")
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
        ["text", "prompt_response", "messages"],
        format_func=lambda value: value.replace("_", " / ").title(),
    )
    data["mapping"] = mapping
    if mapping == "text":
        data["text_column"] = st.text_input("Text column", value="text")
    elif mapping == "prompt_response":
        c1, c2 = st.columns(2)
        data["prompt_column"] = c1.text_input("Prompt column", value="prompt")
        data["response_column"] = c2.text_input("Response column", value="response")
    else:
        data["text_column"] = st.text_input("Messages column", value="messages")
    data["chat_template"] = st.selectbox(
        "Chat template", ["tokenizer", "chatml", "alpaca", "plain"]
    )
    data["validation_fraction"] = st.slider("Validation fraction", 0.05, 0.4, 0.1, 0.05)
    data["seed"] = int(st.number_input("Split seed", min_value=0, value=42))
    st.session_state.dataset = data


def model_page() -> None:
    st.markdown('<div class="fts-kicker">03 · Select</div>', unsafe_allow_html=True)
    st.title("Choose the base model")
    source = st.selectbox("Model source", ["Hugging Face", "Local directory"])
    model: dict[str, Any] = {"source": "hub" if source == "Hugging Face" else "local"}
    model["location"] = st.text_input(
        "Model ID" if source == "Hugging Face" else "Directory",
        value=st.session_state.model.get("location", ""),
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
    if st.button(
        "Inspect Hub metadata", disabled=source != "Hugging Face" or not model["location"]
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
            st.error(f"Model inspection failed: {type(exc).__name__}: {exc}")
    st.session_state.model = model


def training_page() -> None:
    st.markdown('<div class="fts-kicker">04 · Configure</div>', unsafe_allow_html=True)
    st.title("Set the training run")
    c1, c2, c3 = st.columns(3)
    objective = c1.selectbox(
        "Objective", ["sft", "continued_pretraining", "dpo", "kto", "reward", "orpo", "grpo"],
        format_func=lambda value: value.replace("_", " ").upper(),
    )
    method = c2.selectbox("Method", ["qlora", "lora", "full"], format_func=str.upper)
    backends = ["transformers"]
    if importlib.util.find_spec("unsloth") and objective == "sft":
        backends.append("unsloth")
    backend = c3.selectbox(
        "Backend",
        backends,
        format_func=lambda value: (
            "Unsloth" if value == "unsloth" else "Transformers + PEFT + TRL"
        ),
    )
    detected_vendors = {gpu.vendor for gpu in machine_report().gpus}
    suggested = next(
        (name for name, profile in PROFILES.items() if profile.vendor in detected_vendors),
        "cuda",
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
    if PROFILES[runtime_profile].status == "experimental":
        st.warning("AMD ROCm and Intel XPU profiles are experimental until the smoke test passes.")
        training["experimental_acknowledged"] = st.toggle(
            "I understand this runtime is experimental", value=False
        )
    c1, c2 = st.columns(2)
    training["epochs"] = c1.number_input("Epochs", 0.1, 100.0, 1.0, 0.5)
    training["learning_rate"] = c2.number_input("Learning rate", 1e-7, 1.0, 2e-4, format="%.7f")
    training["max_sequence_length"] = st.select_slider(
        "Maximum sequence length", [256, 512, 1024, 2048, 4096, 8192], value=1024
    )
    with st.expander("Advanced training settings"):
        c1, c2 = st.columns(2)
        training["batch_size"] = c1.number_input("Device batch size", 1, 64, 1)
        training["gradient_accumulation_steps"] = c2.number_input(
            "Gradient accumulation", 1, 256, 8
        )
        training["lora_rank"] = c1.selectbox("LoRA rank", [4, 8, 16, 32, 64], index=2)
        training["lora_alpha"] = c2.number_input("LoRA alpha", 1, 256, 32)
        training["lora_dropout"] = c1.number_input("LoRA dropout", 0.0, 0.5, 0.05, 0.01)
        training["packing"] = c2.toggle("Sequence packing", value=False)
        training["gradient_checkpointing"] = c1.toggle("Gradient checkpointing", value=True)
        training["save_steps"] = c1.number_input("Checkpoint interval", 1, 10000, 100)
        training["logging_steps"] = c2.number_input("Logging interval", 1, 1000, 10)
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
    st.markdown('<div class="fts-kicker">05 · Commit</div>', unsafe_allow_html=True)
    st.title("Review and launch")
    evaluation = {
        "before": st.toggle("Evaluate before training", value=True),
        "after": st.toggle("Evaluate after training", value=True),
        "benchmarks": [],
        "benchmark_limit": 100,
    }
    st.caption(
        "Standard benchmark execution is temporarily disabled because the latest stable "
        "Lighteval release resolves a dependency with unfixed security advisories."
    )
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
    supported, reasons = qlora_capability(machine_report())
    with st.expander("Run manifest", expanded=True):
        st.json(manifest.to_dict())
    for message in errors:
        st.error(message)
    for reason in reasons:
        st.warning(reason)
    if st.button("Start training", type="primary", disabled=bool(errors) or not supported):
        upload = st.session_state.upload if manifest.dataset.source == "local" else None
        create_job(manifest, getattr(upload, "name", None), upload.getvalue() if upload else None)
        launch_job(manifest.id)
        st.success(f"Job {manifest.id} started.")


def monitor_page() -> None:
    st.markdown('<div class="fts-kicker">06 · Observe</div>', unsafe_allow_html=True)
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
        request_cancel(job_id)
        st.warning("Cancellation requested. The worker will stop at the next safe callback.")
    events = job_directory(job_id) / "events.jsonl"
    if events.exists():
        lines = events.read_text(encoding="utf-8", errors="replace").splitlines()[-100:]
        st.code("\n".join(lines), language="json")


def export_page() -> None:
    st.markdown('<div class="fts-kicker">07 · Use</div>', unsafe_allow_html=True)
    st.title("Artifacts and Ollama")
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
    st.subheader("Installed Ollama models")
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    if st.button("Refresh Ollama models"):
        try:
            with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as response:
                st.json(json.load(response))
        except (OSError, ValueError, urllib.error.URLError) as exc:
            st.error(f"Ollama is unavailable: {exc}")


initialize_state()
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
    ],
    expanded=True,
)
page.run()
