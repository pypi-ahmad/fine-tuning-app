# Changelog

All notable changes to Fine-Tuning Studio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project governance, security, technical, and usage documentation.

## [0.1.0] - Initial baseline

### Added

- Local-first Streamlit workflow for inspecting a machine, preparing datasets, choosing
  models, configuring runs, monitoring jobs, and browsing artifacts.
- QLoRA supervised fine-tuning for causal language models through Transformers, PEFT,
  TRL, and bitsandbytes on compatible NVIDIA CUDA hardware.
- Hugging Face dataset/model support, local JSON/JSONL/CSV/Parquet upload, dataset
  mapping, validation splitting, checkpointing, cancellation, and loss-based evaluation.
- Adapter export, optional merged-model export, optional Hugging Face Hub adapter push,
  and optional Ollama import and model listing.

### Security

- User tokens are read from the environment and are not displayed or persisted by the
  application.
- Standard benchmark execution is disabled pending a dependency chain without known
  unresolved security advisories.
