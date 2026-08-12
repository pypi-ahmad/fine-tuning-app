# Changelog

All notable changes to Fine-Tuning Studio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.1] - 2026-08-12

### Added

- Added persistent Understand Anything and Graphify architecture graphs, including an
  interactive HTML explorer, GraphRAG-ready JSON, guided tour, structural fingerprints,
  and a plain-language repository audit report.

## [1.0.0] - 2026-08-12

- Stable local GA: schema-v4 manifests, transactional storage migrations, automatic
  pre-migration backups, integrity checks, restore tooling, loopback-only serving,
  a packaged CLI, release attestations, SBOMs, and cross-platform CI.
- Added Hugging Face dataset/model repository URL inputs with canonical ID resolution,
  validation, disk checks, and cache-aware downloads from the Streamlit UI.
- Added pre-training GPU memory inspection and safe cleanup, confirmed termination of
  eligible user-owned GPU processes, and a streamed installed-model Ollama playground.
- Accepted `HF_TOKEN` and the legacy `HUGGING_FACE_HUB_TOKEN` environment variable while
  reporting only credential presence.
- Added a confirmed global stop control with cooperative cancellation, verified
  app-owned process-tree cleanup, preserved job data, and clean local-server exit.

## [0.9.0] - 2026-08-12

- Added single-node Accelerate launch configurations for DDP and FSDP2, physical
  device selection, same-run distributed provenance, and single-GPU Unsloth gating.

## [0.8.0] - 2026-08-12

- Added Python 3.12 support, CPU/CUDA/ROCm/XPU managed profiles, bounded hardware
  probes, automatic precision selection, and explicit stable/beta support tiers.

## [0.7.0] - 2026-08-12

- Added reusable evaluation manifests, JSON/CSV/HTML reports, and two-to-four-run
  comparison data alongside Transformers and optional Inspect AI evaluation.

## [0.6.0] - 2026-08-12

- Added bounded dataset preflight, persistent worker stdout/stderr, process-tree
  cancellation, interrupted-worker reconciliation, and versioned SQLite migrations.

## [0.5.0] - 2026-08-12

- Added DPO, KTO, reward-model, and GRPO recipes, built-in/custom rewards, recipe
  contracts, and optional Inspect AI tasks. ORPO is exposed as experimental and blocked
  because TRL 1.9 does not ship an ORPO trainer.

## [0.4.0] - 2026-08-12

- Added continued pretraining and guarded full fine-tuning with VRAM, disk, and
  one-microbatch safety checks.

## [0.3.0] - 2026-08-12

- Added native Windows/Linux CUDA, ROCm, and XPU runtime profiles and smoke tests. WSL
  remains optional.

## [0.2.0] - 2026-08-12

- Added LoRA, optional Unsloth SFT, schema-v2 provenance, preflight reports, safe child
  resumes, and hashed artifact manifests.

### Added

- Project governance, security, technical, and usage documentation.

## [0.1.0] - 2026-08-12

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
