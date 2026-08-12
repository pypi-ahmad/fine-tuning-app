# Graph Report - .  (2026-08-12)

## Corpus Check
- 105 files · ~121,139 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 404 nodes · 992 edges · 21 communities (14 shown, 7 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 50 edges (avg confidence: 0.58)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Streamlit Application UI
- GPU Memory Management
- Artifacts and Evaluation
- Job Lifecycle
- Training Domain Models
- CLI and Diagnostics
- Project Documentation
- Hugging Face References
- Ollama API Tests
- UA Batch Analysis
- UA Graph Assembly
- UA Fingerprint Input
- UA Final Save
- Linux Launcher
- UA Architecture Analysis
- UA Graph Validation
- UA Tour Analysis
- Dependabot Configuration
- Package Metadata
- Graph Merge Report

## God Nodes (most connected - your core abstractions)
1. `OllamaClient` - 25 edges
2. `RunManifest` - 21 edges
3. `run()` - 20 edges
4. `create_job()` - 18 edges
5. `get_job()` - 18 edges
6. `scan_machine()` - 18 edges
7. `update_job()` - 17 edges
8. `stop_application_processes()` - 17 edges
9. `inspect_gpu_memory()` - 16 edges
10. `FakeProcess` - 16 edges

## Surprising Connections (you probably didn't know these)
- `FakeProcess` --uses--> `DatasetSpec`  [INFERRED]
  tests/test_lifecycle.py → src/fine_tuning_studio/domain.py
- `FakeProcess` --uses--> `ModelSpec`  [INFERRED]
  tests/test_lifecycle.py → src/fine_tuning_studio/domain.py
- `FakeProcess` --uses--> `TrainingSpec`  [INFERRED]
  tests/test_lifecycle.py → src/fine_tuning_studio/domain.py
- `FakeProcess` --uses--> `EvaluationSpec`  [INFERRED]
  tests/test_lifecycle.py → src/fine_tuning_studio/domain.py
- `FakeProcess` --uses--> `ExportSpec`  [INFERRED]
  tests/test_lifecycle.py → src/fine_tuning_studio/domain.py

## Import Cycles
- None detected.

## Communities (21 total, 7 thin omitted)

### Community 0 - "Streamlit Application UI"
Cohesion: 0.07
Nodes (56): Exception, active_worker_pids(), compatibility_rail(), dataset_page(), format_bytes(), gpu_memory_rows(), hub_error_message(), machine_report() (+48 more)

### Community 1 - "GPU Memory Management"
Cohesion: 0.09
Nodes (45): Protocol, SimpleNamespace, _ancestor_pids(), _basename(), _clear_torch_caches(), _current_username(), _find_value(), GpuCleanupResult (+37 more)

### Community 2 - "Artifacts and Evaluation"
Cohesion: 0.07
Nodes (45): RuntimeError, file_sha256(), Path, write_artifact_manifest(), JobStatus, compare_reports(), EvaluationManifest, inspect_command() (+37 more)

### Community 3 - "Job Lifecycle"
Cohesion: 0.12
Nodes (36): dialog, Process, monitor_page(), stop_application_dialog(), ensure_within(), Path, create_job(), _database() (+28 more)

### Community 4 - "Training Domain Models"
Cohesion: 0.17
Nodes (29): build_manifest(), DatasetSpec, EvaluationSpec, ExportSpec, ModelSpec, ProvenanceSpec, Any, RunManifest (+21 more)

### Community 5 - "CLI and Diagnostics"
Cohesion: 0.12
Nodes (28): export_page(), doctor(), _doctor_text(), main(), run_app(), build_diagnostics(), Path, redact() (+20 more)

### Community 6 - "Project Documentation"
Cohesion: 0.10
Nodes (29): Changelog, Code of Conduct, Contributing guide, Application home, Configuration documentation, Dataset validation, Datasets documentation, Development documentation (+21 more)

### Community 7 - "Hugging Face References"
Cohesion: 0.20
Nodes (16): download_hub_repository(), _existing_cache_parent(), HubDownloadPlan, normalize_hub_reference(), plan_hub_download(), Path, RepoType, _validated_repo_id() (+8 more)

### Community 8 - "Ollama API Tests"
Cohesion: 0.16
Nodes (12): FakeResponse, Any, MonkeyPatch, parametrize, Request, request_payload(), test_lists_installed_and_running_models(), test_rejects_unsafe_ollama_hosts() (+4 more)

### Community 9 - "UA Batch Analysis"
Cohesion: 0.21
Nodes (12): edges, extraction, files, fileSummary(), fileTags(), fs, groups, input (+4 more)

### Community 10 - "UA Graph Assembly"
Cohesion: 0.20
Nodes (8): fs, graph, [graphPath, layersPath, tourPath, outputPath, commitHash], layers, nodeIds, output, scan, tour

### Community 11 - "UA Fingerprint Input"
Cohesion: 0.40
Nodes (4): fs, input, [projectRoot, scanPath, outputPath, gitCommitHash], scan

### Community 12 - "UA Final Save"
Cohesion: 0.40
Nodes (4): [assembledPath, graphPath, metaPath, scanPath, gitCommitHash], fs, meta, scan

## Knowledge Gaps
- **40 isolated node(s):** `fs`, `input`, `extraction`, `results`, `nodes` (+35 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `OllamaClient` connect `GPU Memory Management` to `Streamlit Application UI`, `Ollama API Tests`?**
  _High betweenness centrality (0.080) - this node is a cross-community bridge._
- **Why does `RunManifest` connect `Training Domain Models` to `Streamlit Application UI`, `Artifacts and Evaluation`, `Job Lifecycle`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Why does `scan_machine()` connect `Streamlit Application UI` to `CLI and Diagnostics`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `OllamaClient` (e.g. with `GpuCleanupResult` and `GpuDeviceMemory`) actually correct?**
  _`OllamaClient` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `RunManifest` (e.g. with `EventWriter` and `FakeProcess`) actually correct?**
  _`RunManifest` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `fs`, `input`, `extraction` to the rest of the system?**
  _40 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Streamlit Application UI` be split into smaller, more focused modules?**
  _Cohesion score 0.07242063492063493 - nodes in this community are weakly interconnected._