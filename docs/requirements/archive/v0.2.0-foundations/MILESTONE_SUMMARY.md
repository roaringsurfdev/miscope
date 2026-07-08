# v0.2.0 — First Foundational Release

**Released:** 2026-02-06
**Version:** 0.2.0

## Summary

This release takes the project from prototype to a foundational architecture. The MVP proved viability; v0.2.0 establishes the abstractions and infrastructure for sustained research.

## Requirements

| Requirement | Description | Key Files |
|-------------|-------------|-----------|
| [REQ_020](REQ_020_checkpoint_epoch_index_display.md) | Checkpoint epoch-index display in loss curve tooltips and slider | `dashboard/app.py` |
| [REQ_021](REQ_021_model_families.md) | Model Family abstraction (parent requirement) | — |
| [REQ_021a](REQ_021a_core_abstractions.md) | Core abstractions: ModelFamily protocol, Variant, FamilyRegistry | `families/`, `analysis/protocols.py` |
| [REQ_021b](REQ_021b_analysis_library.md) | Analysis library architecture: library/ + analyzers/ separation | `analysis/library/`, `analysis/analyzers/` |
| [REQ_021c](REQ_021c_modulo_addition_family.md) | Modulo Addition 1-Layer family implementation | `model_families/modulo_addition_1layer/` |
| [REQ_021d](REQ_021d_dashboard_integration.md) | Dashboard integration with family-aware Analysis tab | `dashboard/app.py` |
| [REQ_021e](REQ_021e_training_integration.md) | Training integration with family selection | `dashboard/app.py`, `families/` |
| [REQ_021f](REQ_021f_per_epoch_artifacts.md) | Per-epoch artifact storage | `analysis/pipeline.py`, `analysis/artifact_loader.py` |

## Key Decisions

- **ModelFamily as Protocol:** Enables multiple implementation strategies (JSON-driven, code-driven)
- **Variant carries family reference:** Variant knows its family; AnalysisRunConfig does not
- **Per-epoch artifact storage:** Eliminates memory exhaustion; artifacts stored as `epoch_{NNNNN}.npz`
- **ArtifactLoader on-demand loading:** Dashboard loads single epochs per slider interaction
- **Generic library + family-bound analyzers:** Reusable analysis functions compose into family-specific analyzers

## Architecture Established

```
model_families/          # Family definitions (family.json)
families/                # Family protocols and registry
analysis/
  library/               # Generic, reusable analysis functions
  analyzers/             # Family-bound analyzer implementations
  pipeline.py            # AnalysisPipeline orchestrator
  artifact_loader.py     # On-demand artifact access
results/                 # Per-variant checkpoints and artifacts
```

## Also Included

- [analysis_reports.md](analysis_reports.md) — Roadmap document that drove the design of REQ_020–021. Preserved for historical context; the REQ_022/023/024 numbers referenced in this document were reassigned to different requirements in v0.2.1.
