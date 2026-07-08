# Analysis Reports Feature Roadmap

**Status:** Planning
**Last Updated:** 2026-02-03

## Vision

As a mech interp researcher, I would like to define analysis reports by:
- **Model Family** - structurally similar models that share analysis logic
- **Analysis Dataset (Probe)** - the input data used for forward passes during analysis
- **Visualizations** - which views to include in the report

Reports are reusable across variants within a family and persist as saved configurations.

**Key framing:** This is mechanistic interpretability, not hyperparameter optimization. All variants within a family have already solved the problem. The goal is to understand *how* they solve it.

---

## Proposed Requirements

### REQ_020: Checkpoint Epoch-Index Display (Quick Win)
**Status:** Drafted → [active/REQ_020_checkpoint_epoch_index_display.md](../active/REQ_020_checkpoint_epoch_index_display.md)

Short-term usability fix: display epoch↔index mapping in loss curve tooltips and near the slider. Independent of the larger architecture work.

### REQ_021: Model Family Abstraction (Foundation)
**Status:** Draft complete → [drafts/REQ_021_model_families_draft.md](REQ_021_model_families_draft.md)

Define the concept of a Model Family:
- Declared (not inferred) grouping of structurally similar models
- Families share: analyzer registry, visualization options, analysis dataset schema
- Individual **variants** within a family vary by domain parameters (p, seed, etc.)
- Storage: `model_families/<family_name>/family.json`

**Decisions made:**
- Terminology: "Variant" (not "Run") to avoid optimization framing
- Explicit registration via family.json (not auto-detection)
- Generic analysis library + family-bound analyzers
- Dashboard: Family selection filters variants
- Start fresh rather than wrapping ModuloAdditionSpecification

### REQ_022: Shared Cache in Analysis Pipeline
**Status:** Future

Architectural change: share `run_with_cache` results across all analyzers within an analysis run. Currently each analyzer independently processes checkpoints.

**Benefits:**
- Single forward pass per checkpoint regardless of analyzer count
- Enables hierarchical analyzers (transforms operating on base artifacts)
- Better separation: "data extraction" vs "analysis computation"

**Depends on:** May benefit from REQ_021 context, but could be implemented independently.

### REQ_023: Configurable Analysis Datasets
**Status:** Future

Decouple "what input to analyze" from model/checkpoint selection:
- For Modulo Addition: full (a, b) grid or subsets
- For future LLMs: prompt strings
- Dataset definition is family-aware (families define valid dataset types)

**Depends on:** REQ_021 (Model Families)

### REQ_024: User-Defined Reports
**Status:** Future

The payoff: users configure which visualizations appear in a report.
- Save/load report configurations (JSON)
- Reports scoped to a model family
- Apply same report to any variant within the family

**Depends on:** REQ_021, REQ_023

---

## Sequencing

```
REQ_020 (checkpoint display)     ──────────────────────────> Can ship independently

REQ_021 (model families)         ──┬──> REQ_023 (datasets) ──┬──> REQ_024 (reports)
                                   │                          │
REQ_022 (shared cache)           ──┴──────────────────────────┘
```

- REQ_020 is independent and unblocks a usability issue
- REQ_021 is foundational for the family/report concepts
- REQ_022 is an optimization that becomes more valuable as analyzer count grows
- REQ_023 and REQ_024 build on the family abstraction

---

## Design Decisions

### Storage Format: JSON Files (not database)

**Rationale:**
- Appropriate for current scale (< 5 model families near-term)
- Human-readable, version-controllable
- Schema can evolve as understanding develops
- No infrastructure overhead
- Easy to migrate to database later if needed

### Model Families: Declared (not inferred)

**Rationale:**
- What constitutes "structurally similar" is learned over time
- System shouldn't try to auto-detect families
- User explicitly creates and names families
- Family definition can evolve as researcher's understanding deepens

### Terminology: "Variant" (not "Run")

**Rationale:**
- "Run" implies hyperparameter optimization experiments
- Mech interp: all variants have solved the problem; goal is to understand HOW
- "Variant" captures "same model, different domain parameters"
- Variants are compared to find consistent mechanisms, not discarded

### Analyzer Architecture: Generic Library + Family Binding

**Rationale:**
- Some analyses are reusable (Fourier for all modular arithmetic families)
- Library functions compose into family-specific analyzers
- New families can draw from existing library without duplication

### Directory Structure (Proposed)

```
model_families/
  modulo_addition_1layer/
    family.json              # Family definition, analyzer registry
    reports/                 # Saved report configurations
      default.json

results/                     # Variants (existing structure, programmatic)
  modulo_addition/
    modulo_addition_p113_seed42/
      checkpoints/
      artifacts/
      metadata.json
    modulo_addition_p97_seed42/
      ...

analysis/
  library/                   # Generic, reusable analysis functions
  analyzers/                 # Family-bound analyzers
```

---

## Original Notes (Preserved)

### REQ_??? - Short-term fix for checkpoint navigation
> I realized that there's no way to match up the checkpoint epoch with the checkpoint epoch index, which would allow me to jump straight to a specific epoch.

**Resolution:** Captured as REQ_020

### REQ_??? - Refined analysis pipeline
> As a mech interp researcher, I expect that every visualization will require the use of the data generated from the TransformerLens `run_with_cache` method. Additionally, for a given analysis report, all visualizations should share the same input (dataset). It should not be necessary to execute a forward pass for each visualization or analysis pass.

**Resolution:** Captured as REQ_022

### REQ_??? : User-defined analysis reports
> As a mech interp researcher, I would like to be able to create analysis reports. I want to be able to choose which visualizations to show on a given report. I want to be able to save the report configuration and reuse it at a later time.

**Resolution:** Captured as REQ_024

### REQ_??? : Support for Model Families
> As a mech interp researcher, I would like to be able to define a model family such that, within a model family, all models can share the same set of analyses, visualizations, and reports.

**Resolution:** Captured as REQ_021

### REQ_??? : Support for different analysis datasets
> As a mech interp researcher, I want to be able to define a dataset for analysis across all the checkpoints.

**Resolution:** Captured as REQ_023

---

## Related Ideas (from notes/thoughts.md)

See `notes/thoughts.md` section "2026-02-03: Analysis Report Expansion Ideas" for additional brainstorming:
- Report generation (markdown, HTML, PDF export)
- Comparison reports (diff analysis between variants)
- Aggregate metrics (grokking detection, convergence analysis)
- Export formats for reproducibility and sharing
