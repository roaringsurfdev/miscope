# REQ_021: Model Family Abstraction

**Status:** Active
**Priority:** High (foundational for REQ_022, REQ_023, REQ_024)
**Estimated Effort:** Medium-High
**Last Updated:** 2026-02-03

## Problem Statement

The current implementation has a single model type (`ModuloAdditionSpecification`) that conflates model definition, training, and analysis utilities. As the workbench grows to support multiple model types, we need an abstraction that:

1. Groups structurally similar models that share analysis logic
2. Allows parameter variation within a family without duplicating analyzer/visualization code
3. Enables report configurations to be reused across variants in the same family
4. Provides a clear extension point for adding new model families

**Mech Interp Context:** This is not about hyperparameter optimization. All variants within a family have already solved the problem. The goal is to understand *how* they solve it, and whether the mechanism is consistent across parameter variations.

## Core Concepts

### Model Family

A **Model Family** is a declared grouping of models that share:
- **Architecture** - layer count, head count, activation functions (structural properties that determine valid analyses)
- **Analyzers** - which analysis functions are valid for this family
- **Visualizations** - which visualizations can be rendered
- **Analysis dataset schema** - what kind of probe input is valid

Families are **explicitly registered** (not auto-detected) because what constitutes "structurally similar" is learned over time by the researcher.

### Variant

A **Variant** is a specific trained model within a family. Variants share the same architecture and analysis logic but differ in domain-specific parameters.

| Concept | ML Training Mindset | Mech Interp Mindset |
|---------|---------------------|---------------------|
| What varies | Hyperparameters to optimize | Domain parameters to probe mechanisms |
| Goal | Find the best model | Understand how models work |
| Outcome | Discard inferior runs | Compare to find consistent mechanisms |

For Modulo Addition 1-Layer:
- **Domain parameters:** modulus (p), seed
- **Example variants:** p=113/seed=42, p=97/seed=42, p=113/seed=999

### Variant State

A variant progresses through states:
```
Untrained → Trained (checkpoints exist) → Analyzed (artifacts exist)
```

The UI should reflect what state each variant is in.

### Analysis Dataset (Probe)

The input data used during analysis forward passes.

- **Small toy models:** Often one canonical dataset (full (a, b) grid for Modulo Addition)
- **Larger models:** Specific probe datasets that exercise behaviors of interest (e.g., "The capitol of the state containing Dallas is")

The analysis dataset is a family-level concept—families define what probe types are valid.

## Proposed Design

### Directory Structure

```
model_families/
  modulo_addition_1layer/
    family.json              # Family definition (explicit registration)
    reports/                 # Saved report configurations
      default.json

results/                     # Trained variants (existing location, programmatic structure)
  modulo_addition_1layer/
    modulo_addition_1layer_p113_seed42/
      checkpoints/
      artifacts/
      metadata.json
      config.json
    modulo_addition_1layer_p97_seed42/
      ...

analysis/
  library/                   # Generic, reusable analysis functions
    fourier.py              # Modular Fourier basis, FFT utilities
    activations.py          # Gradient magnitude, coarseness metrics
    attention.py            # Attention pattern analysis
  analyzers/                 # Family-bound analyzers composing library functions
    dominant_frequencies.py
    neuron_activations.py
    neuron_frequency_clusters.py
```

### family.json Schema

```json
{
  "name": "modulo_addition_1layer",
  "display_name": "Modulo Addition (1 Layer)",
  "description": "Single-layer transformer trained on modular arithmetic",

  "architecture": {
    "n_layers": 1,
    "n_heads": 4,
    "d_model": 128,
    "d_mlp": 512,
    "act_fn": "relu"
  },

  "domain_parameters": {
    "prime": {
      "type": "int",
      "description": "Modulus for the addition task",
      "default": 113
    },
    "seed": {
      "type": "int",
      "description": "Random seed for model initialization",
      "default": 999
    }
  },

  "analyzers": [
    "dominant_frequencies",
    "neuron_activations",
    "neuron_frequency_clusters"
  ],

  "visualizations": [
    "dominant_frequencies_bar",
    "neuron_activation_heatmap",
    "neuron_activation_grid",
    "frequency_clusters_heatmap"
  ],

  "analysis_dataset": {
    "type": "modulo_addition_grid",
    "description": "Full (a, b) input grid for modular arithmetic"
  },

  "variant_pattern": "modulo_addition_1layer_p{prime}_seed{seed}"
}
```

### Code Architecture

```
ModelFamily (protocol)
├── name, display_name, description
├── architecture: dict
├── domain_parameters: dict[str, ParameterSpec]
├── analyzers: list[str]
├── visualizations: list[str]
├── create_model(params) -> HookedTransformer
├── generate_analysis_dataset(params) -> torch.Tensor
└── get_variant_directory(params) -> Path

Variant
├── family: ModelFamily
├── params: dict (domain parameter values)
├── state: untrained | trained | analyzed
├── checkpoints_dir, artifacts_dir
└── Methods: train(), load_checkpoint(), get_available_checkpoints()

AnalysisSession
├── variant: Variant
├── analysis_dataset: torch.Tensor
├── analyzers: list[Analyzer]
└── Produces artifacts per checkpoint
```

**Migration approach:** Start fresh rather than wrapping `ModuloAdditionSpecification`. Extract learnings but design clean separation.

### Analyzer Architecture

Two layers:

1. **Library functions** - Generic, reusable across families
   - `library/fourier.py`: `get_modular_fourier_basis(p)`, `compute_fft_2d()`
   - `library/activations.py`: `compute_gradient_magnitude()`, `compute_coarseness()`

2. **Analyzers** - Family-bound, compose library functions
   - Declared per-family in `family.json`
   - Can be reused across families that share analysis needs
   - n-layer Modulo Addition could reuse same analyzers as 1-layer

## Sub-Requirements

This requirement is broken into five sub-requirements for cleanly subdivided work:

| Sub-Req | Title | Dependencies | Focus |
|---------|-------|--------------|-------|
| [REQ_021a](REQ_021a_core_abstractions.md) | Core Abstractions | None | ModelFamily protocol, Variant class, directory conventions |
| [REQ_021b](REQ_021b_analysis_library.md) | Analysis Library Architecture | None | Separate library/ from analyzers/ |
| [REQ_021c](REQ_021c_modulo_addition_family.md) | Modulo Addition 1-Layer Implementation | 021a, 021b | Concrete family implementation |
| [REQ_021d](REQ_021d_dashboard_integration.md) | Dashboard Integration | 021a, 021c | Family-aware Analysis UI |
| [REQ_021e](REQ_021e_training_integration.md) | Training Integration | 021a, 021c | Family-aware Training UI |

**Dependency graph:**
```
REQ_021a ──────┬──→ REQ_021c ──→ REQ_021d (Analysis)
               │         ↑            │
REQ_021b ──────┴─────────┘            │
                                      ▼
                               REQ_021e (Training)
```

REQ_021a and REQ_021b can be worked in parallel. REQ_021c integrates both. REQ_021d and REQ_021e add family-aware UI for analysis and training respectively.

## Conditions of Satisfaction

All conditions are tracked in sub-requirements. Summary:

- [x] ModelFamily protocol formally defined (021a) ✓
- [x] Variant concept implemented with state tracking (021a) ✓
- [x] Analysis library separated from family-specific analyzers (021b) ✓
- [x] At least one family (`modulo_addition_1layer`) implemented with `family.json` (021c) ✓
- [x] Dashboard Analysis tab lists families, selecting one filters to its variants (021d) ✓
- [x] Existing analysis functionality preserved (no regression) (021d) ✓
- [x] New family can be created by: adding `family.json` + implementing protocol (021c) ✓
- [ ] Training flows through family abstraction (021e)
- [ ] Dashboard Training tab uses family selection (021e)
- [ ] Full end-to-end: Train → Analyze → Visualize via families (021e)

## Constraints

**Must have:**
- Explicit family registration via `family.json`
- Clear separation: Family (what) → Variant (which) → Analysis (how)
- JSON-based configuration for flexibility
- Analyzers/visualizations are family-level, parameter-agnostic
- `ModelFamily.name` used consistently as directory key under both `model_families/` and `results/`

**Must avoid:**
- Auto-detection that locks in premature structure
- Hyperparameter optimization framing
- Breaking existing trained models

**Flexible:**
- Exact directory structure details beyond the naming convention
- How much existing code is reused vs rewritten

## Decision Log

| Question | Decision | Rationale |
|----------|----------|-----------|
| Migration strategy | Start fresh | ModuloAdditionSpecification conflates concerns; clean separation is better |
| Auto-detection vs explicit | Explicit registration | Family must declare valid analyzers; can't infer from file structure |
| Analyzer binding | Generic library + family-bound | Reuse across families (Fourier for all modular arithmetic) |
| Dashboard flow | Family → Variant | User thinks in terms of model type first |
| Terminology | "Variant" not "Run" | Avoids optimization connotation; captures "same model, different parameters" |
| Directory naming | Use ModelFamily.name consistently | Simplifies lookups; family name is the key under model_families/ and results/ |

## Context & Assumptions

- Currently 1 family, expecting ~5 near-term
- Families added infrequently as researcher understanding grows
- JSON storage appropriate for this scale
- Analysis datasets are mostly fixed per family (small models) or explicitly defined probes (larger models)

## Relationship to Other Requirements

- **REQ_022 (Shared Cache):** Benefits from family-level analyzer registry
- **REQ_023 (Analysis Datasets):** Dataset schema is family-level
- **REQ_024 (Reports):** Reports scoped to families, reusable across variants

---

## Notes

**2026-02-03:** Refined through discussion. Key insights:
- "Variant" terminology preferred over "Run" to avoid ML optimization framing
- Analysis dataset = "probe" for larger models where exhaustive input isn't feasible
- Mech interp goal: understand HOW models solve problems, not WHICH model is best
- Modular Fourier basis is domain-specific (not standard FFT) and belongs in analysis library
- `ModelFamily.name` should be used consistently as directory key for both `model_families/` and `results/` directories
- `variant_pattern` updated to include full family name for consistency

**Implementation phases:**
1. Define `ModelFamily` protocol and `Variant` class (021a) ✓
2. Refactor analysis into library/ + analyzers/ structure (021b) ✓
3. Create `model_families/modulo_addition_1layer/family.json` and implement (021c) ✓
4. Update dashboard Analysis tab to be family-aware (021d) ✓
5. Update dashboard Training tab to use family abstraction (021e)
6. Validate: can add second family with minimal code
