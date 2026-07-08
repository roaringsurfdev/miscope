# MIScope

A mechanistic interpretability research tool for studying how neural networks learn during training. The workbench standardizes training, analysis, and visualization so that researchers can focus on exploring emergent behaviors rather than managing infrastructure or tracking notebook cell execution.

## Purpose

The workbench exists to answer: **"How does behavior X change across training?"**

Train small Transformer models, capture checkpoints at key moments, and analyze how computational structures emerge over time. The platform enforces a scientific invariant — same model variant, same probe, only the checkpoint varies — so that visualizations are directly comparable and scientifically meaningful.

## Current Status

**v0.8.x — Analysis and Visualization Platform**

Thirty Modulo Addition variants across different primes, seeds, and data seeds are actively used for research. The platform includes a full analysis pipeline, an interactive Dash dashboard, and a view catalog API for notebook-based exploration.

See [CHANGELOG.md](CHANGELOG.md) for the full release history.

## Workflow

```
Define Family → Train Variants → Analyze Checkpoints → Explore Visualizations
```

1. **Define** a Model Family with architecture, analyzers, and probes (`data/{family}/family.json`)
2. **Train** Variants with different domain parameters (e.g., different primes or seeds)
3. **Analyze** each variant's checkpoints to generate per-epoch analysis artifacts
4. **Explore** training dynamics through the dashboard or the notebook API

## Key Features

### Model Families and Variants
- JSON-driven family definitions in `data/{family}/family.json` — architecture, analyzers, and parameter schemas
- Variants differ only in domain parameters (modulus, model seed, data seed)
- Supports per-variant intervention sub-variants for targeted experiments

### Analysis Engine
- Per-epoch artifact storage: one `.npz` file per (analyzer, epoch) for constant memory usage
- Resumable analysis: skips already-computed epochs automatically
- Three tiers: **per-epoch analyzers**, **cross-epoch analyzers** (trajectory), **secondary analyzers** (derived metrics)
- Extensible `Analyzer` protocol for adding new analysis functions

### View Catalog API
- Named views for all visualizations (`variant.at(epoch).view("view_name")`)
- Views are universal instruments — they apply to any transformer, not tied to a family
- Supports notebook-inline display (`.show()`), figure access (`.figure()`), and file export (`.export()`)

### Interactive Dashboard
- Dash-based UI with live training and analysis job management
- Synchronized epoch slider across all visualizations
- On-demand per-epoch loading — no full dataset in memory
- Pages: Variant Analysis, Cross-Variant Comparison, Multi-Stream Specialization, Transient Frequencies, Intervention Check, and more

### Research Notebook
- [fieldnotes/](fieldnotes/) — an Astro-based research journal, published to GitHub Pages
- Figures exported from the analysis pipeline embed directly in posts

## Technology Stack

- **Python 3.11+** — supports 3.11, 3.12, and 3.13
- **PyTorch** — training and model execution
- **TransformerLens** — model architecture (`HookedTransformer`)
- **Safetensors** — checkpoint persistence
- **Plotly** — interactive visualizations
- **Dash** — dashboard UI
- **NumPy** — analysis artifact storage (per-epoch `.npz` files)
- **uv** — package management
- **pytest** — testing framework

## Project Structure

The repo is a uv workspace organized into a publishable package and the apps that consume it.

```
/
├── packages/
│   └── miscope/                  # Publishable analysis library (import miscope.*)
│       ├── src/miscope/
│       │   ├── analysis/         # Pipeline, analyzers, artifacts, protocols
│       │   ├── families/         # Family registry, Variant, implementations
│       │   ├── views/            # View catalog API (EpochContext, BoundView)
│       │   └── visualization/    # Renderers and export
│       ├── tests/                # Package unit/component tests
│       └── pyproject.toml        # Publishable project definition
├── apps/
│   ├── dashboard/                # Dash web dashboard
│   │   ├── src/dashboard/
│   │   │   ├── pages/            # Per-page layout and callbacks
│   │   │   ├── app.py            # Application entry point
│   │   │   └── state.py          # Dashboard state management
│   │   ├── tests/
│   │   └── pyproject.toml
│   ├── fieldnotes/               # Research notebook (Astro → GitHub Pages)
│   └── research/                 # Exploratory work
│       ├── notebooks/            # *.ipynb research notebooks
│       └── sketches/             # Exploratory *.py (POCs, one-off analyses)
├── scripts/                      # Operational scripts (run_analysis.py, train_*, etc.)
├── tests/
│   └── integration/              # Cross-cutting tests (placeholder)
├── docs/                         # Documentation, requirements, notes, policies
└── data/                         # Unified per-family data root (REQ_123)
    └── {family}/
        ├── family.json           # Family definition (tracked)
        ├── ideal_frequency_sets.json  # Optional family-level config (tracked)
        ├── variant_registry.json # Compiled aggregate (gitignored)
        └── variants/             # Training outputs (gitignored)
            └── {variant}/        # e.g. p113_seed999_dseed598
                ├── checkpoints/  # Model checkpoints (.safetensors)
                ├── artifacts/    # Per-epoch analysis artifacts
                │   └── {analyzer}/  # epoch_00000.npz, epoch_00100.npz, ...
                ├── metadata.json
                └── config.json
```

## Getting Started

### Prerequisites
```bash
python --version  # Python 3.11+
uv --version      # Package manager (https://docs.astral.sh/uv/)
```

### Installation
```bash
git clone https://github.com/roaringsurfdev/miscope.git
cd miscope
uv sync
```

### Running the Dashboard
```bash
uv run python dashboard/app.py
```

### Running Tests
```bash
uv run pytest
```

### Notebook API

```python
from miscope import load_family

family = load_family("modulo_addition_1layer")
variant = family.get_variant(prime=113, seed=999, data_seed=598)

# Pin an epoch and access any view
ctx = variant.at(epoch=5000)
ctx.view("training.metadata.loss_curves").show()
ctx.view("parameters.pca.trajectory").show()

# Or pass kwargs directly to the view
variant.view("neuron.freq.distribution", site="mlp_post").show()
```

## Architecture

### Separation of Concerns

Two constraints hold across all features:

1. **Views are universal instruments.** Analytical lenses (PCA, Fourier, neuron activations, attention patterns) apply to any transformer. They do not belong to a model family.
2. **Families are context providers, not view owners.** A family contributes probe construction, interpretive context, and task-specific metrics. It does not register or own analytical views.

### Per-Epoch Artifact Storage

```
Training → Checkpoints (.safetensors) → Analysis → Artifacts (.npz) → Views (on-demand)
```

Analysis artifacts are stored one file per (analyzer, epoch). The dashboard loads only what the current view needs.

## Documentation

- [**PROJECT.md**](PROJECT.md) — Project scope, mission, and architectural principles
- [**DOMAIN_MODEL.md**](DOMAIN_MODEL.md) — Core domain objects and relationships
- [**CHANGELOG.md**](CHANGELOG.md) — Release history
- [**requirements/**](requirements/) — Structured requirements

## Acknowledgments

Based on research from:
- Neel Nanda's "A Mechanistic Interpretability Analysis of Grokking"
- TransformerLens library for interpretability research

---
**Last Updated:** 2026-04-04
