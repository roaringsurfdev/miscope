# MIScope (Training Dynamics Workbench)

## Mission

**A dynamics analysis platform that standardizes and hones lenses on models as they learn.**

This is a platform for mechanistic interpretability research, not a model optimization tool. It asks: *how did learning happen?* — not: *did the model learn the task?*

Train/test loss is a starting point for investigation. The platform provides the instruments to go deeper.

## Scientific Strategy

Modulo addition is the **calibration model** — small, well-understood, with known structure (Fourier representations, clear grokking dynamics). It is the model against which analytical instruments are validated. When PCA reveals the expected parameter trajectory and Fourier analysis shows the expected frequency specialization, the instruments are confirmed.

When the next family is introduced, the question shifts to: *what's the same, and what's different?* That comparative question is only scientifically valid if the measurement protocol is identical across families. The platform enforces this.

The platform **accumulates analytical capability over time.** When a lens reveals something meaningful (e.g., the PC2/PC3 trajectory loop as a signature of grokking), it is refined and added to the catalog. When the next family is introduced, the full catalog is immediately available. If a pattern reappears: universal signature. If it doesn't: the difference is the finding.

## Core Invariants

**Scientific invariant:** For any analysis run, the only independent variable is the training checkpoint. The model variant and probe dataset are held constant. Confounds introduced by researcher error — different probes, accidental model variation — make visualizations misleading. The workbench systematizes this so comparisons are meaningful.

**Architectural invariant:** Analytical views are universal instruments. A lens that reveals structure in one transformer applies to any transformer. The instrument does not change shape because of the model family. Families are context providers — they contribute probe construction, interpretive context (e.g., a prime-based Fourier basis for modulo addition), and task-specific performance metrics. They do not own analytical views.

## What This Platform Is Not

- Not a model optimization tool — no hyperparameter search, no benchmark tracking
- Not a "what did the model learn?" tool — final performance is a starting point, not the answer
- Not modulo-addition-specific — that is where we *start* calibrating instruments, not where we stop

## Goals

A mechanistic researcher can:
- Train variants of a model family and save checkpoints at configurable intervals
- Run a standardized analysis pipeline across all checkpoints for any configured set of analyzers
- Explore any analytical view across training via the web dashboard or a notebook
- Compare the same view across multiple variants of the same family
- Apply the full view catalog to a new family without implementing new views

## Scope

**In scope:**
- Systematic analysis of training dynamics (how behavior emerges over training)
- Standardized analytical views applicable across model families (the View Catalog)
- Notebook-based exploration for discovery and visualization design
- Web dashboard for interactive exploration across training checkpoints
- Small-to-medium transformer models (TransformerLens compatible)

**Out of scope:**
- Model performance optimization
- Cloud infrastructure (local execution only, for now)
- Non-transformer architectures (for now)

---

## Domain Concepts

**Model Family:** A declared grouping of models sharing architecture and training protocol. The family contributes: probe construction logic, interpretive context (e.g., Fourier basis for modulo addition), task-specific performance metrics, and training configuration. The family does *not* own analytical views — those belong to the View Catalog and are universal. Families are explicitly registered because what constitutes "structurally similar" is learned over time by the researcher.

**Model Variant:** A specific trained model within a family, differing in domain parameters (e.g., modulus, seed). Variants share architecture and analysis logic. Each variant has its own checkpoints and analysis artifacts. Variants are the unit of comparison.

**Probe:** The input data used during analysis forward passes. For small toy models, one canonical dataset (e.g., the full (a, b) grid for modulo addition). For larger models, targeted probes that exercise specific behaviors. Probe design is part of the research for larger models.

**Checkpoint:** A snapshot of model weights at a specific training epoch. Saved at configurable intervals to enable analysis of how behaviors emerge over training.

**Analyzer:** A module that generates analysis artifacts from a model checkpoint and activation cache. Analyzers are generally applicable across families — they compute a single function and store numpy arrays as per-epoch `.npz` files. Some analyzers may require family-specific context (e.g., a Fourier basis), but the analyzer itself is not family-owned.

**Analysis Run (AnalysisPipeline in code):** Orchestrates artifact generation across checkpoints. Loads each checkpoint, runs a forward pass with the probe, and passes output to each configured analyzer.

**View Catalog:** The registry of named analytical views. Each view definition declares: its name, which analyzer artifact it requires, how to load that artifact, and which renderer to call. Views are available to all variants of any family. Families may contribute task-specific views (e.g., accuracy against ground truth), but the analytical catalog is universal. The catalog grows as new lenses are discovered and validated.

---

## Architecture

```
                         View Catalog
                 (universal + task-specific views)
                              │
              ┌───────────────┼───────────────┐
              │               │               │
          Notebook        Dashboard        Export
          (explore)       (navigate)   (static/gif)
              │               │               │
              └───────────────┼───────────────┘
                              │
                       miscope.views
               (Separated Presentation layer)
                              │
              ┌───────────────┼───────────────┐
              │                               │
       ArtifactLoader                    Renderers
     (data loading)               (plotly figures)
              │
      Analysis Pipeline
   (analyzers → per-epoch .npz)
              │
       Model Families
  (probe, context, training config)
              │
         Checkpoints
     (safetensors files)
```

**Key separation:** `miscope.views` is the connective tissue between data loading and rendering. Neither the dashboard nor notebooks implement their own data→renderer wiring. The dashboard, notebooks, and export utilities all consume the View Catalog. Adding a new analytical view means adding it once to the catalog; all consumers get it immediately.

The interface:
```python
variant.view("parameter_trajectory").show()     # inline in notebook
variant.view("parameter_trajectory").export("png")  # static file
variant.view("parameter_trajectory").figure()   # raw Plotly fig for dashboard
```

---

## Technology Stack

- Python 3.13, PyTorch with CUDA support
- TransformerLens (transformer model support)
- Plotly (visualizations)
- Dash (web dashboard)
- pytest (testing)
- safetensors (checkpoint persistence)
- uv (package management)

## Open Questions

- Async architecture for training and analysis jobs (forcing function: multi-probe support or long-running jobs)
- SQLite for job state management when async jobs arrive
- Probe design methodology for larger models
- Optimization strategy for expensive analyses (representation-space PCA) as model size grows

---

**Last Updated:** 2026-02-21
*(For version history and recent changes, see [CHANGELOG.md](CHANGELOG.md))*
