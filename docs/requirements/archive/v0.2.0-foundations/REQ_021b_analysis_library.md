# REQ_021b: Analysis Library Architecture

**Status:** Active
**Priority:** High
**Parent:** [REQ_021](REQ_021_model_families.md)
**Dependencies:** None (can be worked in parallel with REQ_021a)
**Last Updated:** 2026-02-04

## Problem Statement

Current analysis code is tightly coupled to `ModuloAdditionSpecification`. To support multiple model families, we need:
1. A library of generic, reusable analysis functions
2. Family-bound analyzers that compose library functions
3. Clear separation so new families can reuse existing analysis logic

## Core Principle: The Scientific Invariant

The workbench exists to answer: **"How does behavior X change across training?"**

For that question to be meaningful, analysis must hold constant:
- The trained model instance (Variant)
- The probe (analysis dataset)

The **only independent variable** is the checkpoint (training moment).

This invariant is what the platform enforces. The `AnalysisPipeline` must systematically apply the same probe across all checkpoints to ensure visualizations are scientifically meaningful and not prone to researcher errors in data generation.

## Key Terminology

**Probe (Analysis Dataset):** The input data used during analysis forward passes. For small toy models, often one canonical dataset (e.g., full (a, b) grid for Modulo Addition). For larger models, probes are specific input sets designed to exercise behaviors of interest. Probe design is part of the research.

> **Note:** A single Variant may support multiple Probes in the future. For current toy models, one Probe per Variant is sufficient. Larger models won't allow full coverage (all possible inputs), so multiple specialized probes become necessary.

**AnalysisRunConfig:** Configuration specifying what work the pipeline should perform:
- Which analyzers to run
- Which checkpoints to analyze (optional subset; defaults to all available)

This is variant-agnosti: the same config can be applied to multiple variants.

## Scope

This sub-requirement covers:
1. Creating `analysis/library/` with generic functions
2. Creating `analysis/analyzers/` with composable analyzers
3. Migrating existing analysis code to this structure
4. Defining the Analyzer protocol

## Proposed Design

### Directory Structure

```
analysis/
  __init__.py
  library/                   # Generic, reusable functions
    __init__.py
    fourier.py              # Modular Fourier basis, FFT utilities
    activations.py          # Gradient magnitude, coarseness metrics
    attention.py            # Attention pattern analysis
  analyzers/                 # Family-bound analyzers
    __init__.py
    base.py                 # Analyzer protocol
    dominant_frequencies.py
    neuron_activations.py
    neuron_frequency_clusters.py
```

### Library Functions

**fourier.py** - Modular Fourier analysis utilities:
```python
def get_modular_fourier_basis(p: int, device: torch.device) -> torch.Tensor:
    """Generate Fourier basis for modular arithmetic with period p."""
    ...

def compute_fft_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Compute 2D FFT for activation analysis."""
    ...

def project_onto_fourier_basis(
    weights: torch.Tensor,
    basis: torch.Tensor
) -> torch.Tensor:
    """Project weight matrix onto Fourier basis."""
    ...
```

**activations.py** - Activation analysis utilities:
```python
def compute_neuron_activations(
    model: HookedTransformer,
    inputs: torch.Tensor,
    layer: int
) -> torch.Tensor:
    """Extract MLP neuron activations for given inputs."""
    ...

def compute_activation_grid(
    activations: torch.Tensor,
    grid_shape: tuple[int, int]
) -> torch.Tensor:
    """Reshape activations into (a, b) grid format."""
    ...
```

### Analyzer Protocol

```python
from typing import Protocol, Any
from pathlib import Path
import torch
from transformer_lens import HookedTransformer

class AnalysisResult(TypedDict):
    """Standard structure for analyzer output."""
    data: Any  # The computed analysis data
    metadata: dict[str, Any]  # Computation parameters, timestamps, etc.

class Analyzer(Protocol):
    """Protocol for family-bound analyzers."""

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description."""
        ...

    def analyze(
        self,
        model: HookedTransformer,
        analysis_dataset: torch.Tensor,
        params: dict[str, Any]
    ) -> AnalysisResult:
        """Run analysis and return results."""
        ...

    def save_artifact(
        self,
        result: AnalysisResult,
        artifacts_dir: Path,
        checkpoint_epoch: int
    ) -> Path:
        """Persist analysis result to disk."""
        ...

    def load_artifact(
        self,
        artifacts_dir: Path,
        checkpoint_epoch: int
    ) -> AnalysisResult:
        """Load previously computed result."""
        ...
```

### Analyzer Implementations

**dominant_frequencies.py:**
```python
class DominantFrequenciesAnalyzer:
    """Analyzes which Fourier frequencies dominate the embedding space."""

    name = "dominant_frequencies"
    description = "Identifies dominant frequencies in learned embeddings"

    def analyze(self, model, analysis_dataset, params):
        # Uses library/fourier.py functions
        basis = get_modular_fourier_basis(params["prime"], model.device)
        embedding = model.W_E
        coefficients = project_onto_fourier_basis(embedding, basis)
        norms = coefficients.norm(dim=-1)
        ...
```

**neuron_activations.py:**
```python
class NeuronActivationsAnalyzer:
    """Analyzes MLP neuron activation patterns over input grid."""

    name = "neuron_activations"
    description = "Computes neuron activation heatmaps for (a, b) inputs"

    def analyze(self, model, analysis_dataset, params):
        # Uses library/activations.py functions
        activations = compute_neuron_activations(model, analysis_dataset, layer=0)
        grid = compute_activation_grid(activations, (params["prime"], params["prime"]))
        ...
```

### AnalyzerRegistry

```python
class AnalyzerRegistry:
    """Registry of available analyzers."""

    _analyzers: dict[str, type[Analyzer]] = {}

    @classmethod
    def register(cls, analyzer_class: type[Analyzer]) -> type[Analyzer]:
        """Decorator to register an analyzer."""
        cls._analyzers[analyzer_class.name] = analyzer_class
        return analyzer_class

    @classmethod
    def get(cls, name: str) -> Analyzer:
        """Get analyzer instance by name."""
        return cls._analyzers[name]()

    @classmethod
    def get_for_family(cls, family: ModelFamily) -> list[Analyzer]:
        """Get all analyzers valid for a family."""
        return [cls.get(name) for name in family.analyzers]
```

### AnalysisRunConfig

```python
@dataclass
class AnalysisRunConfig:
    """Configuration for an analysis run.

    Specifies what work the pipeline should perform. This is variant-agnostic;
    the same config can be applied to multiple variants.
    """

    analyzers: list[str]
    """Which analyzers to run (by name)."""

    checkpoints: list[int] | None = None
    """Which checkpoints to analyze. None means all available."""
```

### AnalysisPipeline Interface (Refined)

The pipeline takes a Variant and an AnalysisRunConfig. It obtains everything it needs through the Variant and its Family:

```python
class AnalysisPipeline:
    """Orchestrates analysis across checkpoints.

    Enforces the scientific invariant: same Variant + same Probe across
    all analyzed checkpoints.
    """

    def __init__(self, variant: Variant, config: AnalysisRunConfig):
        """
        Initialize the pipeline.

        The pipeline retrieves what it needs from the variant:
        - variant.family → analyzers, model creation, probe generation
        - variant → checkpoints, checkpoint loading, artifacts_dir
        - config → which (analyzers × checkpoints) to compute
        """
        self.variant = variant
        self.config = config
        self.artifacts_dir = variant.artifacts_dir

        # Generate probe from family (enforces invariant)
        self._probe = variant.family.generate_analysis_dataset(variant.params)

    def run(self, force: bool = False) -> None:
        """Execute analysis, computing only missing artifacts unless force=True."""
        ...
```

This design eliminates the need for adapter classes. The pipeline is family-agnostic—it doesn't need to know about `prime`, `seed`, or other domain-specific parameters. It asks the Variant/Family for what it needs.

### Future: Gap-Filling Pattern

For report rendering, the system can identify missing (analyzer, checkpoint) combinations and construct an `AnalysisRunConfig` to fill gaps:

```python
def get_missing_analyses(
    variant: Variant,
    required_analyzers: list[str]
) -> AnalysisRunConfig | None:
    """Identify missing analyses needed for a report.

    Returns an AnalysisRunConfig for just the missing work, or None if complete.
    """
    ...
```

This enables incremental analysis—only compute what's needed, never redo existing work.

## Conditions of Satisfaction

### Library & Analyzers (Original)
- [x] `analysis/library/` contains extracted generic functions
- [x] `analysis/analyzers/` contains family-bound analyzers
- [x] `Analyzer` protocol defined with analyze/save/load methods
- [x] `AnalyzerRegistry` can discover and instantiate analyzers
- [x] Existing Fourier analysis code migrated to `library/fourier.py`
- [x] Existing activation analysis code migrated to `library/activations.py`
- [x] At least 3 analyzers implemented (dominant_frequencies, neuron_activations, neuron_frequency_clusters)
- [x] Analyzers can be instantiated and run independently of dashboard
- [ ] Unit tests for library functions

### Pipeline Refinement (Added 2026-02-04)
- [x] `AnalysisRunConfig` dataclass defined
- [x] `AnalysisPipeline` accepts `(Variant, AnalysisRunConfig)` instead of `model_spec`
- [x] Pipeline obtains probe via `variant.family.generate_analysis_dataset()`
- [x] Pipeline obtains domain params via `variant.params` (not hardcoded `prime`, `seed`)
- [x] `VariantSpecificationAdapter` eliminated
- [x] Analyzers receive `context: dict` containing params and precomputed values
- [ ] Remaining test files updated for new API (test_remaining_analyzers.py, test_req_003_integration.py, test_artifact_loader.py, test_modulo_addition_family.py)

## Constraints

**Must have:**
- Clear separation: library functions have no family knowledge
- Analyzers compose library functions, don't duplicate logic
- Artifact format supports incremental loading (don't load all checkpoints to view one)
- Pipeline enforces scientific invariant (same probe across all checkpoints)
- Pipeline is family-agnostic (no hardcoded domain parameter names)

**Must avoid:**
- Library functions depending on specific family implementations
- Analyzers with hardcoded model architecture assumptions
- Adapter classes that duplicate logic already in Family/Variant
- Pipeline depending on domain-specific properties like `prime` or `seed` directly

**Flexible:**
- Artifact storage format (numpy, torch, pickle, etc.)
- Whether analyzers are classes or functions with protocol wrapper

## Migration Notes

Code to extract from existing implementation:
- `FourierEvaluation.py` → `library/fourier.py`
- Activation extraction from `ModuloAdditionRefactored.py` → `library/activations.py`
- Visualization-specific analysis → respective analyzer modules

The goal is extraction and reorganization, not rewriting. Preserve working logic while establishing clean boundaries.

## Decision Log

| Date | Question | Decision | Rationale |
|------|----------|----------|-----------|
| 2026-02-04 | How should pipeline get domain params? | Via `variant.params` dict | Pipeline should be family-agnostic; analyzers interpret params |
| 2026-02-04 | Keep or eliminate `VariantSpecificationAdapter`? | Eliminate | Adapter duplicates logic and couples pipeline to Modulo Addition specifics |
| 2026-02-04 | What is the analysis input called? | "Probe" (analysis dataset) | Domain-meaningful term; clarifies it's intentionally designed input for analysis |
| 2026-02-04 | Can same probe be used across checkpoints? | Yes, required | This is the scientific invariant—only checkpoint varies |
| 2026-02-04 | Multiple probes per variant? | Future enhancement | MVP uses one probe; larger models will need multiple specialized probes |

## Notes

**2026-02-04:** Refined based on analysis of `VariantSpecificationAdapter`. Key insights:
- The adapter existed because `AnalysisPipeline` was designed around `ModuloAdditionSpecification`'s interface
- The adapter duplicated ~40 lines of domain-specific data generation
- Proper fix: make pipeline take `(Variant, AnalysisRunConfig)` and ask Family for what it needs
- The "scientific invariant" (same probe across checkpoints) is the core value proposition of the workbench
- "Probe" terminology preferred over "analysis dataset" for clarity
