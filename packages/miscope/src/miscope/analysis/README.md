# Analysis Pipeline

The analysis package provides a modular pipeline for computing and persisting analysis artifacts from model checkpoints.

## Architecture Overview

```mermaid
flowchart TB
    subgraph Training["Training Phase"]
        MS[ModuloAdditionSpecification]
        CP[Checkpoints<br/>*.safetensors]
        MS -->|train| CP
    end

    subgraph Analysis["Analysis Phase (Expensive)"]
        AP[AnalysisPipeline]
        AN1[WeightBasisProjectionAnalyzer]
        AN2[NeuronActivationsAnalyzer]
        AN3[ActivationFrequencyNormAnalyzer]

        AP -->|register| AN1
        AP -->|register| AN2
        AP -->|register| AN3

        CP -->|load_checkpoint| AP
        AP -->|run| ART
    end

    subgraph Artifacts["Artifacts (Disk)"]
        ART[".npz files + manifest.json"]
    end

    subgraph Visualization["Visualization Phase (Cheap)"]
        AL[ArtifactLoader]
        VIZ[Visualizations<br/>REQ_004-006]

        ART -->|load| AL
        AL --> VIZ
    end

    style Analysis fill:#e1f5fe
    style Artifacts fill:#fff3e0
    style Visualization fill:#e8f5e9
```

## Key Concepts

### Separation of Concerns

The pipeline separates **expensive computation** from **cheap visualization**:

1. **Analysis Phase**: Load checkpoints, run forward passes, compute features → save artifacts
2. **Visualization Phase**: Load artifacts, apply visual parameters, display → iterate quickly

This enables fast iteration on visualizations without re-running expensive computations.

### Analyzer Protocol

Analyzers implement a simple protocol:

```python
class Analyzer(Protocol):
    @property
    def name(self) -> str: ...

    def analyze(self, model, dataset, cache, fourier_basis) -> dict[str, np.ndarray]: ...
```

### Resumability

The pipeline tracks completed epochs in `manifest.json`. Re-running analysis skips already-computed checkpoints unless `force=True`.

## Usage

### Running Analysis

```python
from analysis import AnalysisPipeline
from analysis.analyzers import (
    ActivationFrequencyNormAnalyzer,
    NeuronActivationsAnalyzer,
    WeightBasisProjectionAnalyzer,
)

# Create pipeline
pipeline = AnalysisPipeline(model_spec)

# Register analyzers
pipeline.register(WeightBasisProjectionAnalyzer())
pipeline.register(NeuronActivationsAnalyzer())
pipeline.register(ActivationFrequencyNormAnalyzer())

# Run analysis (skips existing artifacts)
pipeline.run()

# Or force recompute
pipeline.run(force=True)

# Or analyze specific epochs
pipeline.run(epochs=[0, 1000, 5000])
```

### Loading Artifacts (Consumer-facing)

Consumers (notebooks, dashboard pages, scripts) reach a configured loader
through ``variant.artifacts`` rather than constructing ``ArtifactLoader``
directly. ``ArtifactLoader`` is an internal storage primitive (REQ_125) — the
class continues to back the pipeline and cross-epoch analyzers, but it is
not re-exported as public API.

```python
from miscope import load_family

family = load_family("modulo_addition_1layer")
variant = family.get_variant(prime=113, seed=999, data_seed=598)

# List available analyzers
print(variant.artifacts.get_available_analyzers())
# ['activation_frequency_norm', 'neuron_activations', 'weight_basis_projection', ...]

# Per-epoch load
epoch_data = variant.artifacts.load_epoch("weight_basis_projection", epoch=1000)

# Summary load
summary = variant.artifacts.load_summary("repr_geometry")

# Cross-epoch load
cross = variant.artifacts.load_cross_epoch("parameter_trajectory")
```

## Package Structure

```
analysis/
  __init__.py              # Exports Analyzer, AnalysisPipeline (ArtifactLoader is internal)
  protocols.py             # Analyzer Protocol definition
  pipeline.py              # AnalysisPipeline orchestrator
  artifact_loader.py       # Internal storage primitive (REQ_125)
  analyzers/
    __init__.py
    weight_basis_projection.py      # Per-site weight Fourier projections
    neuron_activations.py           # MLP activation heatmaps
    activation_frequency_norm.py    # Per-site activation per-frequency energy norm
    ...                             # (24 analyzers total — see registry.py)
```

## Artifacts Produced

Each analyzer writes per-epoch `.npz` files under `artifacts/{analyzer_name}/`.
A representative sample:

| Artifact | Shape | Description |
|----------|-------|-------------|
| `weight_basis_projection/epoch_*.npz` | per-site cos/sin coeffs | Weight Fourier projections (embedding, mlp_in/out, attn sites) |
| `neuron_activations/epoch_*.npz` | (d_mlp, p, p) | MLP activations reshaped to input space |
| `activation_frequency_norm/epoch_*.npz` | per-site `freq_norm` (n_freq, n_units) | Per-frequency activation energy norm (reduced; legacy `neuron_freq_norm` content) |
| `manifest.json` | - | Completion tracking and metadata |

See `analyzers/registry.py` for the full set of 24 analyzers.

## Adding a New Analyzer

1. Create a new file in `analyzers/`
2. Implement the `Analyzer` protocol:

```python
class MyAnalyzer:
    @property
    def name(self) -> str:
        return "my_analyzer"

    def analyze(self, model, dataset, cache, fourier_basis) -> dict[str, np.ndarray]:
        # Your analysis logic here
        result = ...
        return {"data": result.detach().cpu().numpy()}
```

3. Export from `analyzers/__init__.py`
4. Register with the pipeline
