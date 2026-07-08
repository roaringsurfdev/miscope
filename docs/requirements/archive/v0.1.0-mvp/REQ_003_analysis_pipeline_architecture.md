# REQ_003: Analysis Pipeline Architecture

## Problem Statement
Current analysis code exists as a script (ModuloAdditionRefactored.py) that runs ad-hoc visualizations. For the workbench, we need a modular analysis pipeline that can:
- Load model checkpoints systematically
- Run analysis computations on each checkpoint (forward passes, Fourier transforms, etc.)
- Generate and persist analysis artifacts to disk
- Enable visualization rendering from artifacts without recomputation
- Be extensible for future analysis types

The analysis engine should separate concerns:
1. **Analysis computation** (expensive): Load checkpoints, run forward passes, compute features → save artifacts
2. **Visualization rendering** (cheap): Load artifacts, apply visual parameters, display → iterate quickly

This separation enables fast iteration on visualizations without re-running expensive computations.

## Conditions of Satisfaction
- [x] AnalysisPipeline class or module that orchestrates analysis
- [x] Can load checkpoints from a training run by epoch number
- [x] Executes analysis computations across specified checkpoints
- [x] Saves analysis artifacts to disk (persistent, reusable)
- [x] Artifacts stored in organized directory structure alongside checkpoints
- [x] Visualization components can load artifacts independently without recomputation
- [x] Analysis functions are modular and can be composed
- [x] Progress indication during analysis (simple logging acceptable for MVP)
- [x] Can resume/skip analysis if artifacts already exist (avoid redundant computation)

## Constraints
**Must have:**
- Reuses existing FourierEvaluation utilities
- Works with safetensors checkpoint format (from REQ_002)
- Separates analysis computation from visualization rendering
- Artifacts persisted to disk (not memory-only)
- Artifacts stored in structured directory alongside training results
- Suggested artifact formats: NumPy `.npz`, HDF5, or similar efficient formats
- Directory structure:
  ```
  results/
    model_p{prime}_seed{seed}/
      checkpoints/
        checkpoint_epoch_{epoch:05d}.safetensors
      artifacts/
        neuron_activations.npz
        fourier_coefficients.npz
        neuron_freq_norm.npz
      metadata.json
      config.json
  ```

**Must avoid:**
- Tight coupling between analysis computations and visualization rendering
- Re-running expensive computations when only changing visual parameters
- Storing Plotly figures as artifacts (store data, render figures on demand)
- Hard-coded visualization logic mixed with analysis logic

**Flexible:**
- Exact artifact file format (NumPy, HDF5, Parquet all acceptable)
- Whether to store one combined artifact or separate files per analysis type
- Granularity of artifact caching

## Context & Assumptions
- Three analysis types needed for MVP: dominant frequencies, activation heatmaps, neuron clusters
- Each analysis requires forward passes on the full dataset (expensive operation)
- Forward passes need to be run once per checkpoint, results cached as artifacts
- FourierEvaluation.py already provides key utility functions
- Disk storage is cheap, recomputation is expensive (favor caching)
- Assumption: Analysis runs after training completes (not real-time)
- Assumption: Full dataset fits in memory for forward pass computation
- Assumption: Artifacts are significantly smaller than full checkpoint files

## Decision Authority
- [x] Propose options for review

## Success Validation
- Can instantiate AnalysisPipeline with a trained model directory
- Can specify which analyses to run
- Analysis generates and saves artifacts to disk for all checkpoints
- Artifacts persist between sessions
- Can load artifacts directly without re-running analysis
- Changing visualization parameters (colors, thresholds) doesn't trigger recomputation
- Re-running analysis on same model skips checkpoints with existing artifacts
- Adding a new analysis type requires minimal changes to pipeline
- Artifact files are inspectable (can load in notebook for debugging)

---
## Notes

## Implementation Notes (Added by Claude)

**Implementation completed:** 2026-01-31

### Sub-Requirements Completed

| Sub-Requirement | Description | Status |
|-----------------|-------------|--------|
| REQ_003_001 | Core Infrastructure | Complete |
| REQ_003_002 | Dominant Frequencies Analyzer | Complete |
| REQ_003_003 | Remaining Analyzers | Complete |
| REQ_003_004 | Artifact Loader | Complete |
| REQ_003_005 | Polish & Integration Tests | Complete |

### Package Structure Created

```
analysis/
  __init__.py              # Exports Analyzer, AnalysisPipeline, ArtifactLoader
  protocols.py             # Analyzer Protocol definition
  pipeline.py              # AnalysisPipeline class
  artifact_loader.py       # Standalone ArtifactLoader
  analyzers/
    __init__.py
    dominant_frequencies.py
    neuron_activations.py
    neuron_freq_clusters.py
```

### Key Design Decisions

1. **Protocol-based Analyzer interface**: Uses `@runtime_checkable` Protocol for composition-friendly, testable design
2. **NumPy .npz artifacts**: Simple format, no external dependencies, easily inspectable
3. **Manifest.json tracking**: Enables resumability at epoch-level granularity
4. **Atomic writes**: Uses temp file + rename pattern for data integrity
5. **Standalone ArtifactLoader**: Visualization layer can load without pipeline dependency

### Test Coverage

- 100 total tests
- 80 tests for REQ_003 specifically
- 9 integration tests mapping to parent CoS items
- All tests passing

### Artifacts Produced

For each analysis run:
- `dominant_frequencies.npz` - Fourier coefficient norms (n_epochs, n_components)
- `neuron_activations.npz` - MLP activations (n_epochs, d_mlp, p, p)
- `neuron_freq_norm.npz` - Frequency specialization (n_epochs, p//2, d_mlp)
- `manifest.json` - Metadata and completion tracking
