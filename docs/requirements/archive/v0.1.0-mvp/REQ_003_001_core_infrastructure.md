# REQ_003_001: Core Infrastructure

## Problem Statement
REQ_003 requires an analysis pipeline that orchestrates analysis across checkpoints. Before implementing individual analyzers, we need the foundational infrastructure: the pipeline class, the analyzer protocol, and the manifest system for tracking completion state.

This sub-requirement establishes the core patterns that all analyzers will follow.

## Conditions of Satisfaction
- [x] `analysis/` package structure created with `__init__.py` files
- [x] `Analyzer` Protocol defined with `name` property and `analyze()` method
- [x] `AnalysisPipeline` class can be instantiated with a `ModuloAdditionSpecification`
- [x] Pipeline has `register()` method for adding analyzers (supports chaining)
- [x] Pipeline has `run()` method that iterates over checkpoints
- [x] Pipeline creates `artifacts/` directory if it doesn't exist
- [x] Pipeline saves `manifest.json` tracking which epochs are completed per analyzer
- [x] Pipeline loads existing manifest on initialization
- [x] Pipeline skips already-completed epochs (resumability)
- [x] Pipeline `force=True` parameter recomputes even existing artifacts

## Constraints
**Must have:**
- Uses `ModuloAdditionSpecification.load_checkpoint()` and `get_available_checkpoints()`
- Creates fresh model instance per checkpoint (no memory accumulation)
- Runs forward pass once per epoch, shares cache with all analyzers
- Stores artifacts in `model_spec.artifacts_dir`

**Must avoid:**
- Tight coupling to specific analyzers (protocol-based, not inheritance)
- Holding model instances across checkpoints
- Writing artifacts without updating manifest

**Flexible:**
- Exact manifest structure (as long as it tracks epochs per analyzer)
- Whether to save incrementally or at end of run

## Context & Assumptions
- `ModuloAdditionSpecification` already has `artifacts_dir` property pointing to correct location
- Forward pass via `model.run_with_cache(dataset)` returns logits and activation cache
- Fourier basis computed once and passed to all analyzers
- Assumption: All checkpoints fit pattern and are loadable

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Can create pipeline: `pipeline = AnalysisPipeline(model_spec)`
- Can register mock analyzer: `pipeline.register(MockAnalyzer())`
- Running pipeline creates artifacts directory
- Manifest.json exists after run
- Running pipeline twice doesn't duplicate work (uses manifest to skip)
- Running with `force=True` recomputes everything

---
## Notes

## Implementation Notes (Added by Claude)

**Implementation completed:** 2026-01-31

**Package structure created:**
```
analysis/
  __init__.py          # Exports Analyzer, AnalysisPipeline
  protocols.py         # Analyzer Protocol definition
  pipeline.py          # AnalysisPipeline class
  analyzers/
    __init__.py        # Package for individual analyzers
```

**Key design decisions:**
- Used `@runtime_checkable` Protocol for Analyzer interface - enables `isinstance()` checks
- Pipeline runs forward pass once per epoch, shares cache with all analyzers
- Artifacts saved as NumPy `.npz` with atomic writes (temp file + rename)
- Manifest tracks `epochs_completed`, `shapes`, `dtypes`, and `updated_at` per analyzer
- Results buffered in memory during run, saved incrementally every N epochs

**Key code locations:**
- `analysis/protocols.py:11-40` - Analyzer Protocol definition
- `analysis/pipeline.py:28-265` - AnalysisPipeline class
- `tests/test_analysis_pipeline.py` - 22 tests covering all CoS items

**Tests:** 22 tests in 6 classes
- `TestAnalyzerProtocol` - Protocol conformance
- `TestAnalysisPipelineInstantiation` - Instantiation and registration
- `TestAnalysisPipelineRun` - Run mechanics and manifest creation
- `TestAnalysisPipelineResumability` - Skip/force logic
- `TestAnalysisPipelineArtifactLoading` - Artifact loading
- `TestAnalysisPipelineMultipleAnalyzers` - Multiple analyzer support
