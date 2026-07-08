# REQ_022: Family-Specific Summary Statistics

**Status:** Active
**Priority:** High (enables cross-epoch visualization patterns)
**Dependencies:** REQ_021b (Analysis Library), REQ_021f (Per-Epoch Artifacts)
**Last Updated:** 2026-02-06

## Problem Statement

The current analysis pipeline produces per-epoch artifact files containing large arrays (e.g., neuron activations at `(512, 113, 113)` per epoch). These are well-suited for single-epoch visualizations driven by a slider. However, a growing class of analyses requires small values consumed *across all checkpoints at once*:

- Mean coarseness vs epoch (one scalar per checkpoint)
- Blob neuron count over time (one scalar per checkpoint)
- Coarseness distribution per epoch (a small histogram per checkpoint)

There is no infrastructure for this today. To plot "mean coarseness across training," a researcher would need to load hundreds of per-epoch `.npz` files, extract one value from each, and aggregate manually. This is the wrong access pattern — these are summary statistics, not large artifacts.

### Two Kinds of Analysis Data

| | Per-Epoch Artifacts (existing) | Summary Statistics (new) |
|---|---|---|
| **Size per epoch** | Large arrays (MBs) | Scalars or small arrays |
| **Access pattern** | One epoch at a time (slider) | All checkpoints at once (line plots) |
| **Storage** | One `.npz` per epoch | Single file across all checkpoints |
| **Examples** | Neuron activation heatmaps, frequency spectra | Mean coarseness, blob count, distribution histograms |

### Why Family-Specific?

Summary statistics are defined by the Model Family, not the platform. Different families study different phenomena — what constitutes a meaningful summary metric for Modulo Addition (coarseness, blob count) is meaningless for a different family. The family declares which summary statistics its analyzers produce, just as it declares which analyzers are valid.

## Design

### Analyzer Protocol Extension

Analyzers optionally produce summary statistics alongside their per-epoch artifacts. The existing `analyze()` return type (`dict[str, np.ndarray]`) is unchanged. A new method declares what summary statistics an analyzer computes, and a second new method extracts them from the analysis result:

```python
class Analyzer(Protocol):
    @property
    def name(self) -> str: ...

    def analyze(
        self, model, probe, cache, context
    ) -> dict[str, np.ndarray]:
        """Unchanged — returns per-epoch artifact data."""
        ...

    def get_summary_keys(self) -> list[str]:
        """Declare the summary statistic keys this analyzer produces.

        Returns empty list if this analyzer produces no summary statistics.
        Default implementation returns [].
        """
        ...

    def compute_summary(
        self, result: dict[str, np.ndarray], context: dict[str, Any]
    ) -> dict[str, float | np.ndarray]:
        """Compute summary statistics from this epoch's analysis result.

        Called by the pipeline after analyze(), using the same result dict.
        Returns a dict keyed by summary stat names (must match get_summary_keys()).
        Values are scalars (float) or small arrays (e.g., histogram bins).

        Default implementation returns {}.
        """
        ...
```

This keeps summary computation inline — the analyzer already has the data in memory from `analyze()`. No second pass required.

### Pipeline Changes

The pipeline collects summary statistics in memory as it processes epochs, then writes a single summary file after all epochs complete:

```
AnalysisPipeline.run()
├─ ... (existing setup) ...
├─ summary_accumulators = {analyzer_name: {key: [] for key in keys}}
└─ FOR EACH epoch:
    ├─ ... (existing: load model, forward pass) ...
    └─ FOR EACH analyzer:
        ├─ result = analyzer.analyze(model, probe, cache, context)
        ├─ _save_epoch_artifact(analyzer.name, epoch, result)  # existing
        └─ IF analyzer.get_summary_keys():
            ├─ summary = analyzer.compute_summary(result, context)
            └─ accumulate summary values with epoch number
├─ FOR EACH analyzer with summaries:
│   └─ _save_summary(analyzer.name, accumulated_summaries)
└─ ... (existing: update manifest) ...
```

Summary values are small (scalars or small arrays per epoch), so in-memory accumulation is safe even for many checkpoints. This is explicitly different from per-epoch artifacts, which caused memory exhaustion when buffered.

### Storage

Summary statistics are stored as a single `.npz` file per analyzer:

```
artifacts/
  neuron_activations/
    epoch_00000.npz          # existing per-epoch artifact
    epoch_00100.npz
    ...
    summary.npz              # NEW: all summary stats across checkpoints
  dominant_frequencies/
    epoch_00000.npz
    ...
    summary.npz
```

The `summary.npz` contains:
- `epochs`: 1D array of checkpoint epoch numbers
- One array per summary statistic, indexed along the first axis by checkpoint

Example for a coarseness-aware neuron activations analyzer:
```
summary.npz:
  epochs:             shape (N,)    — [0, 100, 500, 1000, ...]
  mean_coarseness:    shape (N,)    — [0.31, 0.33, 0.45, ...]
  blob_count_0_7:     shape (N,)    — [2, 5, 23, ...]
  coarseness_hist:    shape (N, 20) — histogram per epoch
```

### ArtifactLoader Extension

```python
class ArtifactLoader:
    # Existing methods unchanged

    def load_summary(self, analyzer_name: str) -> dict[str, np.ndarray]:
        """Load summary statistics for an analyzer.

        Returns dict with 'epochs' array and one array per summary stat.
        Raises FileNotFoundError if no summary exists for this analyzer.
        """
        ...

    def has_summary(self, analyzer_name: str) -> bool:
        """Check whether summary statistics exist for an analyzer."""
        ...
```

### Gap-Filling Consideration

When the pipeline runs incrementally (computing only missing epochs), it must handle summary statistics correctly:

- Load existing `summary.npz` if present
- Compute summaries for new epochs only
- Merge with existing data (append new epochs, re-sort by epoch)
- Rewrite the summary file

This is cheap — summary data is small — and maintains correctness without recomputing summaries for already-analyzed epochs.

### Checkpoint-Only Resolution

Summary statistics are computed at checkpoint epochs only, not at every training epoch. Training metrics (loss curves) are dense (every epoch). These have different temporal resolutions and must not be naively co-plotted on a shared axis. When visualized:

- Summary stats should be rendered as **discrete markers** (not interpolated lines) when displayed alongside dense training metrics
- Or displayed in **separate visualization panels** with clearly labeled x-axes

This is a scientific integrity constraint, not just a UI preference.

## Scope

This requirement covers:
1. Extending the Analyzer protocol with optional `get_summary_keys()` and `compute_summary()` methods
2. Pipeline changes to collect and persist summary statistics
3. ArtifactLoader support for loading summaries
4. Gap-filling support for incremental summary updates

This requirement does **not** cover:
- Implementing specific summary statistics (e.g., coarseness) — those belong in analyzer-specific requirements
- Dashboard visualization of summary statistics — separate requirement
- Cross-variant summary comparison — future work

## Conditions of Satisfaction

### Analyzer Protocol
- [ ] `get_summary_keys()` method added to Analyzer protocol with default returning `[]`
- [ ] `compute_summary()` method added to Analyzer protocol with default returning `{}`
- [ ] Existing analyzers continue to work unchanged (both methods are optional / have defaults)
- [ ] An analyzer can produce summary statistics without changing its `analyze()` return value

### Pipeline
- [ ] Pipeline detects analyzers with summary keys and collects summary data in memory
- [ ] Summary values accumulated per epoch as pipeline iterates
- [ ] Single `summary.npz` written per analyzer after all epochs complete
- [ ] `summary.npz` contains `epochs` array plus one array per summary key
- [ ] Summary collection does not affect per-epoch artifact saving (existing behavior preserved)
- [ ] Gap-filling: incremental runs merge new summaries with existing `summary.npz`

### ArtifactLoader
- [ ] `load_summary(analyzer_name)` loads and returns summary data
- [ ] `has_summary(analyzer_name)` checks for summary file existence
- [ ] Existing loader methods unchanged

### Integration
- [ ] At least one existing analyzer extended with summary statistics to validate the pattern
- [ ] Summary file is readable from notebook for exploratory analysis

## Constraints

**Must have:**
- Summary computation happens inline during analysis (not as a post-processing step)
- Analyzers without summary statistics are unaffected (backward compatible)
- Summary storage is a single file per analyzer (not per-epoch files)
- `epochs` array in summary file records which checkpoints were analyzed

**Must avoid:**
- Storing summary statistics in `metadata.json` or the manifest (these are family-specific, not generic metadata)
- Requiring all analyzers to produce summary statistics
- Interpolating or implying data between checkpoint epochs

**Flexible:**
- Whether `compute_summary()` receives the full `context` or a subset
- Compression strategy for summary files
- Whether summary keys are validated against `get_summary_keys()` at save time or left to convention

## Parallelization Notes

The current design accumulates summaries in memory within a sequential epoch loop. When epoch processing is parallelized (future requirement):

- Each parallel worker computes its epoch's summary values independently (map step)
- The orchestrator collects all per-epoch summaries after workers complete (reduce step)
- The single `summary.npz` write happens after all workers finish

This is a natural map-reduce pattern. The in-memory accumulation strategy works for both sequential and parallel execution — only the collection mechanism changes.

## Decision Log

| Date | Question | Decision | Rationale |
|------|----------|----------|-----------|
| 2026-02-06 | Inline vs post-processing? | Inline | Analyzers may need raw cache access for summary computation; avoids redundant data loading |
| 2026-02-06 | Storage: per-epoch files vs single file? | Single file per analyzer | Summary data is small; cross-epoch access pattern needs all values at once |
| 2026-02-06 | In-memory accumulation safe? | Yes | Summary values are scalars/small arrays, unlike per-epoch artifacts that caused memory exhaustion |
| 2026-02-06 | Co-plot with loss curves? | Discrete markers only | Summary stats are checkpoint-only (sparse); loss curves are every-epoch (dense). Different resolutions must be visually distinguished |

## Notes

**2026-02-06:** This requirement emerged from coarseness analysis research. The coarseness recommendations document identified five analyses, most of which require small per-epoch values consumed across all checkpoints. The existing per-epoch artifact storage is the wrong access pattern for this class of data. Summary statistics infrastructure enables these analyses and generalizes to any family-specific cross-epoch metric.
