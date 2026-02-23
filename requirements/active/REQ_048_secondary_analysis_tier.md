# REQ_048: Secondary Analysis Tier

**Status:** Draft
**Priority:** High (foundation for REQ_049 and all derived analyses)
**Dependencies:** REQ_021f (Per-Epoch Artifacts), REQ_038 (Cross-Epoch Analyzers)
**Last Updated:** 2026-02-23

## Problem Statement

The current pipeline supports two types of analysis:

1. **Primary (Phase 1):** Runs against model + checkpoint. Receives model, probe, cache. Produces per-epoch artifacts.
2. **Cross-epoch (Phase 2):** Runs after all primary epochs complete. Receives artifacts_dir + epoch list. Produces a single `cross_epoch.npz`.

There is no supported path for an analysis that is:
- Per-epoch (like primary), but
- Derived from existing artifacts (like cross-epoch)

Several high-value analyses share this pattern: they operate on a single epoch's worth of previously-computed artifact data rather than on the model directly. The `parameter_snapshot` analyzer already captures raw weight matrices per epoch. Phase alignment, IPR, neuron specialization, and lottery ticket analyses are all natural derivations of those snapshots — they don't require loading the model or running a forward pass.

Without a dedicated tier, these derived analyses have no clean home:
- Adding them to primary analyzers re-loads models redundantly (defeats the purpose of `parameter_snapshot`)
- Grafting them onto cross-epoch analyzers is semantically wrong — they produce per-epoch outputs, not a single cross-epoch file
- Computing them at render time (in notebooks or dashboard) mixes analysis and presentation concerns, the same anti-pattern REQ_038 was created to fix

### Execution Order

The complete three-tier pipeline execution order is:

```
Phase 1: Primary     — model checkpoint → per-epoch artifact
Phase 1.5: Secondary — primary artifact → per-epoch artifact
Phase 2: Cross-epoch — all per-epoch artifacts → cross_epoch.npz
```

Cross-epoch analyzers remain separate from secondary. A secondary analyzer is still per-epoch. Cross-epoch analyzers need to see the full training run simultaneously. The distinction is meaningful: do not collapse them.

## Design

### New Protocol: SecondaryAnalyzer

```python
@runtime_checkable
class SecondaryAnalyzer(Protocol):
    @property
    def name(self) -> str:
        """Unique identifier (used in artifact naming)."""
        ...

    @property
    def depends_on(self) -> str:
        """Name of the primary analyzer whose per-epoch artifacts this consumes."""
        ...

    def analyze(
        self,
        artifact: dict[str, np.ndarray],
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Run analysis on a single epoch's artifact data.

        Args:
            artifact: Dict of arrays from the dependency analyzer for this epoch.
            context: Same family-provided analysis context as primary analyzers.

        Returns:
            Dict mapping artifact keys to numpy arrays.
        """
        ...
```

Key design choices:
- `depends_on` is a single string (not a list). A dependency DAG is not required now; a two-tier primary → secondary model is sufficient for the current use cases. If multi-dependency cases emerge, this is a known extension point.
- `analyze()` receives `artifact` (not `model`/`probe`/`cache`). The calling convention is explicitly different, making the tier boundary visible in the code.
- Context is the same as primary analyzers. Family-provided values (Fourier basis, prime, etc.) are still available.
- Optional summary statistics follow the same pattern as primary analyzers: `get_summary_keys()` and `compute_summary()` detected via `hasattr()`. The pipeline accumulates and saves `summary.npz` if these methods are present.

### Pipeline Changes

The pipeline gains Phase 1.5 between the existing per-epoch loop and cross-epoch analysis:

```
AnalysisPipeline.run()
├── Phase 1: Per-epoch primary analysis (existing, unchanged)
│   └── FOR EACH epoch: run Analyzers, save per-epoch artifacts
├── Save summary statistics (existing, unchanged)
├── Phase 1.5: Secondary analysis (NEW)
│   └── FOR EACH SecondaryAnalyzer:
│       ├── Determine target epochs from dependency's completed epochs
│       ├── Skip epochs already computed (unless force=True)
│       ├── FOR EACH missing epoch:
│       │   ├── Load artifact from dependency analyzer
│       │   └── result = analyzer.analyze(artifact, context)
│       │   └── Save result to artifacts/{secondary_name}/epoch_{NNNNN}.npz
│       └── Save summary.npz if analyzer supports it
├── Phase 2: Cross-epoch analysis (existing, unchanged)
└── Update manifest, save
```

**Target epoch determination for secondary analyzers:**

Secondary analyzers target epochs that have been computed for their dependency, not `config.checkpoints`. This decoupling is intentional: the checkpoint subset in `config.checkpoints` restricts which epochs incur the cost of model loading (Phase 1). Secondary analysis is cheap (no model loading) and should run on all available dependency epochs.

If a secondary analyzer's dependency has no completed epochs, the pipeline emits a warning and skips the analyzer (not an error — it's valid to add a secondary analyzer before re-running primary analysis on a variant).

**Registration:**

```python
pipeline.register(ParameterSnapshotAnalyzer())          # primary
pipeline.register_secondary(NeuronFourierAnalyzer())     # secondary
pipeline.register_cross_epoch(ParameterTrajectoryPCA())  # cross-epoch
```

`register_secondary()` is a separate method (same rationale as `register_cross_epoch()`: makes the phase boundary explicit and prevents silent miscategorization).

**Storage:**

Secondary analyzers use the same storage pattern as primary:

```
artifacts/
  parameter_snapshot/
    epoch_00000.npz     # primary artifact
    epoch_00100.npz
    ...
  neuron_fourier/
    epoch_00000.npz     # secondary artifact (same naming, different source)
    epoch_00100.npz
    ...
```

The `epoch_{NNNNN}.npz` pattern is reused because secondary results are semantically per-epoch. The artifact loader does not need to distinguish primary from secondary — callers access them by analyzer name.

### ArtifactLoader

No new methods required. `load_epoch()` and `load_epochs()` already work by analyzer name. Secondary artifacts are retrieved the same way as primary artifacts.

### AnalysisRunConfig

The `analyzers` filter in `AnalysisRunConfig` applies to secondary analyzers by name (same as primary). If the filter is non-empty and a secondary analyzer's name is not in it, that analyzer is skipped.

### Family Registration

`ModelFamily.get_analyzers()` returns primary analyzers. A new `get_secondary_analyzers()` method returns secondary analyzers. Families that have no secondary analyzers return an empty list (default). The pipeline calls this alongside `get_analyzers()` to auto-populate registration when no explicit pipeline is constructed.

## Scope

This requirement covers:
1. `SecondaryAnalyzer` protocol definition in `protocols.py`
2. `register_secondary()` method on `AnalysisPipeline`
3. Phase 1.5 execution with dependency epoch detection and skip-if-exists logic
4. Warning behavior when dependency has no completed epochs
5. Optional summary statistics support for secondary analyzers (via hasattr, same as primary)
6. `get_secondary_analyzers()` on `ModelFamily` base class (returns empty list)
7. Family registration of secondary analyzers alongside primary

This requirement does **not** cover:
- Multi-dependency secondary analyzers (DAG support — future)
- Any specific secondary analyzer implementation (covered by REQ_049+)
- Changes to cross-epoch behavior or primary behavior
- Re-running analysis on existing variants (manual re-analysis acceptable)

## Conditions of Satisfaction

### Protocol
- [ ] `SecondaryAnalyzer` protocol defined in `protocols.py` with `name`, `depends_on`, and `analyze()`
- [ ] Protocol is `@runtime_checkable`
- [ ] Distinct from `Analyzer` and `CrossEpochAnalyzer` — no conflation
- [ ] Docstring clearly describes the execution tier and calling convention

### Pipeline
- [ ] `register_secondary()` method accepts `SecondaryAnalyzer` instances
- [ ] Phase 1.5 runs after per-epoch primary phase and summary saving, before cross-epoch phase
- [ ] Secondary analyzers target epochs from their dependency's completed set, not `config.checkpoints`
- [ ] Skips epochs already computed for the secondary analyzer (unless `force=True`)
- [ ] Warns and skips (does not raise) when dependency has no completed epochs
- [ ] Secondary artifacts saved to `artifacts/{analyzer_name}/epoch_{NNNNN}.npz`
- [ ] Optional summary statistics accumulated and saved if `get_summary_keys()` / `compute_summary()` are present
- [ ] `AnalysisRunConfig.analyzers` filter applies to secondary analyzers by name
- [ ] Existing primary and cross-epoch behavior is completely unchanged

### Family
- [ ] `ModelFamily` base class has `get_secondary_analyzers()` returning `[]`
- [ ] Secondary analyzers registered from family alongside primary analyzers in pipeline construction

### Tests
- [ ] Secondary analyzer runs after primary and before cross-epoch
- [ ] Correct dependency epochs used as target (not config.checkpoints)
- [ ] Skip-if-exists logic works correctly
- [ ] Warning emitted when dependency missing; pipeline continues for remaining secondary analyzers
- [ ] Summary statistics collected and saved when secondary analyzer supports them

## Constraints

**Must have:**
- `SecondaryAnalyzer` is a separate protocol — not bolted onto `Analyzer`
- `depends_on` is a single string — no DAG complexity introduced now
- Storage format is identical to primary per-epoch artifacts — no new loading paths needed
- Existing primary and cross-epoch behavior is untouched

**Must avoid:**
- Loading models in the secondary phase (defeats the purpose)
- Conflating secondary with cross-epoch (they produce different artifact shapes)
- Mandatory secondary analyzers — families without them work unchanged

**Flexible:**
- Whether Phase 1.5 shares code paths with Phase 1 (e.g., `_build_work_queue`) or has its own implementation
- Whether the context dict passed to secondary analyzers is identical to primary (it should be, but preparation cost could be skipped if context is already cached)

## Decision Log

| Date | Question | Decision | Rationale |
|------|----------|----------|-----------|
| 2026-02-23 | Single dependency string vs list? | Single string for now | All current use cases have one dependency. List adds complexity (loading, validation) with no current benefit. Extension point is clear. |
| 2026-02-23 | Merge secondary into cross-epoch? | No — keep separate | Cross-epoch produces one result for the full run. Secondary produces per-epoch results. Different output shapes, different loading patterns, different use in downstream analysis. |
| 2026-02-23 | Separate register method? | Yes, `register_secondary()` | Makes the phase boundary visible at call sites. Same rationale as `register_cross_epoch()`. |
| 2026-02-23 | Target epochs from dependency or config? | From dependency's completed epochs | Secondary analysis is cheap (no model loading). Running it on all available dependency epochs is correct and low-cost. The checkpoint subset restriction exists to control expensive model loading. |
| 2026-02-23 | Error vs warning when dependency missing? | Warning + skip | Adding a secondary analyzer to a pipeline before the dependency has run is a valid workflow. Hard failure would break incremental development. |

## Notes

**2026-02-23:** This requirement was motivated by the observation that phase alignment, IPR, neuron specialization (all derived from `parameter_snapshot`) have no natural home in the current two-tier pipeline. The broader value is establishing a pattern for any future analysis where the cost of model loading should not be repeated.

The two-tier primary/secondary model is explicitly a pragmatic choice given current use cases. A DAG is the natural evolution if multi-dependency cases arise. The `depends_on: str` interface leaves room for a `depends_on: list[str]` upgrade without requiring protocol redesign.
