# REQ_038: Cross-Epoch Analyzers

**Status:** Draft
**Priority:** Medium (removes runtime computation from dashboard; enables new analysis patterns)
**Dependencies:** REQ_021f (Per-Epoch Artifacts), REQ_022 (Summary Statistics)
**Last Updated:** 2026-02-13

## Problem Statement

The analysis pipeline processes checkpoints one at a time. Each analyzer receives a single model checkpoint and produces per-epoch artifacts. This is correct for analyses that are local to a training moment — neuron activations, frequency spectra, weight snapshots.

However, some analyses are inherently **cross-epoch**: they require data from all checkpoints simultaneously and cannot be decomposed into independent per-epoch computations. PCA of the parameter trajectory is the motivating example:

1. The `parameter_snapshot` analyzer saves raw weight matrices per epoch
2. The dashboard loads **all** snapshots into memory
3. `compute_pca_trajectory()` runs sklearn PCA across all epochs at render time
4. Every trajectory renderer repeats this computation

This works, but it's architecturally wrong: **analysis is happening in the rendering layer**. PCA projection is not visualization — it's a transformation of raw data into a derived representation. The dashboard should render precomputed results, not compute them.

### Why This Can't Use Existing Summary Statistics (REQ_022)

REQ_022's summary pattern computes one value per epoch inline during the checkpoint loop. PCA cannot work this way — it needs all epochs' parameter vectors to fit the projection basis before any single epoch's projection can be computed. It is a fundamentally different execution pattern: **post-pipeline aggregation** rather than per-epoch accumulation.

### Beyond PCA

Other cross-epoch analyses share this pattern:

- **Representational similarity** (CKA/CCA between early and late checkpoints)
- **Phase transition detection** (requires seeing the full trajectory to identify changepoints)
- **Clustering of training dynamics** (which epochs form natural groups?)

These all require access to the complete set of per-epoch artifacts and produce results that span the full training run.

## Design

### New Protocol: CrossEpochAnalyzer

A second analyzer protocol that runs **after** the per-epoch loop completes. It receives the artifacts directory and the list of completed epochs, rather than a model and cache.

```python
@runtime_checkable
class CrossEpochAnalyzer(Protocol):
    @property
    def name(self) -> str:
        """Unique identifier (used in artifact naming)."""
        ...

    @property
    def requires(self) -> list[str]:
        """Names of per-epoch analyzers whose artifacts this analyzer consumes."""
        ...

    def analyze_across_epochs(
        self,
        artifacts_dir: str,
        epochs: list[int],
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Run cross-epoch analysis.

        Args:
            artifacts_dir: Root artifacts directory for the variant.
            epochs: Sorted list of available epoch numbers.
            context: Family-provided analysis context (same as per-epoch analyzers).

        Returns:
            Dict mapping artifact keys to numpy arrays.
        """
        ...
```

Key differences from `Analyzer`:
- No `model`, `probe`, or `cache` — these are checkpoint-specific
- Receives `artifacts_dir` + `epochs` to load whatever per-epoch data it needs
- `requires` declares dependencies on per-epoch analyzers (pipeline validates these ran first)

### Pipeline Changes

The pipeline gains a second phase after the existing per-epoch loop:

```
AnalysisPipeline.run()
├── Phase 1: Per-epoch analysis (existing, unchanged)
│   └── FOR EACH epoch: run Analyzers, save per-epoch artifacts
├── Save summary statistics (existing, unchanged)
├── Phase 2: Cross-epoch analysis (NEW)
│   └── FOR EACH CrossEpochAnalyzer:
│       ├── Verify required per-epoch analyzers have completed
│       └── result = analyzer.analyze_across_epochs(artifacts_dir, epochs, context)
│       └── Save result to artifacts/{cross_epoch_analyzer_name}/cross_epoch.npz
└── Update manifest, save
```

### Registration

Cross-epoch analyzers are registered separately:

```python
pipeline.register(ParameterSnapshotAnalyzer())          # per-epoch
pipeline.register_cross_epoch(ParameterTrajectoryPCA())  # cross-epoch
```

The family's `get_analyzers()` returns per-epoch analyzers (unchanged). A new `get_cross_epoch_analyzers()` method returns cross-epoch analyzers. This keeps the two concepts separate and avoids muddying the existing Analyzer protocol.

### Storage

Cross-epoch results are stored as a single file per analyzer (not per-epoch):

```
artifacts/
  parameter_snapshot/
    epoch_00000.npz       # per-epoch (existing)
    epoch_00100.npz
    ...
  parameter_trajectory/
    cross_epoch.npz       # NEW: cross-epoch result
```

The `cross_epoch.npz` file contains whatever arrays the analyzer produces. For PCA trajectory:
```
cross_epoch.npz:
  epochs:                    shape (N,)     — epoch numbers
  projections:               shape (N, 3)   — PC1, PC2, PC3 per epoch
  explained_variance_ratio:  shape (3,)     — fraction explained per PC
  explained_variance:        shape (3,)     — eigenvalues
  velocity:                  shape (N-1,)   — parameter velocity between checkpoints
```

### ArtifactLoader Extension

```python
class ArtifactLoader:
    # Existing methods unchanged

    def load_cross_epoch(self, analyzer_name: str) -> dict[str, np.ndarray]:
        """Load cross-epoch analysis results.

        Returns dict of numpy arrays from cross_epoch.npz.
        Raises FileNotFoundError if no cross-epoch results exist.
        """
        ...

    def has_cross_epoch(self, analyzer_name: str) -> bool:
        """Check whether cross-epoch results exist for an analyzer."""
        ...
```

### First Implementation: ParameterTrajectoryPCA

The motivating cross-epoch analyzer. Consumes `parameter_snapshot` artifacts, produces PCA projections and velocity:

```python
class ParameterTrajectoryPCA:
    name = "parameter_trajectory"
    requires = ["parameter_snapshot"]

    def analyze_across_epochs(self, artifacts_dir, epochs, context):
        loader = ArtifactLoader(artifacts_dir)
        snapshots = [loader.load_epoch("parameter_snapshot", e) for e in epochs]

        pca_result = compute_pca_trajectory(snapshots)
        velocity = compute_parameter_velocity(snapshots, epochs=epochs)

        return {
            "epochs": np.array(epochs),
            "projections": pca_result["projections"],
            "explained_variance_ratio": pca_result["explained_variance_ratio"],
            "explained_variance": pca_result["explained_variance"],
            "velocity": velocity,
        }
```

The existing `compute_pca_trajectory()` and `compute_parameter_velocity()` in `analysis/library/trajectory.py` are reused — the library functions don't change, they just move from being called at render time to being called at analysis time.

### Dashboard Impact

Trajectory renderers switch from computing PCA to loading precomputed results. The `get_trajectory_data()` method in `DashboardState` (which currently loads all snapshots and caches them) is replaced by a simple `load_cross_epoch("parameter_trajectory")` call. The renderers receive the precomputed projections directly instead of raw snapshots.

## Scope

This requirement covers:
1. `CrossEpochAnalyzer` protocol definition
2. Pipeline Phase 2 execution with dependency validation
3. `register_cross_epoch()` registration method
4. `cross_epoch.npz` storage pattern
5. ArtifactLoader support (`load_cross_epoch`, `has_cross_epoch`)
6. `ParameterTrajectoryPCA` as the first implementation
7. Dashboard trajectory renderers updated to consume precomputed results

This requirement does **not** cover:
- Changes to per-epoch analyzers or the existing summary statistics pattern
- Additional cross-epoch analyzers beyond PCA trajectory (those are separate requirements)
- Re-running analysis on existing variants (manual re-analysis is acceptable)
- Component-group-specific PCA (e.g., PCA of only attention weights) — future enhancement

## Conditions of Satisfaction

### CrossEpochAnalyzer Protocol
- [ ] Protocol defined with `name`, `requires`, and `analyze_across_epochs()`
- [ ] Protocol is `@runtime_checkable`
- [ ] Distinct from existing `Analyzer` protocol (no conflation)

### Pipeline
- [ ] `register_cross_epoch()` method accepts `CrossEpochAnalyzer` instances
- [ ] Cross-epoch phase runs after all per-epoch analysis and summary saving completes
- [ ] Pipeline validates that required per-epoch analyzers have completed epochs before running cross-epoch analyzers
- [ ] Cross-epoch results saved to `artifacts/{analyzer_name}/cross_epoch.npz`
- [ ] Pipeline skips cross-epoch analysis if results already exist (unless `force=True`)
- [ ] Progress callback reports cross-epoch phase status
- [ ] Existing per-epoch behavior is completely unchanged

### ArtifactLoader
- [ ] `load_cross_epoch(analyzer_name)` loads and returns cross-epoch data
- [ ] `has_cross_epoch(analyzer_name)` checks for `cross_epoch.npz` existence
- [ ] Existing loader methods unchanged

### ParameterTrajectoryPCA
- [ ] Consumes `parameter_snapshot` per-epoch artifacts
- [ ] Produces projections, explained variance ratio, explained variance, velocity, and epochs
- [ ] Results match existing `compute_pca_trajectory()` output (numerical equivalence)
- [ ] Reuses existing library functions (no reimplementation of PCA logic)

### Dashboard Integration
- [ ] Trajectory renderers consume precomputed cross-epoch results
- [ ] `DashboardState` no longer loads all parameter snapshots for trajectory visualization
- [ ] Trajectory visualizations produce the same plots as before (visual equivalence)

## Constraints

**Must have:**
- Cross-epoch analyzers are a separate protocol, not bolted onto `Analyzer`
- Dependency declaration (`requires`) ensures correct execution order
- Existing per-epoch pipeline is untouched — cross-epoch is purely additive
- `cross_epoch.npz` naming convention (not `summary.npz`, which is taken by REQ_022)

**Must avoid:**
- Loading all per-epoch artifacts into memory simultaneously if it can be streamed (analyzer's choice — the protocol doesn't mandate either)
- Breaking existing summary statistics infrastructure
- Making cross-epoch analysis mandatory — per-epoch-only families should work unchanged

**Flexible:**
- Whether `context` for cross-epoch analyzers is identical to per-epoch context or a subset
- Whether cross-epoch results are recorded in the manifest
- Whether `force=True` re-runs only cross-epoch phase or both phases

## Decision Log

| Date | Question | Decision | Rationale |
|------|----------|----------|-----------|
| 2026-02-13 | Extend Analyzer protocol or new protocol? | New protocol | Different execution model (all epochs vs single epoch), different inputs (artifacts_dir vs model/cache). Conflating them would complicate both. |
| 2026-02-13 | Storage: reuse summary.npz or separate file? | Separate `cross_epoch.npz` | Summary statistics (REQ_022) are per-epoch scalars accumulated inline. Cross-epoch results are structurally different — they can be large arrays computed post-pipeline. Different semantics deserve different naming. |
| 2026-02-13 | Should velocity be in the same analyzer as PCA? | Yes | Both consume the same snapshots. Separate analyzers would double the loading cost for no architectural benefit. |
| 2026-02-13 | Registration: same method or separate? | Separate `register_cross_epoch()` | Makes the two-phase nature explicit. Pipeline can validate and execute phases independently. |

## Notes

**2026-02-13:** This requirement was motivated by PCA trajectory computation happening at render time in the dashboard. The broader value is establishing a pattern for any analysis that needs to see the full training run — phase detection, representational similarity, etc. The cross-epoch analyzer pattern is the natural complement to the per-epoch analyzer + summary statistics patterns already in the pipeline.
