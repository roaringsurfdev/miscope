# REQ_029: Parameter Space Trajectory Projections

**Status:** Draft
**Priority:** High (foundation for geometric training analysis)
**Dependencies:** REQ_021a (Model Family), REQ_021f (Per-Epoch Artifacts)
**Last Updated:** 2026-02-09

## Problem Statement

The existing analysis pipeline captures *functional* signals — what Fourier features emerge, which neurons specialize, how attention patterns evolve. These show *what* the model learns and *when*, but not the geometric story of *how* it gets there.

Projecting the model's parameter vector across training checkpoints into a low-dimensional space (via PCA) reveals the optimization trajectory — the actual path the model takes through weight space. This can expose:

- Phase transitions (sharp turns in the trajectory during grokking)
- Whether different weight matrix groups (embedding, attention, MLP) move at different times
- Whether the trajectory structure generalizes across variants or is prime-specific

This requirement also introduces **parameter velocity** — the rate of parameter change between consecutive checkpoints — as a per-epoch scalar that can be plotted alongside functional metrics on a shared epoch axis. Velocity spikes mark phase transitions; velocity settling marks convergence.

### Infrastructure value

The per-epoch parameter snapshots stored by this analyzer are the raw data foundation for all geometric analysis. Effective dimensionality (REQ_030), loss landscape flatness (REQ_031), and future analyses (Riemannian metrics, mode connectivity, geodesic computation) all build from stored weight matrices. The "create-once, cacheable" design means the expensive part (loading checkpoints) happens once; derived analyses are cheap recomputation from artifacts.

## Design

### Analyzer: `ParameterSnapshotAnalyzer`

A new analyzer in `analysis/analyzers/parameter_snapshot.py`.

**Per-epoch artifact:**
```
parameter_snapshot/epoch_{NNNNN}.npz
  W_E:    shape (p+1, d_model)        — token embedding
  W_pos:  shape (n_ctx, d_model)       — positional embedding
  W_Q:    shape (n_heads, d_model, d_head) — query projection
  W_K:    shape (n_heads, d_model, d_head) — key projection
  W_V:    shape (n_heads, d_model, d_head) — value projection
  W_O:    shape (n_heads, d_head, d_model) — output projection
  W_in:   shape (d_model, d_mlp)       — MLP up projection
  W_out:  shape (d_mlp, d_model)       — MLP down projection
  W_U:    shape (d_model, d_vocab_out) — unembedding
```

Weight matrices are stored in their original shapes (not flattened) to support both trajectory projection (which flattens) and dimensionality analysis (which needs matrix structure). Each epoch's artifact is ~540KB for the Modulo Addition model (~135K parameters x 4 bytes).

**Key characteristic:** This analyzer does not use the forward pass, probe, or activation cache. It extracts weights directly from the model. It conforms to the Analyzer protocol by accepting and ignoring those arguments.

**No summary statistics** for this analyzer. All cross-epoch computations (PCA, velocity) are performed at render time from loaded per-epoch artifacts. This is tractable: loading 300 epochs x 540KB = ~162MB, PCA on a 300 x 135K matrix completes in seconds.

### Library Functions

New functions in `analysis/library/weights.py`:

```python
def extract_parameter_snapshot(
    model: HookedTransformer,
) -> dict[str, np.ndarray]:
    """Extract all trainable weight matrices from a model.

    Returns dict mapping weight matrix names to numpy arrays
    in their original shapes. Only includes parameters with
    requires_grad=True (excludes frozen biases).
    """
```

New functions in `analysis/library/trajectory.py`:

```python
def flatten_snapshot(
    snapshot: dict[str, np.ndarray],
    components: list[str] | None = None,
) -> np.ndarray:
    """Flatten selected weight matrices into a single parameter vector.

    Args:
        snapshot: Per-epoch artifact dict from ParameterSnapshotAnalyzer.
        components: Weight matrix names to include. None = all.

    Returns:
        1D array of concatenated, flattened parameters.
    """


def compute_pca_trajectory(
    snapshots: list[dict[str, np.ndarray]],
    components: list[str] | None = None,
    n_components: int = 3,
) -> dict[str, np.ndarray]:
    """Compute PCA projection of parameter trajectory.

    Args:
        snapshots: List of per-epoch snapshot dicts, ordered by epoch.
        components: Weight matrix names to include. None = all.
        n_components: Number of principal components.

    Returns:
        Dict with:
          "projections": (n_epochs, n_components) — coordinates in PC space
          "explained_variance_ratio": (n_components,) — fraction of variance per PC
          "explained_variance": (n_components,) — eigenvalues
    """


def compute_parameter_velocity(
    snapshots: list[dict[str, np.ndarray]],
    components: list[str] | None = None,
) -> np.ndarray:
    """Compute L2 norm of parameter change between consecutive epochs.

    Args:
        snapshots: List of per-epoch snapshot dicts, ordered by epoch.
        components: Weight matrix names to include. None = all.

    Returns:
        1D array of length (n_epochs - 1). velocity[i] = ||theta_{i+1} - theta_i||
    """
```

### Component Groups

For the dashboard, predefined component groups simplify selection:

| Group | Components | Rationale |
|-------|-----------|-----------|
| All | All 9 matrices | Global trajectory |
| Embedding | W_E, W_pos, W_U | Token representation space |
| Attention | W_Q, W_K, W_V, W_O | Attention mechanism |
| MLP | W_in, W_out | Feature computation |

Groups are a renderer/dashboard convenience, not part of the analyzer or library.

### Renderers

New file: `visualization/renderers/parameter_trajectory.py`

**Cross-epoch renderers** (require all epoch snapshots):

```python
def render_parameter_trajectory(
    snapshots: list[dict[str, np.ndarray]],
    epochs: list[int],
    current_epoch: int,
    components: list[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """2D PCA trajectory plot. Points colored by epoch (gradient),
    connected by path. Current epoch highlighted."""


def render_explained_variance(
    snapshots: list[dict[str, np.ndarray]],
    components: list[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Scree plot showing explained variance ratio per principal component.
    Helps the researcher assess whether the 2D projection is faithful."""


def render_parameter_velocity(
    snapshots: list[dict[str, np.ndarray]],
    epochs: list[int],
    current_epoch: int,
    components: list[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Parameter velocity (L2 norm of change) over epochs.
    Global velocity as primary line."""


def render_component_velocity(
    snapshots: list[dict[str, np.ndarray]],
    epochs: list[int],
    current_epoch: int,
    title: str | None = None,
) -> go.Figure:
    """Per-component velocity over epochs. One line per component group
    (embedding, attention, MLP). Reveals timing differences in
    when different parts of the model move."""
```

### Dashboard Integration

Two new panels in the Analysis visualization tab:

1. **Parameter Trajectory** — 2D PCA scatter with path, component group selector (radio or dropdown: All / Embedding / Attention / MLP), explained variance annotation
2. **Parameter Velocity** — velocity over epochs with component breakdown, epoch indicator line synced with slider

Both panels are conditional on `parameter_snapshot` appearing in the variant's available analyzers.

The trajectory panel requires loading all epoch snapshots, which is different from existing per-epoch panels. The dashboard should load snapshots once when the variant is selected (or on first panel interaction) and cache them for the session.

### Family Registration

Add `"parameter_snapshot"` to the Modulo Addition 1-Layer family's analyzers list in `family.json`.

## Scope

**This requirement covers:**
1. Library function: `extract_parameter_snapshot()` in `analysis/library/weights.py`
2. Library functions: `flatten_snapshot()`, `compute_pca_trajectory()`, `compute_parameter_velocity()` in `analysis/library/trajectory.py`
3. Analyzer: `ParameterSnapshotAnalyzer` with per-epoch artifacts
4. Renderers: trajectory, explained variance, velocity, component velocity
5. Dashboard panels: trajectory + velocity
6. Registration in Modulo Addition 1-Layer family
7. Tests

**This requirement does not cover:**
- Cross-variant trajectory comparison (projecting multiple variants onto the same PCA basis)
- UMAP or other nonlinear projections (future enhancement)
- Loss-colored trajectory (requires cross-analyzer data access)
- 3D trajectory rendering (can be added later if 2D is insufficient)

## Conditions of Satisfaction

### Library
- [ ] `extract_parameter_snapshot()` extracts all trainable weight matrices from a HookedTransformer
- [ ] Returns dict with keys matching TransformerLens naming: W_E, W_pos, W_Q, W_K, W_V, W_O, W_in, W_out, W_U
- [ ] Only includes parameters with `requires_grad=True`
- [ ] `flatten_snapshot()` concatenates selected components into a single 1D vector
- [ ] `compute_pca_trajectory()` returns projections and explained variance
- [ ] `compute_parameter_velocity()` returns L2 norms between consecutive snapshots
- [ ] All functions exported from their respective `__init__.py`

### Analyzer
- [ ] `ParameterSnapshotAnalyzer` in `analysis/analyzers/parameter_snapshot.py`
- [ ] Conforms to Analyzer protocol
- [ ] Per-epoch artifact contains all 9 weight matrices in original shapes
- [ ] Registered in `AnalyzerRegistry`

### Renderers
- [ ] `render_parameter_trajectory()` produces a 2D scatter plot with epoch-colored path
- [ ] Current epoch is visually highlighted on the trajectory
- [ ] `render_explained_variance()` produces a scree plot
- [ ] `render_parameter_velocity()` plots velocity over epochs with epoch indicator
- [ ] `render_component_velocity()` shows separate lines for embedding, attention, MLP groups
- [ ] All renderers return `go.Figure` objects

### Dashboard
- [ ] Parameter Trajectory panel with component group selector
- [ ] Parameter Velocity panel with component breakdown
- [ ] Both panels conditional on `parameter_snapshot` artifacts existing
- [ ] Epoch slider syncs with trajectory highlight and velocity indicator

### Family Integration
- [ ] `"parameter_snapshot"` added to Modulo Addition 1-Layer `family.json`

### Tests
- [ ] Library: snapshot extraction returns correct keys and shapes for known architecture
- [ ] Library: flatten produces correct total parameter count
- [ ] Library: PCA returns correct shapes and explained variance sums to <= 1.0
- [ ] Library: velocity has length n_epochs - 1
- [ ] Analyzer: conforms to Analyzer protocol
- [ ] Analyzer: produces correct artifact keys

## Constraints

**Must have:**
- Stores weight matrices in original shapes (not pre-flattened) to support multiple downstream uses
- PCA computed at render time, not stored in artifacts (depends on component selection)
- Only captures `requires_grad=True` parameters
- Per-epoch artifact pattern consistent with existing analyzers

**Must avoid:**
- Pre-computing PCA in the analyzer (component selection is a rendering concern)
- Storing flattened vectors (loses matrix structure needed by REQ_030)
- Hardcoding architecture dimensions (derive from model)

**Flexible:**
- Component group definitions (embedding/attention/MLP split is a starting point)
- Number of PCA components (3 stored, 2 rendered by default)
- Trajectory visualization style (scatter + path is baseline; can enhance)
- Whether to cache loaded snapshots in dashboard state or reload per interaction

## Decision Log

| Date | Question | Decision | Rationale |
|------|----------|----------|-----------|
| 2026-02-09 | Store flattened vectors or original shapes? | Original shapes | Supports both PCA (flattens) and SVD (needs matrix structure) without storing twice |
| 2026-02-09 | PCA in analyzer or renderer? | Renderer | PCA depends on component selection, which is a UI concern |
| 2026-02-09 | Separate velocity analyzer? | No, derive from snapshots | Velocity is a simple derived metric; separate analyzer would duplicate snapshot storage |
| 2026-02-09 | Shared artifacts with REQ_030? | No, independent analyzers | Each analyzer extracts weights independently; overhead is trivial, independence is valuable |

## Notes

**Relationship to REQ_030:** The effective dimensionality analyzer (REQ_030) also needs weight matrices. Rather than introducing a cross-analyzer dependency, REQ_030 extracts weights independently. The per-epoch extraction is trivially cheap (no forward pass), so the duplication is negligible. If storage becomes a concern with larger models, a shared "weight snapshot" artifact could be introduced as a future optimization.

**Analyzer protocol fit:** This analyzer ignores `probe` and `cache` arguments. This is the first analyzer that operates purely on weights rather than activations. If this pattern recurs, a `WeightAnalyzer` protocol variant could be introduced, but for now the existing protocol works fine.

**Loading pattern difference:** Existing dashboard panels load one epoch at a time per slider interaction. The trajectory and velocity panels need all epochs loaded at once. This is a new loading pattern for the dashboard — the implementation should handle this cleanly without breaking the existing on-demand pattern for other panels.
