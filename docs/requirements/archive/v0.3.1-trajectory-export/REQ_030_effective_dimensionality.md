# REQ_030: Weight Matrix Effective Dimensionality

**Status:** Draft
**Priority:** High (manifold learning lens on training dynamics)
**Dependencies:** REQ_021a (Model Family), REQ_021f (Per-Epoch Artifacts), REQ_022 (Summary Statistics)
**Last Updated:** 2026-02-09

## Problem Statement

When a model "finds" a solution manifold, the solution occupies fewer dimensions than the full parameter space. This dimensional collapse is a signature of generalization — the model stops using all available capacity and concentrates on a lower-dimensional structure.

For grokking in modular arithmetic, this predicts:
- During memorization: weight matrices use most of their available rank (high effective dimensionality)
- During/after grokking: weight matrices collapse to lower effective rank as the model discovers the Fourier circuit structure

**Effective dimensionality via participation ratio** measures this directly: for each weight matrix at each checkpoint, how many singular values contribute meaningfully? A participation ratio of 32 on a matrix with 128 singular values means the matrix is effectively rank-32.

This complements the trajectory projection (REQ_029) — trajectory shows *where* the model goes; dimensionality shows *how constrained* the solution is at each point along the path.

## Design

### Metric: Participation Ratio

For a matrix with singular values {sigma_1, sigma_2, ..., sigma_n}, the participation ratio is:

```
PR = (sum(sigma_i))^2 / sum(sigma_i^2)
```

This equals:
- 1.0 when one singular value dominates (rank-1)
- n when all singular values are equal (full rank)

It can be normalized to [0, 1] by dividing by n, but the unnormalized form is more interpretable ("this matrix effectively uses 12 out of 128 dimensions").

### Analyzer: `EffectiveDimensionalityAnalyzer`

A new analyzer in `analysis/analyzers/effective_dimensionality.py`.

**Per-epoch artifact:**
```
effective_dimensionality/epoch_{NNNNN}.npz
  sv_W_E:    shape (min(p+1, d_model),)     — singular values of W_E
  sv_W_pos:  shape (min(n_ctx, d_model),)    — singular values of W_pos
  sv_W_Q:    shape (n_heads, d_head)         — per-head singular values of W_Q
  sv_W_K:    shape (n_heads, d_head)         — per-head singular values of W_K
  sv_W_V:    shape (n_heads, d_head)         — per-head singular values of W_V
  sv_W_O:    shape (n_heads, d_head)         — per-head singular values of W_O
  sv_W_in:   shape (min(d_model, d_mlp),)    — singular values of W_in
  sv_W_out:  shape (min(d_mlp, d_model),)    — singular values of W_out
  sv_W_U:    shape (min(d_model, d_vocab_out),) — singular values of W_U
```

Storing the full singular value spectrum (not just participation ratio) per epoch enables:
- Participation ratio computation at render time
- Singular value spectrum visualization (how dimensionality is distributed)
- Future: effective rank at different thresholds, stable rank, other spectral measures

For the Modulo Addition model, the total storage per epoch is small: ~800 singular values x 4 bytes = ~3.2KB per epoch.

**Attention matrices:** SVD is computed per-head. Each head's W_Q, W_K, W_V, W_O is an independent (d_model, d_head) or (d_head, d_model) matrix. Per-head dimensionality reveals whether different heads use different amounts of their capacity.

**Summary statistics** (via REQ_022 infrastructure):

| Key | Shape | Description |
|-----|-------|-------------|
| `pr_W_E` | scalar | Participation ratio of W_E |
| `pr_W_pos` | scalar | Participation ratio of W_pos |
| `pr_W_Q` | (n_heads,) | Per-head participation ratio of W_Q |
| `pr_W_K` | (n_heads,) | Per-head participation ratio of W_K |
| `pr_W_V` | (n_heads,) | Per-head participation ratio of W_V |
| `pr_W_O` | (n_heads,) | Per-head participation ratio of W_O |
| `pr_W_in` | scalar | Participation ratio of W_in |
| `pr_W_out` | scalar | Participation ratio of W_out |
| `pr_W_U` | scalar | Participation ratio of W_U |

Summary statistics enable the dimensionality trajectory renderer to operate from `load_summary()` without loading per-epoch artifacts.

### Library Functions

New functions in `analysis/library/weights.py` (extending from REQ_029):

```python
def compute_weight_singular_values(
    model: HookedTransformer,
) -> dict[str, np.ndarray]:
    """Compute singular values of all trainable weight matrices.

    For attention matrices (W_Q, W_K, W_V, W_O), computes SVD per head.

    Returns:
        Dict mapping "sv_{matrix_name}" to numpy arrays of singular values.
        Attention matrices have shape (n_heads, d_head).
        Other matrices have shape (min(rows, cols),).
    """


def compute_participation_ratio(
    singular_values: np.ndarray,
) -> float | np.ndarray:
    """Compute participation ratio from singular values.

    PR = (sum(s))^2 / sum(s^2)

    Args:
        singular_values: 1D array of singular values, or 2D array
            where each row is a set of singular values (e.g., per-head).

    Returns:
        Scalar for 1D input, 1D array for 2D input.
    """
```

### Renderers

New file: `visualization/renderers/effective_dimensionality.py`

```python
def render_dimensionality_trajectory(
    summary_data: dict[str, np.ndarray],
    current_epoch: int,
    matrices: list[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Participation ratio over epochs for selected weight matrices.

    One line per matrix. For attention matrices, can show per-head
    or average across heads. Epoch indicator at current_epoch.

    Uses summary data (pr_W_E, pr_W_in, etc.) loaded via
    ArtifactLoader.load_summary().
    """


def render_singular_value_spectrum(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    matrix_name: str = "W_in",
    head_idx: int | None = None,
    title: str | None = None,
) -> go.Figure:
    """Bar chart of singular values for a selected weight matrix at one epoch.

    Shows the distribution of singular values, annotated with
    participation ratio. For attention matrices, head_idx selects
    which head to display.
    """
```

### Dashboard Integration

Two panels:

1. **Dimensionality Trajectory** — participation ratio over epochs, line per weight matrix (or grouped: embedding/attention/MLP). Synced with epoch slider via indicator line. Matrix selector (checkboxes or dropdown).
2. **Singular Value Spectrum** — per-epoch bar chart for a selected weight matrix. Updates with epoch slider. Matrix selector dropdown, optional head selector for attention matrices.

Both panels conditional on `effective_dimensionality` artifacts existing.

### Family Registration

Add `"effective_dimensionality"` to the Modulo Addition 1-Layer family's analyzers list in `family.json`.

## Scope

**This requirement covers:**
1. Library functions: `compute_weight_singular_values()`, `compute_participation_ratio()` in `analysis/library/weights.py`
2. Analyzer: `EffectiveDimensionalityAnalyzer` with per-epoch artifacts and summary statistics
3. Renderers: dimensionality trajectory, singular value spectrum
4. Dashboard panels: trajectory + spectrum
5. Registration in Modulo Addition 1-Layer family
6. Tests

**This requirement does not cover:**
- Hessian-based dimensionality measures (future, more expensive)
- Gradient subspace dimensionality (requires training-time computation)
- Cross-variant dimensionality comparison
- Stable rank or other alternative spectral measures (can be added to renderers later since raw singular values are stored)

## Conditions of Satisfaction

### Library
- [ ] `compute_weight_singular_values()` computes SVD for all trainable weight matrices
- [ ] Attention matrices produce per-head singular values
- [ ] `compute_participation_ratio()` returns correct PR for known inputs
- [ ] PR = 1.0 for a rank-1 matrix, PR = n for n equal singular values
- [ ] Both functions exported from `analysis/library/__init__.py`

### Analyzer
- [ ] `EffectiveDimensionalityAnalyzer` in `analysis/analyzers/effective_dimensionality.py`
- [ ] Conforms to Analyzer protocol
- [ ] Per-epoch artifact contains singular values for all 9 weight matrices
- [ ] Implements `get_summary_keys()` returning participation ratio keys
- [ ] Implements `compute_summary()` returning participation ratios per matrix
- [ ] Registered in `AnalyzerRegistry`

### Renderers
- [ ] `render_dimensionality_trajectory()` plots PR over epochs with matrix selection
- [ ] Current epoch visually indicated
- [ ] `render_singular_value_spectrum()` shows singular values as bar chart with PR annotation
- [ ] Head selector works for attention matrices
- [ ] All renderers return `go.Figure` objects

### Dashboard
- [ ] Dimensionality Trajectory panel with matrix/group selector
- [ ] Singular Value Spectrum panel with matrix and head selectors
- [ ] Both conditional on `effective_dimensionality` artifacts existing
- [ ] Epoch slider syncs with both panels

### Family Integration
- [ ] `"effective_dimensionality"` added to Modulo Addition 1-Layer `family.json`

### Tests
- [ ] Library: SVD produces correct number of singular values per matrix shape
- [ ] Library: singular values are non-negative and sorted descending
- [ ] Library: participation ratio matches manual computation for known inputs
- [ ] Library: per-head SVD produces correct shapes
- [ ] Analyzer: conforms to Analyzer protocol
- [ ] Analyzer: produces correct artifact keys and shapes
- [ ] Analyzer: summary keys match compute_summary output

## Constraints

**Must have:**
- Stores full singular value spectrum (not just participation ratio) per epoch
- Attention matrices analyzed per-head, not as concatenated blocks
- Summary statistics include participation ratio for trajectory visualization
- Per-epoch computation only (no cross-epoch dependencies)

**Must avoid:**
- Computing on the combined (flattened) parameter vector (that's a different question than per-matrix dimensionality)
- Hardcoding matrix dimensions (derive from model architecture)

**Flexible:**
- Whether to normalize participation ratio to [0, 1] or leave as absolute count
- Grouping strategy for dashboard display (per-matrix vs per-group)
- Whether the spectrum renderer shows all matrices or one at a time
- Sort order of singular values in spectrum display

## Decision Log

| Date | Question | Decision | Rationale |
|------|----------|----------|-----------|
| 2026-02-09 | Participation ratio vs Hessian eigenspectrum? | Participation ratio via SVD | Cheap, interpretable, per-matrix; Hessian is future work |
| 2026-02-09 | Per-head or combined attention SVD? | Per-head | Each head is an independent subspace; combined SVD conflates them |
| 2026-02-09 | Store PR or singular values? | Singular values | Raw data enables multiple derived metrics; PR in summary stats for efficiency |
| 2026-02-09 | Share artifacts with REQ_029? | Independent | Trivial computational overhead; preserves analyzer independence |

## Notes

**Interpretation guide for researchers:** A participation ratio drop in W_in from 100 to 15 over training means the MLP input projection went from using most of its 128-dimensional capacity to concentrating on ~15 directions. Those 15 directions likely correspond to the Fourier features the model has learned to compute. Comparing this timing against neuron specialization onset (from existing functional analyzers) would reveal whether dimensional collapse precedes or follows functional specialization.

**Relationship to REQ_029:** Both analyzers extract weight matrices from the model. REQ_029 stores raw matrices for trajectory projection; REQ_030 stores their singular values for dimensionality analysis. The extraction is trivially cheap (no forward pass), so independent computation is preferred over cross-analyzer coupling.

**Future extensions:** The stored singular value spectra support additional metrics without re-running analysis: stable rank (Frobenius/spectral norm ratio), effective rank at various thresholds, nuclear norm tracking, and entropy-based dimensionality measures. These can be added as renderers without changing the analyzer.
