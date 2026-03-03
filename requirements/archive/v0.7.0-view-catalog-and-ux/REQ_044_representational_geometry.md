# REQ_044: Representational Geometry Tracker

**Status:** Draft
**Priority:** High (first activation-space analysis — complements all parameter-space analyses)
**Dependencies:** Analyzer protocol, ArtifactLoader, summary stats (REQ_022), dashboard page pattern (REQ_041)
**Last Updated:** 2026-02-16

## Problem Statement

All existing analyses operate in **parameter space** — tracking weight trajectories, Fourier coefficients, neuron frequency assignments, singular values. These describe what the model *is* but not what it *does* to its inputs. The missing piece is **activation space**: how the model's internal representations of inputs evolve during training.

For modular addition, each output class (residue 0 through p-1) defines a set of inputs that share a label. The activations for those inputs form a cloud (manifold) in activation space. Before grokking, these manifolds are tangled. After grokking, they should be separable with geometric structure reflecting the learned Fourier/circular representation.

### Questions This Analysis Answers

1. **When do class representations become separable?** Track signal-to-noise ratio (between-class vs within-class spread) across training.
2. **What geometric structure emerges?** Do class centroids arrange into a circle in activation space, and when?
3. **Does the circular ordering match the algebraic structure?** Are residue classes ordered by value on the circle (Fourier alignment)?
4. **Where in the network does separability emerge?** Compare geometry at different activation sites (embedding, post-attention, MLP output, final residual stream).
5. **How does representational geometry relate to parameter dynamics?** Enable overlay with existing analyses (neuron commitment, weight PCA, embedding Fourier coefficients).

## Design

### 1. New Analyzer: `RepresentationalGeometryAnalyzer`

Follows the standard Analyzer protocol. Receives `model`, `probe`, `cache`, and `context` from the pipeline. Does not run forward passes — reads activations from the pre-computed cache.

**Analyzer name:** `repr_geometry`

#### Activation Sites

Extract activations at position -1 (last token / prediction position) from 4 sites:

| Site Key | Cache Access | Dimension |
|----------|-------------|-----------|
| `resid_pre` | `cache["resid_pre", 0][:, -1, :]` | `d_model` |
| `attn_out` | `cache["attn_out", 0][:, -1, :]` | `d_model` |
| `mlp_out` | `cache["post", 0, "mlp"][:, -1, :]` | `d_mlp` |
| `resid_post` | `cache["resid_post", 0][:, -1, :]` | `d_model` |

Note: `resid_pre` (residual stream before layer 0) captures post-embedding state. This is more precise than hooking the embed layer directly, since the residual stream at this point includes both token embeddings and positional embeddings.

Exact cache key syntax should be verified against the model's `hook_dict()` during implementation. The keys above follow TransformerLens conventions observed in the existing codebase.

#### Activation Extraction Helper

Add `extract_residual_stream()` to `src/miscope/analysis/library/activations.py`, following the pattern of `extract_mlp_activations()`:

```python
def extract_residual_stream(
    cache: ActivationCache,
    layer: int = 0,
    position: int = -1,
    location: str = "resid_post",
) -> torch.Tensor:
    """Extract residual stream activations from cache.

    Args:
        cache: Activation cache from model.run_with_cache()
        layer: Which transformer layer (default: 0)
        position: Token position to extract (default: -1)
        location: One of "resid_pre", "resid_post", "attn_out"

    Returns:
        Tensor of shape (batch, d_model)
    """
```

#### Geometric Measures

**Per-class measures** (computed for each of the p output classes, at each activation site):

1. **Centroid** — mean activation vector. Shape per site: `(p, d)`.
2. **Effective radius** — RMS distance from centroid: `sqrt(mean(||x - μ_r||²))`. Shape: `(p,)`.
3. **Effective dimensionality** — participation ratio of PCA eigenvalues on centered class activations: `(Σλ_i)² / Σ(λ_i²)`. Shape: `(p,)`.

**Global measures** (scalar per activation site per epoch):

4. **Mean radius** — average of per-class radii.
5. **Mean dimensionality** — average of per-class effective dimensionalities.
6. **Center spread** — RMS distance of centroids from global centroid.
7. **SNR** — `center_spread² / mean_radius²`. The single best summary of representational separability.
8. **Circularity** — how well centroids lie on a circle in their top-2 PCA subspace. Approach: PCA on `(p, d)` centroid matrix → project to top-2 PCs → fit circle (algebraic least-squares / Kåsa method) → score = `1 - (mean_squared_residual / variance_in_2d_plane)`. Range: 0 (no circle) to 1 (perfect circle).
9. **Fourier alignment** — whether the angular ordering of centroids on the fitted circle matches residue class ordering. Compute angles of projected centroids on fitted circle, find the frequency `k` that best explains the ordering (minimize angular residuals for `θ_r ≈ 2πkr/p`), report the R² of that fit. This is a separate measure from circularity — circularity can be high with scrambled ordering.
10. **Fisher discriminant (mean and min)** — pairwise `||μ_r - μ_s||² / (σ_r² + σ_s²)` averaged over all class pairs, and the minimum (bottleneck pair).

#### Per-Epoch Artifact Format

The `analyze()` method returns a flat dict. Keys are prefixed by site name. Example keys for the `resid_post` site:

```python
{
    # Per-class arrays
    "resid_post_centroids": np.ndarray,      # (p, d_model)
    "resid_post_radii": np.ndarray,          # (p,)
    "resid_post_dimensionality": np.ndarray, # (p,)
    # Global scalars
    "resid_post_mean_radius": np.ndarray,    # scalar
    "resid_post_mean_dim": np.ndarray,       # scalar
    "resid_post_center_spread": np.ndarray,  # scalar
    "resid_post_snr": np.ndarray,            # scalar
    "resid_post_circularity": np.ndarray,    # scalar
    "resid_post_fourier_alignment": np.ndarray, # scalar
    "resid_post_fisher_mean": np.ndarray,    # scalar
    "resid_post_fisher_min": np.ndarray,     # scalar
    # ... same pattern for resid_pre, attn_out, mlp_out
}
```

#### Summary Stats

Declare all global scalar keys (8 scalars × 4 sites = 32 keys) via `get_summary_keys()`. This produces a `summary.npz` that the time-series dashboard can load without touching per-epoch centroid data.

Centroids, radii, and dimensionality arrays are per-epoch only — loaded on demand for snapshot visualizations.

### 2. Geometric Computation Library

Add `src/miscope/analysis/library/geometry.py` with pure-function implementations:

- `compute_class_centroids(activations, labels, n_classes)` → `(n_classes, d)`
- `compute_class_radii(activations, labels, centroids)` → `(n_classes,)`
- `compute_class_dimensionality(activations, labels, centroids)` → `(n_classes,)`
- `compute_center_spread(centroids)` → scalar
- `compute_circularity(centroids)` → scalar
- `compute_fourier_alignment(centroids, p)` → scalar
- `compute_fisher_discriminant(activations, labels, centroids)` → `(mean, min)`

All functions take numpy arrays, return numpy arrays/scalars. No torch dependency. No side effects. Independently testable.

### 3. Visualization: Time-Series Dashboard (MVP)

New dashboard page at `/repr-geometry`.

**Layout:** Vertically stacked time-series panels sharing an epoch x-axis, with a site selector dropdown (default: `resid_post`). One panel per measure:

1. **SNR** — the headline number. Log scale y-axis (SNR spans orders of magnitude).
2. **Center spread and mean radius** — dual lines showing the two components of SNR.
3. **Circularity and Fourier alignment** — two lines, both 0-1 range.
4. **Mean dimensionality** — how many effective dimensions the representations use.
5. **Fisher discriminant (mean and min)** — separability measure.

Optionally show all 4 sites simultaneously (one line per site) instead of the dropdown — implementation decides based on visual clarity.

**Data source:** `summary.npz` via `loader.load_summary("repr_geometry")`.

### 4. Visualization: Centroid Geometry Snapshot (MVP)

At a selected epoch (slider), show:

**Panel A — Centroid PCA scatter:** Project `(p, d)` centroids into top-2 PCs. Points colored by residue class using a cyclic colormap. If circular structure has emerged, this will show a polygon/circle with residues in algebraic order.

**Panel B — Pairwise centroid distance heatmap:** `(p, p)` matrix of `||μ_r - μ_s||`. For learned modular structure, this should show circulant pattern (distance depends on `|r - s| mod p`).

**Data source:** Per-epoch artifact via `loader.load_epoch("repr_geometry", epoch)`.

**Site selector:** Same dropdown as time-series, controlling which site's centroids are displayed.

## Conditions of Satisfaction

### Analyzer
1. `RepresentationalGeometryAnalyzer` implements the `Analyzer` protocol with name `repr_geometry`
2. Extracts activations at position -1 from 4 sites: `resid_pre`, `attn_out`, `mlp_out`, `resid_post`
3. Computes per-class centroids, radii, and effective dimensionality for each site
4. Computes global scalars (SNR, circularity, Fourier alignment, Fisher mean/min, center spread, mean radius, mean dimensionality) for each site
5. Per-epoch artifact contains all arrays and scalars with site-prefixed keys
6. Summary stats declared for all 32 global scalar keys; `summary.npz` contains only scalars (no centroid arrays)

### Library
7. `geometry.py` functions are pure numpy, independently testable, no torch dependency
8. `compute_circularity` uses algebraic circle fit (Kåsa method) on top-2 PCA projection
9. `compute_fourier_alignment` finds optimal frequency `k` and returns angular fit R²
10. `extract_residual_stream` added to `activations.py` following existing helper pattern

### Dashboard
11. Time-series page loads from `summary.npz` and renders 5 panels with shared epoch x-axis
12. Centroid snapshot page loads single-epoch artifact and renders PCA scatter + distance heatmap
13. Site selector dropdown controls which activation site is displayed
14. Page follows existing navigation pattern (sidebar link, shared family/variant selector)

### Tests
15. Geometry library functions tested with synthetic data (known circle, known clusters)
16. Analyzer tested with mock cache returning known activations
17. Circularity score ≈ 1.0 for points on a circle, ≈ 0.0 for random point cloud
18. Fourier alignment ≈ 1.0 for correctly ordered circular arrangement

## Constraints

### Must Have
- Standard Analyzer protocol — no pipeline changes
- Per-epoch artifact storage — no stacked format
- Summary stats for dashboard time-series — no loading all epochs for time-series view
- Position -1 extraction — consistent with all existing analyzers
- Pure numpy geometry library — testable without model/torch

### Must Avoid
- Running forward passes in the analyzer (pipeline handles this)
- Storing centroids in summary.npz (too large when stacked across epochs)
- scipy dependency for circle fitting (Kåsa method is ~10 lines of numpy)
- Animated visualizations in MVP

### Flexible
- Exact panel arrangement in dashboard (vertical stack vs grid)
- Whether site selector is dropdown or all-sites-simultaneously
- Colormap choice for centroid PCA scatter (cyclic preferred but implementor decides)
- Whether Fisher discriminant uses pairwise or one-vs-rest formulation

## Decision Authority

- **Claude decides:** renderer details, layout specifics, colormap, helper function signatures, test data construction
- **Discuss first:** adding activation sites beyond the 4 specified, changes to the Analyzer protocol, any new dependencies

## Notes

- The draft spec (`requirements/drafts/representational_geometry_spec.md`) contains extended scientific context, hypotheses about integration with existing analyses, and future extensions. Consult for background but do not treat as binding — this requirement supersedes it.
- The Kåsa circle fit: given 2D points `(x_i, y_i)`, solve the least-squares system for `a, b, c` in `x² + y² + ax + by + c = 0`. Center = `(-a/2, -b/2)`, radius = `sqrt(a²/4 + b²/4 - c)`. Biased for small arcs but fine for our use case (p points expected to span the full circle).
- Future extensions (not in scope): frequency-decomposed manifolds, full manifold capacity (Chung & Abbott 2021), cross-variant comparison, animated centroid evolution. These are documented in the draft spec for future requirements.
