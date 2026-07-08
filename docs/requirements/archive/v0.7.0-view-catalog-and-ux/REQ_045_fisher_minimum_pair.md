# REQ_045: Fisher Minimum Pair Analysis

**Status:** Active
**Priority:** High (extends REQ_044 representational geometry with targeted research question)
**Dependencies:** REQ_044 (repr_geometry analyzer, dashboard page, geometry library)
**Last Updated:** 2026-02-17

*Drafted by Research Claude, edited by Engineering Claude*

## Problem Statement

The representational geometry analyzer computes Fisher discriminant J(r,s) = ||mu_r - mu_s||^2 / (sigma_r^2 + sigma_s^2) across all class pairs but reduces it to two scalars (mean, min). These hide critical information: which specific residue pairs constitute the separation bottleneck, how those bottleneck pairs change over training, and whether the bottleneck pairs are predictable from the model's learned frequency spectrum.

This matters because the Fisher min tells us the model's worst-case vulnerability, but not *where* that vulnerability is. In a model that has learned specific Fourier frequencies, the hardest-to-separate class pairs should be those whose residue difference is poorly resolved by the learned frequencies. Confirming this connection would validate the Fourier interpretation of grokking at the representational level and provide a new diagnostic for understanding partial or failed grokking (e.g., p59/485's near-zero Fisher min despite Fisher mean ~20).

**Origin:** Emerged from analysis of representational geometry results across the 4-variant comparison (p113/999, p101/485, p101/999, p59/485), specifically from observing that p59/485 achieves Fisher mean ~20 but Fisher min ~0 while using only 2 of 3 expected frequency bands.

## Design

### Approach: Recompute at Render Time

The Fisher matrix J(r,s) can be recomputed at render time from per-epoch artifact data already stored: `{site}_centroids` (p, d) and `{site}_radii` (p,). This follows the precedent set by `render_centroid_distances`, which recomputes pairwise L2 distances from stored centroids. Formula: `J(r,s) = ||mu_r - mu_s||^2 / (radii_r^2 + radii_s^2)` where `radii^2 == within-class variance`.

This means zero changes to the per-epoch artifact format. All 6 existing variants with repr_geometry data work immediately.

### New: Argmin Summary Keys

The argmin pair identity over training requires new summary keys (computing argmin needs per-epoch data). Add 3 new scalar keys per site:
- `fisher_argmin_r` — residue class index of argmin pair (first element)
- `fisher_argmin_s` — residue class index of argmin pair (second element)
- `fisher_argmin_diff` — `min(|r-s| mod p, p - |r-s| mod p)` (circular distance in residue space — the number that connects to frequency blind spots)

### Geometry Library

Add `compute_fisher_matrix(centroids, radii) -> np.ndarray` to `geometry.py`. Takes stored data (no raw activations needed), returns symmetric `(p, p)` matrix with zero diagonal.

### Visualization

1. **Fisher heatmap** — `(p x p)` heatmap of J(r,s) at selected epoch, alongside existing centroid distance heatmap. Reversed colorscale so low values (cold spots = vulnerable pairs) stand out.
2. **Argmin time-series** — 6th panel in the existing time-series figure showing `fisher_argmin_diff` over epochs. Hover text includes the actual (r, s) pair.

## Conditions of Satisfaction

### Library
1. `compute_fisher_matrix(centroids, radii)` exists in geometry.py, returns symmetric `(p, p)` ndarray with zero diagonal
2. Function is pure numpy, takes stored data (centroids + radii), no raw activations needed

### Analyzer
3. `_SCALAR_KEYS` extended with `fisher_argmin_r`, `fisher_argmin_s`, `fisher_argmin_diff`
4. `_compute_site_measures` computes argmin pair from Fisher matrix
5. Summary key count increases from 32 to 44 (11 per site x 4 sites)

### Visualization
6. `render_fisher_heatmap` renders p x p Fisher discriminant heatmap from per-epoch data
7. Fisher heatmap hover shows class pair, J value, and `|r-s| mod p`
8. Time-series figure has 6th panel for argmin residue difference (guarded for backward compat)
9. Fisher heatmap appears on repr_geometry dashboard page alongside existing snapshot panels

### Tests
10. `compute_fisher_matrix` tested for symmetry, zero diagonal, and agreement with `compute_fisher_discriminant` mean/min
11. Analyzer produces new argmin keys with valid class indices
12. `render_fisher_heatmap` returns valid Plotly figure

### Batch Notebook
13. Notebook runs repr_geometry pipeline across all variants in a family

## Constraints

### Must Have
- Fisher heatmap recomputed at render time from stored centroids + radii (no artifact format change)
- Backward compatibility: dashboard handles missing argmin summary keys gracefully
- Follows existing renderer API pattern (`epoch_data, epoch, site, p, height`)

### Must Avoid
- Storing the full Fisher matrix in per-epoch artifacts (unnecessary given recompute approach)
- Breaking existing repr_geometry artifacts or summary files

### Flexible
- Fisher heatmap layout (alongside distance heatmap in same row vs new row)
- Colorscale choice for Fisher heatmap
- Whether argmin pair is highlighted on the heatmap with annotation/marker

## Decision Authority
- **Claude decides:** renderer details, colorscale, hover formatting, layout arrangement, test data construction
- **Discuss first:** any changes to per-epoch artifact format, any new dependencies

## Notes
- For p=113, the Fisher matrix is 113x113 = 12,769 entries — trivially small to compute at render time
- The `compute_fisher_discriminant` function already computes the full matrix internally and discards it; `compute_fisher_matrix` extracts this logic using stored data instead of raw activations
- Existing `radii` arrays store RMS distance (not variance). `radii^2 == variance` — confirmed in geometry.py implementation
- The batch notebook serves double duty: re-runs repr_geometry (now with argmin keys) AND provides coverage for the 10 variants that don't yet have repr_geometry data
