# REQ_090: Frequency Group Weight Geometry

**Status:** Complete
**Priority:** Medium
**Branch:** feature/req-090-freq-group-weight-geometry
**Attribution:** Engineering Claude

---

## Problem Statement

We can already track how output class representations evolve geometrically in activation space
(repr_geometry). We can group MLP neurons by their dominant frequency and track their weight-space
coordination (neuron_group_pca). What we cannot yet see is whether the frequency groups are
geometrically well-separated in weight space, and how that separation evolves over training.

The open question: does grokking correspond to the moment when frequency groups become distinct,
well-organized clusters in weight space? Do the group centroids form a ring (analogous to the class
centroid ring in activation space)? Does each group's internal geometry become more elongated
(lower effective dimensionality) as neurons specialize?

This applies the GLUE geometric framework — centroids, radii, dimensionality, Fisher discriminant,
center spread, circularity — to frequency groups in weight space rather than output classes in
activation space. The math is identical; only the grouping criterion and the space change.

---

## Conditions of Satisfaction

### Analyzer

- [ ] New cross-epoch analyzer `FreqGroupWeightGeometryAnalyzer` (`name = "freq_group_weight_geometry"`)
- [ ] Group assignment fixed at the reference epoch (final checkpoint), same pattern as `neuron_group_pca`
- [ ] Analyzes both W_in and W_out weight matrices per checkpoint
- [ ] For each weight matrix, at each epoch, computes per-group:
  - Centroids (group mean weight vector)
  - Radii (RMS distance of neuron weight vectors from group centroid)
  - Effective dimensionality (participation ratio from group covariance eigenvalues)
- [ ] And global (across all groups) per epoch:
  - Center spread (RMS distance of group centroids from global centroid)
  - SNR (center_spread^2 / mean_radius^2)
  - Mean and min Fisher discriminant ratio between group pairs
  - Circularity of group centroids in their top-2 PCA subspace
- [ ] Reuses geometry library functions from `miscope/analysis/library/geometry.py` without modification
- [ ] `requires = ["neuron_freq_norm", "parameter_snapshot"]`
- [ ] Artifact keys (cross-epoch):
  - `group_freqs` int32 (n_groups,) — frequency index per group
  - `group_sizes` int32 (n_groups,) — neuron count per group
  - `Win_centroids` float32 (n_epochs, n_groups, d_model) — per-epoch group centroids in W_in space
  - `Win_radii` float32 (n_epochs, n_groups) — per-epoch group radii in W_in space
  - `Win_dimensionality` float32 (n_epochs, n_groups) — effective dimensionality per group in W_in
  - `Win_center_spread` float32 (n_epochs,)
  - `Win_snr` float32 (n_epochs,)
  - `Win_fisher_mean` float32 (n_epochs,)
  - `Win_fisher_min` float32 (n_epochs,)
  - `Win_circularity` float32 (n_epochs,)
  - Same keys for `Wout_*` (W_out space)
  - `epochs` int32 (n_epochs,)

### Views

- [ ] `weight_geometry.timeseries` — multi-panel time-series for a selected weight matrix (W_in or W_out):
  - Panel 1: SNR over epochs
  - Panel 2: Center spread and mean radius over epochs
  - Panel 3: Circularity over epochs
  - Panel 4: Fisher mean and Fisher min over epochs
  - Epoch cursor support
  - `matrix` kwarg selects `"Win"` or `"Wout"` (default: `"Win"`)
- [ ] `weight_geometry.group_snapshot` — per-group bar chart at a selected epoch:
  - Bars: mean radius and effective dimensionality per group (dual y-axis or normalized)
  - Groups labeled by frequency index
  - `matrix` kwarg selects weight matrix

### Registration

- [ ] Analyzer registered in `modulo_addition_1layer` family config under `analyzers`
- [ ] Views registered in the view catalog (`views/universal.py`)
- [ ] Views accessible from the existing analysis pages or a new lightweight dashboard section

---

## Constraints

**Must:**
- Reuse `geometry.py` functions without modification — this is the validation that those functions
  are truly label-agnostic
- Use the same group assignment logic as `neuron_group_pca` (argmax of norm_matrix at final epoch,
  groups with >= 2 neurons only)
- Store results as a cross-epoch artifact (single file), not per-epoch files

**Must not:**
- Include `fourier_alignment` metric — frequency group indices don't have sequential Fourier
  ordering semantics; omit for this first pass
- Require re-running existing analyzers

**Flexible:**
- Whether W_in and W_out are stored in the same artifact file or separate files (same file is
  simpler; use `Win_` and `Wout_` key prefixes)
- Dashboard page placement — can be a new tab on an existing page

---

## Architecture Notes

**W_in orientation:** `parameter_snapshot` stores W_in as `(d_model, d_mlp)`. Each column is
one neuron's input weight vector in d_model-dimensional space. Transpose to `(d_mlp, d_model)`
before passing to geometry functions as `activations`.

**W_out orientation:** `parameter_snapshot` stores W_out as `(d_mlp, d_vocab)`. Each row is
one neuron's output weight vector. Pass as-is (shape is already `(d_mlp, d_vocab)`).

**Group labels:** `neuron_group_idx` array from `_assign_groups` logic maps each neuron to a
contiguous group index 0..n_groups-1. Pass directly as `labels` argument to geometry functions.
Ungrouped neurons (< 2 members) are excluded; all others get exactly one group.

**Geometry function call pattern:**
```python
# W_in example
W_in = snap["W_in"].T           # (d_mlp, d_model)
centroids = compute_class_centroids(W_in, group_labels, n_groups)
radii     = compute_class_radii(W_in, group_labels, centroids)
dims      = compute_class_dimensionality(W_in, group_labels, centroids)
fisher_mean, fisher_min = compute_fisher_discriminant(W_in, group_labels, centroids)
spread    = compute_center_spread(centroids)
circ      = compute_circularity(centroids)
```

---

## Notes

- This is a detour before REQ_088/REQ_089 (2L MLP) work resumes. Scope is intentionally narrow —
  analyzer + two views. No new dashboard page required for the first pass.
- The key scientific payoff: a direct read on whether grokking is the moment when frequency group
  *weight* geometry organizes, independent of activation geometry. The repr_geometry (class
  centroids in activation space) already shows what happens on the activation side.
- If circularity of group centroids in W_in space rises sharply at grokking, it would suggest
  that the ring structure in activation space is a downstream consequence of ring structure
  already forming in weight space.
- W_out analysis may reveal complementary structure: how output projections per group separate.
