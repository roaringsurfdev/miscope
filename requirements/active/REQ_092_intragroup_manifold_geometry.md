# REQ_092: Intra-Group Manifold Geometry (Quadratic Surface Fit)

**Status:** Active
**Priority:** Medium
**Branch:** feature/req-092-intragroup-manifold-geometry
**Attribution:** Engineering Claude

---

## Problem Statement

Visual inspection of `neuron_group_pca` scatter plots reveals that frequency groups
form distinct geometric shapes in their weight-space PCA subspace — saddles, bowls,
and flat blobs. These shapes are not artifacts of viewing angle; they appear consistently
and correspond to structurally different organizations of neurons within each group.

No metric currently captures this. The existing `neuron_group_pca` analyzer tracks PC1
variance explained (cohesion) and mean spread, but neither discriminates between a
well-defined saddle and a flat blob that happens to have low spread.

The key insight: within each group, neurons are distributed *across* a curved surface
("geometry gets painted"), not clustered at the boundary. This means ring/hollowness
metrics are wrong; the right tool is a quadratic surface fit — measuring how well PC3
is explained by a quadratic function of PC1 and PC2.

---

## Conditions of Satisfaction

### Computation

- [ ] `fit_quadratic_surface(proj_group)` function in
  `src/miscope/analysis/library/manifold_geometry.py`
- [ ] Two-stage fit:
  - Stage 1 (linear): PC3 = d·PC1 + e·PC2 + f → R²_linear
  - Stage 2 (quadratic): adds a·PC1² + b·PC2² + c·PC1·PC2 → R²_quadratic
  - R²_curvature = R²_quadratic − R²_linear (variance explained by curvature alone,
    net of tilt)
- [ ] Shape classification from quadratic coefficients a, b:
  - R²_curvature < 0.05 → `"flat/blob"`
  - same sign(a, b) → `"bowl"`
  - opposite sign(a, b) → `"saddle"`
- [ ] Returns: `r2_linear`, `r2_quadratic`, `r2_curvature`, `a`, `b`, `c`, `shape`

### Analyzer

- [ ] New cross-epoch analyzer `IntraGroupManifoldAnalyzer`
  (`name = "intragroup_manifold"`)
- [ ] `requires = ["neuron_group_pca"]`
- [ ] Runs quadratic fit for each frequency group at the final epoch
- [ ] Artifact keys:
  - `group_freqs` int32 (n_groups,)
  - `group_sizes` int32 (n_groups,)
  - `r2_linear` float32 (n_groups,)
  - `r2_quadratic` float32 (n_groups,)
  - `r2_curvature` float32 (n_groups,)
  - `a` float32 (n_groups,)
  - `b` float32 (n_groups,)
  - `c` float32 (n_groups,)
  - `shape` stored as UTF-8 string array or integer label
- [ ] Registered in `modulo_addition_1layer` family config under
  `cross_epoch_analyzers`

### Views

- [ ] `intragroup_manifold.summary` — bar chart per group: R²_curvature height,
  colored by shape label (saddle / bowl / flat)
- [ ] `intragroup_manifold.surface_fit` — 3D scatter of group neurons with the
  fitted quadratic surface overlaid; group selectable by kwarg

### Cross-variant utility

- [ ] Notebook-level utility (can be promoted later) that loads the artifact across
  all variants and produces a cross-variant summary DataFrame with per-group and
  per-variant mean R²_curvature

---

## Constraints

**Must:**
- Compute from the `projections` field already in `neuron_group_pca` — no additional
  weight loading at analysis time
- Use the two-stage fit (quadratic increment over linear) — the increment isolates
  curvature from tilt, which matters because the projections are not guaranteed to be
  centered on the quadratic axis

**Must not:**
- Use the `c` coefficient (PC1·PC2 cross term) in the shape classification — it
  encodes rotation of the saddle/bowl axes, which is secondary to the bowl/saddle
  distinction itself

**Flexible:**
- Whether `shape` is stored as strings or integer codes in the artifact
- The 0.05 threshold for flat/blob — revisit after seeing the full cross-variant
  distribution

---

## Architecture Notes

The `projections` array in `neuron_group_pca` is already in each group's own PCA
coordinate frame, centered by the final-epoch group centroid. R²_linear will therefore
be near zero by construction (no systematic tilt in a centered frame), making
R²_curvature ≈ R²_quadratic in practice. The two-stage fit is still correct design —
it doesn't assume centering.

---

## Notes

- Motivated by session 2026-04-06. Cross-variant results on 30 variants showed:
  - Saddles outnumber bowls ~3.5:1 (113 vs 32 groups)
  - p59/s485/ds999 (the only non-grokking variant) has mean R²_curvature = 0.132 —
    dramatically below all other variants (next lowest: 0.494). No second descent = no
    manifold geometry. Clean negative control.
  - Canon model (p113/s999/ds598) has *higher* mean R²_curvature (0.839) than the
    reference healthy model p109/s485/ds598 (0.644), despite known diffuse
    specialization. Within-group geometry and between-group functional separation are
    dissociable.
  - Manifold formation (R²_curvature) is a second-descent phenomenon, not a
    first-descent or plateau phenomenon.
- The surface_fit view is the one that makes these results shareable without requiring
  the viewer to trust a specific camera angle on a 3D scatter.
- **Open question — curvature trajectories:** Is manifold formation gradual, or does it
  show phase-transition signatures? The hypothesis: R²_curvature tracks neuron
  commitment (specialization threshold crossings) rather than just time during second
  descent — meaning curvature lags slightly behind the rising neuron specialization
  count, with geometry forming as a consequence of neurons committing rather than
  concurrently. This is testable from existing artifacts: `neuron_freq_norm` carries the
  per-neuron specialization signal at every epoch; `neuron_group_pca` projections are
  already available per epoch. Answering this requires running the quadratic fit per
  epoch rather than only at the final epoch — a natural extension of this requirement's
  scope, not a new one. The cross-variant picture (healthy vs. non-grokking vs.
  anomalous) would show whether trajectory shape (gradual ramp vs. sharp inflection) is
  a reliable signature of grokking quality.
