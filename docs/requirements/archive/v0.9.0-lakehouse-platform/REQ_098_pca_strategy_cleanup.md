# REQ_098: PCA Strategy Cleanup (Single Computational Path)

**Status:** Superseded — primitive scope landed in REQ_109; analyzer-integration scope tracks under REQ_111
**Priority:** High
**Branch:** TBD (incubating on `refactor-dataview`)
**Dependencies:** None blocking. REQ_097 (frequency) is parallel.
**Attribution:** Engineering Claude

> **Superseded by REQ_109 (Measurement Primitives Library).** This REQ's scope (the `pca` / `pca_summary` / `pca_rolling` primitives, `PCAResult` canonical type, removal of scattered PCA implementations, analyzer migration to a single computational path) was consolidated with REQ_097 (Fourier) and REQ_104 (Geometry) into a single primitives-library REQ. The three-mode framing — base PCA, summary/trajectory PCA, rolling PCA — carries forward into REQ_109 unchanged and seeds the `operation_type` enum in REQ_110. Primitive scope landed under REQ_109; the analyzer-integration / view-renderer-migration / removal scope below (new `learned_parameters_pca`, `frequency_group_geometry`, `activation_class_geometry` analyzers, deprecation of `compute_pca_trajectory` / `fit_centroid_pca` / `compute_global_centroid_pca` / `centroid_dmd` PCA path) was carved into **REQ_111 (Parallel Analyzer Build-Out)** and **REQ_102 (Analyzer Deprecation)**, gated on REQ_111 parity validation.

---

## Problem Statement

PCA computation lives in at least four places with subtly different conventions:

- `analysis/library/trajectory.py::compute_pca_trajectory` — uses
  `sklearn.decomposition.PCA`, returns dict.
- `analysis/library/trajectory.py::fit_centroid_pca` — uses raw
  `np.linalg.svd`, returns dict.
- `analysis/library/geometry.py::compute_global_centroid_pca` — separate
  centroid PCA implementation.
- `analyzers/centroid_dmd.py` — own PCA + eigendecomposition path.
- Various renderer-side PCA inside view registrations (e.g., the centroid PCA
  computed inside `render_weight_geometry_centroid_pca`).

The inventory of *what we want* is small: 8 named scenarios spanning weights vs
activations, point vs trajectory, and various groupings (matrix, frequency
group, frequency group centroid, class centroid). The implementations across 
the codebase do not collapse to one primitive, which makes audit difficult and 
makes "is this the same PCA we used in figure X?" a question without a single answer.


---

## Conditions of Satisfaction

### Library

- [ ] `miscope/core/pca.py::PCAResult` is the canonical return type. All PCA
  functions in miscope return this.
- [ ] `miscope/analysis/library/pca.py::pca()` is the single computational
  primitive. Supports both fit and project-onto-existing-basis modes.
- [ ] Convenience functions exist for different PCA types:
  - `compute_pca(sample_set, n_components)`
  - `compute_pca_global_trajectory(sample_sets, n_components)`
  - `compute_pca_folling(sample_set, window_size, stride, n_components)`
- [ ] Domain-specific calls to the Convenience Builders
  - PCA - Global Weight Trajectories - (Weight Matrix Groups, All Combined Weights)
  - PCA - Weight Matrices (Weight Matrix Groups)
  - PCA - Frequency Group (MLP Weights: W_in, W_out)
  - PCA - Frequency Group Centroids (MLP Weights: W_in, W_out)
  - PCA - Class Centroids (Activation Sites: Att_out, MLP_out, Resid_Post)
  - PCA - Class Centroid Global Trajectories (Activation Sites: Att_out, MLP_out, Resid_Post)

- [ ] Each Convenience builder calls `pca()`. Calls to these functions are responsible for creating the correct sample_set(s). No domain logic exists in the convenience functions.
- [ ] Numerics are deterministic (mean-centered SVD via `np.linalg.svd`).

### Analyzer migration
# TODO: Update with final signatures
- [ ] New `learned_parameters_pca` prepares all weight sample_sets and passes to the appropriate convenience functions. This analyzers absorbs and replaces `parameter_trajectory_pca`, `effective_dimensionality`. Global PCA can be run from the same analyzer under the `analyze_across_epochs` call, which is executed as a cross_epoch analyzer call.
- [ ] New `frequency_group_geometry` analyzer prepares frequency group sets and passes to appropriate convenience functions. This analyzer replaces `freq_group_weight_geometry`, and any other analyzers performing frequency group geometry functions (`neuron_group_pca`, `global_centroid_pca`). Global PCA can be run from the same analyzer under the `analyze_across_epochs` call, which is executed as a cross_epoch analyzer call.
- [ ] New `activation_class_geometry` analyzer prepares class group sets and passes to appropriate convenience functions. This analyzer replaces `repr_geometry`. Global PCA can be run from the same analyzer under the `analyze_across_epochs` call, which is executed as a cross_epoch analyzer call.

### View / renderer migration

- [ ] No view renderer calls `np.linalg.svd`, `sklearn.decomposition.PCA`, or
  any direct PCA primitive. PCA happens in analyzers or library functions
  whose output is loaded by the view's `load_data` callback. (Coordinated with
  REQ_099.)
- [ ] View Catalog should point to the new analyzers and their artifacts.

### Removal

Removal of deprecated libraries will be handled in a separate requirement after sufficient regression testing

- [ ] `analysis/library/trajectory.py::compute_pca_trajectory` removed (or
  reduced to a thin re-export of `compute_pca_flattened_snapshots`) once all callers
  are migrated.
- [ ] `analysis/library/trajectory.py::fit_centroid_pca` removed similarly.
- [ ] `analysis/library/geometry.py::compute_global_centroid_pca` removed.
- [ ] `centroid_dmd` PCA + eigendecomposition extracted before deprecation
  (coordinated with REQ_102).

---

## Constraints

**Must:**
- Maintain numerical equivalence with current PCA outputs within reasonable
  tolerance (eigenvector signs may flip; magnitudes and projections must match).
- Validation against existing artifacts on at least 3 reference variants.
- Library primitive remains backend-clean: pure-array in, pure-array out, no
  implicit device coupling. (Permits later GPU/cuML backend without API change.)

**Must avoid:**
- Re-introducing scattered PCA. After this REQ, every PCA call in the codebase
  resolves to one of the convenience builders or the primitive directly.
- Coupling PCA to `sklearn.decomposition.PCA`. The mean-centered SVD path is
  the contract; `sklearn.PCA` is removed from analyzer / library code.

**Flexible:**
- Eigenvector sign convention (PCA basis is arbitrary up to sign; downstream
  consumers should handle sign flips, not assume the SVD's choice is canonical).
- Whether convenience builders accept `n_components` or always return the full
  spectrum and slice downstream. (Discovery uses `n_components` as kwarg.)

---

## Architecture Notes

**Inventory mapping** (see `analysis/library/pca.py` docstring for the canonical
table) — every PCA call site should resolve to a convenience builder
or the bare primitive.

**Single-primitive guarantee.** The convenience builders only handle reshaping
 — they call `pca()` once and never re-implement the SVD path.
This is what makes "the PCA in figure X is the same PCA in figure Y" a defensible
statement.

**PCAResult fields are exhaustive.** `participation_ratio`, `rank`, and
`spread` are on the result so consumers don't compute them separately. This is
the absorption of `effective_dimensionality` into the PCA primitive — those
metrics are derived from singular values, not separate concepts.

---

## Notes

- Discovery code in `analysis/library/pca.py` is the working primitive + builders.
  Migration of analyzers happens under this REQ.
- The numerical tolerance for migration validation should be defined per
  metric — projections and singular values to ~1e-10 relative; eigenvectors
  to within sign flip. Decision deferred until characterization tests are
  written.
- This REQ pairs with REQ_104 (Geometry Consolidation): curvature/sigmoidality
  often consume PCA output. Migrating PCA first lets geometry consolidation
  rest on the canonical primitive.
