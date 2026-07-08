# REQ_109: Measurement Primitives Library

**Status:** Complete — all six primitive categories landed; analyzer-integration scope handed off to REQ_111.
**Priority:** High
**Branch:** Multiple feature branches; phase 2d on `feature/procrustes-primitive` (PR #30).
**Supersedes:** REQ_097 (Frequency Cleanup), REQ_098 (PCA Strategy Cleanup), REQ_104 (Geometry Consolidation).
**Dependencies:** REQ_106 (layering principle — primitives are *measures* with pure-input forms; variant-coupled wrappers are thin convenience layers).
**Successor (analyzer integration):** REQ_111 (Parallel Analyzer Build-Out) — the integration phase that consumes these primitives. Originally written into this REQ as in-place "Analyzer migration"; carved out after the primitive design pass surfaced bugs in pre-REQ_109 derivation paths (arctan vs arctan2, DC fictitious row, dtype regression in `compute_class_centroids`). In-place migration would have silently encoded or fixed those bugs; parallel construction in REQ_111 records the answer.
**Attribution:** Engineering Claude (under user direction)

---

## Problem Statement

The analysis pipeline mixes three concerns inside individual analyzers:

1. **Extract** — load weights / activations from a variant at an epoch.
2. **Transform** — apply a measurement (PCA, Fourier decomposition, circularity, curvature, Procrustes alignment, velocity).
3. **Load** — write artifacts.

The "Transform" step has accreted across the codebase as analyzer-specific code. The same measurement appears in multiple places with subtly different conventions:

- **PCA / Eigendecomposition**: at least four implementations across `analysis/library/trajectory.py`, `analysis/library/geometry.py`, `analyzers/centroid_dmd.py`, and renderer-side computations. (Originally REQ_098.)
- **Fourier decomposition**: three different meanings of "learned frequencies" coexist; the `dominant_frequencies` analyzer is W_E-only despite the generic name; the lissajous-fit bug is a direct symptom. (Originally REQ_097.)
- **Shape characterization** (circularity, curvature, sigmoidality, Lissajous parameters, jerk): scattered across `library/geometry.py`, `library/manifold_geometry.py`, `repr_geometry.py`, `intragroup_manifold.py`, `notebooks/sketch_lissajous_fit.py`, `notebooks/sketch_per_group_kink.py`, and the saddle transport notebooks. (Originally REQ_104.)
- **Shape comparison** (Procrustes): currently lives only in `notebooks/parameter_trajectory_pca.ipynb`. No library home.
- **Clustering metrics** (centroid, radius, Fisher discriminant, dimensionality / participation ratio): live in `library/geometry.py` but mixed with shape descriptors.
- **Velocity / derivative measures**: scattered as inline numpy diff across analyzers and renderers.

The consequence: "is this the same PCA / circularity / Fourier as in figure X?" is a question without a single answer. Auditing requires reading every analyzer.

This REQ extracts these into a single, tensor-friendly **measurement primitives library** under `miscope/analysis/library/`. Primitives are pure functions (ndarray in, typed result out), independent of `Variant` / `Epoch` / `Site`. Analyzers compose them. Renderers do not call them directly — they consume analyzer output (REQ_099).

A note on language: the existing code uses "fit" for several of these (e.g., `compute_circularity`, `sketch_lissajous_fit`). The word *fit* leans toward ML-style parameter learning, which misframes what these measurements do. They are **shape characterizations** — they extract geometric or spectral descriptors from a fixed input. Naming favors *characterize* / *measure* / *describe* over *fit*.

---

## Conditions of Satisfaction

### Primitive categories (the inventory)

The library exposes primitives in six categories. Each primitive is a pure function over arrays; no `Variant` import in pure form.

#### 1. PCA / Eigendecomposition

- [ ] `miscope/core/pca.py::PCAResult` — canonical return type. Fields: `singular_values`, `eigenvalues`, `basis_vectors`, `projections`, `explained_variance`, `explained_variance_ratio`, `participation_ratio`, `rank`, `spread`.
- [ ] `miscope/analysis/library/pca.py` exposes three modes:
  - `pca(X, n_components=None) -> PCAResult` — base function, single sample set.
  - `pca_summary(sample_sets, n_components=None) -> PCAResult` — fits one basis across a stack of sample sets (also called *trajectory PCA*).
  - `pca_rolling(sample_set, window_size, stride, n_components=None) -> List[PCAResult]` — windowed.
- [ ] All three use mean-centered SVD via `np.linalg.svd`. Numerics are deterministic (sign convention documented; downstream consumers handle sign flip).
- [ ] No primitive depends on `sklearn.decomposition.PCA`. (Existing call-site migration is REQ_111 scope.)

#### 2. Fourier Decomposition

- [ ] `miscope/core/frequencies.py` exposes `FrequencySpectrum`, `FrequencySet`, `CommitmentMethod`, and `THRESHOLDS` registry.
- [ ] `miscope/analysis/library/frequency.py` exposes pure-input forms:
  - `site_spectrum_from_matrix(matrix, basis, derivation) -> FrequencySpectrum`
  - `learned_frequencies_from_spectrum(spectrum, threshold, method) -> FrequencySet`
  - `weights_by_frequency_from_matrix(matrix, frequency, basis) -> ndarray`
- [ ] `THRESHOLDS` is the single source of truth: `'canonical': 0.10`, `'transient': 0.05`. All call sites cite the registry; no inline literals.
- [ ] Variant-coupled convenience wrappers (`site_spectrum(variant, epoch, site)` etc.) are 1–3 line shims that load the matrix and call the pure form. Caching (`@cached_artifact`) decorates the wrapper, not the pure form.

#### 3. Clustering Metrics

- [ ] `miscope/analysis/library/clustering.py` (or co-located in `geometry.py` — naming TBD) exposes:
  - `compute_class_centroids(samples, labels) -> ndarray`
  - `compute_class_radii(samples, labels, centroids) -> ndarray`
  - `compute_fisher_discriminant(samples, labels) -> float`
  - `compute_class_dimensionality(samples, labels) -> ndarray` *(participation ratio per class — derived from PCA, not a separate concept)*
  - `compute_center_spread(centroids) -> float`
- [ ] Class-dimensionality computation calls the PCA primitive — does not re-implement SVD.

#### 4. Shape Characterization

- [ ] `miscope/analysis/library/shape.py` exposes characterizations consolidated from existing scattered code:
  - `characterize_circularity(projection_2d, var_explained) -> float` — extracted from `library/geometry.py::compute_circularity` and the residual-stream variant in `repr_geometry.py`. Accepts the projection; does **not** internally re-PCA.
  - `characterize_fourier_alignment(projection_2d, p) -> float` — extracted from `library/geometry.py::compute_fourier_alignment`. Accepts the projection; does **not** internally re-PCA.
  - `characterize_curvature(trajectory) -> ndarray` — point-wise curvature profile (extracted from `library/manifold_geometry.py` and `intragroup_manifold.py`).
  - `characterize_arc_length(trajectory) -> float` — extracted from `library/trajectory.py` (the `# TODO: Move to Geometry` comment).
  - `characterize_sigmoidality(trajectory) -> float` — extracted from saddle transport notebooks. Returns S-shape goodness for transit detection.
  - `characterize_lissajous(trajectory_2d) -> LissajousParameters` — extracted from `notebooks/sketch_lissajous_fit.py`. Returns the geometric parameters (frequency ratio, phase offset, amplitudes); does not couple to family-level frequency context.
  - `characterize_saddle_curvature(trajectory_3d) -> SaddleParameters` — extracted from `library/manifold_geometry.py` quadratic surface fit (R²_curvature). Returns shape parameters.
  - `characterize_jerk(trajectory) -> ndarray` — third-derivative measure (new, research-driven).
  - `detect_self_intersection(trajectory_2d) -> bool` — extracted from existing trajectory code.
  - `compute_signed_loop_area(trajectory_2d) -> float` — extracted from existing trajectory code.
- [ ] Each characterization takes the projected / coordinate-frame trajectory as input. PCA happens upstream, in the caller. (REQ_106 layering rule.)

#### 5. Shape Comparison

- [x] `miscope/analysis/library/comparison.py` exposes:
  - `procrustes_align(a, b) -> ProcrustesResult` — extracted from `notebooks/parameter_trajectory_pca.ipynb`. Wraps `scipy.spatial.procrustes` for bit-exact compatibility; result carries `disparity`, `standardized_a`, `aligned_b`, `n_points`, `n_features`.
  - `compute_procrustes_disparity_matrix(trajectories) -> ndarray` — pairwise N×N convenience over the single-pair primitive; symmetric, zero diagonal, NaN on degenerate input.
- [x] Other comparison primitives (Hausdorff distance, dynamic time warping, etc.) are out of scope until a research use case appears.

#### 6. Velocity / Derivative Measures

- [ ] `miscope/analysis/library/dynamics.py` (naming TBD) exposes:
  - `compute_velocity(trajectory, time_axis=0) -> ndarray` — first derivative, finite-difference.
  - `compute_acceleration(trajectory, time_axis=0) -> ndarray` — second derivative.
- [ ] (Existing inline `np.diff` patterns across analyzers and renderers migrate to these under REQ_111 / REQ_099 — not REQ_109 scope.)

### Interface contract (REQ_106 layering rule)

- [ ] **Pure-input forms.** Every primitive in pure form takes `np.ndarray` (or typed dataclasses composed of arrays) and returns a typed result. No `Variant`, no `Epoch`, no `Site` knowledge.
- [ ] **Documented axis conventions.** Each primitive's docstring states the expected shape (e.g., `(samples, features)`, `(epochs, samples, features)`, `(timesteps, dimensions)`).
- [ ] **Acceptance test (grep).** No file in `miscope/analysis/library/` imports from `miscope.families`. Convenience wrappers that need `Variant` live in a clearly-marked section or sibling module (e.g., `library/frequency.py` exposes both pure forms and wrappers; the wrappers are explicitly tagged in docstrings).
- [ ] **Acceptance test (unit-testable).** Each primitive has at least one unit test that calls it with `np.array(...)` literals and asserts a known-correct numeric result.
- [ ] **Thin adapters, not reimplementations.** Where a primitive wraps an existing library function (sklearn, scipy), the wrapper exists to encode our axis conventions and edge-case handling — not to reimplement the algorithm. The wrapper docstring names the underlying library and the value-add.

### Existing code extraction map

For each primitive, the REQ tracks the existing code location it extracts from. This is the migration manifest.

| Primitive | Extracted from |
|-----------|----------------|
| `pca`, `pca_summary`, `pca_rolling` | `analysis/library/trajectory.py::compute_pca_trajectory`, `analysis/library/trajectory.py::fit_centroid_pca`, `analysis/library/geometry.py::compute_global_centroid_pca`, `analyzers/centroid_dmd.py` PCA path, renderer-side PCA (e.g., `render_weight_geometry_centroid_pca`) |
| `site_spectrum_from_matrix`, `learned_frequencies_from_spectrum`, `weights_by_frequency_from_matrix` | `analyzers/dominant_frequencies.py`, `analyzers/fourier_frequency_quality.py`, `analyzers/neuron_dynamics.py::dominant_freq` derivation, `analyzers/transient_frequency.py` |
| `compute_class_centroids`, `compute_class_radii`, `compute_fisher_discriminant`, `compute_center_spread` | `analysis/library/geometry.py` (existing, mostly stays) |
| `compute_class_dimensionality` | Existing PR-per-class logic; re-routed through the PCA primitive |
| `characterize_circularity`, `characterize_fourier_alignment` | `analysis/library/geometry.py::compute_circularity`, `analysis/library/geometry.py::compute_fourier_alignment`, `analyzers/repr_geometry.py` |
| `characterize_curvature`, `characterize_saddle_curvature` | `analysis/library/manifold_geometry.py`, `analyzers/intragroup_manifold.py` |
| `characterize_arc_length`, `detect_self_intersection`, `compute_signed_loop_area` | `analysis/library/trajectory.py` (the `# TODO: Move to Geometry` items) |
| `characterize_sigmoidality` | Saddle Transport notebooks (`notes/project_saddle_transport.md` references) |
| `characterize_lissajous` | `notebooks/sketch_lissajous_fit.py` |
| `characterize_jerk` | New (research-driven; saddle-transit work) |
| `procrustes_align` | `notebooks/parameter_trajectory_pca.ipynb` |
| `compute_velocity`, `compute_acceleration` | Inline `np.diff` patterns across analyzers and renderers |

### Analyzer integration (deferred to REQ_111)

Analyzer-level integration — building the analyzers that consume these primitives, validating them against the existing analyzers, and arriving at deprecation candidates — is handled separately under **REQ_111 (Parallel Analyzer Build-Out)**. This REQ ships the primitive layer only.

The original draft of this REQ committed to in-place analyzer migration (PCA, Frequency, Geometry, plus consumer rewires for `variant_analysis_summary` and `notebooks/sketch_lissajous_fit.py`). That commitment was withdrawn after the primitive design pass surfaced multiple latent bugs in pre-REQ_109 code (arctan vs arctan2, DC fictitious row, normalization mismatch, float32 accumulation regression). In-place migration would have silently fixed or encoded those bugs without an audit trail. REQ_111 builds new analyzers in parallel, validates outputs against old, and records the answer to *"did the old code do the right thing?"* for each pair before deprecation.

The analyzer-naming proposals from the original draft (`learned_parameters_pca`, `frequency_group_geometry`, `activation_class_geometry`, `frequency_spectrum_per_site`) carry forward into REQ_111 as the starting set; final naming is decided during implementation per REQ_111's *descriptive divergence* convention.

### Removal (deferred to REQ_102, gated on REQ_111)

Removal of deprecated code (`compute_pca_trajectory`, `fit_centroid_pca`, `compute_global_centroid_pca`, `centroid_dmd`'s PCA path, the W_E-only `dominant_frequencies` analyzer, scattered geometry code) is handled under REQ_102 after parity validation completes under REQ_111. No analyzer is listed for retirement under REQ_102 until its REQ_111 validation outcome is recorded. This REQ ships only the additive primitive layer.

---

## Constraints

**Must:**
- Single primitive per measurement. After this REQ, every PCA call resolves to one of `pca` / `pca_summary` / `pca_rolling`. Every circularity call resolves to `characterize_circularity`. Etc.
- Pure-input forms unit-testable with `np.array(...)` literals. No `Variant` import in the pure form module.
- Numerical equivalence with current outputs on reference variants within tolerance (singular values to ~1e-10 relative; eigenvectors to within sign flip; metrics to documented per-metric tolerance).
- Site coupling preserved at the type level for frequency primitives. No site-less "what frequencies does this variant have?" query.
- Storage decoration (`@cached_artifact`) lives on variant-coupled wrappers, not pure forms. Caching is a data-plane concern, not a derivation concern.

**Must avoid:**
- Reinventing wheels. Where sklearn / scipy / pyarrow have a function we need, the primitive is a thin adapter that encodes our axis conventions and edge-case handling — not a reimplementation.
- Renderer-side measurement. After REQ_099, no renderer calls `np.linalg.svd`, `sklearn.decomposition.PCA`, or any primitive directly. Renderers consume analyzer output.
- Coupling primitives to a specific family. Variants supply prime via `variant.prime`; non-modular families either raise informatively or skip.
- Hidden defaults. `CommitmentMethod`, threshold names, and similar are explicit on the result type.
- Premature primitives. Add new primitives only when they're earned by research need (e.g., Hausdorff distance, DTW). This REQ consolidates what exists; new primitives come from research, not speculative API growth.

**Flexible:**
- Module organization within `library/` (single flat module per category vs sub-packages). Default: one module per category, named for the category.
- Whether some categories collapse together (e.g., shape + clustering both under `geometry.py`). Default: separate modules; collapse if the boundary is artificial in practice.
- The exact name of the new frequency analyzer (`frequency_spectrum_per_site`, `site_frequency`, `per_site_fourier`). Not `dominant_frequencies`.
- Eigenvector sign convention. Document the SVD's choice; consumers handle sign flips.

---

## Architecture Notes

### Why "primitives" and not "library functions"

The word *primitive* is load-bearing. A primitive is a function that:
- Takes a typed input (array with documented axis semantics).
- Returns a typed output (`PCAResult`, `FrequencySpectrum`, `LissajousParameters`).
- Performs exactly one measurement.
- Has no knowledge of where the input came from or where the output is going.

Primitives are the **transform** in the analyzer's ETL. The analyzer is the composition: it pulls the right matrix (extract), feeds it to the right primitives (transform), and writes the typed result (load). When measurement-related logic lives outside primitives — inside analyzers, renderers, or notebook code — the audit question "is this the same measurement as in figure X?" loses its single answer.

### Why "characterize" / "measure" instead of "fit"

The existing vocabulary leans on *fit* (`compute_circularity` is implicitly a circle fit; `sketch_lissajous_fit.py` is explicit). In ML, *fit* implies parameter learning with an optimizer. These primitives are not learning parameters — they are extracting geometric or spectral descriptors from a fixed input. Renaming clarifies the intent and avoids inviting "what's the loss function for circularity?" confusions.

### Three-layer pipeline

```
Extract                    Transform (this REQ)             Load
-------                    --------------------             ----
Variant + epoch    →    PCA / Fourier / shape /     →   Typed result
                        clustering / comparison /        (artifact, dataframe)
                        velocity primitives
```

The primitives library is the middle column. Analyzers compose extract + transform + load. Renderers consume load output. This separation is what REQ_106 names; this REQ is the concrete realization of REQ_106 for the transform layer.

### Phasing

The categories have unequal complexity. Phasing as actually executed (with status):

1. ✅ **PCA + clustering metrics + velocity primitives** — landed across REQ_109 phase 1 work. `library/pca.py`, `library/clustering.py`, `library/trajectory.py::compute_parameter_velocity`.
2. ✅ **Fourier groundwork** — `library/fourier_basis.py` (PR #28). `PeriodicFourierBasis`, `project_onto_fourier_basis`, `compute_specialization`. The legacy `library/fourier.py` callers (`dominant_frequencies`, `neuron_fourier`, `attention_freq`, `transient_frequency`) are not retired here — that's REQ_111 work.
3. ✅ **Shape characterization (phase 2a–2c-2)** — `characterize_circularity`, `characterize_fourier_alignment`, `characterize_jerk`, `characterize_surface`, `characterize_lissajous`, `characterize_sigmoidality`, plus curve-shape helpers (`compute_arc_length`, `detect_self_intersection`, `compute_signed_loop_area`, `compute_curvature_profile`). Shipped in PRs #26, #27, #28, #29.
4. ✅ **Shape comparison (phase 2d)** — `procrustes_align` and `compute_procrustes_disparity_matrix` in `library/comparison.py` (PR #30). Validated bit-exact against the notebook implementation on the canon reference triple (p109/s485/ds598, p113/s999/ds598, p101/s999/ds598).

REQ_109 closes with phase 2d. Analyzer-level integration tracks under REQ_111.

### Acceptance test: the grep tests

REQ_106 names grep-test acceptance criteria as part of layering compliance. For this REQ:

- `grep -rn "from miscope.families" src/miscope/analysis/library/` — returns only matches in clearly-marked convenience-wrapper sections.
- `grep -rn "np.linalg.svd\|sklearn.decomposition.PCA" src/miscope/analysis/library/` — returns matches only in `library/pca.py`.
- `grep -rn "_pca_project\|_pca_project_2d" src/miscope/analysis/library/shape.py` — returns zero hits.

(The fourth grep test originally listed — `np.diff` in `analysis/analyzers/` returning zero hits — is an analyzer-level concern; it moves to REQ_111's acceptance criteria.)

The library-side greps are part of CI once REQ_109 closes. Analyzer-side greps move to REQ_111.

---

## Notes

- **The lissajous fit bug is the proof-of-concept test for the whole consolidation.** REQ_109 supplies the primitives (`characterize_lissajous`, `compute_specialization`, etc.); the actual lissajous-bug fix lands when REQ_111 builds the new analyzer that consumes them and validates against the existing pipeline. The architecture pays for itself once that round trip closes.
- The user's audit framing — "if I were a researcher and saw a bunch of scattered implementations of PCA, I'd likely stop review" — is the load-bearing motivation. Single primitive per measurement is what makes external review possible.
- Some primitives may want a *tensor-friendly* path (e.g., `pca_per_matrix` over a stack of `(n_matrices, samples, features)`) in addition to the single-call path. Defer until a real call site needs it; speculative tensor-vectorization invites complexity ahead of demand.
- `manifold_geometry.py` is a candidate for full absorption into `shape.py`. Decide during implementation; default to absorption unless a clear conceptual boundary emerges.
- This REQ is the prerequisite for REQ_110 (Lakehouse Surface) and REQ_111 (Parallel Analyzer Build-Out). Tabular outputs (REQ_110) are downstream of typed measurement results — once primitives return `PCAResult` consistently, the tabular form is a flatten of `PCAResult` over the natural dimension columns. Analyzer integration (REQ_111) is downstream of primitive stability — the parallel-build pattern is what catches latent bugs that in-place migration would have silently encoded.
- **Phase tracking.** Phase 2c-2 (Lissajous + Sigmoidality) landed in PR #29. Phase 2d (`procrustes_align`) landed in PR #30 — the closing primitive. REQ_109 is complete; REQ_111 takes over for analyzer-level integration.
