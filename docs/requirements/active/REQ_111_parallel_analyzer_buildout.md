# REQ_111: Universal Core Pure Renames + Primitive Integration

**Status:** Draft (rescoped after Analysis Atlas (a)(b)(c) audit pass)
**Priority:** Medium — small, low-risk, immediately reduces name drift. The fastest demonstration that the Atlas's (a)(b)(c) annotations drive real code change.
**Branch:** TBD
**Dependencies:**
- REQ_109 (measurement primitives — *staging*; SVD, participation ratio, PCA, finite-difference velocity primitives consumed here).
- REQ_114 (HookedModel analyzer migration — *staging*; analyzers built on the canonical interface).

**Atlas reference:** [docs/analysis_atlas.md](../../analysis_atlas.md) — `weight_spectra` and `parameter_trajectory` entries under Universal Core.

**Attribution:** Engineering Claude (under user direction, after the Analysis Atlas (a)(b)(c) audit pass).

---

## Scope evolution

This REQ was originally scoped as a broad "parallel analyzer build-out" covering frequency analyzers, PCA analyzers, geometry routing, and Lissajous + saddle-transport fits. After the Analysis Atlas (a)(b)(c) annotation pass (2026-05-27), that scope cleaved into multiple distinct shapes with different dependencies and risk profiles:

- **Frequency / basis projection consolidation** → [REQ_126](REQ_126_basis_projection_consolidation.md) (new). Absorbs six existing analyzers into two universal ones; splits fused fields out of `representation_geometry`. Keystone of the phase 1 reorganization.
- **Geometry routing through shape primitives** → completed under REQ_104 (staging).
- **`representation_geometry` scope-tightening refactor and `gradient_site` pipeline integration** → deferred to a future REQ. Both depend on REQ_126's basis column being in place.
- **Lissajous + saddle-transport sigmoidality + saddle-center-center** → each gets its own scoped REQ under Atlas Dynamical Proxies → Phase-space fits. Bucket: `new`.

**What remains in REQ_111 (this document):** the Atlas's two pure-rename entries in Universal Core — `weight_spectra` and `parameter_trajectory`. Both are `existing-rename` with bucket `refactor`. Parity validation against the old analyzers is meaningful and bounded.

This rescope was approved 2026-05-27 in the conversation that drove the Atlas (a)(b)(c) pass. Git history preserves the original (broader) REQ_111 content for archaeology.

---

## Problem Statement

Two existing analyzers carry implementation-detail or scope-overpromising names:

- **`effective_dimensionality`** computes singular values for all weight matrices and surfaces participation ratio as one summary. The name overpromises — the analyzer returns spectra, of which effective dim is one derived metric. Per the Atlas (a)(b)(c) audit, the right name is `weight_spectra`.
- **`parameter_trajectory_pca`** computes cross-epoch PCA on weight trajectories with first-order velocity. The name bakes implementation (PCA) into the contract. Per the Atlas, the right name is `parameter_trajectory` — the PCA is one valid summary; future implementations could swap in a different reduction without invalidating the name.

Both analyzers also predate REQ_109's measurement primitive library. Their internal transform steps currently use ad-hoc numpy calls instead of the canonical primitives.

This REQ renames both and routes their transform steps through REQ_109 primitives. Conceptual shape is unchanged. Parity testing against the old analyzers is meaningful.

---

## Conditions of Satisfaction

### `weight_spectra` (← `effective_dimensionality`)

- [ ] New analyzer registered as `weight_spectra`. Same per-epoch output shape as `effective_dimensionality` plus singular vectors retained (or made available on request — current analyzer discards them).
- [ ] Transform step uses REQ_109 primitives: SVD via the primitive library; participation ratio via the primitive library. No inline `np.linalg.svd` in `analyze()`.
- [ ] Old `effective_dimensionality` analyzer retained for the parallel period; old artifacts remain readable.
- [ ] Parity validation: numerical agreement on the canon reference set (p113/s999/ds598, p109/s485/ds598, p101/s999/ds598). Singular values to ~1e-10 relative tolerance; participation ratio to documented tolerance.
- [ ] Validation outcome recorded in this REQ's Notes section. Outcomes: *matches within tolerance* / *old has bug X (attributed)* / *new has bug Y (attributed)* / *disagreement is real, both kept*.
- [ ] Handoff to REQ_102: `effective_dimensionality` retirement listed only after parity outcome is recorded.

### `parameter_trajectory` (← `parameter_trajectory_pca`)

- [ ] New analyzer registered as `parameter_trajectory`. Same per-variant cross-epoch output shape as `parameter_trajectory_pca`.
- [ ] Transform step uses REQ_109 primitives: PCA via the primitive library; finite-difference velocity via the primitive library. No inline numpy implementations of either.
- [ ] Old `parameter_trajectory_pca` retained for the parallel period; old artifacts remain readable.
- [ ] Parity validation on the same canon reference set. PCA components to within sign flip / intra-eigenspace rotation; velocity to documented tolerance.
- [ ] Validation outcome recorded in this REQ's Notes section.
- [ ] Handoff to REQ_102: `parameter_trajectory_pca` retirement listed only after parity outcome is recorded.

### Migration of consumers

- [ ] Library code referencing the old analyzer names updated to the new ones, or routed through a deprecation pointer.
- [ ] Dashboard pages: migrate to the new names, or hide the churn behind the visualization layer per user direction. Not blocking.
- [ ] Old artifact paths remain readable via `ArtifactLoader` for the deprecation window.

---

## Constraints

**Must:**
- Conceptual shape of each analyzer is unchanged — only the name and primitive routing.
- New analyzers use only REQ_109 primitives for their transform step. Auditable via grep.
- Parity validation outcome recorded before REQ_102 lists the old analyzer for retirement.
- Existing artifacts on disk remain readable for the deprecation window.

**Must avoid:**
- **Bundling primitive extension with rename work.** Primitive gaps surfaced here get filed as REQ_109 follow-ups; the rename doesn't extend the primitive library.
- **Quiet deprecation.** Each rename gets a CHANGELOG entry pointing the old name to the new one.
- **Touching old analyzer code during the parallel period.** The old analyzer is the parity anchor.

**Flexible:**
- Whether to land both renames in one PR or two. Default: two — independent absorption surfaces, smaller diffs.
- Order. Default: `weight_spectra` first (simpler — no trajectory machinery); `parameter_trajectory` after.

---

## Notes

### Validation outcomes (recorded as the work proceeds)

Format: `{old_analyzer} → {new_analyzer}: {outcome}, {date}, {pointer to evidence}`.

- `effective_dimensionality → weight_spectra`: **matches bit-exactly** (max abs diff = 0.0, max rel diff = 0.0) across all 9 weight matrices on the canon reference set (p113/s999/ds598, p109/s485/ds598, p101/s999/ds598) at 5 epochs each. 2026-05-27. Evidence: `tests/test_weight_spectra.py::TestWeightSpectraIntegration::test_parity_singular_values_match_effective_dimensionality` (CI) + canon snapshot diff via `compute_svd` on `parameter_snapshot/*.npz` artifacts. Bit-exactness follows from both paths routing through the same LAPACK `gesdd` call; the old path discarded U/Vt via `compute_uv=False`, the new path retains them via `full_matrices=False`.
- `parameter_trajectory_pca → parameter_trajectory`: **degenerate parity** — the rename was already done at the registered-name level in earlier work (`SPEC.name = "parameter_trajectory"`, artifact directory `artifacts/parameter_trajectory/`). This pass closed the file/class gap: `parameter_trajectory_pca.py` → `parameter_trajectory.py` via `git mv`; `ParameterTrajectoryPCA` → `ParameterTrajectory`. Transform steps already routed through REQ_109 primitives (`pca`, `compute_velocity`) — audit at 2026-05-27 confirmed no inline `np.linalg.svd`, no inline `np.diff` for derivatives. There is no separate "old" analyzer to parity-test; existing on-disk artifacts under `artifacts/parameter_trajectory/` are valid for the renamed analyzer. Evidence: full miscope test suite (1674 passed) including `test_cross_epoch_analyzers.py::TestParameterTrajectory*`.

### Design notes for `weight_spectra`

- New pure primitive: `miscope.analysis.library.pca.compute_svd(matrix) -> SVDResult` (with `SVDResult` in `miscope.core.svd`). Raw, non-mean-centered SVD — distinct from `pca()`, which mean-centers and is intended for sample distributions. The weight matrices are the linear maps themselves, not samples.
- `compute_weight_singular_values` retained for the parallel period; refactored to delegate to a new `compute_weight_spectra(model) -> dict[name, (U, S, Vt)]` helper. No call sites outside `effective_dimensionality` and tests, so the slight overhead of always computing U/Vt is bounded to the deprecation window.
- `WeightSpectraAnalyzer` adds `u_{name}` and `vt_{name}` to the per-epoch artifact alongside the legacy `sv_{name}`. Keys, shapes, and PR-summary semantics for `sv_*` and `pr_*` are unchanged from `effective_dimensionality`.

### Why parallel construction (not in-place rename)

Inherited from the original REQ_111 framing, still load-bearing here. In-place rename of an analyzer implicitly answers the question *"did the old code do the right thing?"* — without recording the answer. Parallel construction makes the answer explicit: run both, compare outputs, record agreement or divergence. Then collapse.

The risk surface is smaller than the original REQ_111 scope (no shape change, no consolidation), but the validation discipline is the same: numerical anchor preserved until parity is recorded.

### What this REQ does NOT do

- Does not touch `representation_geometry` or `gradient_site`. Both are `refactor` bucket but their refactors carry scope changes (defusion of Fourier fields; pipeline integration + basis generalization). Both depend on REQ_126's basis column being in place. Future REQ.
- Does not touch any `reorganization` or `new` bucket analyzers. Those are Atlas entries with their own REQs.
- Does not retire the old analyzers. Retirements live under REQ_102.
