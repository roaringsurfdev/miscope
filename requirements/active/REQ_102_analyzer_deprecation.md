# REQ_102: Analyzer Deprecation (Retire Stale Paths)

**Status:** Draft
**Priority:** Medium
**Branch:** TBD
**Dependencies:** REQ_106 (defines the layering principles whose violation is one of the deprecation criteria); REQ_109 (supplies the primitive layer the new analyzers consume — formerly REQ_097 + REQ_098 + REQ_104, now consolidated); REQ_111 (parallel analyzer build-out + parity validation — gates "migrate-track" retirements: no analyzer is listed for retirement here without a recorded REQ_111 validation outcome).
**Attribution:** Engineering Claude

---

## Problem Statement

Several analyzers have either been superseded by better approaches, never proved their value, or carry layering violations (per REQ_106) that cannot be cleanly migrated. Carrying them forward into the publishable library raises the maintenance burden and confuses external readers about which paths are canonical.

Three analyzers identified for deprecation by the user during discovery:

- **`coarseness`** — superseded by more specific frequency analyses.
- **`fourier_nucleation`** — exploratory, did not yield durable results.
- **`centroid_dmd`** — most of the value lives in two extractable pieces
  (Global / Trajectory PCA, eigendecomposition plot); the DMD framing
  itself is being retired.

The PCA consolidation (REQ_098) implicitly retires `effective_dimensionality`
as a standalone analyzer (its outputs become fields on `PCAResult`).
`dominant_frequencies` is similarly retired by REQ_097 (frequency cleanup).

REQ_106 introduces a third deprecation criterion: **layering-principle violations that cannot be migrated within the consolidation effort.** An analyzer that re-implements an upstream derivation, mixes data-plane access into measure code, or cannot conform to declared-dependencies discipline is a deprecation candidate if migration would amount to a rewrite. Audit before declaring; some violations are migrations, not retirements.

**Two retirement tracks** (clarified after REQ_111 carve-out):

- **No-replacement retirements** — analyzers with no canonical successor: `coarseness`, `fourier_nucleation`, exploratory paths in `centroid_dmd` not extracted. These can be retired here directly once the audit confirms no live consumers.
- **Migrate-track retirements** — analyzers whose canonical successor is being built in parallel under REQ_111: `dominant_frequencies` → new site-aware Fourier analyzer; `effective_dimensionality` → absorbed into PCA-result fields; `parameter_trajectory_pca`, `freq_group_weight_geometry`, `repr_geometry`, `neuron_group_pca`, `global_centroid_pca`, `centroid_dmd` (PCA paths) → consolidated PCA analyzers in REQ_111. **No migrate-track analyzer is listed for retirement here until its REQ_111 parity validation outcome is recorded** (matches / old-has-bug / new-has-bug-fixed / both-kept).

---

## Conditions of Satisfaction

### Extractions (do these BEFORE deletion)

- [ ] `centroid_dmd`: extract Global PCA and Trajectory PCA paths into
  appropriate library functions / analyzers. Decide whether they're
  consumed by an existing analyzer (e.g., `parameter_trajectory_pca`)
  or warrant a new one.
- [ ] `centroid_dmd`: extract the eigendecomposition plot. If it's still
  research-useful, port to a standalone view backed by a small library
  function. If not, retire with a note in this REQ's Notes.

### Retirements

- [ ] `coarseness` analyzer: removed from `analysis/analyzers/`,
  `registry.py`, `__init__.py`. Existing `coarseness` artifacts left in
  place (read-only legacy); loader continues to read them on request.
- [ ] `fourier_nucleation` analyzer: removed similarly.
- [ ] `centroid_dmd` analyzer: removed once extractions are complete.
- [ ] `dominant_frequencies` analyzer: removed after REQ_097's new
  analyzer + view migrations are in place.
- [ ] `effective_dimensionality` analyzer: removed after REQ_098's PCA
  consolidation absorbs its outputs into per-matrix PCA results.

### Layering audit (new — REQ_106 criterion)

- [ ] For each analyzer in the registry, run REQ_106's grep tests against its source: does its `analyze()` call `np.linalg.svd` directly? Does it inline filesystem path construction? Does it re-derive a field that exists upstream?
- [ ] For each violation, classify: **migrate** (rewrite to use the canonical primitive — e.g., move SVD to PCA library), or **retire** (the analyzer's value doesn't justify the rewrite).
- [ ] Migration candidates land under their owning REQ (PCA → REQ_098, frequency → REQ_097, geometry → REQ_104). This REQ owns the retire decisions.
- [ ] Layering audit results recorded in this REQ's Notes section as the audit completes.

### Cleanup

- [ ] Family configurations updated: each `family.json` removes references
  to deprecated analyzers from its `analyzers` list.
- [ ] `analysis/analyzers/__init__.py` no longer imports / exports
  deprecated analyzer classes.
- [ ] `analysis/analyzers/registry.py` no longer registers deprecated
  analyzer classes.
- [ ] Renderers / views that referenced deprecated analyzers either
  deleted or migrated to the canonical replacements.
- [ ] CHANGELOG entry on next release describing the retirement and pointing
  to replacements.

### Documentation

- [ ] `analysis/README.md` updated to reflect the canonical analyzer set.
- [ ] Each retired analyzer file (in git history) has a final commit
  with deprecation notice + pointer to replacement, before deletion.

---

## Constraints

**Must:**
- Extractions happen before retirement. No information loss in flight.
- Existing artifacts on disk remain readable. Researchers with old `results/`
  trees should not lose access to historical data — `ArtifactLoader` continues
  to load `coarseness` etc. if asked, even after the analyzer is gone.
- Family configurations stay valid throughout. No release with a
  configuration referencing a removed analyzer.

**Must avoid:**
- Silent retirement. Each retirement gets a CHANGELOG entry with a clear
  pointer to what supersedes it (or a note that nothing does, if the
  capability was abandoned).
- Removing analyzers whose outputs are still being read by views or
  notebooks. Audit the consumer surface before deleting.

**Flexible:**
- Order of retirement. Suggest: do extractions first, then retire
  `coarseness` and `fourier_nucleation` (no consumers), then `centroid_dmd`
  (after extractions land), then `dominant_frequencies` (after REQ_097),
  then `effective_dimensionality` (after REQ_098).
- Whether to keep deprecated-analyzer code around in a `deprecated/`
  subdirectory for one release before deletion. Default: no — git history
  is sufficient.

---

## Architecture Notes

**Audit before deleting.** For each candidate analyzer:
1. Grep the codebase for the analyzer name (analyzer string, class name).
2. Confirm no view, no `load_data`, no notebook consumes its artifacts.
3. Confirm the family configurations don't list it.
4. Run the REQ_106 layering grep tests to determine migrate-vs-retire.
5. Then delete (or migrate, with a follow-up commit under the owning REQ).

**`coarseness` and `fourier_nucleation` audit:** likely have no live consumers
based on user note. Verify before deleting.

**`centroid_dmd` audit:** has at least the eigendecomposition plot to
preserve. The Global / Trajectory PCA pieces likely overlap with what
`parameter_trajectory_pca` already does — verify before re-extracting to
avoid duplication.

---

## Notes

- This REQ is mostly subtraction and is low-risk if the audit step is
  honored. The deletion of `coarseness` and `fourier_nucleation` should
  happen first (safe, no extractions needed).
- The user expressed mild uncertainty about `centroid_dmd` ("maybe
  centroid_dmd, though I'd like to pull the Global and Trajectory PCA
  from that, as well as the eigendecomposition plot"). Treat as
  conditional: extract the keepers, then delete the rest.
- This REQ pairs naturally with REQ_103 (PyPI Publication Hardening) —
  the publishable library should not include retired analyzers.
