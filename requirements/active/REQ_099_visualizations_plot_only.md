# REQ_099: Visualizations Plot-Only (No Computation in Renderers)

**Status:** Draft
**Priority:** Medium
**Branch:** TBD
**Dependencies:** REQ_106 (layering principle — REQ_099 is the migration mechanism that enforces the rule on the renderer/loader side); REQ_097 (frequency primitives), REQ_098 (PCA primitives) — both need to be available so renderer-side compute has somewhere to migrate to.
**Attribution:** Engineering Claude

---

## Problem Statement

Several view renderers and `load_data` callbacks perform analytical computation
inline — Fourier projections, PCA, geometry calculations. This has three costs:

1. **Render slowdown.** Computation runs every time the dashboard loads the view,
   not once at analysis time.
2. **Data quality risk.** A computation done in the renderer is not the same
   path that produces an artifact, so two callers (dashboard vs notebook) can
   silently diverge.
3. **Audit difficulty.** "Where does this number come from?" has multiple
   answers if rendering, loaders, and analyzers all compute things.

A clean separation of concerns: analyzers compute and persist; library functions
compute on demand from already-loaded artifacts (deterministic, cacheable);
loaders fetch and shape; renderers plot. Each layer has one responsibility.

---

## Conditions of Satisfaction

### Audit

- [ ] All view renderers under `src/miscope/visualization/renderers/` audited
  for compute calls (`np.linalg.*`, `sklearn.*`, manual SVD/eigendecomp,
  in-renderer Fourier projection, etc.).
- [ ] All `load_data` callbacks in `src/miscope/views/universal.py` and
  `dataview_universal.py` audited similarly.
- [ ] Audit results documented (list of offending sites + planned migration
  target). Lives in this REQ's Notes section after audit completes.

### Migration

- [ ] All renderer-side computation migrates either to:
  - A library function (returns a dataclass / array) called from `load_data`, or
  - An analyzer that persists the result as an artifact.
- [ ] `load_data` callbacks are allowed to call library functions for
  on-the-fly computation (e.g., rolling window metrics over an existing
  artifact). They are not allowed to perform numerical work inline.
- [ ] Renderer functions take only structured data + render parameters.
  No numpy linear algebra, no sklearn, no Fourier basis construction.

### Validation

- [ ] After migration, view rendering time on a representative variant
  improves measurably (target: 25%+ reduction on views that had compute
  in the renderer; specific numbers documented post-audit).
- [ ] Visual output of migrated views unchanged within tolerance (verified
  against current screenshots or characterization renders).

---

## Constraints

**Must:**
- Renderers remain pure presentation. The simple test: `import` of
  `numpy.linalg`, `sklearn`, or any miscope library function that returns
  more than formatted plot data should not appear in renderer modules.
- `load_data` callbacks may call library functions. They may not perform
  inline numerical work. The distinction: library calls are auditable to
  one canonical path; inline numpy is bespoke.

**Must avoid:**
- Pushing computation into "helper" functions inside renderer modules to
  evade the rule. Helpers move to `library/` if they exist.
- Breaking visual outputs. Any migrated view requires a side-by-side check
  against the prior render before the migration is considered done.

**Flexible:**
- Whether some computation moves to `load_data` (still on-demand) vs an
  analyzer (cached). Decision per case: cache if expensive and cross-variant
  reusable; on-demand library call if cheap and parameter-dependent.
- Order of migration. Suggest tackling renderers with the largest compute
  surface first (e.g., the centroid PCA inside
  `render_weight_geometry_centroid_pca`).

---

## Architecture Notes

**The three-layer rule (subsumed by REQ_106):**

```
analyzer (persists)  →  library (cached compute on artifacts)  →  loader (fetch + shape)  →  renderer (plot)
```

A renderer that needs a number it doesn't have should walk back the chain:
ask the loader; if loader can't shape it, ask library; if library doesn't
have the path, build the analyzer. Never compute inline.

REQ_106 supersedes the principle statement and generalizes it across the analysis surface (data plane / derivation / measure). REQ_099 remains the *migration* mechanism specific to the renderer/loader boundary — it enforces the principle for visualization-side code where the rule was originally articulated.

**Known offenders (preliminary, pre-audit):**
- `render_weight_geometry_centroid_pca` calls `compute_global_centroid_pca`
  inline (visible in `notebooks/dataview_analysis.ipynb`).
- `dimensionality.timeseries` view's `load_data` (per REQ_096) computes
  rolling PR₃ and class-centroid PR₃ inline. Acceptable per the rule
  (loader-side library calls), but worth confirming the library path is
  the canonical one after REQ_098.

---

## Notes

- This REQ is mostly subtraction work. Each migration is small; the audit is
  the load-bearing step.
- Renders that currently feel slow are good audit-prioritization signals.
- This REQ depends on REQ_097 and REQ_098 because some computation can only
  migrate once the canonical library functions exist.
