# REQ_104: Geometry Consolidation (Curvature, Sigmoidality, Circularity, Jerk)

**Status:** Superseded — primitive scope landed in REQ_109; analyzer-integration scope tracks under REQ_111
**Priority:** Medium
**Branch:** TBD
**Dependencies:** REQ_106 (layering principle — geometry functions are *measures*, not derivations); REQ_098 (PCA primitives — geometry consumes PCA output, not raw matrices).
**Attribution:** Engineering Claude

> **Superseded by REQ_109 (Measurement Primitives Library).** This REQ's scope (curvature, sigmoidality, circularity, jerk, arc length, self-intersection, loop area, Lissajous, Procrustes — consolidated under one geometry library) was folded into REQ_109's "shape characterization" and "shape comparison" categories. The layering-violation framing (geometry functions silently re-fitting their own PCA) is preserved in REQ_109 verbatim — the fix is the same: the caller materializes `PCAResult` once; characterization functions consume it. The "fit" → "characterize" / "describe" rename also originated from feedback on this REQ. Primitive scope landed under REQ_109; the caller-side migration scope below (updating callers to consume the new primitives, renderer-side migration coordinated with REQ_099, removal of unused exports in `trajectory.py`) was carved into **REQ_111 (Parallel Analyzer Build-Out)** and tracks there.

---

## Problem Statement

Geometric calculations on trajectories — curvature, arc length, self-intersection,
loop area, sigmoidality, circularity, jerk — are scattered across several
locations:

- `analysis/library/trajectory.py` mixes trajectory PCA and lemniscate-style
  geometry (the file already has a `# TODO: Move to Geometry` comment).
- `analysis/library/geometry.py` has a separate set of geometry helpers
  (centroids, radii, Fisher discriminant, circularity).
- `analysis/library/manifold_geometry.py` has another set.
- Some renderers compute geometry inline (REQ_099 audit will catch these).

These are all *shape-of-trajectory* operations, distinct from PCA (which
produces the trajectory's coordinate frame) and from frequency analysis
(which is spectral, not geometric). They deserve a coherent, single-purpose
library module.

**Layering violation (REQ_106 framing).** Several existing geometry functions silently fit their own derivations: [`compute_circularity(centroids)`](../../src/miscope/analysis/library/geometry.py#L141) calls `_pca_project_2d(centroids)` internally; [`compute_fourier_alignment(centroids, p)`](../../src/miscope/analysis/library/geometry.py#L177) does the same. With `compute_summary` separately re-fitting PCA on the same centroids, that's three SVDs per (epoch, site) on the same matrix. These functions are *measures* but they are doing *derivation* work — they should accept the projection as input, not produce it.

The consolidation under this REQ is the moment to fix that. Functions move under REQ_106's measure-takes-typed-input rule: the caller materializes the `PCAResult` once; the geometry function consumes it.

---

## Conditions of Satisfaction

### Library

- [ ] `miscope/analysis/library/geometry.py` is the single home for
  trajectory and shape geometry, organized by concept:
  - **Arc-length & curvature**: `compute_arc_length`,
    `compute_curvature_profile`.
  - **Self-intersection & loops**: `detect_self_intersection`,
    `compute_signed_loop_area`.
  - **Shape descriptors**: `compute_circularity`, `compute_sigmoidality`
    (new — promotes from saddle/transit work), `compute_jerk` (new —
    third-derivative measure for transit detection).
  - **Class / group geometry**: existing `compute_class_centroids`,
    `compute_class_radii`, `compute_class_dimensionality`,
    `compute_fisher_discriminant`, `compute_center_spread` remain.
- [ ] `analysis/library/trajectory.py` reduced to trajectory-specific
  helpers only (`flatten_snapshot`, `compute_parameter_velocity`).
  Lemniscate geometry moves out.
- [ ] `analysis/library/manifold_geometry.py` audited; either folded into
  `geometry.py` or scoped to a clearly distinct concern (manifold-level vs
  trajectory-level).

### Layering compliance (REQ_106)

- [ ] Every geometry function takes typed structured inputs and does no derivation. Specifically:
  - `compute_circularity(projection_2d, var_explained)` — accepts the PCA projection and explained-variance ratio; does not call `_pca_project_2d` internally.
  - `compute_fourier_alignment(projection_2d, p)` — accepts the projection; does not call `_pca_project_2d` internally.
  - All other shape descriptors take typed inputs (arrays of known shape and meaning), not raw centroids that they re-derive from.
- [ ] Acceptance test: grep `_pca_project`, `_pca_project_2d`, `np.linalg.svd`, `np.linalg.eigh` inside `library/geometry.py` returns zero hits. PCA happens in REQ_098's primitive; geometry consumes its output.
- [ ] Each geometry function has at least one unit test calling it with `np.array(...)` literals and asserting a known-correct numeric result. (See REQ_106's measure-test rule.)

### Migration

- [ ] All callers of geometry functions updated to import from the
  consolidated `geometry.py`. Backward-compatible re-export shims in
  `trajectory.py` until callers migrate, then removed.
- [ ] Callers that previously passed raw centroids to `compute_circularity` / `compute_fourier_alignment` now compute the PCA projection once (via REQ_098's primitive) and pass it. The single PCA fit replaces three.
- [ ] Renderer-side geometry calculations migrated under REQ_099.

### Validation

- [ ] Numerical equivalence with current outputs for the migrated
  functions on at least one reference trajectory per metric.
- [ ] No unused exports remain in `trajectory.py` after migration.

---

## Constraints

**Must:**
- All geometry functions in one place. The audit question "where does
  curvature come from?" has one answer.
- Functions remain pure: take arrays, return arrays / floats. No I/O,
  no caching (geometry is cheap; if a specific operation gets expensive,
  cache via `@cached_artifact` at the analyzer level, not in the library).

**Must avoid:**
- Coupling geometry to PCA. Geometry operates on the *output* of PCA
  (typically a 2D or 3D trajectory); the input shape and origin are
  the caller's concern.
- Adding new geometry primitives without a clear research use case.
  This REQ consolidates what exists; new primitives come from research
  needs, not speculative API growth.

**Flexible:**
- Internal organization of `geometry.py` (single flat module vs sub-modules).
  Default: single module with section headers (matches current style).
- Whether `manifold_geometry.py` survives as a distinct module. Default:
  fold into `geometry.py` unless there's a clear conceptual boundary.

---

## Architecture Notes

**The natural three-layer pipeline:**

```
PCA library  →  geometry library  →  shape descriptors / metrics
(coordinate frame)   (shape ops)         (curvature, jerk, ...)
```

Geometry consumes PCA output. PCA does not consume geometry. Keeping the
modules separate makes the data flow obvious. This is the REQ_106 layering rule applied to a specific module: the *measure* (geometry function) takes the *derivation* (PCA result) as input and returns a scalar. No layer-crossing.

**Sigmoidality and jerk are new (research-driven).** The user's saddle-transit
research has produced two new quantities:
- *Sigmoidality*: a measure of how S-shaped a transit trajectory is, used
  in the on-arc / off-arc decomposition (saddle transport analysis).
- *Jerk*: the third derivative of position (rate of change of acceleration),
  expected to spike during regime transitions.

Both should land in the consolidated geometry module with clear docstrings
about their research origin.

---

## Notes

- This REQ is lower priority than REQ_097 / REQ_098 / REQ_100 / REQ_103
  because it's mostly housekeeping. Sequence it after the load-bearing REQs.
- The `# TODO: Move to Geometry` comment in `trajectory.py:115` is the
  marker for one of the explicit migrations.
- Pairs naturally with REQ_098 — once PCA returns `PCAResult` consistently,
  geometry functions can take `PCAResult.projections` as input directly.
