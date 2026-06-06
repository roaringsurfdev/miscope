# REQ_099: Visualizations Plot-Only (No Computation in Renderers)

**Status:** Draft — **audit complete (2026-06-05)**, findings in Notes. Execution (migration) recommended **after REQ_110 merges to `develop`**, so the migration targets are stable. Not bundled into the REQ_110 branch (keeps that branch's smoke-test surface frozen, and several targets only stabilize once 110 lands).
**Priority:** Medium
**Branch:** TBD
**Dependencies:** REQ_106 (layering principle — REQ_099 is the migration mechanism that enforces the rule on the renderer/loader side). **REQ_109** (measurement primitives) and **REQ_110** (warehouse + DuckDB query surface) — these now provide the canonical migration *targets* and **subsume the earlier REQ_097/098 dependency**: a derived metric keyed by coordinates (proximity, rolling PR₃, per-band counts) now has a natural home as a warehouse column / query-surface aggregation, not only a library function. **REQ_107** (registry/discoverability) — before re-homing an offender into a new library function, check whether the warehouse already computes it (e.g. 110-D's conformed `(epoch, neuron) → freq` dimension), so the migration *joins* an existing field rather than re-deriving it.
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

- [x] All view renderers under `src/miscope/visualization/renderers/` audited
  for compute calls (`np.linalg.*`, `sklearn.*`, manual SVD/eigendecomp,
  in-renderer Fourier projection, etc.). *(2026-06-05; results in Notes.)*
- [x] All view renderers under `src/miscope/visualization/renderers/` audited
  for imports of `miscope.analysis.library` in order to re-compute data that is already stored as pre-computed artifacts.
- [x] All `load_data` callbacks in `src/miscope/views/universal.py` and
  `dataview_universal.py` audited similarly.
- [x] Audit results documented (list of offending sites + planned migration
  target). Lives in this REQ's Notes section.

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

**Migration targets shift after REQ_110.** The original rule offered two homes —
a library function, or an analyzer artifact. REQ_109/110 add a third and now-preferred
home for derived metrics that are *keyed by coordinates*: a **warehouse column /
query-surface aggregation**. Several offenders below are exactly that shape
(proximity per epoch+pair, rolling PR₃ per epoch+site, per-band committed-neuron
counts per epoch), and one (`multi_stream`) re-derives a dimension 110-D already
conformed. Re-home via the REQ_107 registry: check `reg.search(...)` / `reg.field(...)`
before writing a new library function, so the renderer *joins* an existing field.

---

## Notes

- This REQ is mostly subtraction work. Each migration is small; the audit is
  the load-bearing step.
- Renders that currently feel slow are good audit-prioritization signals.
- ~~This REQ depends on REQ_097 and REQ_098~~ — superseded: REQ_109 (measurement
  primitives) + REQ_110 (warehouse/query surface) are the canonical homes now.

### Audit results (2026-06-05)

Grep across `visualization/renderers/*.py` (30 files) and the two loader modules
(`views/universal.py`, `views/dataview_universal.py`) for `np.linalg.*`, `sklearn`,
SVD/eig, FFT, and `miscope.analysis.library` imports that return analytical results.

**Renderer-side compute (true offenders — renderers must be plot-only):**

| Site | What it computes inline | Migration target (REQ_110-aware) |
|---|---|---|
| `parameter_trajectory.py` ~L775 (group-proximity view) | pairwise group-trajectory L2 distance with sign-flip correction (`min(‖a−b‖, ‖a+b‖)`) per epoch | derived measure keyed by `(epoch, group-pair)` → **warehouse column** or library fn; renderer plots it |
| `dimensionality_dynamics.py` L51–97, L183 (`_compute_pr3`, `_compute_rolling_trajectory_metrics`) | rolling-window PR₃ + f_top3 over PC-space projections — *helper-inside-renderer* anti-pattern | PR₃ is a REQ_109 primitive; rolling metric keyed by `(epoch, site)` → **warehouse column** / loader-library call |
| `repr_geometry.py` L13, L810 | imports `library.pca.pca` + `library.geometry.compute_fisher_matrix`; runs PCA + Fisher in the renderer | move to `load_data` (loader→library) or read the `repr_geometry` artifact/warehouse |
| `multi_stream_specialization.py` L31–95 (`_compute_mlp_band_counts`, `_compute_attn_aggregate`, `_compute_embedding_dim_counts`) | per-frequency committed-neuron counts, mean-QK Fourier fraction, embedding-dim counts | **overlaps 110-D**: counts over the conformed `(epoch, neuron) → freq` dim are a `GROUP BY` on `neuron_frequency_attribution` — join, don't re-derive (check REQ_107 registry first) |
| `effective_dimensionality.py` L13, L151 | renderer imports + calls `compute_participation_ratio(sv)` | cheap REQ_109 primitive; move the call to the loader |

**Loader-side inline numpy (lower priority — loaders may call libraries but not do inline numerical work):**

| Site | Inline work | Disposition |
|---|---|---|
| `views/universal.py` L143–147 | `np.linalg.norm` to band-normalize attn V magnitudes | confirm whether this is *shaping* (acceptable) vs analytical derivation; if the latter, move to a library fn |
| `views/universal.py` L166–167 (`_adapt_embedding_coefficients_legacy`, REQ_127) | `np.linalg.norm` reconstructing the legacy `coefficients` shape | likely acceptable shaping (legacy-format adapter); flag for confirmation, low priority |

The loader layer is otherwise clean: `universal.py` already routes real compute
through library functions (`compute_summary_from_dynamics`,
`compute_band_concentration_trajectory`, `compute_data_compatibility`) — those are
compliant (loader → library). `dataview_universal.py` showed no inline-numpy hits.

The earlier pre-audit guess (`render_weight_geometry_centroid_pca` calling
`compute_global_centroid_pca` inline) was **not** found in the current renderers —
appears already migrated; verify no residual in notebooks before closing.

### Sequencing recommendation

Execute **after REQ_110 merges to `develop`** (audit is captured above; migration is
deferred), because: (1) it keeps REQ_110's pre-merge smoke test — which exercises the
visualization layer heavily — from validating a moving target; (2) it respects branch
atomicity (110 = build the surface; 099 = migrate compute off renderers); (3) the
preferred targets (warehouse columns, the 110-D conformed dimension, the query
surface) are only stable once 110 lands. Prioritize the renderer offenders
(largest first: `multi_stream` and `dimensionality_dynamics`, both of which most
benefit from the 110-D / warehouse home); the loader-side norms are a confirm-and-
maybe-leave pass.
