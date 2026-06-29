# REQ_155: Dashboard views for circuit spectra (OV/QK copy-transform & rank dynamics)

**Status:** Complete on `feature/REQ_155_deferred_views` (awaiting merge). The MVP
(trajectory view + page) merged earlier; this branch finishes the three deferred
follow-ups: the **copy/transform head-ranking bar** (`circuits.spectra.ranking`),
the **circuit-matrix heatmap** (`circuits.spectra.matrix`, resolved through the
tensor catalog), and the **cross-variant overlay** (`miscope.views.circuit_spectra`
prep + `render_circuit_spectra_cross_variant`). All renderers are plot-only; the
dashboard Circuit Spectra page now wires site / metric / matrix-head dropdowns plus
a Load-Overlay control (data-seed / model-seed peers, event-relative on grok).

One **OPEN** item from CoS line 50–53: the per-head dominant-frequency
cross-reference (the 2+2 pairing) is **not** drawn on the ranking bar.
`weight_basis_projection.dominant_frequency` is head-keyed but lives on the Fourier
sites (resid_pre / attn_out / …), not the circuit sites — mapping "the head's
dominant frequency" onto a circuit site is a semantic-coupling decision, deferred
rather than guessed. The copy/transform split itself (the load-bearing CoS) is
fully legible without it.
**Priority:** Medium — makes the new Layer 4 circuit data (REQ_152/154) explorable;
the underlying findings (OV copy/transform split, QK→rank-1 sparsening, two-timescale
decoupling) are currently only visible via ad-hoc queries.
**Branch:** TBD (`feature/REQ_155_circuit_spectra_views`).
**Attribution:** Engineering Claude (under user direction).
**Depends on:** REQ_152 (`circuit_spectra` analyzer, `full_ov`), REQ_154 (generic
instrument + `full_qk`/`ov`/`qk`/`direct_path` sites).

---

## Problem Statement

The `circuit_spectra` analyzer now materializes, per head and per training epoch, the
spectral invariants of five composed circuits (`full_ov`, `full_qk`, `ov`, `qk`,
`direct_path`): `copying_score`, `effective_rank`, `operator_norm`, plus the composed
matrix and eigenvalues as tensors. This data already exposes two findings
(`finding-ov-copy-transform-head-split`):

1. An **emergent copy/transform head split** on the OV side, crystallizing at grok.
2. A **two-timescale decoupling**: QK effective_rank collapses to ~1 (sparse,
   single-frequency) *early/pre-grok*, while OV stays diffuse (rank ~4–5) and its
   copy/transform roles differentiate *at grok*.

But none of this is visible in the platform — it lives only in ad-hoc query scripts.
The View Catalog (REQ_047, universal instruments) and the Dashboard are where this
should surface so the dynamics are explorable across variants without re-deriving.

## Conditions of Satisfaction

- [ ] **Per-head circuit-spectra trajectory view** (universal, View-Catalog
  registered): a metric (`copying_score` / `effective_rank` / `operator_norm`) for a
  chosen circuit `site`, plotted per head over training epochs, with a **grok marker**
  (test-loss crossing). This is the view that shows the emergence/decoupling directly.
  Reachable as `variant.at(epoch).view(...)` per the catalog contract (`.figure()` /
  `.show()` / `.export()`), keyed off the warehouse `circuit_spectra` table — not
  re-reading artifacts.
- [ ] **QK-vs-OV decoupling view**: effective_rank of `full_qk` vs `full_ov`
  (head-mean or per-head) over training on shared axes, the headline two-timescale
  story (QK→1 early, OV stays diffuse). May be a parameterization of the trajectory
  view rather than a separate renderer.
- [x] **Copy/transform head ranking** at a selected epoch: per-head `copying_score`
  for `full_ov` (bar / sorted), so the copy↔transform split at a checkpoint is
  legible. **Done** (`circuits.spectra.ranking`, sorted descending, true head index
  preserved on each bar). The dominant-frequency cross-reference is **OPEN** (see
  Status — semantic-coupling deferral, not built).
- [x] **Circuit-matrix heatmap** (optional within this REQ): the composed `(p, p)`
  `full_ov` / `full_qk` / `direct_path` matrix at an epoch, resolved via the tensor
  catalog (demonstrates the columnar→blob hop in a consumer view). **Done**
  (`circuits.spectra.matrix`; `_load_circuit_matrix` selects descriptors then
  resolves only those — no payload touched for the metadata filter).
- [x] **Dashboard page** surfacing the above with the standard left-nav controls
  (variant, epoch, circuit site, metric, head selection), following the existing
  analysis-page template. **Done** — trajectory + ranking + matrix graphs + a
  cross-variant overlay section. **Export caveat:** the per-graph export reuses the
  shared `export_panel` (the REQ_150 stub) which exports BoundView defaults, not the
  current left-nav `view_parameter`. This REQ does **not** add export logic, so it
  inherits — does not re-introduce — the REQ_150 limitation; the proper fix is
  REQ_150's (export must read live left-nav state).

## Constraints

- **Universal instrument (invariant 1).** The views are universal — they take a
  circuit `site` + metric as parameters; they do not bake in `full_ov`. A renderer
  works for any `circuit_spectra` site.
- **Read through the API (invariant 3).** Views read the warehouse `circuit_spectra`
  table via `miscope.query` / `variant.warehouse` and resolve tensors via the tensor
  catalog — no artifact-path literals, no direct `ArtifactLoader`.
- **Grok marker is task/variant context.** The test-loss crossing used for the grok
  overlay comes from the `losses` table per variant — supplied as context, not
  hardcoded per prime.
- **Plot-only renderers (REQ_099 line) — load-bearing here.** Renderers return
  figures from *already-prepared* data; **no data-processing logic hidden in the
  view.** Aggregation, joins, the grok-epoch lookup, head-mean reductions, and any
  derived metric live in the query / view-data layer (or a derived table), not inline
  in the renderer. **When modeling a new view on an existing one, check how much
  processing that reference does — a viz that does a lot of in-renderer processing is
  NOT the right pattern to copy.** Prefer a thin renderer reading a conformed query.

## Notes

- Strongest single visual: per-head `copying_score(full_ov)` and
  `effective_rank(full_qk)` over training with the grok line — the two-timescale story
  in one panel. Good fieldnotes figure once built (the deferred capture from REQ_152).
- Cross-variant comparison (p113/p109/p101 on shared axes, event-relative time) is a
  natural extension but can follow; this REQ targets the per-variant views + the
  dashboard page first.
- Data is per-head; `direct_path` is head-less (single series) — the view must handle
  both (uniform-rank head axis, head=0).
- **Tensor-catalog staleness found (and locally fixed).** The tensor catalog for
  the baselines still indexed the **old REQ_152 analyzer name** `full_ov_circuit`,
  not the renamed `circuit_spectra` — so `circuits.spectra.matrix` found **0**
  `circuit_matrix` descriptors until the catalog was re-materialized. The columnar
  warehouse was refreshed by REQ_154/156 but `tensor_catalog.materialize()` was not:
  it is **not on the REQ_145 freshness signature** (same class as
  `finding-site-addition-not-in-refresh-signature`). Cheap fix applied to the three
  baselines (header reads only, no recompute); the **mass refresh is REQ_137's job**.
  Worth folding tensor-catalog materialization into the REQ_145 signature so an
  analyzer rename invalidates it automatically.
- **copying_score is OV-meaningful only.** For QK circuits it is degenerate (rank-1,
  non-positive-real eigenvalues → pinned ~0). Resolved in the **view** (QK sites
  default to `operator_norm`), NOT in the analyzer: a value-based NaN guard was
  considered and **rejected** — its trigger (copying ≈ 0) is exactly a legitimate OV
  *pure-transform* head (p113 h1 = 0.001, p109 h3 = 0.004, p101 h2 = 0.032), so it
  would erase the transform end of the copy/transform finding. There is no
  divide-by-zero in `_copying_score` (the earlier "spike" was a diagnostic CV
  artifact, not the metric). The renderer is `connectgaps=False`, so a future
  **per-site metric applicability** mechanism (mark copying_score N/A on QK sites —
  the correct analyzer-level honesty fix, logged as a small future item, not built)
  would surface as gaps without reopening this REQ.
