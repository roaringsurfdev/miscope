# REQ_155: Dashboard views for circuit spectra (OV/QK copy-transform & rank dynamics)

**Status:** MVP implemented on `feature/REQ_155_circuit_spectra_views` (awaiting
merge). Done: the per-head circuit-spectra **trajectory view** (`circuits.spectra.trajectory`,
plot-only, grok marker) parameterized by `site` + `metric` — which covers the
trajectory CoS *and* the QK-vs-OV decoupling (switch site/metric) — plus the
**Circuit Spectra dashboard page** (circuit + metric dropdowns). Deferred follow-ups:
the dedicated copy/transform head-ranking bar, the circuit-matrix heatmap (tensor
resolve), and cross-variant overlays.
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
- [ ] **Copy/transform head ranking** at a selected epoch: per-head `copying_score`
  for `full_ov` (bar / sorted), so the copy↔transform split at a checkpoint is
  legible; cross-reference the head's dominant frequency (the 2+2 pairing) to show the
  axes are orthogonal.
- [ ] **Circuit-matrix heatmap** (optional within this REQ): the composed `(p, p)`
  `full_ov` / `full_qk` / `direct_path` matrix at an epoch, resolved via the tensor
  catalog (demonstrates the columnar→blob hop in a consumer view).
- [ ] **Dashboard page** surfacing the above with the standard left-nav controls
  (variant, epoch, circuit site, metric, head selection), following the existing
  analysis-page template. Respect `view_parameter` from the left nav on export
  (don't repeat the REQ_150 export-default bug).

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
