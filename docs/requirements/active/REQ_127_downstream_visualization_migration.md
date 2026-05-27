# REQ_127: Downstream Visualization Migration & DataView Layer

**Status:** Draft — *Problem statement only.* Conditions of satisfaction, constraints, and scoping decisions to be developed in a dedicated session.
**Priority:** High — gates REQ_102's analyzer retirements and is the necessary follow-on to REQ_126's three PRs. Until consumers port over, the absorbed analyzers stay live and the cleanup gain from REQ_126 is only partly realized.
**Branch:** TBD
**Dependencies:**
- REQ_126 (Family Basis Projection Consolidation — *staging*; this REQ migrates the consumers of the analyzers REQ_126 absorbed).
- REQ_106 (Analysis Layer Architecture — *active*; relevant to the DataView axis below).
- REQ_047 (View Catalog — already shipped; the `BoundView` pattern is the foundation this REQ extends).

**Hands off to:**
- REQ_102 (Analyzer Deprecation — *active*; once this REQ ports the consumers, the absorbed analyzers can retire). The REQ_126 gate checks already passed on canon; REQ_102's outstanding work is the cleanup itself, not further verification.

**Attribution:** Engineering Claude (drafted 2026-05-27 immediately after REQ_126 PR 3 shipped, while the consumer surface for the absorbed analyzers was fresh in context).

---

## Problem Statement

REQ_126 absorbed six analyzers and defused a fused field into a new dedicated analyzer:

- `dominant_frequencies`, `attention_fourier`, `neuron_fourier`, and `fourier_nucleation`'s one-shot projection → `weight_basis_projection`.
- `attention_freq`, `neuron_freq_norm` → `activation_basis_projection`.
- `repr_geometry.*_fourier_alignment` (fused field) → `centroid_fourier_alignment`.

The new analyzers are universal instruments parameterized by family-supplied sites; the old ones are Fourier-locked artifacts of an earlier design. REQ_126 deliberately left the old analyzers live and their on-disk artifacts intact: the user direction was that *downstream visualization churn may be hidden behind the visualization layer for now* and *dashboard / notebook migration is a separate cleanup track, not blocking.*

That carve-out keeps the REQ_126 ship small and honest, but it means the codebase now carries two overlapping representations of the same signal — the old per-analyzer artifacts that the dashboard, view catalog, and renderers consume, and the new per-site outputs that no consumer reads yet. The cleanup gain from REQ_126 is structural (Family Basis column is real; the abstraction is in place), but not yet realized in the surfaces a researcher interacts with.

The downstream cleanup also surfaces a second-order question. The renderers that consume analyzer artifacts have accreted a pattern: the first ~30 lines of each renderer cobble together data from one-or-more analyzer artifacts, then the remainder constructs the figure. With REQ_126's split — where a single dashboard panel may now source from two or three analyzer artifacts (e.g., `repr_geometry` centroids + `centroid_fourier_alignment` summary + the activation-side basis projection) — the cobbling cost is going up in lockstep with the analyzer granularity gain. The right fix may not be "rewire the renderers to read more artifacts"; it may be **add a DataView layer that composes analyzer artifacts into domain-meaningful structures before they reach a renderer**, and slim renderers down to pure plotting.

A related observation: as the renderers are touched anyway during this migration, some shared visual idioms become legible — eigenvalue-on-unit-circle plots (`activation_dmd` / `parameter_dmd`), per-site multi-panel time series, centroid 2D PCA scatter, residual-with-peak-markers. Whether these warrant extraction into shared visualization primitives is a tractable question to answer *while* the renderers are open.

The opportunity, then: REQ_126 gives the consumer-migration work a natural trigger and a fresh-context window. The pages and renderers that consumed the absorbed analyzers must be touched; while they're open, the DataView layer and shared-primitive questions become concrete rather than speculative. The risk is the trap of unbounded "let's refactor visualization" — discipline about which surfaces are in scope and which heuristics graduate to shared abstractions belongs in this REQ's scoping pass.

This REQ does **not** attempt to answer those scope questions in its problem statement. It captures the trigger (REQ_126's consumers are stale), the axes (consumer migration; possible DataView layer; possible shared visualization primitives), and the constraints implied by the dependencies (REQ_102 retirements gated on this REQ; REQ_047 / REQ_106 are the architectural neighbors). The Conditions of Satisfaction and scope discipline are deferred to a dedicated scoping session.

---

## Context: What changed and what's stale

**Live consumers of the absorbed analyzers (the immediate migration surface):**

- Renderers in [packages/miscope/src/miscope/visualization/renderers/](../../../packages/miscope/src/miscope/visualization/renderers/) — at minimum `repr_geometry.py`, `neuron_freq_clusters.py`, `attention_freq.py`, `dominant_frequencies.py`, `attention_fourier.py`, `neuron_fourier.py`, `fourier_nucleation.py`, `coarseness.py`. Each is a candidate either for migration to the new analyzer's output schema or for retirement once the dashboard layer no longer surfaces the corresponding view.
- View Catalog entries in [packages/miscope/src/miscope/views/universal.py](../../../packages/miscope/src/miscope/views/universal.py) and [packages/miscope/src/miscope/views/dataview_universal.py](../../../packages/miscope/src/miscope/views/dataview_universal.py) — `BoundView`s declare an `AnalyzerRequirement` set; entries that name the absorbed analyzers need to either be retired or re-pointed at the new analyzers.
- Dashboard pages under [apps/dashboard/src/dashboard/pages/](../../../apps/dashboard/src/dashboard/pages/) — page-level state and callbacks that load analyzer artifacts via `Variant.artifacts.load_*` and route to renderers. Specific pages that consume the absorbed analyzers were not enumerated in REQ_126; a survey pass belongs in this REQ's scoping.
- Notebooks under [apps/research/notebooks/](../../../apps/research/notebooks/) — research artifacts; per the REQ_126 carve-out, *signals that diminish after refactor are a finding worth surfacing, not a regression to mask.* These don't block the migration but may inform decisions about which renderers carry their weight.

**Already-graceful surfaces (degrade silently when the field disappears):**

- [packages/miscope/src/miscope/visualization/renderers/repr_geometry.py:257](../../../packages/miscope/src/miscope/visualization/renderers/repr_geometry.py#L257) reads `f"{s}_fourier_alignment"` from `summary_data` with an `if key in summary_data:` guard; the panel branch silently inactivates when the field is removed from new `repr_geometry` summaries. Re-wiring this to read `centroid_fourier_alignment.summary` is a representative micro-task.

**Float64 precision shift (already documented):**

REQ_126's new analyzers accumulate intermediates in float64 (REQ_109's primitives cast on entry) where the legacy analyzers accumulated in float32. The largest discrepancies appear at the highest frequencies (rapid basis oscillation amplifies float32 round-off). Documented inline in the new analyzers' test modules and in [feedback_req126_float64_parity](../../../../.claude/projects/-home-megano-projects-mechinterp-training-dynamics-workbench/memory/feedback_req126_float64_parity.md). Any *shape-of-behavior* change in renderers after migration is a real finding; last-few-bits discrepancies are not regressions.

---

## Scope considerations (to address in the dedicated session)

Two axes surfaced during REQ_126's close-out, in user direction (2026-05-27). Both belong in this REQ's scoping but are recorded here only as framing, not as decisions:

### 1. Shared visualization primitives

Reasonable candidates observed across the active renderers: eigenvalue unit-circle plot (`activation_dmd` / `parameter_dmd`); per-site multi-panel time series (`repr_geometry`, `freq_group_weight_geometry`); class-centroid 2D PCA scatter; residual-with-peak-markers from the DMD regime detector. Tension noted by the user: *Plotly has a tendency to be fidgety and may fight standardization.* The layout-level idioms (the unit circle frame, the panel grid) are likely stable enough to share; trace styling is brittle enough that forcing it would create more friction than it saves. Survey pass + selective extraction (1-3 primitives, not "standardize all visualization") is the scope discipline that surfaces naturally.

### 2. DataView layer

User observation (2026-05-27): *With Analyzers becoming more granular, I'm wondering if it makes sense to start creating dataviews for visualizers instead of cobbling together the data within the visualization functions.*

The current renderer pattern is "load artifacts → cobble data → render figure" with the cobbling growing as analyzers split. A DataView abstraction would: read from one-or-more analyzer artifacts; compose into a domain-meaningful structure (e.g., a `RepresentationGeometryView` that holds centroids + circularity + fourier alignment for a site); hand the structure to a renderer that does pure plotting. This extends the REQ_047 `BoundView` pattern downward into composition rather than just routing.

The two axes interact. **Axis 2 done first reduces the case for Axis 1**: if data composition is centralized in a DataView, individual renderers become smaller and the case for sharing layout primitives shrinks. Ordering may matter — the scoping session should consider whether to lead with the DataView pass and let primitive extraction fall out as a follow-on.

The trap to avoid: open-ended "refactor visualization." A tight first pass — pick 2-3 dashboard pages most touched by REQ_126's analyzer changes, write DataViews for them, validate the abstraction holds — gives the evidence needed before extending. Pages that consumed the absorbed analyzers (and whose renderers are already on the migration list) are natural first candidates.

---

## Notes

- The trigger for this REQ is the REQ_126 carve-out, but the surface it touches is broader than just "consumers of the absorbed analyzers." If the DataView axis is included, the design ripples into surfaces that weren't part of REQ_126 (e.g., `repr_geometry`'s dashboard page consumes `repr_geometry` artifacts already; a DataView for that page would touch them even though they weren't absorbed). Scope discipline matters: which surfaces are in for this REQ, which are deferred.
- The `repr_geometry → representation_geometry` rename is *not* this REQ. It's a separate REQ_111-style pure-rename PR (86-file ripple; bookkeeping). Could land before, during, or after this REQ; doesn't gate on it.
- REQ_102's close-out blocks on this REQ landing far enough that the absorbed analyzers no longer have live consumers. "Far enough" is a judgment call for REQ_102 — it doesn't necessarily require *every* consumer migrated, but the user-visible surfaces (dashboard, fieldnotes-published figures) likely do.
- Possible interaction with REQ_110 (Lakehouse Surface): REQ_110's `to_wide(...)` and DataFrame contracts are the *publication* face of analyzer data; the DataView discussed here is the *visualization* face. They're related but distinct surfaces — confirm during scoping whether they share a substrate or coexist.
