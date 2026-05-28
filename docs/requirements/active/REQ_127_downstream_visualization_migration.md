# REQ_127: Downstream Visualization Migration & DataView Layer

**Status:** Scoped (2026-05-27) — *Two-phase plan. Phase A (page/renderer deletions + 1-to-1 view re-points) unblocks bulk of REQ_102 retirements; Phase B (single composition object for `*_fourier_alignment`) unblocks the remainder. `effective_dimensionality` migration folded in per user direction.*
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

## Scoping decisions (2026-05-27 session)

A scoping pass on 2026-05-27 ran the consumer-surface audit, resolved the three axes from the problem statement, and folded in dramatic dashboard-page deletions per user direction. Outcomes:

**Axis 1 — Consumer migration:** In scope. Surface significantly reduced by deleting four legacy pages outright (see Phase A).

**Axis 2 — DataView layer:** In scope, but **anti-designed**. Only one true composition case survives the scope reduction (`*_fourier_alignment` panel reading from `repr_geometry` centroids + `centroid_fourier_alignment` summary). It ships as one concrete dataclass — no abstract base class, no registry, no protocol. The existing `dataview_universal.py` / `dataview_catalog.py` are treated as experimental and do not constrain the design; they may be removed if doing so reduces noise. The discipline: build the object the renderer needs, validate it works, generalize only when a second concrete case proves the shape worth lifting.

**Axis 3 — Shared visualization primitives:** Dropped from this REQ. With Phase A reducing the renderer set substantially and Phase B containing only one composition object, there is no concentration of redundant idioms left to extract. Re-evaluate post-REQ_127 if any actually accumulates.

**Class centroid trajectories preservation:** *Updated 2026-05-27: clarified during implementation that the class-centroid-trajectories rendering was bundled into `geometry.dmd_reconstruction` (one of the three centroid_dmd views). Per user direction, none of the three centroid_dmd views needs catalog preservation; a clean centroid-trajectories view (independent of `centroid_dmd` analyzer) will be created in a future REQ.*

**`effective_dimensionality` (REQ_111-gated) migration:** Folded into this REQ per user direction so the analyzer can retire alongside the REQ_126-gated set.

**Notebooks / sketches / fieldnotes:** Per the REQ_126 carve-out, signals that diminish in research notebooks after the migration are findings worth surfacing, not regressions to mask. Fieldnotes is clean (no published-figure references to any retiring analyzer). Notebooks and sketches are explicitly out of scope.

---

## Conditions of Satisfaction

### Phase A — Page/renderer deletions + 1-to-1 view re-points

**Dashboard page deletions** (with nav/registration cleanup in `apps/dashboard/src/dashboard/app.py` and `components/sitenav.py`):
- [ ] `apps/dashboard/src/dashboard/pages/summary.py` deleted.
- [ ] `apps/dashboard/src/dashboard/pages/visualization_archive.py` deleted.
- [ ] `apps/dashboard/src/dashboard/pages/centroid_dmd.py` deleted.
- [ ] `apps/dashboard/src/dashboard/pages/neuron_dynamics_archive.py` deleted.

**Renderer deletions** (no surviving consumer after page deletions):
- [ ] `packages/miscope/src/miscope/visualization/renderers/coarseness.py` deleted.
- [ ] `packages/miscope/src/miscope/visualization/renderers/neuron_fourier.py` deleted.

**View catalog cleanup — orphans from the above deletions** (in `packages/miscope/src/miscope/views/universal.py`):
- [ ] Remove view registrations: `activations.mlp.coarseness_distribution`, `coarseness_by_neuron`, `coarseness_trajectory`, `blob_count_trajectory`, `neuron_fourier_heatmap`, `neuron_fourier_heatmap_output`, `dominant_frequencies_over_time`, `head_frequency_range`.
- [ ] Remove the `_dmd_req` block and all three dependent views (`geometry.dmd_eigenvalues`, `geometry.dmd_residual`, `geometry.dmd_reconstruction`) feeding the deleted `centroid_dmd.py` page. *Updated 2026-05-27: the class-centroid-trajectories view was combined into `geometry.dmd_reconstruction`; per user direction, none of the three views needs catalog preservation. A clean centroid-trajectories view will be created in a future REQ.*

**View re-points (loader swaps to new analyzers):**
- [ ] `parameters.embeddings.fourier_coefficients` → reads from `weight_basis_projection` (W_E site).
- [ ] `parameters.attention.qk_fourier_heatmap`, `v_fourier_heatmap`, `head_alignment_trajectory` → `weight_basis_projection` (attention sites).
- [ ] `activations.mlp.neuron_frequency_clusters`, `neuron_freq_distribution`, plus the 6 inline imports in `universal.py` for `neuron_freq_clusters` threshold-driven views → `activation_basis_projection` (MLP site).
- [ ] `activations.attention.head_frequency_clusters`, `frequency_clusters` → `activation_basis_projection` (attention sites).
- [ ] `parameters.effective_dimensionality`, `parameters.singular_value_spectrum` → `weight_spectra`.
- [ ] `multi_stream_specialization` composite view's internal loads of `attention_fourier` + `effective_dimensionality` → both replaced.
- [ ] `analysis.band_concentration.rank_alignment` view's internal `dominant_frequencies` load → `weight_basis_projection`.

**Renderer disposition** (per re-point, decide: port to new artifact schema, or retire in favor of the new analyzer's renderer):
- [ ] `dominant_frequencies.py`: ported or retired.
- [ ] `attention_fourier.py`: ported or retired.
- [ ] `attention_freq.py`: ported or retired.
- [ ] `neuron_freq_clusters.py`: ported or retired. Clustering presentation logic stays in the renderer; not hoisted into a DataView.

**Summary-emitter parity (preservation check before `effective_dimensionality` retires):**
- [ ] Verify `weight_spectra` emits an equivalent field for `effective_dimensionality_cross_over_epoch` (consumed at `packages/miscope/src/miscope/views/universal.py:1439` and `:1475`). If not, add it.
- [ ] Same for `effective_dimensionality_crossover_W_E_pr` (consumed at `apps/dashboard/src/dashboard/pages/viability_certificate.py:72`).

**Validation:**
- [ ] Surviving dashboard pages render end-to-end against the canon variant (`p113_seed999_dseed598`): `visualization.py`, `activation_heatmaps.py`, `neuron_dynamics.py`, `dimensionality.py`, `dimensionality_dynamics.py`, `multistream.py`, `repr_geometry.py`, `viability_certificate.py`. No silent panel inactivation; no broken loader paths.
- [ ] After Phase A merges to `develop`, REQ_102 unblocked for retirement of: `coarseness`, `dominant_frequencies`, `attention_fourier`, `neuron_fourier`, `attention_freq`, `neuron_freq_norm`, `effective_dimensionality`.

### Phase B — Composition object for `*_fourier_alignment`

- [ ] One concrete dataclass (e.g., `RepresentationGeometryView`) composes `repr_geometry` centroids + `centroid_fourier_alignment` summary into a single domain-meaningful structure. Plain Python; no layer.
- [ ] [packages/miscope/src/miscope/visualization/renderers/repr_geometry.py:257](../../../packages/miscope/src/miscope/visualization/renderers/repr_geometry.py#L257) re-wired to consume the dataclass; the panel actively renders again instead of silent inactivation.
- [ ] No abstract base class, no registry, no protocol. If `dataview_universal.py` / `dataview_catalog.py` are touched, it is to remove experimental code, not to extend it.
- [ ] Validation: `repr_geometry` page shows the fourier_alignment panel on the canon variant; the panel does not graceful-degrade.
- [ ] After Phase B merges to `develop`, REQ_102 unblocked for the `*_fourier_alignment` field absorption close-out.

### Documentation

- [ ] CHANGELOG entry for the release describing each deletion and migration with a pointer to its replacement.
- [ ] This REQ's Notes section records: the consumer-surface audit summary, tracked-deferred items (class-centroid-trajectories re-homing; shared-primitives re-evaluation).

---

## Constraints

**Must:**
- Surviving dashboard pages render against the new artifact set without regression in panel availability (other than known float64-precision shifts documented in `feedback_req126_float64_parity`).
- Summary-emitter fields consumed by surviving code are preserved — either continued emission by the new analyzer, or replaced by an equivalent named field with the consumer updated to the new name.
- Storage encapsulation invariant (REQ_122) holds: re-points happen via `variant.artifacts.load_*` and the view catalog, not via filesystem path construction.

**Must avoid:**
- **DataView layer design.** No abstract base class, no registry, no protocol — just the concrete dataclass for the one composition case. If a second case surfaces mid-Phase-B, it gets its own dataclass; do not abstract.
- **Shared visualization primitives extraction.** Out of scope; defer to a future REQ if accumulation actually warrants it.
- **Migrating notebooks or sketches.** Per the REQ_126 carve-out, diminished signal in research notebooks is a finding, not a regression to mask.
- **Re-homing the class-centroid-trajectories view in this REQ.** Catalog preservation is sufficient; a new dashboard surface is a future REQ.

**Flexible:**
- Phase A and Phase B may ship as separate PRs or as one. Preference: two PRs for diff readability; bundling allowed if Phase B turns out to be a small addendum.
- Ordering within Phase A: page deletions can lead (reduce subsequent migration scope) or trail. Default: deletions first.

---

## Architecture Notes

**Anti-design rationale.** The existing `dataview_universal.py` / `dataview_catalog.py` files were built as a designed layer before its users landed. Post-REQ_126, only one true composition case remains. A registry + protocol + abstract base for a single-element set is overkill that pays its design cost on speculation; the cost shows up later as drift between the layer and how renderers actually want to consume composed data. The discipline: build the concrete object the renderer needs, validate it works, and only generalize when a second concrete case proves the shape worth lifting.

**Why the bulk of the work lives in `universal.py`.** The view catalog is the spine — most retirements are loader re-points inside view registrations, not renderer rewrites. The catalog absorbs the analyzer-naming churn; renderers that consume `dict[str, np.ndarray]` artifacts often don't notice when the underlying analyzer name changes, as long as the artifact schema matches.

**Storage encapsulation as a quiet beneficiary.** Deleting `summary.py` resolves three direct `load_epoch(...)` calls (lines 86, 253) that bypassed the view catalog and violated REQ_122. With those gone, no surviving dashboard page should be doing artifact composition outside the catalog. A grep check for `variant.artifacts.load_` calls in surviving pages after Phase A is the verification.

**REQ_102 sub-phasing benefit.** After Phase A merges, REQ_102 can begin retiring 7 of its 9 candidates (coarseness, dominant_frequencies, attention_fourier, neuron_fourier, attention_freq, neuron_freq_norm, effective_dimensionality) without waiting on Phase B. The composition-dependent retirements (`*_fourier_alignment` absorption gating, `centroid_dmd` wrapper cleanup) wait for Phase B.

---

## Notes

- The trigger for this REQ is the REQ_126 carve-out, but the surface it touches is broader than just "consumers of the absorbed analyzers." If the DataView axis is included, the design ripples into surfaces that weren't part of REQ_126 (e.g., `repr_geometry`'s dashboard page consumes `repr_geometry` artifacts already; a DataView for that page would touch them even though they weren't absorbed). Scope discipline matters: which surfaces are in for this REQ, which are deferred.
- The `repr_geometry → representation_geometry` rename is *not* this REQ. It's a separate REQ_111-style pure-rename PR (86-file ripple; bookkeeping). Could land before, during, or after this REQ; doesn't gate on it.
- REQ_102's close-out blocks on this REQ landing far enough that the absorbed analyzers no longer have live consumers. Phase A unblocks 7 of 9 retirements; Phase B unblocks the `*_fourier_alignment` absorption close-out.
- Possible interaction with REQ_110 (Lakehouse Surface): REQ_110's `to_wide(...)` and DataFrame contracts are the *publication* face of analyzer data; the dataclass introduced in Phase B is the *visualization* face. They're related but distinct surfaces — confirm during REQ_110 scoping whether they share a substrate or coexist. Given Phase B's anti-design discipline, the visualization-face dataclass should not constrain REQ_110's choices.

### Consumer-surface audit summary (2026-05-27)

Audit ran across all 8 retirement-bound names (`dominant_frequencies`, `attention_fourier`, `neuron_fourier`, `attention_freq`, `neuron_freq_norm`, `coarseness`, `*_fourier_alignment`, `effective_dimensionality`, `centroid_dmd`) over surviving dashboard pages, the view catalog (`packages/miscope/src/miscope/views/universal.py`), renderers, fieldnotes, notebooks, and scripts.

**Surviving consumers driving the Phase A migration list:**
- `visualization.py` — heaviest; 9 view names on retiring analyzers.
- `activation_heatmaps.py` — 3 views (`neuron_freq_norm`, `attention_freq`).
- `neuron_dynamics.py` — 2 views (`neuron_freq_norm`).
- `dimensionality.py` — 2 views (`effective_dimensionality`).
- `multistream.py` — composite view (`attention_fourier` + `effective_dimensionality`).
- `viability_certificate.py` + `dimensionality_dynamics.py` — summary-emitter field reads (`effective_dimensionality_cross_over_epoch` / `_crossover_W_E_pr`).

**Clean surfaces (no retiring-analyzer dependency):** `analysis_run`, `checkpoint_schedule`, `geometry_weights`, `initialization_sweep`, `input_trace`, `intervention_check`, `loss_landscape`, `neuron_group`, `parameter_dmd`, `activation_dmd`, `peer_comparison`, `transient_frequency`, `training`, `variant_table`, `_analysis_page_template`.

**Fieldnotes:** clean — no published figures or posts reference any retiring analyzer name. No regression risk for the deployed research-notebook surface.

### Tracked deferred (post-REQ_127)

- **Clean class-centroid-trajectories view.** All three centroid_dmd-backed views (`dmd_eigenvalues`, `dmd_residual`, `dmd_reconstruction`) removed during Phase A. Future REQ builds an independent centroid-trajectories view (sourced from `repr_geometry` or equivalent, not `centroid_dmd`) and finds it a dashboard home.
- **Shared visualization primitives extraction.** Dropped from this REQ. Re-evaluate if redundant-idiom concentration actually surfaces post-migration.
- **Removal of experimental `dataview_universal.py` / `dataview_catalog.py`.** If touched during Phase B, removal is preferred over extension. Otherwise stays as-is until a future cleanup REQ revisits.
