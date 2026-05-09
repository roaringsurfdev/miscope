# REQ_117: DMD Reorganization — Activation and Parameter, Windowed and Per-Regime

**Status:** In Progress (phase 1: activation track)
**Priority:** High — central narrative for first external research communication. The unified DMD treatment is the mathematically grounded backbone that other dynamical analyses will reference as supporting evidence.
**Branch:** `feature/req-117-dmd-reorganization`
**Supersedes:** REQ_073 (Weight-Space DMD). Absorbs the Research Claude drafts at `docs/requirements/drafts/research_claude/requirements_DMD_pipeline.md` (REQ_001 / REQ_002) and `requirements_windowed_DMD.md` (REQ_003).
**Dependencies:**
- REQ_109 (measurement primitives) — DMD-side primitives live in `analysis/library/dmd.py`; this REQ may extend them, following REQ_109's pure-input pattern.
- REQ_111 (parallel analyzer build-out) — this REQ inherits REQ_111's parallel construction philosophy. New analyzers are built alongside `centroid_dmd`; the existing analyzer is not modified.
- REQ_118 (neuron grouping primitive) — **hard dependency for the parameter-side track.** The parameter DMD CoS items cannot complete until `neuron_grouping` lands. The activation-side track has no such dependency and can proceed first.
- REQ_102 (analyzer deprecation) — picks up retirement of `centroid_dmd` after both tracks complete and validate.
**Attribution:** Engineering Claude (under user direction)

---

## Problem Statement

### Communication problem

The platform is approaching a point where its results need to be defensible to readers outside the daily working context. Sharing a heterogeneous mix of geometry views, frequency views, and trajectory views and asking a reader to assemble the picture is a high-friction ask. Each visualization carries its own interpretive overhead; the reader is asked to trust the assembly.

DMD inverts this problem. The mathematics is established and agnostic. The interpretive heft does not have to be borrowed from the reader — it is carried by the operator-theoretic framing itself. Used as a backbone, DMD lets the project's other visualizations function as supporting evidence for a single mathematically grounded narrative, rather than competing primary lenses.

### Technical problem

The existing `centroid_dmd` analyzer was a first pass. It computes a single global DMD across all checkpoints on the cross-class centroid trajectories, projected through a single global PCA. Three structural issues:

1. **Single-window DMD averages out regime structure.** The dynamics during memorization, the grokking transition, and post-grokking are qualitatively different. A single DMD operator fits a compromise that captures none of them well. Residual norms across the full run are noisy aggregates; the regime boundaries that should be visible are smeared.

2. **Short transition windows get smoothed away by global fits.** A PCA basis or a DMD operator fit across the full trajectory is dominated by the long stable phases. The short windows where the internal representation is actively reorganizing — tuning, rotating, deconstructing, reconstructing — appear as small perturbations against the global structure rather than as their own dynamical regime. The most informative dynamics are also the most vulnerable to being smoothed away. The dimensionality dynamics view already exhibits this property: features visible at window scale are invisible at trajectory scale. Windowing recovers them by making the operator local in time.

3. **Activation-only.** The mechanism of grokking lives in weight space. Centroid DMD captures the downstream consequence; it does not see the parameter dynamics that drive that consequence.

### What this REQ does

Reorganizes DMD into a **two-track, windowed, regime-aware** treatment:

- **Activation DMD** (`activation_dmd`) — per-site windowed DMD on per-class centroid stacks. Eigenvalue tracking across windows. Residual-driven regime detection. Per-regime DMD as a recursive second pass that fits a clean linear operator inside each detected regime.

- **Parameter DMD** (`parameter_dmd`) — same structural shape (windowed → regime detection → per-regime DMD), but operating per-frequency-group on slices of `W_in` columns and `W_out` rows. The grouping comes from REQ_118's `neuron_grouping` primitive (with modadd's family-supplied Fourier grouping as the initial concrete implementation).

- **Splits existing `centroid_dmd`.** The trajectory-PCA portion is absorbed by REQ_111's `representation_trajectory` work. The modal portion is the new `activation_dmd`. After both this REQ and REQ_111 complete and validate, REQ_102 retires `centroid_dmd`.

The hypothesis the work tests: residual norm spikes from windowed DMD will surface real dynamical regime boundaries. Per-regime DMD inside those boundaries will reveal eigenvalue structure that maps recognizably onto the geometric phenomena already documented in the notes (saddle transit, ring formation, Fourier commitment). If the hypothesis holds, both analyzers earn promotion to Universal Core in a future Atlas revision. Until that validation lands, they sit in Dynamical Proxies → Operator dynamics where they are today.

---

## Conditions of Satisfaction

### Cross-cutting (applies to both tracks)

- [ ] **Parallel construction.** Both new analyzers are built alongside `centroid_dmd` per REQ_111's pattern. The existing analyzer code is not modified during this work.
- [ ] **Primitive-grounded.** All new code consumes REQ_109 primitives. Where a primitive is missing (windowed DMD operator, eigenvalue tracking across windows, residual-spike segmentation), the gap is either filed as a REQ_109 follow-up or implemented inside `analysis/library/dmd.py` following REQ_109's pure-input convention. No analyzer-internal SVDs or inline numerical machinery.
- [ ] **Per-class state vector.** Both tracks operate on per-class structure, not cross-class averages. The DMD state vector preserves the per-class dimension; cross-class summarization is a downstream view operation, not a pre-DMD step.
- [ ] **Eigenvalue tracking primitive.** A reusable primitive matches eigenvalues across adjacent windows (nearest-neighbor for v1; Hungarian-matching documented as a possible upgrade). Returns per-mode trajectories of `|λ|`, `arg(λ)`, conjugate-pair status.
- [ ] **Residual-driven regime detection.** A primitive that ingests the per-window residual time series and returns regime boundaries. v1 accepts caller-supplied thresholds (with a default heuristic from the residual distribution); auto-detection with confidence bands is a follow-up.
- [ ] **Per-regime DMD recursive pass.** Once segments are identified, a standard DMD is computed inside each segment as its own analysis. Each segment carries its own eigenvalue spectrum, modes, amplitudes, residuals, and reconstruction quality. Regime boundaries are stored as part of the analyzer artifact so views can render them on overlays.
- [ ] **Discrete-time DMD.** Snapshots are checkpoints; the DMD operator maps one checkpoint's state to the next. Visualizations use real epoch labels, not window indices.
- [ ] **Configurable window size and stride.** Defaults documented; minimum window enforced (≥ ~5 checkpoints per window so DMD estimates are not degenerate).
- [ ] **View catalog integration.** New views land in `views/universal.py` with names that follow existing conventions (`activation_dmd.*` and `parameter_dmd.*` prefixes). Old `geometry.dmd_*` view IDs remain functional during the parallel period; their migration to the new IDs is part of the dashboard handoff.
- [ ] **Dashboard integration.** Side-by-side comparison with the existing `centroid_dmd` page during the parallel period. After validation, the dashboard switches to the new analyzers; the old page is removed under REQ_102.

### Activation track — `activation_dmd`

- [ ] **Per-site execution.** Windowed DMD runs separately at each of the four sites (post-embed, attn out, MLP out, resid post — same site set as `representation_geometry`). Each site has its own residual time series, regime boundaries, and per-regime spectra. Cross-site comparison is a downstream view, not a coupling solved inside the analyzer.
- [ ] **Per-class centroid stack as state.** State vector is `(n_classes × d_model)` (or its global-PCA projection if the projected form is preferred for tractability — decide during implementation, document the choice). The cross-class structure is preserved.
- [ ] **Global PCA projection (REQ_001 absorbed).** A single PCA basis fit across all checkpoints' centroids (per site) provides the projected state vector for windowed DMD. Number of retained components is data-driven (95% variance default). The global PCA artifact is reused; it is not refit per analyzer call.
- [ ] **Per-site cross-epoch artifact.** Output structure per site: window list (with epoch ranges), per-window eigenvalues / modes / amplitudes / residual norm, eigenvalue trajectories, regime boundaries, per-regime DMD spectra and reconstruction error.
- [ ] **Mode-Fourier projection (modadd family context).** For sites whose modes carry Fourier interpretability (modadd-specific), each per-window or per-regime mode can be projected onto the family's Fourier basis. Surfaces as a derived view (universal mechanism, family-supplied basis); not part of the analyzer artifact.
- [ ] **Reference-variant sanity check across the spectrum.** Run on the four reference variants spanning known behavior:
  - **p109/s485/ds598** — clean grokker, one end of the spectrum. Eigenvalues should migrate from real-axis-dominated to stable conjugate pairs on the unit circle, with timing matching the known grokking window. Residual spikes should align with documented regime boundaries.
  - **p113/s999/ds598** — canon, middle case. Eigenvalue migration should be present but with the slower, more diffuse character documented in the canon notes (diffuse neuron specialization, late frequency commitment).
  - **p59/s485/ds598** — failure mode, other end of the spectrum. Eigenvalue migration should show qualitatively different behavior: wandering angles, modes that fail to reach the unit circle, regime boundaries that do not align cleanly with documented training events.
  - **p101/s999/ds598** — degenerate case. Open-loop PC2/PC3 geometry, slow grokking, poor specialization. Included not as a spectrum endpoint but as a structurally interesting failure mode the analyzer should be able to observe distinctly from p59.

  The analyzer surfaces these signatures as observations; interpreting them is downstream work.

### Parameter track — `parameter_dmd` *(blocked on REQ_118)*

- [ ] **Per-group execution.** Windowed DMD runs separately per neuron group, where the grouping comes from `neuron_grouping` (REQ_118). For modadd, the family supplies a Fourier-derived grouping by default; the architectural seam allows substitution with any `neuron_grouping` output.
- [ ] **Group-sliced weight state.** State vector for a given group is the concatenation of `W_in` columns and `W_out` rows for the group's neuron set (and optionally `W_E`-derived input projections — decide during implementation). The choice of which weight matrices contribute to a group's state is documented.
- [ ] **Per-matrix variant.** In addition to per-group, support per-matrix windowed DMD (W_E, W_in, W_out, W_Q/K/V) for cases where the question is "which weight matrix is actively reorganizing during this regime?" — kept as a separate output, not a replacement for per-group.
- [ ] **Mode-Fourier projection.** For modes associated with W_E, W_in, or W_out, project onto the family-supplied Fourier basis. Connects each DMD mode to a Fourier-frequency identity. Carries forward REQ_073 CoS 5.
- [ ] **Per-group cross-epoch artifact.** Output structure per group: same shape as the activation track's per-site artifact (windows, eigenvalues, modes, amplitudes, residuals, regime boundaries, per-regime DMD).
- [ ] **Group-aware regime boundaries.** Each group has its own residual time series and its own detected regime boundaries. The analyzer does not collapse groups into a global regime list. Group-vs-group regime alignment (or misalignment) is itself a derived observation.
- [ ] **Reference-variant validation.** Per-group windowed DMD on canon (p113/s999/ds598) should surface regime boundaries that align with the documented neuron commitment epochs (from `neuron_dynamics`). Eigenvalue structure during the second-descent window should match the documented frequency-group reorganization timing.

### Validation and handoff

- [ ] **Parity component (where applicable).** For the activation track, run the new `activation_dmd` and the existing `centroid_dmd` on the reference variant set (p109/s485/ds598, p113/s999/ds598, p101/s999/ds598, p59/s485/ds598). Where the new analyzer's single-window mode is configured to match `centroid_dmd`'s settings, eigenvalues and residuals should agree within REQ_111's documented tolerances. Document any divergence (with bug attribution to old or new) per REQ_111's validation pattern.
- [ ] **No parity for parameter track.** No predecessor analyzer exists. Validation reduces to: reference-variant sanity checks pass, mode-Fourier projection produces interpretable frequency assignments on canon, regime boundaries align with documented commitment epochs.
- [ ] **Validation outcomes recorded** in this REQ's Notes section, following REQ_111's format.
- [ ] **Atlas update.** The Atlas entries for `activation_dmd` and `parameter_dmd` are updated to point at this REQ. The `representation_trajectory` entry already cites REQ_111 for the trajectory portion of the split; that text remains as-is.
- [ ] **Handoff to REQ_102.** After validation, REQ_102 lists `centroid_dmd` (analyzer + page + view IDs) for retirement. The new analyzers become the dashboard default.

---

## Constraints

**Must:**
- Build new analyzers parallel to `centroid_dmd` per REQ_111. Do not modify the existing analyzer during this work.
- Operate on per-class state vectors. Cross-class averaging is not allowed at the DMD-input layer.
- Operate per-site (activation) and per-group (parameter). Aggregation across sites or groups is a downstream view operation.
- Discrete-time DMD only. Continuous-time interpretation is a future extension if it becomes useful.
- All new code consumes REQ_109 primitives. New DMD-specific primitives follow REQ_109's pure-input convention and live in `analysis/library/dmd.py`.

**Must avoid:**
- **Single-window global DMD.** The existing failure mode. The new analyzers compute windowed DMD as the primary operation; a single-window setting may exist for parity validation, not as a default.
- **Global-average state vectors.** Cross-class averaging at the input layer destroys the signal the analysis is trying to surface.
- **Bundling primitive extension with analyzer construction.** Primitive gaps surfaced during this work get filed as REQ_109 follow-ups (or implemented in `library/dmd.py` to REQ_109's standards), not embedded ad-hoc in analyzer code.
- **Promoting either analyzer to Universal Core in this REQ.** Promotion is conditional on validation outcomes. A future Atlas revision handles the promotion; this REQ leaves both analyzers in Dynamical Proxies → Operator dynamics where the Atlas places them today.
- **Bypassing REQ_118.** The parameter track does not ship with a hardcoded Fourier grouping path that ignores the planned `neuron_grouping` interface. Fourier grouping is the family's contribution through `neuron_grouping`'s family-override mechanism — not a parallel code path.

**Flexible:**
- Window size and stride defaults. Initial values documented in CoS; tunable per analyzer call.
- Eigenvalue matching algorithm (nearest-neighbor v1; Hungarian later if needed).
- Whether the activation-track state vector is the raw `(n_classes × d_model)` form or the global-PCA-projected form. Implementation decision; document the choice and its tradeoffs in the analyzer.
- Where regime-boundary detection lives. Default: a primitive in `library/dmd.py` (or a new `library/regimes.py` if the boundary-detection logic generalizes beyond DMD residuals). Refactor freely if generalization emerges.
- Dashboard layout for the new pages. Side-by-side comparison with the old page during the parallel period; final layout decided when the new analyzers are the default.

---

## Architecture Notes

### Implementation phasing — activation track first

The hard dependency on REQ_118 already sequences the parameter track behind the activation track. This is not just a consequence of the dependency graph — it is the right shape for a workstream whose hypothesis still needs first confirmation.

Phase 1: ship the activation track. Run it across the four reference variants. Read the eigenvalue migration story. Confirm or disconfirm the hypothesis at the cheapest possible point in the work.

Phase 2 (conditional on phase 1): ship REQ_118, then the parameter track. The parameter side reproduces the activation side's structural shape in weight space, so most of the analyzer-design risk is already burned down by the time it starts.

If phase 1 produces the expected signature — eigenvalues migrating from the real axis to stable conjugate pairs on the unit circle, residual spikes aligning with documented regime boundaries, qualitative contrast across the spectrum — the case for promoting both DMD analyzers to the central narrative is made. If it does not, the parameter buildout is the work that is *not* incurred. The phasing exists precisely to make this stopping point cheap.

### Why DMD as the backbone for first external communication

The platform's analytical surface today is broad and heterogeneous. For a researcher inside the working context, the geometry views and the frequency views and the trajectory views form a coherent picture. For a researcher outside that context — the audience for the project's first results communication — the assembly cost is high. Each visualization carries interpretive baggage that the reader is asked to absorb.

DMD is mathematically established. The framing is operator-theoretic and agnostic to the model class. A reader does not need to be persuaded that DMD is meaningful; they need to be persuaded that the application is sound. That is a much smaller persuasion budget than asking them to accept a heterogeneous mix of bespoke lenses.

If the windowed + per-regime treatment produces the eigenvalue-migration story the hypothesis predicts, every other analyzer in the platform becomes evidence in support of a DMD-derived narrative, rather than an independent claim that has to defend itself separately. The platform's communication problem is solved structurally, not visually.

### Why per-site for activation and per-group for parameter

These are not symmetric choices. The asymmetry comes from how the dynamics are organized in each space.

**Activation space** factors naturally by site. The four analyzed sites (post-embed, attn out, MLP out, resid post) carry distinct functional roles. They reorganize on different timelines — empirical evidence already in the project's notes (Fourier alignment commits at MLP first; resid_post inherits the structure last). A single DMD that flattens across sites would smear their timelines together. Per-site DMD respects the structure.

**Parameter space** factors by neuron group, not by weight matrix. The dynamics that windowed DMD is trying to surface — frequency commitment, second-descent reorganization, saddle transit — operate on subsets of neurons that share a frequency identity. A windowed DMD on a flattened `W_in` matrix smears those subsets together and the residual signal becomes a mixture of overlapping regime boundaries. Per-group DMD respects the structure.

The user's framing was direct: "summarizing across an entire weight-group creates invalid segments." This REQ encodes that as a hard architectural rule.

### Why the hard dependency on REQ_118

The parameter track could in principle ship today with a Fourier-derived grouping hardcoded for the modadd family. Doing so would let the activation and parameter sides ship in lockstep and accelerate the central narrative.

The user chose the hard-dependency path. The reasoning, in the user's words: "I'd rather start with the clean intended base than have to refactor if the results are what I expect they'll be." If the DMD analysis produces the expected mechanistic narrative, both analyzers will be promoted to Universal Core. A Universal-Core analyzer that internally hardcodes a family-specific grouping path is architecturally wrong. The substitution from "hardcoded Fourier" to "family-supplied grouping via `neuron_grouping`" is not zero-cost — it touches the analyzer's contract, its tests, and its artifacts. Doing it once, correctly, costs less than doing it twice.

The cost of the choice: parameter DMD is sequenced behind REQ_118. Activation DMD has no such dependency and can proceed first.

### Per-regime DMD as a recursive operation

Windowed DMD answers "where do the dynamics change?" Per-regime DMD answers "what are the dynamics inside each regime?" The two operations together produce a piecewise-linear approximation of nonlinear training dynamics — each piece a clean linear operator with a clean eigenvalue spectrum, joined at the regime boundaries that windowed DMD discovered.

The recursive shape is what makes the analysis defensible as a backbone. The first pass surfaces structure; the second pass characterizes that structure with a tighter linear fit. Reviewers can interrogate either layer independently.

### Relationship to phase-space fits

The Atlas defines a sub-category for phase-space fits — `lissajous_fit`, `saddle_center_center_fit`, `saddle_transport_sigmoidality`. Those analyzers fit assumed dynamical-system structures to observed trajectories. DMD is upstream of all of them: a phase-space fit assumes the local dynamics has a particular eigenvalue signature; DMD measures the eigenvalue signature without the assumption. When the phase-space fits land, their characterizations should be interpretable in the DMD eigenvalue picture and vice versa — they are different lenses on the same operator-theoretic structure.

This is part of why DMD is positioned as the backbone. The phase-space fits become natural extensions of the DMD picture rather than independent claims.

---

## Notes

### Validation outcomes

Format: `{track} on {variant}: {outcome}, {date}, {pointer to evidence}`.

**Activation track** (phase 1a complete, 2026-05-08; phase 1b complete, 2026-05-08):

- `activation_dmd` on **p113/s999/ds598** (canon, 2026-05-08): per-site signal-check passed. Residual structure cleanly identifies the grokking window — `resid_post` shows a wide bump centered at ~10k matching the documented grokking transition; `mlp_out` shows two spikes (early reorganization + grokking peak); `attn_out` and `resid_pre` show coordinated early/grokking-window structure. Eigenvalue migration at zoom shows a "pendant" of paired complex conjugate excursions near (1, 0) at `mlp_out`, smooth wing arcs at `resid_pre` and `attn_out`. Evidence: `apps/research/exports/req_117/residual_p113.png`, `eigenvalues_p113.png`, `tracks_p113_mlp_out.png`.
- `activation_dmd` on **p109/s485/ds598** (clean grokker, 2026-05-08): signal-check passed. Sharp early reorganization peaks at 1.5k–5.5k (matches fast-grokker timing). Late-training boundaries near 22k–24k confirmed by user as anticipated end-of-training reactivation, not threshold artifacts. Eigenvalue migration shows a discrete fan + paired conjugates qualitatively distinct from canon's pendant. Evidence: `residual_p109.png`, `eigenvalues_p109.png`.
- `activation_dmd` on **p101/s999/ds598** (degenerate, 2026-05-08): triple-confirmation case. The residual lens recovers two prior independent findings without prior knowledge of either: (1) the failed-transit-with-retry from saddle transit analysis surfaces as a stuttering peak pair at 12–14k across all four sites; (2) the freq-5 abandonment at ~25k from transient frequency tracking surfaces as a coordinated sub-regime bump at ~24.1k (below the boundary detector's height threshold but visible in the signal at every site simultaneously, with a corresponding small bump in the test loss curve). Three different mathematical operations converge on the same picture at appropriate scales. Eigenvalue migration shows a teardrop + isolated complex pairs at `resid_post`. Evidence: `residual_p101.png`, `eigenvalues_p101.png`; full narrative in `apps/fieldnotes/src/content/drafts/dmd-at-zoom.mdx`; project memory at `findings_dmd_convergence.md`.
- `activation_dmd` on **p59/s485/ds598** (failure mode, 2026-05-08): signal-check shows qualitatively different character from the three grokkers. `mlp_out` produces only the initial spike then stays quiet; `resid_pre`, `attn_out`, and `resid_post` show multiple late-training boundaries past 20k consistent with churning that doesn't commit. Eigenvalue migration is sparse near-real with very little complex structure — the lens that produced loops and fans elsewhere produces almost a flat line. Evidence: `residual_p59.png`, `eigenvalues_p59.png`.

**Detection-algorithm refinement (2026-05-08).** The original rising-edge boundary detector missed signals that started above threshold and decayed without re-crossing (Problem A — visible in p109 `resid_post`'s peak at ~5k producing zero boundaries). Refactored to peak-based detection via `scipy.signal.find_peaks` with edge-padding, `height = median + 3·MAD`, and `prominence = 1·MAD`. Returns `peak_prominences` per boundary as a quality signal. Re-run on all four reference variants confirms: Problem A closed (p109 `resid_post` now 3 boundaries), late-training p109 signal preserved (real signal per user's prior context), canon `mlp_out` initial spike now detected, p101 stuttering peak pair correctly resolved, p59 character preserved. Tests: 70/70 passing in `test_dmd_primitives.py`.

**Parameter track** (phase 2 complete, 2026-05-09):

- `parameter_dmd` on **p113/s999/ds598** (canon, 2026-05-09): per-(group, matrix) signal-check passed at default `reference_epoch=24999` (last available). 4 populated groups recovered (canon's documented `{9, 33, 38, 55}`). Per-group eigenvalue migration is meaningfully distinct across groups — not all four show the same character. W_out has consistently wider vertical spread than W_in across all four groups. Evidence: `apps/research/exports/req_117_parameter/residuals_p113.png`, `eigenvalues_p113.png`, `win_vs_wout_p113.png`.
- `parameter_dmd` on **p109/s485/ds598** (clean grokker, 2026-05-09): 3 populated groups. Group 3 (k=4) shows clear W_in / W_out *desynchronization* during early training — W_out drops to near-zero around epoch 2.5k while W_in is still elevated. This is the cleanest empirical justification for treating W_in and W_out as separate matrices rather than concatenating them. Late-training mlp_out activity (visible in the activation-vs-parameter overlay) doesn't have a parameter-side counterpart — candidate neural-collapse signature flagged for future investigation.
- `parameter_dmd` on **p101/s999/ds598** (degenerate, 2026-05-09): 5 populated groups when run with **pinned** `reference_epoch=20000` (where freq 5 was still committed). Default-last-epoch run shows 4 groups (freq 5 already abandoned). Pinned run captures freq 5 (group 4, 39 neurons) with a clear residual bump at ~25k in both W_in and W_out, confirming the freq-5 abandonment shows up at the parameter level when we pin to a snapshot where the frequency was still committed. Group 4's W_out drop after the abandonment peak is steeper than other groups' W_out — flagged as a candidate "transient frequency abandonment" signature for future cross-variant investigation. Evidence: `residuals_p101.png` post-pinning + `overlay_p101.png`.
- `parameter_dmd` on **p59/s485/ds598** (failure mode, 2026-05-09): 3 populated groups. W_in and W_out track each other tightly — but in a failure-mode trajectory where neither matrix is doing meaningful learning, so the agreement is uninformative. Consistent with the "nothing structured to disagree about" interpretation.

**Within-DMD multi-lens convergence (2026-05-09).** The activation+parameter overlay (`overlay_*.png`) shows that mlp_out residual peaks (from `activation_dmd`) align with per-group W_in/W_out residual peaks (from `parameter_dmd`) on the major training events across all four reference variants. The two analyzers, applied to fundamentally different mathematical objects (activation centroids vs weight slices), recover the same regime structure. Magnitude differences encode event scope: localized parameter events (e.g., freq 5 abandonment in p101) show more relative magnitude on the parameter side because they are not diluted by averaging over uninvolved neurons. mlp_out is consistently smoother than W_in/W_out (rounder peaks vs sharper); this is part of a smoothness gradient where activation residuals smooth what parameter residuals sharpen.

**`extra_context` infrastructure (2026-05-09).** Added `extra_context: dict[str, Any] | None = None` kwarg to `AnalysisPipeline.run()` that merges into the family-supplied analysis context after `prepare_analysis_context()`. Caller-supplied keys override family-supplied keys. Used to inject `parameter_dmd_reference_epoch` for the pinned-canary use case without touching family code. Generic — any future analyzer that reads context can take experiment-specific config the same way.

### Phase 2b — dashboard integration (2026-05-09)

- Renderers in `packages/miscope/src/miscope/visualization/renderers/parameter_dmd.py`: `render_parameter_dmd_residuals_with_regimes`, `render_parameter_dmd_eigenvalue_migration`, `render_parameter_dmd_per_regime_vs_windowed`, `render_parameter_dmd_track_trajectories`. The first three auto-scale height to the variant's populated-group count; the fourth takes a (group_id, matrix) pair selected via dashboard dropdowns.
- View catalog: four `parameter_dmd.*` view IDs registered in `views/universal.py`. The `track_trajectories` view accepts `group_id` + `matrix` kwargs.
- Dashboard page `apps/dashboard/src/dashboard/pages/parameter_dmd.py` at route `/parameter-dmd`. Group dropdown is dynamic — populated from the loaded variant's `populated_groups` so canon shows 4, p101 shows 4 or 5 depending on `reference_epoch`, etc. Matrix dropdown (W_in / W_out). Reference epoch indicator in the left nav surfaces which `neuron_grouping` snapshot the analyzer used.
- Smoke-tested HTTP 200 on `/parameter-dmd`; user verified rendering in the live dashboard.

### Phase 1b — dashboard integration (2026-05-08)

- Renderers in `packages/miscope/src/miscope/visualization/renderers/activation_dmd.py`: `render_activation_dmd_residuals_with_regimes`, `render_activation_dmd_eigenvalue_migration`, `render_activation_dmd_track_trajectories`, `render_activation_dmd_per_regime_vs_windowed`.
- View catalog: `activation_dmd.residuals_with_regimes`, `activation_dmd.eigenvalue_migration`, `activation_dmd.track_trajectories`, `activation_dmd.per_regime_vs_windowed` registered in `views/universal.py`.
- Dashboard page `apps/dashboard/src/dashboard/pages/activation_dmd.py` at route `/activation-dmd`. Built fresh, no side-by-side with legacy `centroid_dmd` page (validation already complete via the notebook signal-check + multi-lens convergence; coexistence period not load-bearing for this analyzer pair). Smoke-tested HTTP 200 + browser-verified by user.
- Hand-off to REQ_102: legacy `centroid_dmd` analyzer + page + view IDs (`geometry.dmd_*`) can be retired in REQ_102's deprecation pass when convenient. The new `activation_dmd` is the canonical home for windowed + per-regime modal analysis on activation centroids.

### Open questions

- **Window size defaults.** REQ_003 (absorbed) suggested 10–15 checkpoints with stride 1. With ~94 checkpoints per variant, that yields ~80 windows. Whether this resolution is right for every site / group is unknown until the analysis runs on the reference set.
- **Regime-boundary thresholding.** Caller-supplied thresholds in v1, with a default heuristic derived from the residual distribution. Whether auto-detection with confidence bands is worth building depends on how stable the residual signature is across variants.
- **Activation-track state representation.** `(n_classes × d_model)` raw vs. global-PCA-projected. Tractability is fine either way at workbench scale; the question is which form makes the eigenvalue interpretation cleanest. Decide during implementation.
- **Parameter-track scope of W matrices.** Whether per-group state should include `W_E`-derived input projections in addition to `W_in` columns and `W_out` rows. The Fourier-mode projection in REQ_073 used `W_E` directly; preserving that connection may argue for inclusion.
- **Promotion timing.** Both analyzers stay in Dynamical Proxies → Operator dynamics in this REQ. A future Atlas revision considers promotion to Universal Core after validation. The signal to promote is "the eigenvalue migration story holds across canon and at least two failure modes, and other analyzers in the platform start citing DMD outputs as their reference timing."
- **REQ_111 boundary.** REQ_111 already lists `centroid_dmd` (PCA paths only) as part of its `learned_parameters_pca` and friends migration. The trajectory-portion split is REQ_111's. The modal-portion split is this REQ's. The boundary is the SVD step: REQ_111 owns "fit PCA on centroids"; this REQ owns "compute DMD on the projected centroids." Coordinate during implementation if the line blurs.

### Cross-references

- [REQ_073](REQ_073_weight_space_dmd.md) — superseded by this REQ. Status updated.
- [REQ_111](REQ_111_parallel_analyzer_buildout.md) — the trajectory portion of the `centroid_dmd` split lives in REQ_111. This REQ's parameter and activation tracks consume the projected-centroid outputs that REQ_111's `representation_trajectory` produces.
- [REQ_118](REQ_118_neuron_grouping.md) — the grouping primitive that unblocks the parameter track.
- [Analysis Atlas](../../analysis_atlas.md) — updates to `activation_dmd` and `parameter_dmd` entries land as part of this REQ's CoS.
- Research Claude drafts at `docs/requirements/drafts/research_claude/requirements_DMD_pipeline.md` and `requirements_windowed_DMD.md` — absorbed by this REQ.
