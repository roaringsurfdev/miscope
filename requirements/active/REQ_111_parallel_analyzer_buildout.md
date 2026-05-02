# REQ_111: Parallel Analyzer Build-Out on Measurement Primitives

**Status:** Draft
**Priority:** High — the integration phase that pays off REQ_109's primitive work without prematurely committing to in-place migration.
**Branch:** TBD
**Supersedes:** None (carves out the analyzer-migration content originally in REQ_109).
**Dependencies:** REQ_109 (the measurement primitives library that new analyzers consume); REQ_106 (layering principle — primitives are the transform step in analyzer ETL); REQ_102 (analyzer deprecation, which this REQ gates with parity validation); REQ_110 (lakehouse surface — new analyzers respect the tabular output contract where applicable).
**Attribution:** Engineering Claude (under user direction)

---

## Problem Statement

REQ_109 stabilizes a measurement primitives library. The existing analyzers were built before that library existed — each encodes its own derivation path, axis conventions, and (as the REQ_109 design pass surfaced for Fourier and clustering) latent bugs:

- `arctan` vs `arctan2` in the Fourier coefficient phase recovery.
- DC fictitious-row in the original `sin_basis` (inconsistent with the cosine basis).
- `n_frequencies` field disagreement with basis row count.
- `float32` accumulation in `compute_class_centroids` causing silent precision loss in re-analysis runs.

Each was caught **before** an in-place migration would have either silently fixed it (with no audit trail) or silently encoded it forward (preserving the bug as the canonical path).

The original REQ_109 plan called for in-place analyzer migration as the integration phase. That plan implicitly answers the question *"did the old analyzer do the right thing?"* — without recording the answer. For analyzers built across many months on top of primitives that have only just been audited, that's a risky default.

This REQ reframes integration as **parallel construction**:

1. Build new analyzers alongside existing ones, consuming only the REQ_109 primitive layer.
2. Validate new outputs against old on a reference variant set. Record agreement, principled divergence (with bug-attribution to old or new), or persistent disagreement.
3. Only after parity validation is recorded does REQ_102 retire the corresponding old analyzer.

This honors the user's framing: *expand the platform during the parallel period, then collapse back to the canonical set once validation is complete.* It also matches the ETL spirit of the platform — artifacts are filesystem-backed, additive, and have no transactional coupling to the analyzer that produced them, so two analyzers writing parallel artifacts is structurally clean.

---

## Conditions of Satisfaction

### Parallel analyzer construction

- [ ] **One new analyzer per modernized old analyzer.** Each new analyzer is created alongside the existing one in `analysis/analyzers/`. The existing analyzer code is **not** modified during the parallel period.
- [ ] **New analyzers consume only REQ_109 primitives** for their transform step. Audit via `grep`: a new analyzer's `analyze()` should not call `np.linalg.svd`, `sklearn.decomposition.PCA`, raw `np.diff` on a trajectory, or any pre-REQ_109 derivation paths.
- [ ] **REQ_106 layering compliance.** Each new analyzer follows the extract → transform → load pattern: pull the right matrix (extract), feed it to the right primitives (transform), write the typed result (load).
- [ ] **Naming convention.** Default: *descriptive divergence* — pick a name that reflects the new analyzer's actual scope, not its temporal relation to the old one. Avoid `_v2` suffixes (they encode timing rather than meaning). When the old name was misleading (e.g., `dominant_frequencies` is W_E-only despite the generic name), the new analyzer takes a more accurate name (e.g., `frequency_spectrum_per_site`). When the old name was already accurate, the new analyzer takes a related name with explicit scope (e.g., `learned_parameters_pca` for the consolidated PCA replacement).
- [ ] **Per-epoch artifacts written to a new directory.** Old artifact directories remain untouched throughout the parallel period.

### Modernization scope

The initial scope mirrors what REQ_109's draft "Analyzer migration" section called out, plus research-active additions surfacing from the primitive work. Expand as new research needs emerge.

**Frequency:**
- [ ] New `frequency_spectrum_per_site` (name TBD, must avoid `dominant_frequencies` collision) replaces / absorbs the W_E-only `dominant_frequencies`. Composed weight sites (MLP_INPUT, ATTN_*) and ActivationSite spectra (mlp_out, attn_out, resid_post, resid_pre, embed, logits) all populated. Consumes `library.fourier_basis`.
- [ ] `transient_frequency` analyzer rewires to consume `frequency_trajectory(...)` rather than its own ad-hoc derivation. (May still be in-place if the rewire is a one-line change — judgment call during implementation; otherwise build parallel.)

**PCA:**
- [ ] New `learned_parameters_pca`, `frequency_group_geometry`, `activation_class_geometry` analyzers consume `library.pca`. Replace `parameter_trajectory_pca`, `effective_dimensionality`, `freq_group_weight_geometry`, `repr_geometry`, `neuron_group_pca`, `global_centroid_pca`, `centroid_dmd` (PCA paths only).

**Geometry / shape:**
- [ ] All geometry-touching analyzers route through `library/shape.py` primitives. Callers that previously passed raw centroids to `compute_circularity` now compute the PCA projection once via `library.pca` and pass it. The single PCA fit replaces three.

**Research-active additions (new — no old analyzer to validate against):**
- [ ] Lissajous analyzer: per-epoch `characterize_lissajous` on activation-site centroid PCA. Produces tracked `LissajousParameters` over training. Skips parity step (no old analyzer); validates against the sketch notebook's reported numbers on canon variant.
- [ ] Saddle-transport sigmoidality analyzer: per-segment `characterize_sigmoidality`. Segment discovery remains an open research problem (acknowledged at REQ scope); the new analyzer accepts segments from caller-configured boundaries (manual or from a future segment-discovery primitive). Reference parity: framework-notebook reported numbers on canon variants.

**Out of scope (handled by REQ_102, not here):**
- Retirement of `coarseness`, `fourier_nucleation`, `centroid_dmd` (the analyzer wrapper, after PCA paths extracted).
- Removal of family.json entries for retired analyzers.

### Parallel validation

- [ ] **Reference variant set:** p109/s485/ds598, p113/s999/ds598, p101/s999/ds598 (matches REQ_109's reference set). Expand to additional variants when a divergence on the reference set warrants it.
- [ ] **For each new-vs-old pair:** run both on the reference set; compute numerical agreement (singular values to ~1e-10 relative, eigenvectors to within sign flip, scalar metrics to documented per-metric tolerance, frequency lists exact match where applicable).
- [ ] **Validation outcomes recorded** in this REQ's Notes section. Possible outcomes:
  - *Matches within tolerance* — proceed to deprecation under REQ_102.
  - *Old has bug X (attributed)* — new is canonical; document the bug; proceed to deprecation under REQ_102.
  - *New has bug Y (attributed)* — fix in new analyzer before deprecation.
  - *Disagreement is real and both are kept* — document the conceptual difference; both analyzers stay; no deprecation.
- [ ] **Validation harness lives in `notebooks/parallel_analyzer_validation/`** (or similar — naming during implementation). Reusable per analyzer. Discarded or archived after the modernization phase completes.

### Dashboard integration

- [ ] Dashboard pages can show old + new side-by-side during the parallel period (when useful — not required for every analyzer pair).
- [ ] Once a new analyzer's parity validation is recorded as *matches* or *old-has-bug*, the dashboard switches to consuming the new analyzer.
- [ ] Old artifact paths remain readable via `ArtifactLoader` for the deprecation window per REQ_102 (existing on-disk artifacts are not invalidated).

### Handoff to REQ_102

- [ ] **No deprecation listing without recorded validation.** REQ_102's analyzer-retirement list cites the corresponding REQ_111 validation entry for each migrated analyzer.
- [ ] **Old analyzer's family.json entries removed only after** the new analyzer is the dashboard default and the validation outcome is recorded.
- [ ] **CHANGELOG entries** describe the new analyzer and the validation outcome (matches / bug attributed / etc.).

---

## Constraints

**Must:**
- New and old analyzers coexist throughout the parallel period. The existing analyzer code is *not* modified.
- Each new analyzer uses only REQ_109 primitives for its transform step. Auditable via grep tests inherited from REQ_106.
- Validation outcome is recorded *before* an analyzer is listed for retirement under REQ_102.
- New analyzers respect REQ_110's tabular output contract where applicable (long-format with `group_type` / `operation_type` discriminator columns).
- The validation harness is reusable. Same pattern applies whenever a primitive change ripples through an analyzer.

**Must avoid:**
- **Premature deprecation.** Don't list an old analyzer for retirement under REQ_102 until parity (or principled-divergence resolution) is recorded.
- **Touching old analyzer code.** The point of parallel construction is to preserve the original as a numerical anchor.
- **Bundling primitive extension with analyzer construction.** New analyzers consume primitives; they do not extend them. Primitive gaps surfaced during this work get filed as REQ_109 follow-ups, not extended here.
- **Treating "matches old" as the only success criterion.** Some old analyzers may have only ever produced wrong output. The validation should be honest about this rather than mechanical.

**Flexible:**
- Naming convention per analyzer. Default: descriptive divergence; case-by-case if a name collision is unavoidable.
- Order of analyzer construction. Default: start with the analyzer most aligned with active research (Lissajous + saddle-transport sigmoidality, since those have the user's open research questions on top of them). Frequency and PCA modernizations can run in parallel tracks.
- Whether validation lives in tests, notebooks, or both. Default: notebook for variant-by-variant inspection (research-grade evidence), plus a unit-test layer for known synthetic cases (regression coverage).
- Granularity of parallel artifact directories. Default: each new analyzer writes to its own directory at `artifacts/{new_name}/`. Old artifacts at `artifacts/{old_name}/`. Don't share directories.

---

## Architecture Notes

### Why parallel construction (rather than in-place migration)

The primitive design pass for REQ_109's Fourier work surfaced bugs in the original `library/fourier.py` (arctan vs arctan2, DC fictitious row, normalization inconsistency) and a dtype regression in `compute_class_centroids`. Each was caught **before** an in-place migration would have either silently fixed it (no audit trail) or silently encoded it forward (preserving the bug as the canonical path).

The analyzers built on those primitives carry the same risk: they encode conventions and derivation choices that may have been correct, may have been wrong, or may now make assumptions the new primitives no longer support.

In-place migration of an analyzer is a refactor that *looks* mechanical. It is not — it implicitly answers the question *"did the old code do the right thing?"* — without recording the answer.

Parallel construction makes the answer explicit. Run both. Compare outputs. Record agreement or divergence. Only then collapse.

### Expand, then collapse

The ETL framing already supports this: artifacts are filesystem-backed, additive, and have no transactional coupling to the analyzer that produced them. Two analyzers can write parallel artifacts to disk; loaders can read either; dashboards can render either.

The platform expands during the parallel period (new + old analyzers coexist on disk and in the registry) and collapses back to the canonical set once validation is complete and REQ_102 retires the deprecated analyzers.

### Validation isn't a one-time gate

Surfacing a divergence months after deprecation would invalidate archived results. The harness should be re-runnable: same pattern applies whenever a primitive change rolls out. This makes the harness valuable beyond REQ_111's lifetime — it becomes the standing tool for "did this primitive update silently change downstream analyzer outputs?"

### Research-active additions skip the parity step

Lissajous and saddle-transport sigmoidality have no canonical "old analyzer" to compare against — the existing implementations live in notebooks (`sketch_lissajous_fit.py`, `saddle_transport_framework.ipynb`). For these, validation reduces to: "does the new analyzer reproduce the notebook's reported numbers on the reference variants?" That's still a parity check, just against a different anchor.

### Decoupling research questions from refactoring

The user's open research questions — *which frequency source(s)* for the saddle / Lissajous / heteroclinic ties, *which eigenvalues*, *which segments* — are scientific decisions, not refactoring decisions. Forcing them into the integration phase (as in-place migration would have) couples mechanical work with hard scientific choices.

Parallel construction lets each move at its own pace. The new analyzers can be implemented with conservative initial choices (matching the old analyzer's conventions where possible); research can iterate on top of them without blocking the deprecation path.

---

## Notes

### Validation outcomes (recorded as the work proceeds)

Format: `{old_analyzer} → {new_analyzer}: {outcome}, {date}, {pointer to evidence}`.

- *(empty until first new analyzer ships)*

### Open questions

- **Granularity of parallel artifact directories.** Default proposal: each new analyzer writes to `artifacts/{new_name}/` while old writes to `artifacts/{old_name}/`. No shared directories. May need to revisit if disk usage becomes an issue (REQ_109 follow-ups already note ~6 GB per variant for an `activation_snapshot` parity gap; some new analyzers may add comparable storage).
- **Dashboard convention for showing old + new side-by-side.** Could be a temporary toggle, a comparison view, or just two adjacent pages. Decide during implementation per page; the comparison is most valuable for analyzers where divergence is suspected (Fourier first; PCA likely "matches within tolerance").
- **When does the parallel period end?** Each analyzer pair has its own timeline. The overall REQ_111 closes when the last pair's validation outcome is recorded. After that, REQ_102 takes over for the actual retirement.

### Relationship to REQ_109 closure

REQ_109 closes cleanly once the primitive layer is complete (phase 2d / `procrustes_align`). The analyzer-migration content originally in REQ_109's "Conditions of Satisfaction" moves here. This avoids the situation where REQ_109 stays open for analyzer work that's actually research-active integration rather than primitive extraction.
