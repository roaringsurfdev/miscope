# REQ_126: Family Basis Projection Consolidation

**Status:** Completed — staging (2026-05-27). Shipped across three PRs to `develop`:
- PR 1 (`weight_basis_projection`): weight-side absorption of `dominant_frequencies`, `attention_fourier`, `neuron_fourier`, and `fourier_nucleation`'s one-shot projection.
- PR 2 (`activation_basis_projection`): activation-side absorption of `attention_freq` and `neuron_freq_norm`; REQ_102 blob-vs-plaid gate verified on canon (coarseness recoverable transitively through reconstructed `neuron_freq_norm`).
- PR 3 (`centroid_fourier_alignment`): defused `fourier_alignment` field out of `repr_geometry` into a dedicated secondary analyzer; `circularity` retained on `repr_geometry` (geometric, not basis projection).

Outstanding follow-ups, separable from this REQ:
- `repr_geometry → representation_geometry` rename (86-file ripple; bookkeeping).
- Downstream visualization migration to consume the new analyzers' outputs.
- REQ_102 stays in `active`; coarseness/attention_freq/neuron_freq_norm etc. retire once downstream visualizations have ported over.

**Priority:** High — keystone of the Atlas-driven phase 1 reorganization. Unlocked REQ_102's `coarseness` retirement and cleared the Family Basis Projection column for downstream work (`gradient_site` generalization, `transient_frequency` generalization, future family-basis analyzers).
**Branch:** Merged to `develop` via `feature/req-126-weight-basis-projection`, `feature/req-126-activation-basis-projection`, `feature/req-126-repr-geometry-defusion`.
**Dependencies:**
- REQ_109 (measurement primitives — *staging*; the projection-onto-orthonormal-basis primitive consumed here).
- REQ_114 (HookedModel analyzer migration — *staging*; new analyzers built on the canonical interface).
- REQ_118 (neuron grouping — *staging*; downstream consumer for some basis-projection outputs).

**Hands off to:**
- REQ_102 (analyzer deprecation — `coarseness` retirement gated on this REQ's verification that `activation_basis_projection` preserves the blob-vs-plaid signal).

**Atlas reference:** [docs/analysis_atlas.md](../../analysis_atlas.md) — `weight_basis_projection` and `activation_basis_projection` entries under Family Basis Projections; `representation_geometry` entry under Universal Core (fused Fourier fields).

**Attribution:** Engineering Claude (under user direction, after the Analysis Atlas (a)(b)(c) audit pass).

---

## Problem Statement

The (a)(b)(c) audit pass on `docs/analysis_atlas.md` surfaced a load-bearing reorganization that has no owning REQ today. The Family Basis Projection column in the Atlas is currently scattered across six existing analyzers plus fused fields inside `representation_geometry`:

- **Weight-side:** `dominant_frequencies`, `attention_fourier`, `neuron_fourier`, and the one-shot projection step of `fourier_nucleation`.
- **Activation-side:** `attention_freq`, `neuron_freq_clusters`.
- **Fused into Universal Core analyzer:** `fourier_alignment` and `circularity` fields inside `representation_geometry`'s output dictionary.

This is the **keystone** of the next phase because:

1. It absorbs six analyzers into two universal ones — the largest single consolidation in the Atlas.
2. It splits family-basis content out of `representation_geometry`, tightening that analyzer's contract (its (a)(b)(c) audit flagged this as a refactor signal).
3. It unlocks REQ_102's `coarseness` retirement, which requires demonstrating that `activation_basis_projection` preserves the blob-vs-plaid signal before `coarseness` can be honestly retired.
4. The Family Basis column being a real place is a precondition for downstream refactors (`gradient_site`'s per-frequency → per-basis generalization; eventual `transient_frequency` generalization).

The architectural principle being honored: *families are context providers; views and analyzers are universal instruments* (per [PROJECT.md](../../../PROJECT.md)). The current analyzers have Fourier-locking in their names or implementations; consolidating into `weight_basis_projection` / `activation_basis_projection` with the basis as a parameter restores the universal/family-context split.

---

## Conditions of Satisfaction

### Universal analyzers

- [x] `weight_basis_projection` analyzer implemented as a universal instrument, parameterized by `(site, family_basis)`. Modadd family supplies a Fourier basis; other families supply different bases or none. Consumes `parameter_snapshot` artifacts.
- [x] `activation_basis_projection` analyzer implemented with the same shape, activation-side. Per the Q4 direction, reads the hook cache directly (`ModelInput(needs_cache=True)`) rather than `activation_snapshot` — hooks are the long-term primary path because LLM-scale models may not be able to materialize full activation snapshots.
- [x] Both analyzers consume only REQ_109 primitives for their transform step (projection onto orthonormal basis). Verified by grep test (`test_analyzer_imports_only_req109_basis_primitives`) in each test module.
- [x] The basis is a family-supplied parameter, not embedded in the analyzer. Verified: composer functions live in family implementations; the analyzer iterates the family's declared sites via `weight_basis_projection_sites` / `activation_basis_projection_sites` properties. Note: the analyzer still constructs `get_fourier_basis(prime)` internally — a slight Fourier-coupling acknowledged for PR 1 and left open until a non-Fourier family arrives (per user direction).

### Absorptions (weight side, into `weight_basis_projection`)

- [x] `dominant_frequencies` outputs reproducible from `weight_basis_projection` (`embedding` site sin/cos band norms across `d_model`).
- [x] `attention_fourier` outputs reproducible from `weight_basis_projection` (`attn_v` site for V band; `attn_qk` 2D site diagonal magnitudes for QK).
- [x] `neuron_fourier` outputs reproducible from `weight_basis_projection` (`mlp_in` and `mlp_out` site magnitudes).
- [x] One-shot projection step of `fourier_nucleation` reproducible from `weight_basis_projection` (`mlp_in` aggregate power normalized). **The iterative refinement of `fourier_nucleation` is preserved** — only the one-shot projection portion is absorbed; the iterative-refinement value lives on in `fourier_nucleation` per the Atlas (retain).

### Absorptions (activation side, into `activation_basis_projection`)

- [x] `attention_freq` outputs reproducible from `activation_basis_projection` (`attn_pattern` 2D site joint diagonal + per-axis marginals).
- [x] `neuron_freq_clusters` (`neuron_freq_norm`) outputs reproducible from `activation_basis_projection` (`mlp_out` 2D site joint diagonal + per-axis marginals). The "neuron clusters" semantic — argmax-by-frequency assignment — lives downstream in `neuron_grouping`'s modadd override (already shipped in REQ_118), so the cluster-assignment portion has its own home.

### `representation_geometry` defusion

- [x] `fourier_alignment` removed from `repr_geometry`'s output schema. **`circularity` retained** (Q3 direction: it's geometric — Kåsa circle fit on 2D PCA — not basis projection). The analyzer's contract narrows from 11 to 10 scalar keys per site.
- [x] `fourier_alignment` signal re-emerges from the new `centroid_fourier_alignment` secondary analyzer (per the Q2 direction: separate analyzer, composition through analyzer chaining, cleaner audit). Reproducibility verified on canon at epoch 24999 across all four sites.

### REQ_102 gate condition

- [x] Verified `activation_basis_projection` outputs preserve the blob-vs-plaid signal currently produced by `coarseness`. Verified on canon (p113/s999/ds598) via `test_req102_coarseness_recoverable_from_activation_basis_projection`: reconstructed coarseness matches legacy element-wise; blob/plaid classification (threshold 0.7) matches element-wise. Expanded variant coverage (p109/s485/ds598, p101/s999/ds598) deferred to REQ_102's own close-out per user direction (2026-05-27).

### Migration of consumers

- [ ] Library code that referenced the absorbed analyzers migrates to the new ones. **Deferred** — existing analyzers stay live in the deprecation window; consumers continue to read the original artifacts. The new analyzers are additive.
- [ ] Dashboard pages that consumed the absorbed analyzers either migrate or are noted as future cleanup. **Deferred to a downstream-visualization-migration requirement** per user direction (2026-05-27).
- [ ] Notebook / export usage migrates to the new analyzers. **Deferred** with the dashboard work.
- [x] Old artifact paths remain readable via `ArtifactLoader` (existing on-disk artifacts not invalidated; only the analyzer source code changed for `repr_geometry`, which still emits all retained per-site keys).

---

## Constraints

**Must:**
- Family supplies the basis; the analyzer encodes no basis internally.
- Both analyzers use only REQ_109 primitives for their transform step. Auditable via grep.
- Every output field of the six absorbed analyzers is recoverable from the new analyzers' outputs (possibly after a per-site slice).
- `fourier_nucleation`'s iterative refinement is preserved as a distinct analyzer; only the one-shot projection step is absorbed.
- Existing artifacts on disk remain readable for the deprecation window. No release in which `ArtifactLoader` loses the ability to read the old artifacts.

**Must avoid:**
- **Re-encoding Fourier-specific assumptions** in the new analyzers' names or core logic. The basis parameter is the only family-specific input.
- **Silent information loss** during the `representation_geometry` defusion. Verify that `fourier_alignment` and `circularity` signals are reproducible from the new outputs before removing them.
- **Coupling this work to dashboard cleanup.** The dashboard migration is acknowledged as future work and not blocking. Don't conflate "two new analyzers exist + verification passes" with "all surface area migrated."

**Flexible:**
- Whether `representation_geometry` outputs the basis-projection results as a sub-key (compositional reference) or stops outputting them entirely. Default: stop outputting; consumers go to the basis-projection artifacts directly.
- Order of weight-side vs. activation-side. Default: weight-side first (cleaner numerical anchor — `dominant_frequencies` outputs are easy parity targets); activation-side after, since the blob-vs-plaid gate for REQ_102 is on the activation side and benefits from seeing the weight-side pattern first.
- Whether to land both analyzers in one PR or two. Default: two — they share a primitive but their absorption sets are independent.

---

## Context & Assumptions

### Why this is the keystone

Per the Atlas (a)(b)(c) audit pass:

- Six existing analyzers are bucketed `reorganization` and target `weight_basis_projection` / `activation_basis_projection`. None individually warrants its own REQ; together they're the largest single consolidation in the Atlas.
- The Family Basis Projection column is conceptually a stub today. After this REQ, it's a real place that other analyses can build on.
- The `representation_geometry` refactor (REQ_111 phase 2 in the proposed phasing) depends on the Fourier fields having a new home.
- The `coarseness` retirement (REQ_102) is blocked on demonstrating blob-vs-plaid preservation, which can only happen once `activation_basis_projection` exists.

### Disruption is acceptable

The user has explicitly accepted the migration cost. Verbatim:
- *"This disruption is necessary, and better now than later."*
- *"My expectation for the dashboard is that we might be able to get away with hiding some of the churn under the visualization layer."*
- *"If analysis result disappear or are diminished after refactor, then I'm glad I waited."*

This REQ takes that authorization at face value. Implementation prioritizes structural correctness over preserving every existing consumer's exact output shape.

### Verification spirit

This REQ's deliverable is not just "two new analyzers exist" — it's "the Family Basis column is a real place, and consolidation didn't lose signal." The blob-vs-plaid preservation check is the most concrete instance. Analogous checks for other absorbed content (Fourier alignment from `representation_geometry`, per-site spectra from the four weight-side analyzers) belong in implementation.

### What this REQ does NOT do

- Does not retire any analyzer. Retirements live under REQ_102, which cites this REQ's verification outcomes.
- Does not generalize `gradient_site` (separate refactor; depends on this REQ's basis column being in place).
- Does not generalize `transient_frequency` (still Fourier-locked per Atlas; rename + generalization in a future REQ once `neuron_grouping` consumption pattern stabilizes).
- Does not touch `fourier_nucleation`'s iterative refinement (retained as-is per Atlas).
- Does not migrate dashboard pages comprehensively — only the minimum to keep the build green.

---

## Notes

- The phasing this REQ slots into was derived from the Analysis Atlas (a)(b)(c) annotation pass. See the Atlas's consolidation map for the canonical list of which existing analyzers fold into which new ones.
- This REQ pairs naturally with the narrowed REQ_111 (Universal Core pure renames) and the gated REQ_102 (analyzer deprecation). Together they form phase 1 of the Atlas-driven reorganization.
