# REQ_126: Family Basis Projection Consolidation

**Status:** Draft
**Priority:** High — keystone of the Atlas-driven phase 1 reorganization. Unlocks REQ_102's `coarseness` retirement and clears the Family Basis Projection column for downstream work (`gradient_site` generalization, `transient_frequency` generalization, future family-basis analyzers).
**Branch:** TBD
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

- [ ] `weight_basis_projection` analyzer implemented as a universal instrument, parameterized by `(site, family_basis)`. Modadd family supplies a Fourier basis; other families supply different bases or none. Consumes `parameter_snapshot` artifacts.
- [ ] `activation_basis_projection` analyzer implemented with the same shape, activation-side. Consumes `activation_snapshot` (or the current activation capture path).
- [ ] Both analyzers consume only REQ_109 primitives for their transform step (projection onto orthonormal basis). Verified by grep: no inline `np.fft`, `np.einsum`-rolled-projection, or other ad-hoc basis math in `analyze()`.
- [ ] The basis is a family-supplied parameter, not embedded in the analyzer. Verified by reading the analyzer source: no `import` of Fourier-specific helpers, no `if family_name == "modadd"` branches.

### Absorptions (weight side, into `weight_basis_projection`)

- [ ] `dominant_frequencies` outputs reproducible from `weight_basis_projection` (W_E site).
- [ ] `attention_fourier` outputs reproducible from `weight_basis_projection` (attention-site weight projections).
- [ ] `neuron_fourier` outputs reproducible from `weight_basis_projection` (MLP-input weight projections).
- [ ] One-shot projection step of `fourier_nucleation` reproducible from `weight_basis_projection`. **The iterative refinement of `fourier_nucleation` is preserved** — only the one-shot projection portion is absorbed; the iterative-refinement value lives on in `fourier_nucleation` per the Atlas (retain).

### Absorptions (activation side, into `activation_basis_projection`)

- [ ] `attention_freq` outputs reproducible from `activation_basis_projection` (attention-site activation projections).
- [ ] `neuron_freq_clusters` outputs reproducible from `activation_basis_projection` (MLP activation projections; cluster assignment may remain as a downstream derivation).

### `representation_geometry` defusion

- [ ] `fourier_alignment` and `circularity` fields removed from `representation_geometry`'s output schema. The geometry analyzer's contract narrows to pure class-manifold geometry (centroids, radii, dimensionality, SNR, Fisher discriminants, PCA variance per PC).
- [ ] These signals re-emerge from `activation_basis_projection` (Fourier alignment) and downstream geometry on basis-projected representations (circularity, if still wanted as a derived metric). Verify reproducibility on the canon reference set before deletion.

### REQ_102 gate condition

- [ ] Verify `activation_basis_projection` outputs preserve the blob-vs-plaid signal currently produced by `coarseness`. Compare on the canon reference set (p113/s999/ds598, p109/s485/ds598, p101/s999/ds598). Outcome recorded as evidence under REQ_102 before `coarseness` retirement proceeds.

### Migration of consumers

- [ ] Library code that referenced the absorbed analyzers (e.g., `ArtifactLoader.load_*` calls naming the old analyzers) migrates to the new ones, or returns a deprecation pointer.
- [ ] Dashboard pages that consumed the absorbed analyzers either migrate to the new ones or are noted as future cleanup. **Per user direction, dashboard churn may be hidden behind the visualization layer for now** — this is acknowledged as a separate cleanup track and is not blocking.
- [ ] Notebook / export usage migrates to the new analyzers. **Per user direction, notebooks/exports are not blocking** — sharing has been waiting on a stable basis; any signal that diminishes after refactor is a finding worth surfacing, not a regression to mask.
- [ ] Old artifact paths remain readable via `ArtifactLoader` for the deprecation window (existing on-disk artifacts not invalidated).

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
