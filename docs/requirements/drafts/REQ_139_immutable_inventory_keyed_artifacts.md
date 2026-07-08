# REQ_139 (draft): Immutable, Inventory-Keyed Cross-Epoch Artifacts

**Status:** Draft — captured 2026-06-05 from the REQ_138 review discussion. Direction
agreed (keep REQ_138's default-at-root now; deliver the full snapshot model here).
**Relationship to REQ_138:** Completes the snapshot/lifecycle axis that REQ_138
deliberately left at "default-at-root, recompute-in-place." REQ_138 made a *generation
parameter* part of artifact identity (recipe-addressed storage); this makes the
*checkpoint inventory* part of cross-epoch artifact identity, so training extension
yields a new immutable snapshot rather than overwriting the prior analysis.
**Attribution:** Engineering Claude (under user direction).

---

## Problem Statement

A cross-epoch artifact is overwritten in place whenever the checkpoint inventory
grows: `planner._plan_cross_epoch_item` marks any cross-epoch artifact stale when
`available > covered` and the pipeline rewrites it. So extending a training run
**destroys the pre-extension analysis** — the prior trajectory analysis is gone, not
preserved as a dated observation. For a platform whose mission is *how did learning
happen?* and whose philosophy is *codified, audit-bearing analysis as inheritable
infrastructure* (see `project_codified_analysis_philosophy`), silently overwriting a
prior analysis on extension is the wrong default.

The REQ_138 review surfaced the precise structure of the problem: there are **two
extension-sensitive dimensions**, and only one is a parameter.

1. **Reference epoch** (e.g. `parameter_dmd`'s grouping snapshot) — a *declared
   generation parameter*, so REQ_138 can already address it (a floating `max_epoch`
   default vs an explicit pin).
2. **Trajectory length** — the cross-epoch artifact spans *all* available
   checkpoints. This is **not** a parameter and never appears in a recipe, so REQ_138
   cannot snapshot it.

Therefore value-addressing the floating default alone (the alternative considered in
the REQ_138 review) would preserve the reference-axis snapshot but still overwrite the
trajectory axis — a confusing half-snapshot. The clean lever is to make the
**inventory itself part of cross-epoch artifact identity.**

## Core idea (to design)

- A cross-epoch artifact is keyed by (recipe ⊕ **inventory signature**), where the
  inventory signature canonicalizes the set of checkpoints it was computed over.
- Extension → new inventory signature → a **missing** artifact, planned natively by
  the scheduler. The "missing > stale" principle (the p101 lesson) applied to the
  inventory axis: no bespoke staleness comparison needed.
- The prior artifact remains on disk as an **immutable, addressable snapshot** of the
  pre-extension analysis — reproducible and comparable (pre/post extension is a
  legitimate dynamics question).
- The floating-default reference then value-addresses *naturally and uniformly*
  alongside the inventory, with no special case.

## Payoff / simplification
- **Retire `freshness.check_reference_freshness`** (REQ_138 Phase 5): the
  inventory-derived staleness it computes by byte-comparison collapses into the
  planner's native missing-artifact handling once the inventory is part of identity.
- One uniform snapshot model across both extension-sensitive axes — no half-snapshot.

## Open questions / constraints (for the full write-up)
- **Inventory signature canonicalization** — exact checkpoint set vs (first, last,
  stride)? Coexistence + GC story for accumulating inventory snapshots (reuse the
  REQ_138 `_run_sets`/liveness + prune model, extended to inventory snapshots).
- **Scope** — cross-epoch artifacts only (per-epoch artifacts are already epoch-keyed
  and immutable by construction). Per-epoch artifacts unaffected.
- **Latest pointer** — a stable "current/latest" handle so consumers and the dashboard
  don't have to resolve an inventory signature to read "the live analysis."
- **Migration** — pre-v1.0.0, gitignored/regeneratable artifacts; one-time relocation
  of cross-epoch artifacts into inventory-keyed addresses, or regenerate.
- Honor the three architectural invariants (universal views, families-as-context,
  storage-internal-to-API) — inventory→path composition lives only in the storage
  primitive (`analyzer_dir` / `warehouse.paths`).

## Sequencing
After REQ_138 (recipe-addressed storage) and within the REQ_110 lakehouse line.
Builds on the REQ_133 scheduler (missing = planned) and the REQ_138 run-set
registry/liveness/GC machinery (extended to inventory snapshots).
