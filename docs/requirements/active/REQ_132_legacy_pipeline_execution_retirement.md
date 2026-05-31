# REQ_132: Legacy Pipeline-Execution & Query-Surface Retirement

**Status:** Stub — *findings parked from the REQ_129 cull; scope and CoS to be developed in a dedicated session.*
**Priority:** Low — close-out track for v1.0.0. Not blocking; pairs with REQ_103.
**Branch:** TBD
**Dependencies:**
- REQ_129 (v1.0.0 Dead-Code Cull — retired the legacy *authoring* scaffolding; deferred these two surfaces because they are API/structure decisions, not dead-code removal).

**Attribution:** Engineering Claude (stubbed 2026-05-30 from follow-ups flagged during the REQ_129 implementation).

---

## Motivation

REQ_129 removed the legacy `category`-authoring scaffolding and the registry's dead synthesis path. In doing so it deliberately drew a line: two surfaces are *dead in production but test-exercised*, and removing them changes either a **public API surface** or the **pipeline's execution structure** — decisions that deserve their own deliberation rather than riding along in a dead-code cull.

Both surfaces share one root: they are remnants of the **pre-unification category-phased model**. Before REQ_121 unified analyzers under a single `Analyzer` protocol + structural `inputs` declaration, the pipeline ran analyzers in three separate phases (primary / secondary / cross-epoch) and the registry indexed them by authored category. Authoring is now unified; the execution and query structure that mirrored it has not been collapsed.

The forcing function is the same as REQ_129's: a clean v1.0.0 published surface. Carrying a test-only legacy query API and a spec-less execution path into the first PyPI release confuses external readers about which paths are canonical.

## Findings parked (from the 2026-05-30 REQ_129 implementation)

Each is a *candidate*, not confirmed scope — the REQ_102/REQ_129 "audit before deleting" discipline applies.

1. **Pipeline spec-less analyzer support.**
   - The `spec is None` conservative branch in `AnalysisPipeline._materialize_per_epoch_inputs` (populates model + cache + logits + probe for analyzers with no registered Spec).
   - The `extra_allowed` / `depends_on` widening in `_run_secondary_analyzers` (widens the deps scope for spec-less secondaries via the legacy `depends_on` attribute).
   - These fire **only** for analyzers with no registered Spec. All 32 production analyzers register via `@register_analyzer`, so the path is exercised only by ad-hoc test analyzers.
   - Coupled to the pipeline's separate `_secondary_analyzers` execution phase and the `depends_on` attribute convention. Retiring it likely means collapsing the per-phase execution loops into a single dependency-ordered pass — a *structural* change. Confirm no test or research entry point relies on registering a bare analyzer instance without a Spec.

2. **Legacy class-based registry query API.**
   - `AnalyzerRegistry.get` / `get_secondary` / `get_cross_epoch` / `get_for_family` / `get_secondary_for_family` / `get_cross_epoch_for_family` / `list_all` (`registry.py`).
   - Post-REQ_129 these are test-only; `get_cross_epoch` / `get_secondary_for_family` / `get_cross_epoch_for_family` have **zero** consumers anywhere. The spec-based API (`get_spec` / `list_specs` / `list_specs_by_category` / `create` / `list_for_family`) is the canonical surface.
   - Removing them changes `AnalyzerRegistry`'s public surface — hence the explicit-decision framing. Migrating the test callers (`test_modulo_addition_family.py`, `test_analysis_library.py`) to the spec-based API is the bulk of the work.

## Decision needed (before scoping)

- **Collapse the pipeline's category-phased execution** into a single dependency-ordered pass, or keep the phases and only drop the spec-less branches? (Item 1 hinges on this.)
- **Retire the legacy class-based query API**, or keep it as a thin convenience layer over the spec-based API? (Item 2.)

## Conditions of Satisfaction

*(Deferred — stub. Develop once the two decisions above are made.)*

## Notes

- **Not urgent.** Both surfaces are inert in production; carrying them is a clarity cost, not a correctness risk.
- **Sequencing.** Independent of REQ_106/099. Slots into the v1.0.0 close-out track alongside REQ_103.
