# REQ_129: v1.0.0 Dead-Code Cull (Scaffolding Retirement)

**Status:** Stub — *findings parked, scope and CoS to be developed in a dedicated session.*
**Priority:** Medium — close-out track for v1.0.0. Not blocking; pairs with REQ_103.
**Branch:** TBD
**Dependencies:**
- REQ_102 (Analyzer Deprecation — retires superseded *analyzers*; this REQ retires superseded *scaffolding*).
- REQ_128 (Analyzer Input Provisioning — removes dead code **on the input path**; this REQ catches the rest).

**Relationships:**
- REQ_103 (PyPI Publication Hardening). The publishable library should not ship dead coexistence scaffolding. This REQ feeds REQ_103's release notes.

**Attribution:** Engineering Claude (stubbed 2026-05-28 from findings surfaced during the REQ_128 design session).

---

## Motivation

The work approaching v1.0.0 (first PyPI release) is the forcing function. The user has confirmed **no external back-compat needs** and **tolerance for system instability** while converging on a cleaner base. Several REQ-era coexistence layers (REQ_120/121 spec/IO unification, the ETL/planner introduction) left scaffolding that is now dead or near-dead. Carrying it into the published library raises maintenance burden and confuses external readers about which paths are canonical.

This REQ is the **ruthless cull** track the user flagged — deliberately separated from REQ_128 so that REQ_128 stays atomic ("the input contract changed") and this stays atomic ("dead scaffolding removed"). REQ_128 removes only the dead code co-located on the input-provisioning path; everything else lands here.

## Findings parked (from the 2026-05-28 REQ_128 audit)

These are *candidates*, not confirmed scope — each needs a use-check before removal (the REQ_102 "audit before deleting" discipline applies):

- **Legacy `category`-authoring scaffolding (the portion outside the input path).** 0 of 34 analyzers author the legacy `AnalyzerSpec.category` style; all are unified (`inputs=` / `output_scope=`). REQ_128 removes the input-path branches (`is_unified`, `effective_*` reads in `_materialize_*`); the planner's category-descriptor machinery (`planner.py` `effective_category`, the `category` field on its descriptor, the primary/secondary/cross-epoch classification) can likely simplify once authoring is gone. Confirm nothing external classifies by authored `category`.
- **REQ_121 Phase 2C three-protocol dispatcher.** `spec.py` documents a legacy three-protocol dispatch path kept "until Phase 2C retires it." If no legacy-style Specs remain (they don't), the dispatcher is dead.
- **General v1.0.0 surface review.** A dedicated pass over `packages/miscope/` for re-export shims, deprecated aliases, and `# removed`/back-compat comments left by prior REQs (e.g. the `modadd_intervention.py` deprecated re-export shim noted in project memory; the `TDW_*` legacy env-var aliases).

## Conditions of Satisfaction

*(Deferred — stub. Develop after REQ_102 and REQ_128 land, since they remove the analyzer- and input-path subsets and clarify what scaffolding actually remains.)*

## Notes

- **Audit before deleting.** Mirror REQ_102's safeguard: grep for each candidate, confirm no consumer (view, notebook, downstream analyzer, app), confirm family configs and `__init__`/`registry` don't reference it, then remove.
- **Sequencing.** Best run *after* REQ_102 and REQ_128 so the remaining dead set is well-defined and not a moving target.
