# REQ_133: Fluid Dependency Scheduler

**Status:** Stub — *deferred from REQ_132. High discovery risk; scope/CoS to be developed after a spike, not pre-specified.*
**Priority:** Low — internal architecture; follows REQ_132. Not blocking v1.0.0.
**Branch:** TBD
**Dependencies:**
- REQ_132 (collapses the phase taxonomy out of the contract — the boundary this work hides behind).

**Attribution:** Engineering Claude (stubbed 2026-05-30 from the REQ_132 scoping session).

---

## Problem

The pipeline executes analyzers in three fixed layers — primary → secondary → cross-epoch — even though analyzers already declare their real dependencies structurally via `ArtifactInput` ([inputs.py](../../../packages/miscope/src/miscope/analysis/inputs.py)). The layered model cannot express arbitrary dependency edges:

- A per-epoch analyzer cannot depend on a cross-epoch artifact.
- A cross-epoch analyzer cannot depend on another's per-epoch (secondary) output.
- `plan_analysis` only topo-sorts *within* the secondary layer (REQ_130's `_order_secondaries`); cross-layer order is hard-coded.

The goal is a **fluid dependency tree**: execution order determined solely by the `ArtifactInput` DAG, with the only structural axis being `output_scope` (per-epoch vs cross-epoch iteration). "Secondary" dissolves — it is just a per-epoch analyzer whose inputs are all `ArtifactInput`s, topo-ordered inside the epoch loop.

## Discovery risk (why this is a spike, not a spec)

This was deliberately *not* folded into REQ_132 because the full collapse carries real discovery risk — we don't yet know what the layered model is quietly relying on. Open unknowns to resolve **before** committing CoS:

- **Freshness/staleness over a real DAG.** Today staleness is per-layer (`_plan_cross_epoch_item`'s covered-epoch heuristic). What does "stale" mean when an arbitrary upstream changes? Does invalidation need to propagate transitively?
- **Cross-epoch → per-epoch edges.** Are they actually desired, and what iteration structure do they imply (a per-epoch analyzer reading a whole-trajectory artifact at each epoch)?
- **Hidden ordering assumptions.** What in the planner, freshness module, `_absorb_plan_references`, or the dashboard's plan preview assumes exactly three phases?
- **Scheduling within the epoch loop.** Per-epoch artifact-dependent analyzers must run after their upstreams *at the same epoch* — confirm same-epoch artifacts are written before dependents read them.

## Likely shape (provisional)

- Teach `plan_analysis` to topo-sort the full `ArtifactInput` DAG, not just the secondary layer.
- Replace the three `_run_*_from_plan` loops with two `output_scope`-keyed passes, each executing a topological order of the DAG.
- Rework freshness for transitive invalidation.

These are *candidates*, not committed scope. The most-tested planner/freshness logic is in the blast radius, so the audit-before-change discipline (REQ_102/REQ_129) applies in full.

## Conditions of Satisfaction

*(Deferred — develop after the spike resolves the discovery-risk questions above.)*

## Notes

- Independent of v1.0.0 close-out; can land any time after REQ_132.
- Recommend a throwaway spike branch first to surface the freshness and cross-edge questions empirically before writing CoS.
