# REQ_133: Fluid Dependency Scheduler

**Status:** Spike complete — *CoS drafted from spike findings (2026-06-01). Ready for scoping review.*
**Priority:** Low — internal architecture. Not blocking v1.0.0. Sequenced **after REQ_134** (see Dependencies).
**Branch:** TBD
**Dependencies:**
- REQ_132 (collapses the phase taxonomy out of the contract — the boundary this work hides behind).
- REQ_134 (regression harness) lands **first**, by decision 2026-06-01: it restores the byte-regression
  safety net on the stable `plan_analysis` interface this work preserves, and becomes the oracle for an
  ordering-only refactor of the most-tested planner/freshness logic. A clean-from-scratch regression mode,
  if wanted, depends on *this* requirement (see Spike finding 2).

**Attribution:** Engineering Claude (stubbed 2026-05-30 from the REQ_132 scoping session; spike run 2026-06-01).

---

## Problem

The pipeline executes analyzers in three fixed layers — primary → secondary → cross-epoch — even though analyzers already declare their real dependencies structurally via `ArtifactInput` ([inputs.py](../../../packages/miscope/src/miscope/analysis/inputs.py)). The layered model cannot express arbitrary dependency edges:

- A per-epoch analyzer cannot depend on a cross-epoch artifact.
- A cross-epoch analyzer cannot depend on another's per-epoch (secondary) output.
- `plan_analysis` only topo-sorts *within* the secondary layer (REQ_130's `_order_secondaries`); cross-layer order is hard-coded.

The goal is a **fluid dependency tree**: execution order determined solely by the `ArtifactInput` DAG, with the only structural axis being `output_scope` (per-epoch vs cross-epoch iteration). "Secondary" dissolves — it is just a per-epoch analyzer whose inputs are all `ArtifactInput`s, topo-ordered inside the epoch loop.

## Spike findings (2026-06-01)

The four discovery-risk questions below were resolved empirically by walking all 23 family
specs' `ArtifactInput` edges against upstream `output_scope`, and by exercising the planner
primitives against a clean (empty) artifacts dir. Results:

1. **Cross-epoch → per-epoch edges are YAGNI.** Zero such edges exist across the 23 specs, and
   none are latent. The riskiest design question — a per-epoch analyzer reading a whole-trajectory
   artifact at each epoch — has no consumer. **Out of scope:** `output_scope` stays a clean
   2-valued axis; the topo-sort never crosses scopes downward. Building for this now would be
   speculative.

2. **The real gap is cross-epoch → cross-epoch ordering.** Three such edges exist:
   `neuron_group_pca → intragroup_manifold`, `neuron_dynamics → transient_frequency`,
   `global_centroid_pca → activation_dmd`. The planner does **not** order them:
   `cross_epoch_descs` is built in list order with no topo-sort (only secondaries get
   `_order_secondaries`), and the cross-epoch loop never feeds cross-epoch outputs back into
   `projected_completed` ([planner.py:294-304](../../../packages/miscope/src/miscope/analysis/planner.py#L294-L304)).
   Confirmed consequence on a clean artifacts dir — all three come out **blocked** in a single pass:

   ```
   intragroup_manifold   blocked_by=('neuron_group_pca',)
   transient_frequency   blocked_by=('neuron_dynamics',)
   activation_dmd        blocked_by=('global_centroid_pca',)
   ```

   They succeed today only because the upstream `cross_epoch.npz` already exists on disk from a
   prior incremental run (every analyzed variant in `data/` masks the gap). This is the concrete
   "cross-layer order is hard-coded" fragility — and the reason a clean-from-scratch regression
   run (REQ_134) is gated on this requirement.

3. **The three-phase assumption surface is contained.** It lives in exactly three places: the
   `Plan` dataclass's three lists + `format`/`to_dict` ([planner.py](../../../packages/miscope/src/miscope/analysis/planner.py)),
   the three run passes + `_absorb_plan_references` in
   [pipeline.py:214-219](../../../packages/miscope/src/miscope/analysis/pipeline.py#L214-L219),
   and freshness — which **already** merges per_epoch+secondary into one bucket
   ([freshness.py:221-222](../../../packages/miscope/src/miscope/analysis/freshness.py#L221-L222)).
   The **dashboard does not branch on the three phases** — it only calls `plan.format()` and reads
   `FreshnessReport`. The preview surface is safe.

4. **Same-epoch write-before-read is new, not pre-existing.** Secondaries currently run as a
   separate full pass *after* all primaries complete, reading finished per-epoch artifacts from
   disk — so write-before-read is trivially satisfied. Collapsing "secondary" into per-epoch
   topo-order *inside the epoch loop* introduces a new invariant: at epoch N, an upstream
   per-epoch analyzer (e.g. `neuron_grouping`) must write `epoch_N` before its dependent
   (e.g. `fourier_frequency_quality`) reads it. This warrants an explicit test, not a redesign.

## Direction

Informed by the spike (candidates above are now firmer, not speculative):

- Teach `plan_analysis` to topo-sort the full `ArtifactInput` DAG within each `output_scope`,
  not just the secondary layer. Seed `projected_completed` with cross-epoch outputs so downstream
  cross-epoch items are not falsely blocked (the Finding 2 fix).
- Replace the three `_run_*_from_plan` loops with two `output_scope`-keyed passes, each executing
  a topological order of the DAG. "Secondary" dissolves into per-epoch topo-order.
- Rework freshness for transitive invalidation: when an upstream regenerates, dependents
  (transitively) replan as stale.
- Hold `output_scope` to two values; do **not** add cross-epoch→per-epoch support (Finding 1).

The most-tested planner/freshness logic is in the blast radius, so the audit-before-change
discipline (REQ_102/REQ_129) applies in full. REQ_134's restored byte-regression net is the oracle:
this is an *ordering-only* change — analyzer outputs must remain byte-identical.

## Conditions of Satisfaction

1. **DAG-driven order.** Execution order is derived solely from the `ArtifactInput` DAG,
   topologically sorted within each `output_scope`. "Secondary" is no longer a distinct phase in
   the planner, pipeline, or `Plan`.
2. **Cross-epoch → cross-epoch satisfiable in one pass.** On a clean variant with the full family
   spec set, a single `plan_analysis` + `pipeline.run` produces `intragroup_manifold`,
   `transient_frequency`, and `activation_dmd` (no `blocked_by`). Regression test asserts this.
3. **`projected_completed` accounts for cross-epoch outputs**, so downstream cross-epoch items are
   not falsely blocked when their upstream is planned in the same run.
4. **Transitive staleness.** When an upstream artifact is regenerated (or absent), its transitive
   dependents are replanned as stale/missing. Test covers a 2-hop chain.
5. **Same-epoch write-before-read.** Per-epoch artifact-dependent analyzers run after their
   per-epoch upstreams at the *same* epoch; a test asserts the dependent reads the upstream's
   `epoch_N` written earlier in the same epoch iteration.
6. **No cross-epoch → per-epoch support** is added (Finding 1); the 2-valued `output_scope` axis is
   documented as the only structural axis.
7. **Byte-regression parity.** Analyzer outputs are byte-identical to pre-REQ_133 — validated by the
   REQ_134 regression harness (ordering-only change).
8. **Consumer surfaces intact.** The dashboard plan preview and `FreshnessReport` consumers continue
   to work after the three-phase `Plan` surface is reworked.

## Notes

- Independent of v1.0.0 close-out; can land any time after REQ_134.
- Spike was non-destructive (read-only planner probes); no throwaway branch was needed — the
  findings above replace the originally-recommended spike branch.
- REQ_134 sequenced first per decision 2026-06-01 (restores the byte-regression oracle on the stable
  interface this work preserves).
