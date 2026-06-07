# REQ_145: Signature-Based Incremental Refresh — One Freshness Predicate for Artifacts and Tables

**Status:** Active (design surfaced across a 2026-06-06 architecture session; promoted
from the `platform_ideas.md` "Data Refresh" note).
**Priority:** High — a full dumb re-run + re-materialize of one variant is ~40 min,
which currently quarantines all work to 3 pinned variants and blocks an affordable
full rebuild (REQ_137). This is the lever that makes both cheap.
**Branch:** TBD (`feature/REQ_145_signature_based_incremental_refresh`); lands **after
REQ_141 merges to `develop`** and **before REQ_144** (sequencing decided 2026-06-06).
**Parent:** REQ_133 (Fluid Dependency Scheduler) — this evolves the freshness DAG from
a coverage check into a signature-based predicate. Builds on REQ_107 (version/output
schema), REQ_138 (recipe coordinate), REQ_110A/C (warehouse + query), REQ_141 (derived
tables). Interacts with REQ_140 (materializer scope/isolation) and REQ_137 (full refresh).
**Attribution:** Engineering Claude (under user direction)

---

## Problem Statement

The pipeline has exactly two ways to decide whether work must run, and **neither is
both correct and selective**:

1. **`force` — correct but blunt.** Recomputes every applicable epoch for every
   analyzer regardless of on-disk state, then the warehouse wipes and rebuilds the
   whole columnar + derived plane. This is the ~40-minute path. It is the only way to
   *guarantee* a changed analyzer's output (and everything downstream) is rebuilt.

2. **Coverage-freshness — cheap but blind.** The planner's non-force path
   (`_plan_cross_epoch_item`, `_per_epoch_target_epochs`) decides staleness purely by
   **epoch counts**: a cross-epoch artifact is stale only when more checkpoints exist
   than it covers (`threshold = max(max_dep_epochs, len(available_epochs)) > covered`),
   and a per-epoch artifact is fresh when its `epoch_*.npz` files are present. There is
   **no mtime, content, version, or recipe comparison between an upstream and its
   consumer.** Refactor an analyzer and recompute it over the *same* epoch set and
   every downstream still reports fresh — the gap diagnosed in this session.

The motivating failure: split one analyzer into two (or change one analyzer's logic).
The new analyzer shows "absent"; the original still looks "fresh"; downstream
consumers — which read the old output's *meaning* — are never flagged. The only
recourse is `force`, which rebuilds everything.

The warehouse layer inherits the same blind spot, currently masked by brute force.
`materialize_variant_columnar` calls `_wipe_columnar_outputs(variant)` and rebuilds
the entire columnar plane from the family's current analyzers; `materialize_variant_derived`
rebuilds every registered derived table whose inputs exist. When triggered, this is
*correct* (full rebuild from current artifacts), but it is total — and the only
incremental behavior that exists, the **absent-table self-heal**, is destructive: it
otherwise does a full wipe, a hazard observed mid-migration during REQ_141 smoke
testing (see `platform_ideas.md`, 2026-06-06 Data Refresh note). There is **no
artifact-vs-table staleness detection.**

**Thesis (the consolidation, not an addition):** a per-output **provenance signature**,
stamped at write time and compared at plan time, is a single freshness predicate that
*replaces* both mechanisms above — correct like `force`, selective like coverage. It
makes "change one analyzer → recompute that analyzer + its transitive dependents +
re-materialize only the affected tables" the default, and demotes `force` to an
explicit "ignore signatures" override.

This is a deliberate **build, not buy** (verdict recorded in the build-vs-buy layering
memo). The staleness model itself is well-established: the same triple of *code
version + data version + automatic staleness propagation across a dependency graph*
recurs across build and orchestration systems (Dagster's software-defined assets, DVC,
Nix, Bazel). REQ_145 is **inspired by** that established pattern and implements it
natively rather than depending on it. A review of off-the-shelf orchestration tools 
found each a poor fit for our needs: dbt is SQL-model-centric and doesn't fit a Python analyzer + npz / tensor model; DVC would impose a parallel pipeline-DAG declaration duplicating our `requires` graph; DuckLake/Delta/Iceberg version *storage*, not *compute*; and a full asset framework (e.g. Dagster) solves it but requires adopting a dependency in
the orchestration layer the project deliberately keeps thin. Native implementation is
favored because the substrate is already owned: the `requires` DAG and its topological
sort (`planner._topo_order`), `AnalyzerSpec.version` (REQ_107), the `recipe` coordinate
(REQ_138), the declared `outputs` schema, and the publish schema gate's per-field diff
logic (`publish/schema_gate.diff_table`) all already exist. The marginal build is
signature compute + a write-time stamp + a plan-time compare; the marginal
*integration* cost of any off-the-shelf tool (re-expressing the pipeline in its
asset/stage model) is far larger.

## Mechanism (proposed — see Decision Authority for the open forks)

A node's recompute decision is an **input-derived signature**:

```
sig(node) = hash(
    code_version,              # AnalyzerSpec.version (manual) and/or a source hash
    recipe,                    # REQ_138 parameterization, already first-class
    checkpoint_set_identity,   # NEW: what makes model-driven primaries skippable
    sorted(stored_sig(u) for u in declared upstream inputs),   # Merkle edge up the DAG
)
```

- **Stamp** `sig` into the produced artifact (and its catalog row) at write time, via
  the storage primitives — never a path literal (constraint 3).
- **Compare** at plan time: this rides the planner's existing topo pass — a
  `projected_signature` map alongside `projected_completed`. For each node in
  topological order, compute `would_sig` from current code/recipe/checkpoints and the
  *projected* signatures of upstreams (their stored sig if not being recomputed, else
  their new `would_sig`). Differ → emit a recompute PlanItem; equal → skip. Because a
  downstream folds in its upstreams' signatures, **a changed upstream automatically
  yields a changed downstream signature — forward propagation through the DAG does the
  invalidation.** No reverse walk is needed for *execution*.
- The **reverse-edge index** (inverting `requires` / `input_tables`) earns its keep
  only for *reporting* — human-facing impact preview ("this change touches X, Y, Z,
  ~N min") and *scoping* the warehouse's selective re-materialize — not for the core
  recompute decision. It is therefore secondary, not load-bearing.

Two load-bearing notes carried from the design discussion:
- **`checkpoint_set_identity` is what makes the dominant cost skippable.** The 40 min
  is mostly model-driven primaries doing forward passes per checkpoint; they have no
  analyzer upstream to key on. Without a checkpoint fingerprint they can never be
  proven fresh, and we are back to `force`.
- **The refactor case is caught without output hashing**, *provided code changes are
  reflected in the code-version component* (manual `spec.version` bump, or an auto
  source hash). An optional output **data-version hash** is the robustness layer for
  value-drift-at-constant-shape (e.g. nondeterminism) — not a requirement, and it is
  non-circular only because the recompute *decision* signature is input-derived (it
  never depends on the node's own output).

## Conditions of Satisfaction

- [ ] **Every produced artifact and materialized table carries a stamped provenance
  signature**, written through the storage primitives and readable through an accessor
  (no consumer composes a path).
- [ ] **The planner recomputes a node iff its signature changed** — replacing the
  count-based coverage check as the freshness predicate. A node that is merely
  coverage-complete but whose code/recipe/upstreams changed **is recomputed**; a node
  that is coverage-complete *and* signature-equal **is skipped**.
- [ ] **A single-analyzer change recomputes that analyzer and its transitive dependents
  only** — demonstrated end-to-end: change one analyzer, observe the others skip.
- [ ] **Model-driven primaries are skippable** when checkpoints and code/recipe are
  unchanged (i.e. `checkpoint_set_identity` participates in the signature). A re-run
  with no changes is a no-op, not a 40-minute rebuild.
- [ ] **The warehouse re-materializes only tables whose source signature changed**
  (surgical re-materialize), replacing the destructive full wipe as the default. The
  absent-only self-heal's destructive-wipe hazard is removed.
- [ ] **`force` becomes an explicit "ignore signatures" override**, not the primary
  refresh path — one predicate, one override.
- [ ] **Skip transparency** — the plan/event surface reports *why* each node was
  skipped or recomputed ("fresh: signature unchanged" / "stale: upstream `X` changed"
  / "stale: code v2→v3"), answering the "why didn't X re-run?" question from the
  Pipeline Observability note.
- [ ] **`scripts/run_analysis.py` reaches refresh parity with the dashboard path** —
  analyze → materialize → summarize as one signature-aware flow (the (b) item from the
  Data Refresh note); no lingering explicit two-step that leaves tables stale.
- [ ] **No second staleness mechanism** — the signature *is* the freshness DAG's
  predicate (REQ_133 evolved in place); REQ_141/REQ_144 derived tables defer to it,
  not to a parallel check.

## Validation

- [ ] **Byte-parity on the three baselines** (p113/s999/ds598, p109/s485/ds598,
  p101/s999/ds598; never `find | head`). A full signature-aware rebuild produces
  artifacts/tables value-identical to a `force` rebuild (`rtol=1e-3` per REQ_126 only
  where a float recompute is unavoidable; integer counts/epochs exact). A
  shape-of-behavior change is a finding, not noise.
- [ ] **Selectivity demonstrated, not asserted** — a measured before/after: a
  single-analyzer change costs (that analyzer + dependents + affected tables), and a
  no-change re-run costs ~0, against the ~40-min `force` baseline.
- [ ] **Invalidation correctness** — bump a producer's `version` (or change its code)
  and confirm every transitive dependent restages and no non-dependent does; confirm a
  changed upstream propagates through a multi-hop chain
  (`activation_basis_projection → neuron_frequency_attribution → neuron_dynamics`).
- [ ] **Stale-but-coverage-complete is caught** — the turn-1 regression: an artifact
  with full epoch coverage but a changed upstream is correctly flagged stale (the case
  the old predicate missed).
- [ ] **Regression net stays green** — `test_freshness.py`, `test_planner.py` (if
  present), `test_warehouse.py`, plus the REQ_141 derived-table parity harness.

## Constraints

**Must have:**
- Signature stamped/read **only through storage primitives + accessors** (constraint 3,
  storage-encapsulation invariant).
- The signature must capture, at minimum: code version, recipe (REQ_138), and upstream
  signatures; and a checkpoint identity for model-driven primaries.
- A **swappable component boundary**: a `signature(node, upstream_sigs) -> str` plus
  stamp/read accessors, with **zero bleed into analyzer bodies** — so a later
  adoption of an off-the-shelf orchestration tool (should that ever be re-justified)
  maps onto this seam rather than unwinding it.

**Must avoid:**
- Taking an **orchestration dependency** (build-vs-buy verdict) or otherwise thickening
  the orchestration layer.
- A **second staleness mechanism** running alongside coverage — this *replaces* it.
- Breaking determinism / parity, or making the signature depend on the node's own
  output (the circularity trap).
- A destructive full wipe as the default warehouse path.

**Flexible:**
- Where the signature physically lives (npz key vs. sidecar vs. catalog-row-only).
- How rich the impact-preview / reverse-edge reporting is in v1.
- Whether the optional output data-version hash ships now or is deferred.

## Context & Assumptions

- This requirement is the codification of a 2026-06-06 design discussion. The relevant
  current-state findings (verify before building):
  - One-way DAG: `planner._topo_order` walks `requires` (upstream) edges only; there is
    no persisted reverse (dependents) index.
  - Coverage-only staleness: `_plan_cross_epoch_item` / `_per_epoch_target_epochs`
    compare epoch *counts*, never content/version/mtime.
  - `AnalyzerSpec.version` (REQ_107) is **code-only** — never stamped into artifacts —
    and its compat check (`registry._index._check_source`, `producer.version <
    src.min_version`) is code-vs-code at `registry.load()`. `min_version` lives **only
    on `DataViewSource`**; `ArtifactInput` has no version field, so the
    analyzer→analyzer edge has no version awareness today (a small adjacent fix:
    mirror `min_version` onto `ArtifactInput`).
  - The publish schema gate (`publish/schema_gate.py`) is a **structural** column diff
    against a *committed baseline manifest* — its `diff_table` classification
    (identical/additive/breaking) is reusable for invalidation *severity*; its
    publish-baseline plumbing is the wrong anchor for at-rest invalidation.
  - Warehouse `materialize_variant_columnar` is wipe-and-rebuild; `materialize_variant_derived`
    rebuilds all present-input tables. Recipe coordinate (REQ_138) is already threaded.
- Parking-lot lineage: `platform_ideas.md` — **2026-06-06 "Data Refresh — Guide First,
  Then Consolidation"** (this REQ is its narrowed scope (a) surgical/incremental +
  (b) `run_analysis.py` parity; the destructive-wipe hazard is its motivating
  episode) and **2026-06-06 "Pipeline Observability"** (the skip-transparency CoS, and
  per-analyzer duration as the natural way to *measure* the selectivity win).
- **Sequencing — why REQ_145 before REQ_144:** REQ_144 introduces *more* derived
  tables (`variant_outcomes`, window metrics) downstream of analyzers. Building them on
  a coverage-blind DAG means they are born with the same staleness blindness; landing
  the signature predicate first means REQ_144's new tables inherit a correct
  invalidation regime from day one. REQ_144 is also a large refactor whose own dev loop
  suffers the 40-min cost — incremental refresh makes that iteration affordable.
- **Sequencing — relationship to REQ_137 (full refresh):** the mass rebuild of all
  ~28 stale variants the user is strategically deferring *is* REQ_137. REQ_145 should
  land **before** that rebuild executes, so artifacts are born signature-stamped (and
  the rebuild itself becomes resumable/incremental), and so that after the rebuild a
  future single-analyzer change never re-triggers a full 40-min pass.
- Assumes the v1.0.0 no-back-compat cutoff: stamping a new signature field and changing
  the freshness predicate needs no coexistence shim; a one-time rebuild re-stamps
  existing artifacts (folds into REQ_137).

## Decision Authority
- [x] **Propose options for review** — open forks:
  - **(a) Code-version source:** manual `AnalyzerSpec.version` (honest, simple,
    discipline-dependent) vs. an auto source hash (no discipline, but brittle to
    cosmetic edits) vs. both. Default leaning: manual `version` for v1, source hash as
    a later robustness layer.
  - **(b) Checkpoint identity:** what to fingerprint (epoch set + safetensors content
    hashes vs. mtime+size vs. a per-training-run id) and how cheaply, given this is
    read at every plan.
  - **(c) Output data-version hash:** ship now (catches value-drift-at-constant-shape
    / nondeterminism) or defer (rely on code-version + `force`).
  - **(d) Reverse-edge index + impact preview:** build in v1 (better UX, scopes
    warehouse re-materialize) or defer (forward propagation already covers execution).
  - **(e) Signature stamp location:** npz key vs. sidecar file vs. catalog-row-only.

## Success Validation

"Done" looks like: changing a single analyzer and re-running produces *only* that
analyzer's artifacts, its transitive dependents', and the warehouse tables sourced from
them — measured to be a small fraction of the ~40-min `force` baseline — with byte-parity
to a `force` rebuild on the three baselines, and a plan that *explains* every skip and
recompute. A no-change re-run is a near-instant no-op. `force` still exists, but only as
the explicit "rebuild everything regardless" override.

The deeper success state is the **absence of a manual refresh runbook.** Today a
contributor must know which of several entry points to run to bring artifacts + tables
back in sync (the two-layer trap in the Data Refresh note). Once REQ_145 is in place
that knowledge should be unnecessary: staleness is detected and the correct work runs,
whether triggered automatically or through a single entry point. A manual "what do I
run to refresh X" guide is justified only **transitionally, during a refactor phase**
(e.g. a migration in flight, or the REQ_137 mass rebuild) when the automatic path is
not yet established — a short-lived doc, not a standing policy.

---
## Notes
- Framing to hold onto: this is a *consolidation* (one predicate replacing `force` +
  coverage), not new surface area — which is what keeps the effort and risk bounded.
- The hardest design risk concentrates in fork (a) code-version and fork (b) checkpoint
  identity; both should be settled before implementation, as the rest of the mechanism
  is mechanical given the existing topo pass.
- The "next already-sorted problem around the corner" hedge is the swappable-component
  boundary (Must-have): keep invalidation logic behind one seam so a future tool, if
  ever justified, maps onto it instead of requiring an unwind.

---
## Implementation guidance (carried from the REQ_141 implementation session, 2026-06-06)

Concrete starting points from having just built REQ_141 and lived the gap this REQ
closes. Verify against the code before relying on them.

1. **REQ_141 already hands you a live multi-hop fixture — use it as the first
   invalidation test.** Reshaping `neuron_dynamics` bumped its `AnalyzerSpec.version`
   to **2** (see `analysis/analyzers/neuron_dynamics.py`), and introduced the real
   3-hop chain `activation_basis_projection → neuron_frequency_attribution →
   neuron_dynamics`. That is exactly the code-version-delta + transitive-propagation
   case the predicate must catch — and the historical situation (neuron_dynamics
   "fresh" by coverage despite the reshape) is the regression to reproduce. Pin the
   invalidation-correctness validation to this chain.

2. **Fork (b) checkpoint identity — strong lean to mtime+size (or a train-time
   manifest id), NOT a content hash.** A single full materialize was measured to blow
   past a 5-minute timeout under contention; re-hashing safetensors at *every plan*
   would reintroduce the very cost this REQ removes. Read-cheap is the constraint.
   Also stamp the **epoch set**, not just the count, so adding dense checkpoints
   invalidates only the new epochs.

3. **Per-epoch stamp granularity is where the win actually is — make it a CoS, not an
   afterthought.** Per-epoch analyzers (e.g. the new `neuron_frequency_attribution`)
   write one npz per epoch. If the signature/stamp is per-*analyzer*, adding one
   checkpoint re-runs all N epochs — i.e. no better than the coverage check that
   exists. Signature-incremental only beats coverage if per-epoch nodes are stamped
   and compared per-`(analyzer, epoch)`. Per-epoch primaries are the bulk cost.

4. **Name derived tables as a producer kind in `sig()`.** A `DerivedTableSpec`'s
   "code" is its **SQL text**, so its signature is `hash(query) + spec.version +
   sorted(input-table sigs)`. Changing the query must invalidate. The formula as
   written keys on `code_version`; spell out that for derived tables that folds the
   query text (or require a `version` bump on query change).

5. **Two concrete reuse points already in the tree:**
   - The dashboard's *dumb* full re-materialize added in REQ_141
     (`apps/dashboard/src/dashboard/pages/analysis_run.py`, commit 9aea734 — the
     `materialize_variant_columnar` + `materialize_variant_derived` calls after
     `pipeline.run`) is the exact call site to make signature-scoped (CoS: re-materialize
     only changed tables). `scripts/run_analysis.py` is the parity sibling to do the same.
   - `apps/research/sketches/validate_req141_parity.py` (read-only, recompute-in-memory
     vs. stored) is the template for the "signature-aware rebuild == `force` rebuild"
     byte-parity validation.

6. **Conservative-recompute is correct; it justifies deferring fork (c).** REQ_141
   proved old vs new `neuron_dynamics` are *value-identical* despite the v1→v2 bump
   (`validate_req141_parity.py`). So the predicate will recompute it on the version
   delta even though values won't change — the honest signal beating a missed
   optimization. This is precisely why the optional output data-version hash (fork c)
   is genuinely deferrable: the input-derived code-version is the load-bearing trigger.
