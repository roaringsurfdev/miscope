# REQ_133: Fluid Dependency Scheduler

**Status:** Implemented on `feature/REQ_133_fluid_dependency_scheduler` (2026-06-01) — *code complete and byte-neutral; **merge held** pending separate REQ_134 baseline reconciliation of p101 stale references (decision 2026-06-01, see Finding).*
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

## Resolution (2026-06-01)

Implemented as an ordering-only refactor across the four files the spike identified.

**Files:**
- [inputs.py](../../../packages/miscope/src/miscope/analysis/inputs.py) — added
  `derive_has_model_input`; documented `output_scope` as the only structural axis
  and demoted `derive_category` to freshness-only name-bucketing (CoS 6).
- [planner.py](../../../packages/miscope/src/miscope/analysis/planner.py) — `plan_analysis`
  now splits descriptors by `output_scope` into two passes, each ordered by a single
  scope-agnostic topo-sort (`_topo_order`, generalized from `_order_secondaries` to walk
  all `ArtifactInput` edges). Per-epoch coverage branches on `has_model_input`
  (`_per_epoch_target_epochs`): model-driven analyzers cover all checkpoints; purely
  artifact-derived ones (former secondaries) follow the intersection of their upstreams'
  epochs — byte-identical to the old secondary logic for single-upstream chains. After
  planning each cross-epoch item, `projected_completed[name]` is seeded with the full
  available-epoch set (unless blocked), so a cross→cross dependent isn't falsely blocked
  in the same pass (CoS 2/3). `Plan.secondary` removed; `format`/`to_dict`/`is_empty`
  updated (CoS 1, 8).
- [pipeline.py](../../../packages/miscope/src/miscope/analysis/pipeline.py) — `register`
  routes by `output_scope` into two buckets; `_secondary_analyzers` and
  `_run_secondary_from_plan` deleted. Former secondaries execute inside the per-epoch
  loop in the planner's topo order, so an upstream writes `epoch_N` before its dependent
  reads it in the same iteration (CoS 5). Blocked per-epoch items are logged and skipped
  (preserving the old warning-not-raise behavior). The `config.analyzers` asymmetry is
  preserved via `_filter_per_epoch_by_config`.
- [freshness.py](../../../packages/miscope/src/miscope/analysis/freshness.py) — dropped the
  `plan.secondary` merge; former secondaries now arrive in `plan.per_epoch` (CoS 8).
- [library/pca.py](../../../packages/miscope/src/miscope/analysis/library/pca.py) — added
  `_canonicalize_svd_sign`, applied in `pca()` and `compute_svd()`, pinning the SVD sign gauge so
  PCA/SVD bases are reproducible across BLAS/LAPACK builds. Surfaced by this refactor's regression
  run (see Finding); implemented here by owner decision with the rationale captured in-code.

**CoS status:**
1. ✅ DAG-driven order; "secondary" gone from planner/pipeline/Plan.
2. ✅ `test_cross_to_cross_satisfiable_in_one_pass` (planner.py) — clean tree, no `blocked_by`.
3. ✅ Covered by the same test + `test_cross_to_cross_blocked_propagates_when_root_absent`
   (a blocked upstream is *not* seeded, so its dependents stay blocked).
4. ✅ `test_transitive_staleness_two_hop_per_epoch` — 2-hop replanning via projected coverage.
5. ✅ `test_same_epoch_write_before_read_in_one_pass` (test_secondary_analyzers.py) — clean
   single pass; dependent value matches its own-epoch upstream.
6. ✅ No cross→per-epoch support added; 2-valued `output_scope` documented in inputs.py.
7. ✅ Byte-parity established. `--no-recompute` integrity green (9436 artifacts). Full forced
   recompute: **p113 and p109 pass** (2775 each). **p101/s999/dseed598 reports 4 cross-epoch
   mismatches** (`global_centroid_pca`, `parameter_trajectory`, `parameter_dmd`, `activation_dmd`)
   — diagnosed as **pre-existing stale references, not a REQ_133 effect** (see Finding below).
   REQ_133 itself is byte-neutral: recompute digests for all 4 analyzers are *identical* between
   the pre-REQ_133 base tree and this branch.
8. ✅ Dashboard uses only `plan.format()` + `FreshnessReport`; full dashboard + miscope
   suites green (1469 + 54 passed).

### Finding — p101 cross-epoch stale references (2026-06-01)

A determinism probe (run each analyzer twice over the canonical on-disk artifacts; compare a
pipeline-identical save to the stored reference sha) established, on **both** the base tree and
this branch (identical digests):

| analyzer | determinism | recompute vs reference |
|---|---|---|
| `global_centroid_pca` | deterministic | **differs** from stored ref |
| `parameter_trajectory` | deterministic | **differs** from stored ref |
| `parameter_dmd` | deterministic | **differs** from stored ref |
| `activation_dmd` | deterministic | matches ref **when fed canonical `global_centroid_pca`** |

The three roots' `.analyze()` code is untouched by REQ_133 (identical recompute digests on both
trees), so REQ_133 is byte-neutral. A deeper element-wise diff (stored vs current recompute, both at
352 checkpoints) then resolved each root to a **distinct, concrete mechanism** — superseding the
initial "low-order FP drift" guess:

- **`global_centroid_pca` and `parameter_trajectory` — SVD sign-gauge instability.** Shapes and
  eigenvalues match (~1e-15–1e-5); only `basis`/`projections` differ, at `max_rel == 2.000` — the
  `v` vs `-v` signature. `np.linalg.svd` is deterministic/seedless but a singular pair is defined only
  up to a shared sign, and which sign LAPACK returns is not stable across BLAS builds (p101's
  near-degenerate spectrum tipped it). **Fixed under this REQ** by pinning the gauge in the `pca()` /
  `compute_svd()` primitive — `_canonicalize_svd_sign` (largest basis loading made positive; flips
  `u`/`v` together; eigenvalues/center untouched), with the full rationale in a comment at the helper.
  `weight_spectra` also stores SVD vectors, so `compute_svd` got the same treatment for uniformity.
- **`activation_dmd` — pure cascade** from `global_centroid_pca`; resolves once the upstream basis is
  canonical (also the CoS-2 fix correctly feeding it the freshly-recomputed upstream).
- **`parameter_dmd` — reference-epoch mismatch, NOT a sign issue.** `_resolve_reference_epoch` defaults
  to the *last* checkpoint; the canonical p101 artifact was generated with `parameter_dmd_reference_epoch`
  pinned to **20000**, but its last checkpoint is **34999** (extended to 35k epochs), so the unpinned
  harness recompute partitions at a different epoch → different groups/components/regimes. p113/p109
  pass only because their pin (24999) equals their last checkpoint. The harness can't reproduce
  artifacts built with non-default `extra_context`. **Addressed separately** (out of this REQ): planned
  via simultaneous pinned+default data cuts; near-term the pinned variants are refreshed for parity.

**Consequence:** the sign-gauge fix changes the bytes of *every* PCA/SVD-vector artifact across *all*
variants (one-time, deliberate), so the pinned-variant reference checksums must be regenerated before
the byte-regression checker is green again. Owner is reconciling the baseline; "all models will need
refreshes" (decision 2026-06-01).

**Decision (2026-06-01):** hold REQ_133 merge; reconcile the regression baseline in a separate
REQ_134 follow-up rather than refreshing references inside this branch. REQ_133 code is committed on
the feature branch and stays unmerged until that follow-up confirms the p101 references. The follow-up
should also settle whether the drift is a one-time environment/FP shift (refresh once) or recurring
nondeterminism (e.g. BLAS-thread-count sensitivity in SVD/DMD on ill-conditioned p101) needing a
determinism guard before any refresh.

## Notes

- Independent of v1.0.0 close-out; can land any time after REQ_134.
- Spike was non-destructive (read-only planner probes); no throwaway branch was needed — the
  findings above replace the originally-recommended spike branch.
- REQ_134 sequenced first per decision 2026-06-01 (restores the byte-regression oracle on the stable
  interface this work preserves).
