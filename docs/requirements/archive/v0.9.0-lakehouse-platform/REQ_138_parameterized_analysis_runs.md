# REQ_138: Parameterized Analysis Runs (Recipe-Addressed Storage + Run-Set Registry)

**Status:** Completed — merged to `develop` 2026-06-06 (REQ_110 lakehouse line). Phases 1–6 (6 commits); full miscope suite green. Open questions resolved (see Resolution).
**Priority:** High — **blocks REQ_110-D**. Establishes the parameterization coordinate that consumer migration must land on *once*. Must be addressed before the rest of REQ_110 continues.
**Branch:** TBD (suggested: continuation of `feature/REQ_110_lakehouse_surface`, since it extends the 110-A/B/C coordinate model).
**Relationship to REQ_110:** Extends the coordinate model that 110-A (columnar warehouse), 110-B (tensor catalog), and 110-C (DuckDB query surface) established. Inserts **between 110-C and 110-D** so that 110-D re-points renderers/summaries onto an already-parameterized surface rather than migrating twice. Accepts bounded, one-time thrash to 110-C's view layer (one coordinate added to data views + the `catalog` union; one new `run_sets` registry view).
**Dependencies:**
- REQ_133 (Fluid Dependency Scheduler) — the DAG resolution engine. A reference binding is a DAG edge; the per-artifact recipe is a fold over the walk REQ_133 already performs.
- REQ_107 (Discoverability Registry) — supplies per-field `kind` + coordinate declarations. The parameterization signature becomes a new coordinate; the registry gains a parameterization-aware "which parameterizations of this field exist?" answer.
- REQ_110-A/B/C — the warehouse/catalog/query surfaces the parameterization coordinate threads through.
- REQ_128 (Analyzer Input Provisioning) — `ResolvedInputs` / `ArtifactInput` are the structural inputs this builds parameter bindings alongside.
**Attribution:** Engineering Claude (under user direction, design session 2026-06-05).

---

## Problem Statement

Analysis-generation parameters are currently injected through `extra_context` — a single untyped `dict[str, Any]` merged flat into the family analysis context immediately before a run ([pipeline.py:180-181](../../../packages/miscope/src/miscope/analysis/pipeline.py#L180-L181)). One analyzer consumes it for real today: `parameter_dmd` reads `parameter_dmd_reference_epoch` to choose which `neuron_grouping` snapshot partitions its frequency groups, defaulting to "last checkpoint" ([parameter_dmd.py:197-219](../../../packages/miscope/src/miscope/analysis/analyzers/parameter_dmd.py#L197-L219)). The same "reference epoch" idea recurs, hardcoded to last, in `neuron_group_pca` ([neuron_group_pca.py:157](../../../packages/miscope/src/miscope/analysis/analyzers/neuron_group_pca.py#L157)) and `freq_group_weight_geometry`.

This mechanism has a precise defect class, demonstrated by the REQ_133 p101 regression: **a parameter that changes output bytes is supplied through an unrecorded, untyped, global side-channel.** The artifact records `reference_epoch` as an *output* field (you can read what was used) but nothing records it as an *input* that drives a recompute. The canonical p101 artifact was built with a pinned value; the pin lived only in the ephemeral invocation; an unpinned recompute fell back to the default, split at a different epoch, and diverged. The regression harness correctly flagged byte-divergence — but the divergence was **lost provenance, not data drift**, and the fix-by-retrain treated a provenance problem as a data problem (see REQ_133 Finding, 2026-06-04).

The bigger need this exposes is **parameterized data generation**, with two concrete drivers:

1. **Analyst-supplied parameters.** A researcher defines a parameter as part of kicking off a run. The marquee future case is a **probe = a prompt (or small prompt set) into an LLM** — there is no enumerable full dataset to send the way ModAdd does today. Comparing **model internals across probes** requires the artifacts to **coexist side-by-side**, not overwrite.
2. **Upstream-derived parameters.** A downstream analyzer pinned to a dynamical regime that can only be *discovered* by prior analysis — e.g. a window split pinned to a DMD-discovered regime boundary. The parameter value is produced by an upstream analyzer (DMD is a *source* of downstream parameters).

Both drivers must satisfy the same three constraints: **reproducibility** (a recorded, replayable recipe), **side-by-side coexistence** (multiple parameterizations of one analyzer on one variant), and **queryability** (ask which parameterizations exist / which used probe X).

This must be solved **before larger-model analysis**, where full dumps of parameters/activations are infeasible and naive duplication per parameterization is the cost that would force a later refactor.

---

## Core Model (decided)

### The conceptual cut

The analysis context conflates two kinds of object. The discriminating test is: **does changing it change the output bytes?**

- **Deterministic context** — derivable from variant + family (prime, ideal frequency sets, group basis). Not a choice; same every run. Stays exactly where it is.
- **Generation parameter** — a free analyst choice *or* an upstream-derived value that *selects which artifact you get*. This is part of the artifact's **identity**, not its context. Two `parameter_dmd` runs at different reference epochs are not one artifact recomputed; they are two distinct analytical objects.

### Parameterization, bindings, recipe

A **Parameterization (run set)** is an ordered set of parameter **bindings**, each either:
- a **literal** — an analyst's free value (`probe="ds_A"`, `ref_epoch=20000`, a tensor/layer selection like `W_out only`), or
- a **reference** — a path into an upstream artifact field (`ref_epoch = activation_dmd.regime_boundaries[0]`), carrying that upstream's own recipe transitively.

Bindings have two scopes:
- **Run-level** — declared once at kickoff, propagating to every analyzer whose closure includes them (probe; "what's under study": specific layers, specific weight matrices).
- **Analyzer-local** — declared on one analyzer's Spec (`parameter_dmd`'s `reference_epoch`).

A default is itself a binding (a literal default, or a reference like `max_epoch(neuron_grouping)`). There is no "unrecorded default" — which is what makes the p101 failure mode structurally impossible.

**A "last/max epoch" default must be a *reference* binding, never a captured literal.** Its resolution depends on the **checkpoint inventory, which is mutable — training can be extended.** A reference binding's *resolution inputs* (here, the inventory) are therefore part of the staleness graph: when training extends and `max_epoch` re-resolves to a later value, the floating-default artifact is replanned. Recording the *resolved value* in the recipe (open question #4's lean) is what makes this detectable — stored `ref_epoch=40000` vs a default that now re-resolves to `50000` → stale. An **explicitly pinned** earlier epoch (`ref_epoch=20000`) is *not* invalidated by extension — it pins a value that still exists; only the floating default moves. (The superseded old default becomes an orphan — see the GC open question.)

### The single storage rule

> An artifact's **recipe** is the projection of the run set's bindings onto that artifact's **transitive upstream closure**.
> — empty projection → parameter-independent → **one shared copy**
> — projection matches an already-stored binding set → **reuse**
> — projection is new → **write under the new recipe**

**Storage is addressed by recipe** (per-artifact, content-addressed by its generation recipe). The **run set** is the researcher-facing handle and the comparison/query unit, recorded in a registry that maps `run_set → {bindings, resolved values, the recipes it realized}`.

### Tiering falls out of the rule (it is not separate machinery)

- **Training outcomes** (checkpoints, `parameter_snapshot`, activation snapshots) have no analysis parameter in their closure → empty recipe → shared globally, never duplicated. `parameter_snapshot` stays shared even when `probe` varies, because it reads *weights*, not the probe forward pass.
- **Universal / non-parameterizable analyzers** declare no parameters and sit downstream of nothing parameterized → empty recipe → shared.
- **Parameterized analyzers** have a live parameter in their closure → recipe non-empty → coexist per parameterization.

"Parameterized" is therefore **dynamic, not a static label**: an analyzer is parameterized *in a given run set* iff a live binding sits in its closure. When `probe` is promoted, the 9 model-driven analyzers split automatically — weight-readers (`parameter_snapshot`, `weight_spectra`) stay shared; activation-readers (`neuron_activations`, `attention_patterns`, `repr_geometry`, `activation_basis_projection`, `input_trace`) get one artifact per probe — with zero per-analyzer configuration.

### Separation of concerns (each piece extends something that exists)

| Responsibility | What it does | Extends |
|---|---|---|
| **Parameter declaration** | Spec gains a `parameters` block: name, type, scope (run-level/local), default *binding* (literal or reference). An analyzer that reads an undeclared parameter fails registry load — same discipline as `outputs`. | REQ_107/132 Spec |
| **Parameterization resolution** | Planner resolves a requested parameterization + DAG into concrete bindings + per-artifact recipe, reading upstream fields for reference bindings. | REQ_133 scheduler |
| **Recipe-addressed storage + run-set registry** | Recipe becomes a coordinate; paths carry it; accessors take a parameterization; a `run_sets` registry relation maps handle → bindings/provenance/recipes. | REQ_110 warehouse/catalog |

---

## Conditions of Satisfaction

### Parameter declaration
- [ ] **Spec-declared parameters.** Each analyzer that consumes a generation parameter declares it on its Spec (name, type, scope, default binding). An analyzer reading an undeclared parameter **fails registry load** (parallel to the `outputs` discipline).
- [ ] **Literal and reference default bindings.** A default may be a literal value or a reference to an upstream field (e.g. `max_epoch(neuron_grouping)`). `parameter_dmd`'s "last checkpoint" default is expressed as an explicit reference binding, not a code fallback.
- [ ] **Run-level vs analyzer-local scope.** Run-level parameters (probe, tensors/layers under study) declared at kickoff propagate to every analyzer whose closure includes them; analyzer-local parameters are declared on one Spec. Both are bindings in the recipe.

### Recipe-addressed storage
- [ ] **Recipe = closure projection.** An artifact's recipe is the projection of the run set's bindings onto its transitive upstream closure; computed as a fold over the REQ_133 DAG walk.
- [ ] **Parameter-independent artifacts are shared, not duplicated.** An artifact with an empty recipe (e.g. `parameter_snapshot`, universal analyzers) has exactly one copy across all run sets. *Storage cost of an N-way sweep of a narrow parameter is the changed artifact(s) only, not the whole tree.*
- [ ] **Coexistence.** Two parameterizations of one analyzer on one variant produce two artifacts addressed by their distinct recipes; neither overwrites the other.
- [ ] **Reuse.** A recipe matching an already-stored binding set resolves to the existing artifact (no recompute, no rewrite).
- [ ] **Near-zero migration.** Parameter-independent artifacts keep their current path (empty recipe ≡ today's address), so the existing ~9.4k artifacts do **not** relocate; only parameterized artifacts gain a recipe segment. (Pre-v1.0.0: no back-compat shim owed.)

### Run-set registry & query surface
- [ ] **Run-set registry as a third index relation.** Alongside `_catalog` (110-A) and `_tensor_catalog` (110-B), a registry maps `run_set → {bindings, resolved reference values, realized recipes, provenance}`. "Which run sets used probe X?" is a `SELECT`; the data views join on the recipe/run-set column.
- [ ] **Query surface gains the dimension once.** 110-C data-table views and the `catalog` union carry a `run_set` (and/or recipe) column; a `run_sets` view translates human bindings → signature. Cross-parameterization comparison is `WHERE run_set = ...` / `GROUP BY run_set`.
- [ ] **Storage encapsulation preserved.** No consumer composes a recipe/run-set path; all addressing flows through accessors and `warehouse.paths` (architectural invariant 3). A parameterized view is still universal (invariant 1); families remain context providers, not parameter owners (invariant 2).

### Provenance & reproducibility (the p101 fix)
- [ ] **Replayable recipe.** Every parameterized artifact's full recipe (bindings, including transitive upstream recipes and resolved reference values) is persisted as an input-side record; a recompute reconstructs the same artifact without out-of-band knowledge.
- [ ] **p101-class divergence is impossible.** A recompute of a parameterized artifact cannot silently fall back to a different default — the default is an explicit recorded binding. Regression covers: pin → record → recompute → byte-identical.

### Scheduler / freshness integration
- [ ] **Reference binding = DAG edge.** A reference binding participates in REQ_133 topo-order; its value is read at run time, its edge known at plan time.
- [ ] **Recipe-aware freshness.** A new recipe is a missing artifact (planned natively). When an upstream re-runs and a reference's resolved value moves, dependents' recipes change and the stale ones are replanned/invalidated (transitive, per REQ_133).
- [ ] **Inventory-derived references re-resolve on extension.** For an artifact whose recipe contains an inventory-derived reference (e.g. the `max/last epoch` default), freshness re-runs the resolver against the *current* checkpoint inventory and compares to the stored resolved value; a moved value marks the floating-default artifact stale. An explicitly pinned earlier epoch is unaffected (its value still exists).

---

## Scope boundary (what this REQ does and does not build)

**In scope now — the mechanism, proven on the live parameter.** Implement parameter declaration, recipe-addressed storage, the run-set registry, and scheduler/freshness integration, and apply them to the **three existing `reference_epoch` sites** (`parameter_dmd`, `neuron_group_pca`, `freq_group_weight_geometry`). This is verifiable today against the three baselines and directly closes the p101 provenance gap.

**Named forward target, not implemented here — probe-as-parameter.** The design is shaped for the LLM-prompt-probe future (probe = an input to the HookedModel forward pass, supplied at kickoff; ModAdd's dataset becomes the *default* probe binding). But promoting probe requires a family-API change to `generate_analysis_dataset` and the HookedModel/LLM substrate that does not yet exist (REQ_105/112/113 land the boundary; the LLM family does not). Implementing probe-as-parameter is a **follow-on REQ**. This REQ must leave probe-as-parameter a drop-in (no refactor of the recipe/registry model to add it). Deferral is low-risk: the shape of a probe-as-parameter is already well-established by existing mech-interp practice (a prompt / small prompt set fed through a hooked model), so the forward target is concrete enough to design against without building it now. Promoting probe widens the scientific invariant from "same Variant + same Probe across all checkpoints" to "same Variant + same **parameterization**" — to be ratified in the follow-on.

---

## Resolution (2026-06-05 — implementation)

Built in six phases, one commit each, on `feature/REQ_110_lakehouse_surface`:

1. **Parameter declaration** — `miscope.analysis.parameters` (`ParameterSpec`,
   `LiteralBinding`/`ReferenceBinding`, `Reducer`/`FieldIndex` selectors,
   `Parameterization`); `AnalyzerSpec.parameters`; registry load-time gate.
2. **Recipe** — `miscope.analysis.recipe` (`project_recipe` closure fold,
   `recipe_signature`, `RecipeResolver`); planner folds reference sources into edges.
3. **Recipe-addressed storage** — one `analyzer_dir(...)` path primitive adopted by
   `ArtifactLoader`, pipeline writes/scans, and the planner (`recipe_map`); empty
   recipe ≡ today's path (no migration). `Pipeline.run(parameterization=...)`
   replaces `extra_context`; `ResolvedInputs.parameters`; the three `reference_epoch`
   sites migrated. Regression: pinned recompute byte-identical; coexistence verified.
4. **Run-set registry** — `_run_sets` relation (`warehouse/run_sets.py`); pipeline
   records the run set; `run_set` coordinate column on every columnar table + catalog
   row; `query.open()` registers a `run_sets` view.
5. **Liveness/GC + freshness** — `live_recipe_signatures` / `orphaned_recipe_dirs`;
   `prune_deprecated_artifacts.py --recipes`; `freshness.check_reference_freshness`
   re-resolves inventory-derived defaults.
6. **Consumer API** — `variant.parameterize(...)` → `ParameterizedVariant` (recipe-
   scoped loader; universal views read the recipe plane transparently).

**Resolved decisions:** consumer API = `parameterize()` handle (OQ #5); selector
grammar = minimal `Reducer` + `FieldIndex` (OQ #2); GC = liveness query folded into
the prune script (OQ #3). Flagged-for-review: signature = canonical-JSON over binding
*specs*, ints exact / floats quantized 12dp (OQ #1, #4); registry persists named
bindings (JSON) + opaque `recipe_signature` (OQ #6); recipe path = a `__rs_<sig>`
subdir, omitted when empty.

**Snapshot/lifecycle axis — decided to defer (review 2026-06-05).** The floating
`max_epoch` default at the **root** path is recomputed *in place* on training
extension (`check_reference_freshness` surfaces the move; the prior analysis is not
auto-preserved). Value-addressing the floating default alone would snapshot the
reference axis but not the **trajectory-length** axis (the cross-epoch artifact spans
all checkpoints, which is not a parameter), yielding a confusing half-snapshot. The
clean end-state — making the **checkpoint inventory** part of cross-epoch artifact
identity so extension yields a uniform immutable snapshot and the bespoke staleness
check collapses into the planner's native missing-artifact handling — is captured as a
follow-on (`docs/requirements/drafts/REQ_139_immutable_inventory_keyed_artifacts.md`).
REQ_138 keeps default-at-root.

**One deferred slice (explicitly out of this REQ, 110-D-adjacent):** materializing
*parameterized* artifacts into the **columnar** tables. The `run_set` column exists
on every table/catalog row (the coordinate is present once, which is what 110-D
needs); coexistence + provenance are fully covered by the recipe-scoped `.npz`
artifacts and the run-set registry; the columnar writer's wipe-regenerate model
materializes the default plane today. Folding parameterized rows into the columnar
tables lands with the 110-D consumer migration.

## Open Questions (resolved — see Resolution above)

1. **Recipe hash canonicalization.** A DMD-derived regime boundary may be a *float*; hashing floats into a stable signature is fragile. Needs a canonical representation (rounding/quantization policy) so the same logical binding addresses stably across runs/environments. *Decision authority: reasonable decision, flag for review.*
2. **Reference-binding selector grammar.** `activation_dmd.regime_boundaries[0]` implies a small grammar for "field + selector." What is expressible — positional index, named regime, predicate? Keep minimal; grow on demand. *Propose options.*
3. **Lifecycle / GC of orphaned recipes.** Coexistence accumulates; when an upstream re-runs and a derived value moves, the prior downstream recipe is orphaned. Needs a "which recipes are live / referenced by a run set" + prune story. The untracked `scripts/prune_deprecated_artifacts.py` is likely already circling this. *Propose options.*
4. **Signature: recipe-addressed vs value-addressed (decided-with-lean).** Lean: **address by recipe, record the resolved value in provenance** — computable before upstream runs (composes with the scheduler) while staying human-legible via the recorded value. Confirm.
5. **Consumer API for selecting a parameterization.** `variant.parameterize(probe="A", ref_epoch=...)` returning a parameterized handle vs. threading a param argument through `.at(epoch).view(name)`. Touches the dashboard `_VIEW_LIST` dispatch. *Propose options.*
6. **Run-set granularity of the registry coordinate.** Address by recipe signature, but persist *named* bindings as columns so the warehouse can filter on `probe` / `ref_epoch` directly (not just on an opaque hash). Confirm the named-columns-plus-signature shape.

---

## Constraints

**Must have:**
- Recipe-addressed physical storage with sharing of empty-recipe artifacts (the property that keeps large-model parameterized analysis tractable).
- Full replayable provenance for every parameterized artifact (closes p101).
- All three architectural invariants intact (universal views, families-as-context-providers, storage-internal-to-API).
- Verifiability against the three pinned baselines (per `feedback_verify_against_baselines_only`).

**Must avoid:**
- Single-signature-keys-storage: duplicates parameter-independent artifacts per run set and re-runs the full pipeline on every narrow-parameter change — quantified below as the model that forces a later large-model refactor.
- Recipe/run-set path literals leaking outside the storage primitive.
- Implementing probe-as-parameter here (out of scope; must remain a drop-in).
- An untyped global side-channel (the `extra_context` pattern) for any byte-affecting parameter.

**Flexible:**
- The selector grammar (start minimal), the hash canonicalization mechanics, and the exact registry column layout — all resolvable during design.

---

## Decision Authority
- [x] Propose options for review (selector grammar, GC, consumer API)
- [x] Make reasonable decisions and flag for review (hash canonicalization, registry column layout)
- [ ] Full autonomy to proceed

---

## Success Validation
- **`ref_epoch` proving ground.** On a baseline, two run sets differing only in `parameter_dmd`'s reference epoch write **one** new artifact (the `parameter_dmd` cross-epoch file) and **reuse** the other ~3.2k; the recompute is one analyzer, not the pipeline. Both coexist and are queryable by run set.
- **p101 reproduction test.** Pin → record → recompute → byte-identical; an unpinned recompute cannot silently diverge (the default is a recorded binding).
- **Derived-parameter walk.** A reference binding (e.g. `parameter_dmd.reference_epoch ← <upstream field>`) resolves through the REQ_133 scheduler at run time, and the downstream artifact's recipe transitively includes the upstream recipe.
- **Training-extension re-resolution.** After extending a baseline with later checkpoints, a floating `max/last epoch` default re-resolves to the new last and its artifact is replanned as stale; an explicitly pinned earlier epoch's artifact remains valid and is reused.
- **Query surface.** "Which run sets exist for this variant / used binding X?" answerable as a `SELECT` against the run-set registry, joinable to data views — verified across the three baselines.
- **Encapsulation audit.** No new recipe/run-set path literal outside `warehouse.paths` / accessors.

---

## Notes

**Alternatives considered (rejected, with the quantification that settled it).** On the p113 baseline (3278 artifacts, 251 checkpoints, 25 analyzers, 9 model-driven), the cost of a single-signature storage model equals the number of artifacts *independent* of the changed binding:
- *Narrow parameter* (`ref_epoch`; sole consumer `parameter_dmd`, terminal, 1 artifact): recipe-keying writes 1 + reuses 3277 and re-runs 1 analyzer; single-signature writes 3278 (3277 byte-identical duplicates) and re-runs all 25 incl. the 9 forward-pass analyzers — ~99.97% wasted writes and ~all wasted compute.
- *Broad parameter* (`probe`; feeds the forward pass): both models write ~3278 and re-run ~all — **identical**, no difference.

The decisive asymmetry: single-signature is **free exactly when you don't need it** (broad params — everything changes anyway) and **ruinous exactly when you do** (narrow downstream params — which is *by construction* what the derived-regime feature is). A content-dedup hybrid (run-set key + write-time dedup) recovers disk but not recompute (you still re-run forward-pass analyzers to discover identical bytes) — strictly dominated for the case the design is for, and for large models recompute is the binding cost. Recipe-keying pays one recipe column on a registry needed anyway, and leaves the researcher-facing handle/comparison UX byte-for-byte identical.

**Origin.** Stubbed from the 2026-06-05 design/discovery session that traced the `extra_context` use case (resurfaced as the REQ_133 p101 finding) to its proper boundaries. See REQ_133 "Resolution (2026-06-04)" for the triggering incident.
