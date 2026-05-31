# REQ_132: Legacy Pipeline-Execution & Query-Surface Retirement

**Status:** Implemented on `feature/REQ_132-legacy-pipeline-retirement` (2026-05-30). Full miscope (1465 passed, 29 skipped) + dashboard (45 passed) suites green; audit greps clean. Awaiting merge approval. One CoS item (byte-identical regression artifacts) is code-verified but not executed this session — see note below.
**Priority:** Low — close-out track for v1.0.0. Not blocking; pairs with REQ_103.
**Branch:** `feature/REQ_132-legacy-pipeline-retirement` (TBD at start)
**Dependencies:**
- REQ_129 (v1.0.0 Dead-Code Cull — retired the legacy *authoring* scaffolding; deferred these two surfaces because they are API/structure decisions, not dead-code removal).

**Attribution:** Engineering Claude (stubbed 2026-05-30 from REQ_129 follow-ups; scope developed 2026-05-30 with the user).

---

## Motivation

REQ_129 removed the legacy `category`-authoring scaffolding and the registry's dead synthesis path. It deliberately deferred two surfaces that are *dead in production but test-exercised*, because removing them changes either a **public API surface** or the **pipeline's execution structure**.

Both surfaces share one root: they are remnants of the **pre-unification category-phased model**. Before REQ_121 unified analyzers under a single `Analyzer` protocol + structural `inputs` declaration, the pipeline ran analyzers in three authored phases (primary / secondary / cross-epoch) and the registry indexed them by authored category. Authoring is now unified — category is *derived* from `Spec.inputs` + `output_scope` via `derive_category` ([inputs.py](../../../packages/miscope/src/miscope/analysis/inputs.py)) — but the taxonomy still **leaks into the consumer-facing contract** in several places it no longer needs to.

**Guiding principle (user, 2026-05-30):** the primary/secondary/cross-epoch vocabulary must *not* live in the API contract. Consumers declare *which analyzers they want*; the internal dependency-list logic decides ordering. Keeping the taxonomy out of the contract lets that internal logic evolve freely without forcing consumers to track how it works.

The forcing function is the same as REQ_129's: a clean v1.0.0 published surface. Pre-v1.0.0 there is no back-compat obligation (no production consumers), so the taxonomy can be hard-collapsed without coexistence shims.

## Current state (audited 2026-05-30)

The canonical entry-point pattern is **already category-free** — `run_analysis.py`, `run_analysis_regression.py`, and the dashboard's `analysis_run.py` all do:

```python
specs = AnalyzerRegistry.list_for_family(variant.family)   # flat list of specs
plan  = plan_analysis(variant, specs, force=...)           # planner derives order from inputs
pipeline.run(plan=plan)                                    # _absorb_plan_references instantiates
```

`list_for_family` already concatenates the family's three buckets into one flat list, and the planner re-derives category from each spec's `inputs`. The three family buckets are therefore **redundant pre-sorting the planner ignores**. The taxonomy survives only in these surfaces:

| Surface | Reach | Disposition |
|---|---|---|
| Family declaration triad `analyzers` / `secondary_analyzers` / `cross_epoch_analyzers` (protocol + `base_model_family` + 3 `family.json`) | Author-facing | Collapse to single `analyzers` list |
| Pipeline `register` / `register_secondary` / `register_cross_epoch` | API | Collapse to single `register` (routes internally) |
| Legacy registry query API `get` / `get_secondary` / `get_cross_epoch` / `get_for_family` / `get_secondary_for_family` / `get_cross_epoch_for_family` / `list_all` | Test-only | Retire |
| `Spec.category` property + `Category` public type + `list_specs_by_category` | Tests + planner | Make internal — drop from public Spec/registry surface; planner derives via `derive_category` |
| Pipeline spec-less materialization (`spec is None` branch, `extra_allowed`/`depends_on` widening) | Test-only | Retire — Spec mandatory at execution time |
| `Plan.per_epoch` / `.secondary` / `.cross_epoch` + private `_run_*_from_plan` loops | Internal | **Keep** — out of the contract; the internal phase structure stays |

Note: `Spec.output_scope` (`per_epoch` / `cross_epoch`) is a **different axis** — it determines artifact storage shape (`epoch_*.npz` vs `cross_epoch.npz`) and is load-bearing. It stays, despite the `cross_epoch` name overlap.

## Decisions (resolved 2026-05-30)

1. **Scope width → Full: collapse the contract.** Family declares one flat `analyzers` list; collapse the register triad to a single `register`; retire the legacy query API and `list_specs_by_category`; drop `Spec.category` / `Category` from the public surface. The planner keeps deriving execution order from `inputs` internally.

2. **Internal execution → Spec mandatory, keep phase loops.** Remove the spec-less `_materialize_*` branch and the legacy `depends_on` widening so every *executed* analyzer must carry a registered Spec. The three private `_run_*_from_plan` phase loops stay as an implementation detail (they are not part of the contract). No collapse into a single topological pass — that is explicitly out of scope.

**Boundary that makes both decisions coherent:** *planning is lenient, execution requires a Spec.* `plan_analysis` / `_describe` keep their analyzer-instance fallback (freshness sentinels rely on it), but `_materialize_per_epoch_inputs` / `_materialize_cross_epoch_inputs` and the per-epoch run loop assume a non-`None` Spec.

## Scope

**In scope**
- Family declaration collapse: `families/protocols.py`, `families/base_model_family.py`, the three tracked `data/*/family.json` files, and `registry.list_for_family`.
- Pipeline `register` collapse: single `register(analyzer)` that routes into the internal phase lists by derived category (Spec-driven). Remove `register_secondary` / `register_cross_epoch`. Update `scripts/run_regression_check.py`.
- Retire the legacy registry query API and `list_specs_by_category`; migrate test callers to the spec-based API.
- Make `Spec.category` / `Category` internal: drop the public property/type and route the planner's one remaining read through `derive_category`.
- Remove the spec-less pipeline branches; migrate test fakes that ran on them to carry registered Specs.

**Out of scope**
- Collapsing the three internal `_run_*_from_plan` loops into one dependency-ordered pass (Decision 2 — deliberately deferred to REQ_133).
- Any change to `Spec.output_scope`, the `Plan` dataclass shape, or `_absorb_plan_references`.
- Touching REQ_103 / REQ_106 / REQ_099 work.

## Conditions of Satisfaction

**Family declaration is a single flat list.**
- [x] `ModelFamily` protocol exposes only `analyzers: list[str]`; `secondary_analyzers` and `cross_epoch_analyzers` are gone.
- [x] `base_model_family.py` reads a single `analyzers` list from config; the two derived getters are removed.
- [x] The three `family.json` files (`modulo_addition_1layer`, `modulo_addition_2layer_mlp`, `modulo_addition_learned_emb_mlp`) merge their three arrays into one `analyzers` array, preserving every existing analyzer name (23 / 15 / 8 — the exact bucket sums).
- [x] `list_for_family` reads only `family.analyzers`; family-unit tests (`test_modulo_addition_*`) assert membership against the flat list and pass.

**Pipeline registration is a single verb.**
- [x] `AnalysisPipeline.register(analyzer)` is the only registration method; `register_secondary` / `register_cross_epoch` are removed.
- [x] `register` routes the analyzer into the correct internal phase list via its registered Spec's derived category. Registering an analyzer with no Spec raises `ValueError` — covered by `test_pipeline_register_requires_spec`.
- [x] `scripts/run_regression_check.py` uses `register(...)` for all analyzers. **Code-verified only:** the register-routing is exercised by unit tests, but the full byte-identical regression harness (needs trained variants + checkpoints) was not run this session — run it before/at merge.

**Legacy query API retired.**
- [x] `get` / `get_secondary` / `get_cross_epoch` / `get_for_family` / `get_secondary_for_family` / `get_cross_epoch_for_family` / `list_all` / `list_specs_by_category` are removed from `AnalyzerRegistry`.
- [x] Retained surface: `get_spec`, `has_spec`, `list_specs`, `get_factory`, `create`, `list_for_family`, `list_all_names`, `is_registered`, `clear`.
- [x] `test_modulo_addition_family.py`, `test_analysis_library.py`, and `test_spec_registry.py` migrated to the retained spec-based API and pass.

**Category vocabulary is internal-only.**
- [x] `Spec.category` public property and the `Category` public type/export are removed; `Category` + `derive_category` now live in `inputs.py` (internal); the planner's `_describe` derives via `derive_category(spec.inputs, spec.output_scope)`, and `freshness.py` likewise.
- [x] Audit grep returns no consumer-facing hits — only the private `_secondary_analyzers`/`_cross_epoch_analyzers` phase lists (kept by Decision 2) and the planner's internal descriptor remain.

**Spec mandatory at execution; spec-less branches gone.**
- [x] `_materialize_per_epoch_inputs` drops the `spec is None` branch and the `extra_allowed` parameter; `_run_secondary_from_plan` drops the `depends_on` widening (the blocked-by warning now reads `item.depends_on`).
- [x] Test fakes previously relying on the spec-less path register a Spec via a snapshot-isolated helper (`_register_fake_spec` in `test_secondary_analyzers.py`; an autouse `_auto_spec_register` interceptor in `test_analysis_pipeline.py`) and pass.

**Whole-suite gate.**
- [x] Full `pytest`: `packages/miscope` 1465 passed / 29 skipped; `apps/dashboard` 45 passed.

## Trajectory (why this is the on-ramp, not a half-measure)

The end goal (user, 2026-05-30) is a **fluid dependency tree**: execution order determined solely by the `ArtifactInput` DAG, with the only structural axis being `output_scope` (per-epoch vs cross-epoch iteration), not the three-way phase taxonomy. The artificial part is *primary vs secondary* — "secondary" is just a per-epoch analyzer whose inputs are all `ArtifactInput`s; in a fluid model it dissolves into a topo-ordering inside the epoch loop.

REQ_132 is the **boundary work** that unblocks that: once the contract no longer speaks the taxonomy, the scheduler can be rebuilt entirely behind it with zero consumer churn ("adaptability lives in the boundaries"). Nothing here is throwaway against the fluid future — the single flat family list, single `register`, and Spec-mandatory rule are exactly what the DAG scheduler wants. The one reversible internal seam left behind is "`register` routes into three phase lists," which REQ_133 collapses.

**REQ_133 (the fluid scheduler)** carries the deeper change: teach `plan_analysis` to express arbitrary `inputs` edges (today it only topo-sorts within the secondary layer), replace the three `_run_*_from_plan` loops with two `output_scope`-keyed passes each running a topological sort of the DAG, and rework freshness/staleness for a real DAG. That touches the most-tested planner/freshness logic, so it gets its own audit + CoS.

## Notes

- **Not urgent.** Both surfaces are inert in production; carrying them is a clarity cost, not a correctness risk. Slots into the v1.0.0 close-out track alongside REQ_103.
- **No back-compat shims** — pre-v1.0.0 cutoff (no production consumers), so the collapse is a hard break by design.
- **Audit-before-delete discipline** (REQ_102/REQ_129): the consumer audit above is the evidence; re-run the greps after implementation as the CoS gate.
- **Sequencing.** Independent of REQ_106/099. The `family.json` edits touch tracked data — call them out explicitly in the merge request.
