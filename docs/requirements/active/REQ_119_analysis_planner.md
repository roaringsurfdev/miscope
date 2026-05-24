# REQ_119: Analysis Planner — Plan/Execute Separation

**Status:** Draft
**Priority:** High — the duplication between [`pipeline._build_work_queue`](../../../packages/miscope/src/miscope/analysis/pipeline.py#L258) and [`freshness.py`](../../../packages/miscope/src/miscope/analysis/freshness.py) is the source of stealth bugs across multiple entry points. Each new entry point reinvents the decision tree with subtle variation. Step 1 of 5 in the ETL exploration's forward plan.
**Branch:** `feature/req-119-analysis-planner`
**Supersedes:** None.
**Dependencies:** None blocking.
**Coordinates with:**
- REQ_106 (analysis layer architecture) — REQ_106 names the layering *inside* an analyzer (data plane / derivation / analyzers). REQ_119 addresses orchestration *across* analyzers. Complementary, not competing.
- REQ_111 (parallel analyzer build-out) — REQ_111's parallel construction philosophy is preserved. This REQ does not modify or migrate any analyzer. Close coordination with REQ_111 will be needed at step 1.5 (Specs touch every analyzer file) and step 2 (I/O unification); flagged here so the sequencing is visible.
**Downstream consumers:**
- *Step 1.5 — analyzer Spec + Registry* (future REQ). Each analyzer gains a declarative `Spec` (name, protocol category, dependencies, capability flags). Once Specs exist, the Planner can make load-time decisions (skip `run_with_cache` when no analyzer needs the cache) and the hard-coded `pipeline.register(...)` blocks in entry points get replaced by registry lookups.
- *Step 2 — analyzer I/O unification* (future REQ). Collapses the three protocols into one; structured `inputs` declarations replace step 1.5's capability flags. `Plan`'s shape evolves to consume them.
- *Dashboard analysis path.* A `Plan` is previewable before execution.
- *Future Store work* (REQ_100 / REQ_101). The `Plan` describes inputs/outputs in terms of `(analyzer, scope)` rather than file paths — the seam a future Store replaces.
**Attribution:** Engineering Claude (under user direction). Outcome of the design dialogue captured in `feature/generic_analyzer` [BRANCH_NOTES.md](../../../BRANCH_NOTES.md).

---

## Problem Statement

The current [`AnalysisPipeline`](../../../packages/miscope/src/miscope/analysis/pipeline.py) conflates three concerns. The first is where stealth bugs live and is the focus of this REQ:

1. **What work needs to happen.** Spread across [`_build_work_queue`](../../../packages/miscope/src/miscope/analysis/pipeline.py#L258), the per-phase loops in [`pipeline.run()`](../../../packages/miscope/src/miscope/analysis/pipeline.py#L129), the secondary-phase target-epoch logic ([`_run_secondary_analyzers`](../../../packages/miscope/src/miscope/analysis/pipeline.py#L481)), the cross-epoch freshness check ([`_run_cross_epoch_analyzers`](../../../packages/miscope/src/miscope/analysis/pipeline.py#L560)), **and a second implementation in [`freshness.py`](../../../packages/miscope/src/miscope/analysis/freshness.py)**.
2. *How to get inputs* — checkpoint → model → cache, or artifact-dir lookup. (Out of scope here; touched in steps 2 and 4.)
3. *How to persist outputs* — three near-identical atomic-save helpers. (Out of scope here.)

The first concern is the metastasis. Multiple entry points each rebuild the work-queue decision tree with subtle variations:

- [`scripts/run_analysis.py`](../../../scripts/run_analysis.py)
- [`scripts/run_analysis_regression.py`](../../../scripts/run_analysis_regression.py)
- [`scripts/fill_checkpoints.py`](../../../scripts/fill_checkpoints.py)
- The dashboard's analysis-progress path

And `freshness.check_freshness()` is a *second* implementation of the same decision tree as `pipeline._build_work_queue()`. The two implementations drift — the freshness report can claim "fresh" while the pipeline still has work to do, or vice versa. The bug surface is the gap between them.

### What this REQ does

Extracts the decision tree into a single **`Planner`** that produces a **`Plan`** object. All entry points construct a `Plan`; the existing pipeline consumes a `Plan` rather than building its own work queue. Freshness reporting becomes "describe the `Plan` over the current disk state" — the same code, different presentation.

This is a **mechanical refactor** with one new object. No analyzer protocol changes. No artifact location changes. No new storage backends. The intent is to kill the duplication and lock in a single source of truth for "what would run if I called `pipeline.run()` right now."

---

## Conditions of Satisfaction

### The Plan object

- [ ] **`miscope/analysis/planner.py::Plan`** — a passive data class. Fields:
  - `per_epoch: list[PlanItem]` — `(analyzer_name, epochs_to_compute)` for primary analyzers.
  - `secondary: list[PlanItem]` — `(analyzer_name, depends_on, epochs_to_compute)` for secondary analyzers.
  - `cross_epoch: list[PlanItem]` — `(analyzer_name, requires, available_epochs)` for cross-epoch analyzers.
- [ ] **`Plan.is_empty -> bool`** — true if no work in any phase.
- [ ] **`Plan.format() -> str`** — human-readable, three-section display matching the existing `FreshnessReport.format()` shape, so it can substitute in CLI / log output.
- [ ] **`Plan.to_dict() -> dict`** — serializable form for logging and (future) dashboard preview.
- [ ] **Plan preserves the existing three-phase distinction** (primary per-epoch / secondary per-epoch / cross-epoch) as separate fields. v1 does not unify these; that is step 2's job.

### The Planner

- [ ] **`miscope/analysis/planner.py::plan_analysis(variant, analyzers, force=False, checkpoints=None) -> Plan`** — single entry point.
  - `variant: Variant`
  - `analyzers: list[Analyzer | SecondaryAnalyzer | CrossEpochAnalyzer]` — accepts all three existing protocols; classification by `isinstance` against the protocols.
  - `force: bool` — if true, every applicable epoch is included regardless of disk state.
  - `checkpoints: list[int] | None` — restrict to these epochs (None = all available).
- [ ] **Decision logic consolidates** what's currently in `pipeline._build_work_queue` (per-epoch missing-check), the secondary-phase target-epoch logic, and the cross-epoch freshness check ([`freshness.cross_epoch_is_stale`](../../../packages/miscope/src/miscope/analysis/freshness.py)).
- [ ] **No side effects.** No model loading, no analyzer execution, no writes. Reads only the artifacts directory state.
- [ ] **Cross-epoch prerequisite reporting.** If a cross-epoch analyzer's `requires` declares a per-epoch analyzer with no completed epochs, the Plan records it as a *blocked* item naming the specific missing prerequisite. Execute-time behavior preserves today's raise from [`_run_cross_epoch_analyzers`](../../../packages/miscope/src/miscope/analysis/pipeline.py#L596) (the configurable raise-vs-skip choice waits for step 1.5, when richer declarations exist). The Plan field is the substrate this REQ delivers; the behavior choice is downstream.

### Freshness re-expressed

- [ ] **`freshness.check_freshness()` becomes a thin wrapper** over `plan_analysis(force=False)`. Anything in the resulting Plan is "stale or missing"; everything not in the Plan is "fresh."
- [ ] **Existing `FreshnessReport` / `PerEpochFreshness` / `CrossEpochFreshness` shapes preserved.** Callers (CLI freshness command, dashboard freshness view) see no API change. The internals derive the report from a Plan.
- [ ] **`freshness.cross_epoch_is_stale` deleted.** Its logic is absorbed by the Planner. The function is dead code after this refactor.

### Pipeline consumption

- [ ] **`AnalysisPipeline.run()` accepts an optional `plan: Plan` parameter.** If not provided, the pipeline calls `plan_analysis(self.variant, [...registered analyzers...], force=force, checkpoints=self.config.checkpoints)` internally.
- [ ] **The pipeline's per-phase loops are driven by the Plan**, not by re-running the decision logic inline. `_build_work_queue` deleted; the inline secondary-phase target-epoch computation deleted; the inline cross-epoch freshness check deleted.
- [ ] **Backwards compatible at the call-site level.** Existing callers that do `pipeline.register(...).run()` continue to work without code changes.

### Entry-point migration

- [ ] **[`scripts/run_analysis.py`](../../../scripts/run_analysis.py)** — explicit Planner usage: `plan = plan_analysis(variant, analyzers, force=args.force)`; log the plan; pass to `pipeline.run(plan=plan)`.
- [ ] **[`scripts/run_analysis_regression.py`](../../../scripts/run_analysis_regression.py)** — same pattern.
- [ ] **[`scripts/fill_checkpoints.py`](../../../scripts/fill_checkpoints.py)** — same pattern (to the extent it touches the analysis pipeline; if it only runs training, this item is a no-op and gets recorded as such).
- [ ] **Dashboard analysis path** — `apps/dashboard` analysis-progress flow uses Planner explicitly. The Plan is logged before execution (foundation for future preview UI).

### Tests / validation

- [ ] **Planner unit tests** against fixture variants with known artifact state covering: (a) all epochs computed → empty plan, (b) some epochs missing per-epoch → plan contains them, (c) cross-epoch artifact present but built on fewer epochs → plan contains it as stale, (d) secondary analyzer with missing dependency → plan surfaces the prerequisite gap, (e) `force=True` → plan contains everything regardless of disk state.
- [ ] **Parity test:** pre-REQ pipeline's effective work queue matches `Planner.plan_analysis()` output on the same fixture inputs. Implemented as a snapshot test recording the work queue from the existing code path, then asserting the Planner produces the same output. This validates the refactor is lossless.
- [ ] **Freshness parity:** `FreshnessReport` for a canon variant produces field-identical output before and after the refactor. Cross-check against an existing canon variant's artifact state.
- [ ] **End-to-end smoke test:** run a small reference variant (one of the canon set) through the full pipeline using the new Planner path; verify artifacts produced match the pre-refactor result byte-for-byte.

---

## Constraints

**Must:**
- Backwards compatible at the pipeline API level. Existing scripts that call `pipeline.register(...).run()` continue to work without code changes.
- No new analyzer protocols. The three existing protocols (`Analyzer`, `SecondaryAnalyzer`, `CrossEpochAnalyzer` in [protocols.py](../../../packages/miscope/src/miscope/analysis/protocols.py)) are preserved unchanged. Unification is step 2's scope.
- No artifact location changes. The data layout (`results/<family>/<variant>/artifacts/<analyzer>/...`) is preserved unchanged. Step 4 covers data-root reorganization.
- No new storage backends. `.npz` files remain the only on-disk format.
- No changes to existing analyzers themselves.
- Single source of truth: after the refactor there is exactly one implementation of "what would run." `freshness.cross_epoch_is_stale` and `pipeline._build_work_queue` both gone.

**May:**
- Plan format may evolve in step 2 to support unified `inputs` declarations. The v1 shape is allowed to be tightly fitted to today's three-protocol structure — we are not pre-designing for step 2 here.
- Plan's `format()` output is a presentation detail and can change between v1 and step 2 without breaking the data model.

**Must Not:**
- Introduce an `AbstractStore` or storage-backend abstraction. Triggered by REQ_100 / REQ_101, not by this REQ.
- Modify or retire `LoadedFamily` / `FamilyRegistry`. That's step 3.
- Move artifacts or family configs. That's step 4.

---

## Notes

- The duplication being killed: [`pipeline._build_work_queue`](../../../packages/miscope/src/miscope/analysis/pipeline.py#L258) vs [`freshness.cross_epoch_is_stale`](../../../packages/miscope/src/miscope/analysis/freshness.py) + the inline per-epoch missing-check.
- The `Plan` object is intentionally passive in v1 — it is a description, not an executor. Execution stays in `pipeline.run()`. The shape will gain capability flags (model/cache requirements) in step 1.5 once analyzer Specs exist, and will evolve again in step 2 when input declarations get structured.
- **What this REQ does *not* solve.** The hard-coded analyzer instantiation pattern in [scripts/run_analysis.py:67-89](../../../scripts/run_analysis.py#L67-L89) (the long block of commented-out `pipeline.register(...)` lines, with three separate `register` / `register_secondary` / `register_cross_epoch` entry points) remains. So does the always-load behavior for model state on every epoch — `model.run_with_cache(probe)` runs even when no registered analyzer consumes the cache. Both are explicit step 1.5 scope; they cannot be solved cleanly without analyzer-side declarations.
- The user's framing from the design dialogue: "ETL lives in parallel with the Family/Variant analysis navigation that is more user-friendly. A user navigates a family and its variants. A background process (eventually) extracts the data for analysis." This REQ respects that framing — Planner takes a `Variant` as input; it does not replace variant navigation.
- The notebook scaffold at [scripts/run_analysis.ipynb](../../../scripts/run_analysis.ipynb) on `feature/generic_analyzer` captures the buggy run-logic intent ("multiple paths for running the analyzer pipeline, and there are some stealth bugs in logic in several places"). After this REQ lands, that notebook's content is encoded in code and the notebook itself is no longer useful as a reference — it can be deleted along with the exploratory branch when the learnings are incorporated.

---

## Out of Scope

These belong to subsequent steps and explicitly do not land here:

- **Step 1.5:** Analyzer Spec + Registry. Each analyzer gets an explicit `Spec` (name, protocol category, dependencies, capability flags for model/cache needs). Registry replaces hard-coded `register(...)` calls in entry points. Three protocols preserved at this stage. Enables Planner's load-decision optimization and discoverable analyzer lists. Touches every analyzer file; close scope coordination with REQ_111.
- **Step 2:** Unify the three analyzer protocols under a single `Analyzer` with structured `inputs` declarations replacing step 1.5's capability flags. `Plan`'s shape evolves to consume the unified declarations. Coordinates with REQ_111.
- **Step 3:** Retire `LoadedFamily`; let the family own its own variant lookup. Independent of this REQ; can land in parallel.
- **Step 4:** Unified data root (`data/<family>/family.json` + `data/<family>/variants/...`). Largest blast radius; lands last, after the abstractions settle.
- **`AbstractStore` abstraction.** Not introduced here. The Plan's `(analyzer, scope)` addressing is the seam a future Store will use; introducing the abstraction now would lock to the one concrete backend that exists today.
