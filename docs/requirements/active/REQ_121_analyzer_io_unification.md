# REQ_121: Analyzer I/O Unification

**Status:** Draft (preliminary — refinement expected after REQ_120 implementation lands)
**Priority:** Medium-high — closes out the analyzer protocol fragmentation that has been a recurring friction surface for the platform. Step 2 of 5 in the ETL exploration's forward plan.
**Branch:** `feature/req-121-analyzer-io-unification`
**Supersedes:** None — replaces the three existing protocols at retirement time, but does not retire any standing requirement.
**Dependencies:**
- **REQ_120 (Analyzer Spec + Registry) — hard dependency.** This REQ evolves REQ_120's capability flags into structured `inputs` declarations. REQ_120 must land first.
- REQ_119 (Plan/Planner) — Planner's shape evolves with the unified inputs; the load-decision and prerequisite-resolution machinery established by REQ_119/120 carries forward.
**Coordinates with:**
- REQ_111 (parallel analyzer build-out) — close coordination required. REQ_111's new-construction analyzers can adopt the unified interface from birth once this REQ lands; existing analyzers covered by REQ_102 deprecation skip migration entirely. The crossover point between REQ_111's parallel-period analyzers and REQ_121's migration scope needs explicit scope demarcation before either work starts.
- REQ_106 (analysis layer architecture) — alignment on the "analyzer is the transform step" framing. REQ_121's unified protocol *is* the transform-layer interface REQ_106 names.
**Downstream consumers:**
- Future cross-variant analyzers — the unified `inputs` declaration must compose to cross-variant scope cleanly. This REQ ensures it does.
- Future Store work (REQ_100 / REQ_101) — unified inputs reference artifacts by `(analyzer, scope)`, not by file path; that's the seam a Store replaces.
**Attribution:** Engineering Claude (under user direction). Outcome of the design dialogue captured in `feature/generic_analyzer` [BRANCH_NOTES.md](../../../BRANCH_NOTES.md).

---

## Problem Statement

After REQ_120, every analyzer has a `Spec` with capability flags. But the underlying *I/O contract* still varies by protocol:

| Protocol | `.analyze()` signature |
|---|---|
| `Analyzer` | `(ctx: ActivationContext) -> dict[str, np.ndarray]` — receives model + cache + probe |
| `SecondaryAnalyzer` | `(artifact: dict, context: dict) -> dict[str, np.ndarray]` — receives one epoch's upstream artifact |
| `CrossEpochAnalyzer` | `(artifacts_dir: str, epochs: list[int], context: dict) -> dict[str, np.ndarray]` — receives a directory and epoch list |

Three input shapes; three dispatch paths in the pipeline; three sets of CoS items in every analyzer's tests. The Planner from REQ_119 + REQ_120 can decide *which* analyzers run, but the pipeline still has three execution branches threaded through it. New analyzer shapes (cross-variant being the most plausible next one) force a new protocol; the fragmentation amplifies.

### What this REQ does

Collapses the three protocols into one. Each analyzer declares its inputs structurally:

```python
SPEC = AnalyzerSpec(
    name="freq_group_weight_geometry",
    inputs=[
        ArtifactInput("parameter_snapshot", scope="epoch"),
        ArtifactInput("neuron_grouping", scope="epoch"),
    ],
    output_scope="cross_epoch",
    ...
)
```

The pipeline reads `inputs` to materialize whatever the analyzer needs, then invokes a single `.analyze(inputs, context) -> dict[str, np.ndarray]` method. Output shape (per-epoch / cross-epoch / summary) is declared in the Spec, not implicit in the protocol category. Three execution paths collapse to one.

This is the *collapse* in the "expand then collapse" arc REQ_120 began.

---

## Conditions of Satisfaction

> **Implementation-shape note.** The data classes proposed below (`InputSpec`, `ResolvedInputs`, the unified `Analyzer` protocol shape) are illustrative. The load-bearing design decision is the precise structure of `inputs` and the materialized form passed to `.analyze()`. Final shape requires implementation experience from REQ_119 + REQ_120 and may differ from this draft.

### The unified protocol

- [ ] **`miscope/analysis/protocols.py::Analyzer`** — single protocol replacing the existing three:
  ```python
  class Analyzer(Protocol):
      @property
      def name(self) -> str: ...
      @property
      def inputs(self) -> list[InputSpec]: ...
      @property
      def output_scope(self) -> Literal["per_epoch", "cross_epoch", "per_epoch_summary"]: ...
      def analyze(
          self,
          inputs: ResolvedInputs,
          context: dict[str, Any],
      ) -> dict[str, np.ndarray]: ...
  ```
- [ ] **`miscope/analysis/inputs.py`** — input declaration types:
  - `ModelInput(needs_weights: bool = True, needs_cache: bool = True)` — model state at the analyzer's current epoch (scope is implicit; cross-epoch analyzers do not declare `ModelInput`).
  - `ArtifactInput(analyzer_name: str, scope: Literal["epoch", "all_epochs", "summary"])` — upstream artifact at the named scope.
  - *(Reserved for future:)* `CrossVariantInput(...)` — placeholder for cross-variant analyzers; not implemented in this REQ but the structure leaves room.
- [ ] **`miscope/analysis/inputs.py::ResolvedInputs`** — the materialized form passed to `.analyze()`. Holds the materialized model state, per-epoch artifact dicts, all-epochs artifact stacks, and so on, indexed by the declared `InputSpec` items.

### Migration path

- [ ] **Phase 2A — coexistence.** The new `Analyzer` protocol coexists with the three legacy protocols. The pipeline dispatches by `isinstance`: new-shape analyzers go through the new path; legacy analyzers stay on the existing three-path dispatcher. Both produce identical artifacts.
- [ ] **Phase 2B — migration.** Each analyzer scheduled for retention (not under REQ_102 deprecation) is migrated to the new protocol. Per analyzer:
  - Convert the protocol-specific signature to the unified one.
  - Translate the analyzer's dependency declarations (`requires` / `depends_on`) into `inputs` declarations.
  - Remove the legacy protocol class inheritance / structural typing.
  - Verify byte-identical artifact output against the legacy implementation on the reference variant set (parity bar from REQ_111).
- [ ] **Phase 2C — retirement.** Remove the three legacy protocols and their three dispatch paths from `pipeline.py`. Pipeline has one execution path.

### Plan + Planner integration (REQ_119 + REQ_120 evolution)

- [ ] **Plan shape simplification.** Plan's `per_epoch` / `secondary` / `cross_epoch` fields collapse into a single `items: list[PlanItem]` field, with `output_scope` on each item distinguishing them. (Or — preserve the three-field shape for readability — that's an implementation-time choice; the simplification is allowed but not required.)
- [ ] **Capability flags become derived.** REQ_120's `requires_model_weights` / `requires_activation_cache` are no longer authored on the Spec; they're computed from `inputs` (a Spec with any `ModelInput(needs_cache=True)` has `requires_activation_cache = True`). One source of truth.
- [ ] **First-class prerequisite resolution.** REQ_120's placeholder for auto-queueing prerequisites becomes a first-class capability: when analyzer X declares `ArtifactInput("Y", scope="epoch")` and Y is not in the disk state or the current Plan, the Planner adds Y to the Plan automatically — gated by an explicit `auto_queue_prerequisites: bool` flag that defaults to True under the unified protocol.

### Tests / validation

- [ ] **Per-analyzer migration parity.** For each migrated analyzer, run the new-shape implementation side-by-side with the legacy one on the reference variant set (canon: p113/s999/ds598, p109/s485/ds598, p101/s999/ds598, p59/s485/ds598). Verify byte-identical artifact output.
- [ ] **Pipeline-execution parity.** Run a full canonical analysis pipeline on a reference variant under the legacy three-path dispatcher and the new unified dispatcher; verify the produced artifact set is identical.
- [ ] **Cross-protocol-merge test.** Existing analyzers spanning all three categories (e.g., `parameter_snapshot` + `fourier_frequency_quality` + `intragroup_manifold`) migrated to the new protocol; verify the pipeline produces the same artifacts as the legacy three-path version.

---

## Constraints

**Must:**
- **Migration is incremental.** Phase 2A allows old and new analyzers to coexist. The pipeline does not require all analyzers to migrate simultaneously. Phase 2C (retirement of the three legacy protocols) only happens after every retained analyzer has migrated.
- **Each migrated analyzer preserves byte-identical artifact output.** Parity validation against the reference variant set is the bar.
- **`inputs` declaration is the single source of truth** for what an analyzer reads. REQ_120's capability flags become derived properties after this REQ; they are no longer authored independently.
- **Analyzers covered by REQ_102 deprecation are not migrated.** They retire under REQ_102 rather than passing through this REQ.

**May:**
- The `ResolvedInputs` type may be opinionated about how to bundle (e.g., a `NamedTuple` with `model`, `artifacts: dict[str, Any]`, `cross_epoch_artifacts: dict[str, dict[int, Any]]`). v1 shape is a tractable starting point; refinement allowed.
- Cross-variant `inputs` placeholders may be left as a deferred TODO if cross-variant analyzers aren't yet in flight. The structure should leave room for them; the implementation doesn't have to deliver them.
- Plan's shape may simplify (three fields → one) or retain three-field readability — implementation choice.

**Must Not:**
- Modify any analyzer's `.analyze()` computational logic during migration. Migration is a signature change and dependency-declaration change; the math is identical. Bug fixes discovered during migration are filed separately and applied to both old and new before parity validation.
- Touch artifact format or location.
- Introduce a Store abstraction (REQ_100 / REQ_101 territory).
- Retire the legacy protocols before every retained analyzer migrates. The three-path dispatcher coexists until Phase 2C.

---

## Notes

- **This REQ is preliminary.** The exact shapes of `InputSpec`, `ResolvedInputs`, and the unified `Analyzer.analyze()` signature are the load-bearing design decisions. The shapes proposed in CoS are illustrative; final shape requires implementation experience from REQ_119 + REQ_120. Expect a refinement pass on this REQ once those land.
- After this REQ lands, an analyzer's "protocol category" becomes a derived attribute computed from `output_scope`. REQ_120's `category` field on Spec can be removed.
- **Cross-variant analyzers** are an explicit future case. Without unified inputs, adding a fourth protocol would amplify the fragmentation this REQ corrects. The `InputSpec` hierarchy reserves room for `CrossVariantInput` without committing to its v1 implementation.
- The collapse the REQ achieves: three protocols + three `register_*` methods + three dispatch paths → one protocol + one register method + one dispatch path. Every entry point and every test that touched the three-shaped API gains a single shape to write against.
- This REQ pairs with REQ_111's "expand then collapse" philosophy at the protocol layer: REQ_120 expanded (Specs alongside protocols); REQ_121 collapses (one protocol replaces three).

---

## Out of Scope

- **Cross-variant analyzers themselves.** A future REQ. This REQ ensures the unified `inputs` shape can express them; it does not implement any.
- **Store abstraction (REQ_100 / REQ_101 territory).** The `(analyzer, scope)` addressing established here is the seam a Store will eventually replace, but no Store is introduced in this REQ.
- **`LoadedFamily` retirement** — step 3 of the ETL refactor sequence; future REQ.
- **Data root unification** — step 4 of the ETL refactor sequence; future REQ.
- **Per-analyzer computational logic changes.** Any analyzer whose math should change in conjunction with this migration is a separate REQ.
