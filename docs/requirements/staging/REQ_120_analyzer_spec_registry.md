# REQ_120: Analyzer Spec + Registry

**Status:** Completed (implementation; awaiting merge approval)
**Priority:** High — unblocks the Planner's load-decision and transitive-dependency capabilities (REQ_119) and replaces the hard-coded analyzer instantiation pattern in entry points. Step 1.5 of 5 in the ETL exploration's forward plan.
**Branch:** `feature/req-120-analyzer-spec-registry`
**Supersedes:** None.
**Dependencies:**
- REQ_119 (Plan/Planner) — REQ_120 enriches the Plan's information sources with declarative Specs. REQ_119 can land first; REQ_120 makes the Planner reach its full potential.
**Coordinates with:**
- REQ_111 (parallel analyzer build-out) — close coordination required. REQ_120 touches every existing analyzer file (adds a Spec). REQ_111's parallel-construction philosophy is preserved: new analyzers can ship with Spec from birth; old analyzers under REQ_111's retention scope get a Spec retrofit.
- REQ_106 (analysis layer architecture) — REQ_106's "analyzer is the transform step" framing aligns with the spec/impl split this REQ introduces.
**Downstream consumers:**
- *Step 2 (REQ_121, drafted alongside this REQ).* Capability flags introduced here evolve into structured `inputs` declarations under the unified `Analyzer` protocol.
- *Planner (REQ_119).* Gains load-time decisions (skip `model.run_with_cache(probe)` when no analyzer needs the cache) and transitive prerequisite resolution.
**Attribution:** Engineering Claude (under user direction). Outcome of the design dialogue captured in `feature/generic_analyzer` [BRANCH_NOTES.md](../../../BRANCH_NOTES.md) and the REQ_119 drafting follow-up.

---

## Problem Statement

The current Analyzer is opaque about its own needs. Three protocols ([`Analyzer` / `SecondaryAnalyzer` / `CrossEpochAnalyzer`](../../../packages/miscope/src/miscope/analysis/protocols.py)) plus three `pipeline.register_*` methods leak the protocol distinction into every caller. The "what does this analyzer need" information is *latent* — buried in the body of `.analyze()` — so the Planner cannot:

1. **Decide whether to skip the forward pass.** Some analyzers (e.g., `parameter_snapshot`, `effective_dimensionality`) consume only model weights. Today they still pay for [`model.run_with_cache(probe)`](../../../packages/miscope/src/miscope/analysis/pipeline.py#L300) on every epoch because the pipeline has no way to know in advance whether the cache will be consulted.
2. **Resolve dependencies transitively.** If analyzer X requires analyzer Y's artifact and Y's artifact isn't present on disk, the Planner can detect the gap but cannot reason about whether Y itself should be added to the current run.

Meanwhile, every entry point hard-codes analyzer instantiation:

```python
# scripts/run_analysis.py:67-89
pipeline.register_cross_epoch(IntraGroupManifoldAnalyzer())
# pipeline.register(AttentionFreqAnalyzer())
# pipeline.register(AttentionPatternsAnalyzer())
# pipeline.register(DominantFrequenciesAnalyzer())
# ... 17 more commented-out lines ...
```

Adding a new analyzer requires editing every script that wants to use it. Listing what's available requires reading the script. The protocol category leaks out — callers must know which of `register` / `register_secondary` / `register_cross_epoch` to invoke for each analyzer.

### What this REQ does

Separates *declaration* from *implementation*. Each analyzer gains a `Spec` — a small declarative object capturing what the Planner needs to know (name, protocol category, dependencies, capability flags). Specs live in a Registry: analyzers self-register, so the Registry can be queried for "give me all analyzers for family X" or "give me the analyzer named Y" without anyone hand-importing class definitions.

The three existing protocols are preserved unchanged. This is *not* I/O unification — that's step 2 (REQ_121). It is the substrate that makes the Planner's downstream optimizations possible and that retires the discoverability pain in entry points.

---

## Conditions of Satisfaction

### The Spec object

- [ ] **`miscope/analysis/spec.py::AnalyzerSpec`** — frozen data class:
  - `name: str` — unique identifier (used in artifact naming and registry keys).
  - `category: Literal["primary", "secondary", "cross_epoch"]` — which of the three protocols this analyzer implements.
  - `requires: list[str]` — analyzer names whose artifacts this analyzer consumes. Empty for primary; a single item for what `SecondaryAnalyzer` calls `depends_on`; multi-item for `CrossEpochAnalyzer.requires`.
  - `requires_model_weights: bool` — does `.analyze()` access `model.get_weight(...)` or `model` directly?
  - `requires_activation_cache: bool` — does `.analyze()` access `ctx.cache[...]`?
  - `produces_summary: bool` — does this analyzer also produce per-epoch summary statistics (REQ_022's optional surface)?
  - `architecture_support: frozenset[str] | None` — which architectures this analyzer can run on. Promoted from the existing class attribute introduced by REQ_088 onward.

### Spec → analyzer association

- [ ] **Each analyzer module exposes a `SPEC` attribute** — a module-level `AnalyzerSpec` instance. Adding a Spec is a 5–10 line addition per file.
- [ ] The existing protocol-defined `name` property continues to work; the Spec's `name` field mirrors it. (Mild duplication accepted in v1.)
- [ ] **Audit.** A grep across [`packages/miscope/src/miscope/analysis/analyzers/`](../../../packages/miscope/src/miscope/analysis/analyzers/) confirms every analyzer file has a `SPEC`. Import-time CI check: every concrete `Analyzer` / `SecondaryAnalyzer` / `CrossEpochAnalyzer` in the module has a corresponding `SPEC` whose `category` matches the analyzer's protocol class.

### The Registry

- [ ] **`miscope/analysis/registry.py::AnalyzerRegistry`** — module-level registry holding `dict[str, AnalyzerSpec]` plus `dict[str, Callable[[], Analyzer]]` (factories). Registry is a singleton accessed through the module's functions, not instantiated per-call.
- [ ] **Registration call** in each analyzer module's import path: `register_analyzer(SPEC, factory=lambda: AnalyzerClass())`. Self-registration triggered by importing the module.
- [ ] **Auto-discovery via [`analyzers/__init__.py`](../../../packages/miscope/src/miscope/analysis/analyzers/__init__.py).** Today's `__init__.py` already imports each analyzer class for re-export; this REQ adds Spec registration to those same imports. No filesystem-walk magic.
- [ ] **Query API:**
  - `AnalyzerRegistry.get_spec(name: str) -> AnalyzerSpec`
  - `AnalyzerRegistry.list_specs() -> list[AnalyzerSpec]`
  - `AnalyzerRegistry.get_factory(name: str) -> Callable[[], Analyzer]`
  - `AnalyzerRegistry.list_for_family(family: ModelFamily) -> list[AnalyzerSpec]` — filtered by `architecture_support` (set membership against the family's architecture) plus the family's declared analyzer list (from `family.json`'s `analyzers` field if present).

### Planner integration (REQ_119 enrichment)

- [ ] **Planner accepts Specs.** `plan_analysis(variant, specs_or_analyzers, force=False, checkpoints=None) -> Plan` — when fed Specs, looks up factories at execute time. Backwards compatible: still accepts Analyzer instances directly.
- [ ] **Load-decision.** Planner computes per-epoch requirements as the OR across all analyzers running at that epoch: `needs_weights = any(spec.requires_model_weights for spec in active_specs)`; same for cache. Pipeline consumes these decisions and skips `model.run_with_cache(probe)` when `needs_cache = False`.
- [ ] **Plan format extended.** Plan gains per-epoch `needs_model_weights: bool` and `needs_activation_cache: bool` fields, derived from the resolved analyzer set. Backwards compatible: defaults to `True` for both when running pre-REQ_120 Analyzer instances without Specs.
- [ ] **Transitive prerequisites.** When the Planner sees a cross-epoch analyzer requiring an absent per-epoch artifact, it looks up the per-epoch analyzer's Spec via the Registry and surfaces the gap in the Plan with the resolved analyzer name. v1: surface as a suggestion; auto-queueing is a configurable option (`auto_queue_prerequisites: bool = False`).

### Entry-point migration

- [ ] **[`scripts/run_analysis.py`](../../../scripts/run_analysis.py)** — replace the hard-coded `pipeline.register(...)` block (lines 67–89) with registry-driven enumeration:
  ```python
  # All analyzers for the family:
  specs = AnalyzerRegistry.list_for_family(family.family)
  # Or explicit selection:
  specs = [AnalyzerRegistry.get_spec(n) for n in ["parameter_snapshot", "repr_geometry"]]
  plan = plan_analysis(variant, specs, force=FORCE)
  pipeline.run(plan=plan)
  ```
- [ ] **[`scripts/run_analysis_regression.py`](../../../scripts/run_analysis_regression.py)** — same pattern.
- [ ] **[`scripts/fill_checkpoints.py`](../../../scripts/fill_checkpoints.py)** — same pattern if it touches the analysis pipeline; otherwise record as a no-op.
- [ ] **Dashboard analysis path** — registry-driven; the analyzer-selection UI (if/when added) pulls from `AnalyzerRegistry.list_for_family()`.

### Tests / validation

- [ ] **Spec audit test.** Imports every analyzer module and verifies each defines a `SPEC` whose `category` matches the analyzer's protocol class via `isinstance`.
- [ ] **Load-decision test.** Plan containing only `parameter_snapshot` (a weight-only analyzer) → pipeline does not call `run_with_cache`. Verify by patching/mocking and asserting non-call; sanity-check the time delta against the cache-loaded baseline as a regression-test-of-intent.
- [ ] **Registry consistency test.** For each Spec in the Registry, calling its factory produces an Analyzer instance whose `.name` matches the Spec's `name`. Catches drift between Spec and impl.
- [ ] **Backwards compatibility test.** Pre-REQ_120 pattern (hand-constructed Analyzer instances passed to `pipeline.register(...)`) continues to work without Specs. Plan defaults `needs_weights = needs_cache = True` for these analyzers.

---

## Constraints

**Must:**
- The three existing protocols (`Analyzer`, `SecondaryAnalyzer`, `CrossEpochAnalyzer`) are preserved unchanged. Their signatures, dispatch logic, and `.analyze()` contracts are untouched.
- Spec is purely additive — adding a Spec to an analyzer module requires no changes to the analyzer's `.analyze()` method, name, dependency declarations on existing protocol attributes, or summary-statistics methods.
- **Capability flags must honestly reflect actual usage.** Each Spec verified by inspection (or static analysis where feasible) against what the analyzer's `.analyze()` body actually reads from `ctx`. A wrong `requires_activation_cache = False` would cause silent runtime errors when the cache isn't loaded but is accessed.
- Registry discoverable via the package's `__init__.py` import. No filesystem-walk auto-discovery.
- Backwards compatibility at the call site: existing scripts that hand-construct Analyzer instances continue to work.

**May:**
- Spec format may gain fields in step 2 (REQ_121, when structured `inputs` declarations replace flat capability flags). v1 capability flags are allowed to be tightly fitted to today's needs.
- Family-aware filtering (`list_for_family`) may evolve as families' `analyzers` field gets richer. v1 honors the existing simple list.

**Must Not:**
- Modify any analyzer's `.analyze()` implementation. This REQ is purely additive at the declaration layer.
- Introduce a fourth protocol. The Spec is metadata, not a protocol.
- Touch artifact format or location. Step 4 territory.
- Unify the three protocols. That's REQ_121.

---

## Notes

- The protocol category leak in entry points (`register` / `register_secondary` / `register_cross_epoch`) is the visible pain that this REQ retires. The deeper issue — three input shapes baked into the `.analyze()` signatures themselves — waits for REQ_121.
- **Registration mechanism choice (decided 2026-05-24):** `@register_analyzer(SPEC)` decorator on the class. Module-level `SPEC` is the source of truth, decorator wires it into the Registry at class-definition time. The legacy `register_default_analyzers()` is retained as the import hub (importing every analyzer module triggers the decorators); its now-redundant explicit `AnalyzerRegistry.register*(cls)` calls are no-ops because the decorator-path Spec is preserved.
- **Capability flag honesty matters more than convenience.** An analyzer that incorrectly declares `requires_activation_cache = False` would cause silent failures. v1 audited exhaustively by reading each `.analyze()` body; six primary analyzers ended up cache-free and eligible for forward-pass skip (`parameter_snapshot`, `effective_dimensionality`, `dominant_frequencies`, `attention_fourier`, `fourier_nucleation`, `landscape_flatness`). Capability flags surface on `PlanItem`; the Plan aggregate `needs_activation_cache` drives the pipeline's `model.run_with_cache(probe)` skip decision.
- **Architecture-support field replaced with `required_hooks` (2026-05-24):** REQ_120 as drafted referenced `architecture_support: frozenset[str]`, but that class attribute was already superseded codebase-wide by `required_hooks: list[str]` (per-hook canonical-name compatibility check at `pipeline._run_single_epoch`). The Spec carries `required_hooks: tuple[str, ...]` instead — same compatibility semantics in finer grain.
- **Transitive prerequisite default `auto_queue=False`:** Plan surfaces blocked-by-known-Spec dependencies in a new `transitive_prerequisites` tuple (informational). Per the design choice, the planner does not auto-add them — the user runs the suggested prereqs explicitly. This matches the "surface, don't decide" stance.
- This REQ takes the "expand then collapse" philosophy ([feedback_primitive_workflow.md] design pattern): expand by adding Specs everywhere, accept the duplication, then collapse in REQ_121 when the three protocols become one.

---

## Out of Scope

- **REQ_121 (step 2):** Unifying the three protocols into a single `Analyzer` interface with structured `inputs` declarations. Capability flags from this REQ evolve into the unified `inputs` shape there.
- Moving the family-aware analyzer list into a more structured place than `family.json`'s `analyzers` field. The current field is honored as-is.
- Per-analyzer family customization beyond what `architecture_support` and the family's `analyzers` field already express.
- Introducing a Store abstraction (REQ_100 / REQ_101 territory).
