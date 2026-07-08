# REQ_114: HookedModel Analyzer Migration

**Status:** Implemented; validation passed
**Priority:** Medium-High — completes the analyzer side of the `HookedModel` boundary. Lower than REQ_112/113 because the canary in REQ_112 already demonstrates the pattern; this REQ scales it.
**Branch:** TBD.
**Dependencies:**
- REQ_105 (HookedModel base class + canonical vocabulary).
- REQ_112 (HookedTransformer + one canary analyzer migrated end-to-end). The canary is the migration template; this REQ inherits the pattern.
- REQ_113 (Both MLP HookedModel subclasses). All three architectures must be on `HookedModel` before bulk analyzer migration; otherwise an analyzer's `required_hooks` declaration cannot be checked uniformly.
- **Coordinates with REQ_111 (Parallel Analyzer Build-Out).** Ordering is mutable based on REQ_111's progression — see *Coordination with REQ_111* below.
- Hands off to REQ_102 (Analyzer Deprecation). REQ_102 retires old analyzers based on REQ_111's parity validation outcomes; REQ_114 ensures the survivors run on `HookedModel`.

**Attribution:** Engineering Claude (under user direction).

---

## Problem Statement

After REQ_105/112/113, the model side of the `HookedModel` boundary is in place: all three trained architectures expose a canonical hook + weight vocabulary, and one canary analyzer (from REQ_112) has been migrated end-to-end through it. The remaining analyzers continue to work via the legacy path:

- Cache reads via raw TL strings (`cache["blocks.0.mlp.hook_pre"]`, `cache["post", layer, "mlp"]`) or family-specific bundle accessors (`bundle.mlp_post(...)`, `bundle.weight("W_E")`).
- The `architecture_support` class attribute on each analyzer declaring which architectures it supports.
- Per-family `*ActivationBundle` plumbing — partially retired by REQ_113 for the MLP families; still present in any analyzer call site that hasn't migrated.

This REQ migrates all surviving analyzers onto `HookedModel`. After this REQ:

- Every analyzer reads activations via `cache[canonical_name]` and weights via `model.get_weight(canonical_name)`.
- Every analyzer declares `required_hooks: list[str]` instead of `architecture_support: set[str]`.
- The pipeline filters analyzers per family by checking `all(h in model.hook_names() for h in analyzer.required_hooks)`.
- `analysis/library/activations.py` and `analysis/library/weights.py` either fold into `HookedModel` (their concerns are model-side) or rewrite to take `HookedModel` instead of `(model, cache)` tuples or bundle objects.
- The quarantine smoke test from REQ_112 holds across the entire codebase: TL imports remain confined to `architectures/hooked_transformer.py`.

---

## Coordination with REQ_111

REQ_111 is the parallel analyzer build-out: new analyzers are constructed alongside existing ones, validated for parity on a reference variant set, and only then handed off to REQ_102 for retirement of the old. The ordering relationship between REQ_111 and REQ_114 is **deliberately flexible** because outcomes from REQ_111's validation phase may shift the migration scope here.

**Three plausible orderings**, evaluated on the day REQ_111 ships its first new-vs-old analyzer pair with a recorded validation outcome:

### Option A (default): REQ_114 lands after REQ_111 substantially complete

- REQ_111 builds new analyzers against the legacy interface during the parallel period (since old analyzers are the parity anchor and they're on the legacy interface).
- REQ_111 validates each new-vs-old pair; outcomes are recorded.
- REQ_102 retires analyzers per validation outcomes.
- REQ_114 then migrates the survivors (whatever's left after retirement) onto `HookedModel` in one pass.
- **Pro:** No analyzer is migrated twice. Migration scope is minimized.
- **Con:** Long-running parallel period during which the legacy path must keep working. Two interfaces in flight (legacy + canary).

### Option B: New REQ_111 analyzers born on `HookedModel` from day one

- REQ_111's new analyzers consume `HookedModel` directly (canonical-name reads, `required_hooks` declaration). Old analyzers stay on the legacy path per REQ_111's "old code is not modified" rule.
- Parity validation compares new (`HookedModel`-native) outputs against old (legacy-path) outputs on the reference variant set. The interface difference is irrelevant to parity — both paths produce per-epoch artifacts in the same schema.
- REQ_102 retires old analyzers per validation outcomes; REQ_114's scope shrinks to migrating only the analyzers that REQ_111 doesn't touch.
- **Pro:** New analyzers are born correct. No double migration. REQ_114's scope shrinks.
- **Con:** Requires REQ_111 to depend on REQ_112 having shipped the canary; REQ_111's first analyzer cannot land before REQ_112 does.

### Option C: REQ_114 partial pass before REQ_111 begins; finish later

- REQ_114 migrates a small, well-understood subset of analyzers (e.g., the ones not in REQ_111's modernization scope) immediately after REQ_113 lands. This validates the migration template at scale.
- REQ_111 begins its parallel build-out with the rest. New analyzers born on `HookedModel`.
- A residual REQ_114 pass cleans up any analyzer not covered by either of the above.
- **Pro:** Validates migration template under load early; smooths the long tail.
- **Con:** Three coordination points instead of two. More complexity if scope shifts mid-flight.

### Recommendation

**Default to Option B** if REQ_111's first analyzer ships after REQ_112's canary. The shared decision rule: *new analyzers should be born on the canonical interface they will live on; there is no benefit to constructing them against an interface they will need to migrate off.*

If REQ_111 ships its first analyzer before REQ_112's canary lands (e.g., REQ_111 has urgent research-active scope on Lissajous / saddle-transport sigmoidality that doesn't wait), default to Option A: those new analyzers are born on the legacy interface, and REQ_114 sweeps them up later.

**The decision is recorded in REQ_111's Notes section** (specifically, in the Validation outcomes log) when the first new analyzer ships. The recorded decision is binding for subsequent REQ_111 analyzers; revisiting requires explicit user direction.

---

## Conditions of Satisfaction

### Analyzer migration

- [ ] **All analyzers in `analysis/analyzers/`** consume `cache[canonical_name]` and `model.get_weight(canonical_name)`. Raw cache string keys and bundle-style reads are removed from analyzer code.
- [ ] **`architecture_support` class attribute is retired.** Replaced by `required_hooks: list[str]` declaring the canonical hook paths the analyzer reads. Optionally also `required_weights: list[str]` for analyzers that read weights — implementation choice; if `required_hooks` is sufficient (because the pipeline already enforces hook availability and weight reads `KeyError` cleanly), don't introduce a parallel attribute.
- [ ] **Pipeline filtering.** The pipeline checks `all(h in model.hook_names() for h in analyzer.required_hooks)` before invoking each analyzer. Analyzers whose required hooks are not published by the current family's model are skipped with a structured log entry (analyzer name, missing hooks, family, variant). No exception.
- [ ] **Architecture-conditional analyzers.** For the small number of analyzers that need explicit branching (e.g., a frequency analyzer that handles transformer's shared `embed.W_E` *and* embedding-MLP's `embed.embed_a + embed.embed_b`), the resolution is **two analyzer subclasses with different `required_hooks`** registered separately. The pipeline picks the runnable one per family. **Adding `model.architecture_kind` or equivalent is a regression** and forbidden by review.

### Library migration

- [ ] **`analysis/library/activations.py`** — either folds into `HookedModel` (preferred, since its concerns are model-side) or rewrites to take `HookedModel` instead of `(model, cache)` tuples / bundle objects. Decision made during implementation based on what falls out cleanly.
- [ ] **`analysis/library/weights.py`** — same. Either folds into `HookedModel` (preferred) or rewrites to take `HookedModel` instead of `ActivationBundle` arguments.
- [ ] **`analysis/core/groups.py`** — **stays as an analysis-side helper.** Reverses the original REQ_105's "fold into adapter" instruction. `groups.py` carries domain knowledge (frequency groups, weight matrix groupings) that takes canonical-name inputs but is not part of the model's published surface. Folding it into `HookedModel` would push frequency-group knowledge into model code.
- [ ] **`analysis/core/sites.py`** — same as `groups.py`. Stays as an analysis-side helper. Sites are an analysis-side concept; the model publishes hooks, the analysis layer organizes those hooks into sites.

### Quarantine

- [ ] **Quarantine smoke test from REQ_112 still passes.** `grep -rn "transformer_lens" src/miscope/` returns hits only inside `architectures/hooked_transformer.py`. Any other location is a regression.
- [ ] **No raw TL hook strings outside `architectures/hooked_transformer.py`.** A grep test catches strings like `"hook_attn_out"`, `"hook_pre"`, `"hook_post"` (TL's spelling) outside the translation table in the transformer subclass. Canonical strings are allowed everywhere; legacy TL strings are confined to the translation table.

### Stub-model substitution

- [ ] **Stub-model test from REQ_112 generalizes.** A `StubHookedModel` (no concrete architecture) can be substituted for *any* migrated analyzer's input. The analyzer either runs (if its `required_hooks` happen to be satisfied by the stub) or skips cleanly via the pipeline filter. No analyzer crashes deeper than the model boundary. This proves architecture-agnostic behavior across the migrated population.

### Validation

- [ ] **REQ_086 regression scaffold byte-identical** on canon variants for *all* migrated analyzers. The full reference variant set: `p109/s485/ds598`, `p113/s999/ds598`, `p101/s999/ds598` (matches REQ_109's reference set per the consolidation overview). Outputs are byte-identical or differ only within documented per-metric tolerance before and after migration.
- [ ] **All existing tests pass.** Including dashboard tests, fieldnotes export scripts, and any notebook in active use.
- [ ] **Per-family analyzer coverage** — for each of the three families, the set of analyzers that ran before REQ_114 still runs (filtered down to those whose `required_hooks` are publishable by that family's `HookedModel`). The pipeline's skip log records any analyzer that *should* run but doesn't (helpful for catching missed migrations during development).

---

## Constraints

**Must:**

- Every analyzer migrates. No leftover analyzer with an `architecture_support` flag and bundle-style reads.
- The `groups.py` / `sites.py` modules **stay** in the analysis layer. The original REQ_105's "fold into adapter" instruction is reversed here per the user-aligned framing that those modules are domain knowledge, not model-surface knowledge.
- Pipeline filtering is structural (the filter step exists, with a structured skip log), not advisory. Analyzers cannot opt out of having their `required_hooks` checked.
- Architecture-conditional analyzers resolve via *two analyzer subclasses with different `required_hooks`*, not a `model.architecture_kind` accessor.

**Must avoid:**

- **Re-introducing architecture awareness in analyzers.** `if model is HookedTransformer: ...` is forbidden. So is `model.config.architecture == "transformer"`. So is any predicate testable on the model object beyond `hook_name in model.hook_names()`. The boundary is the canonical name, not the type.
- **Changing analyzer outputs.** This REQ is a refactor of *how* analyzers read inputs, not *what* they compute. Per-epoch artifact schemas, file paths, and numerical outputs are byte-identical before and after.
- **Migrating analyzers that REQ_111 is rewriting in parallel.** Per the coordination rule above, REQ_111's old analyzers stay on the legacy path; REQ_114 sweeps them only after REQ_102 has retired the corresponding old analyzer (or determined it survives). Migrating an analyzer here that REQ_111 is about to rewrite produces wasted work.
- **Touching the view layer.** Views are universal instruments, untouched by this REQ. They consume the same canonical-name surface analyzers do, but no view-side change is required for analyzer migration to succeed.

**Flexible:**

- **Ordering relative to REQ_111.** Decided per the *Coordination with REQ_111* section above. Default: Option B (new REQ_111 analyzers born on `HookedModel`); fall back to Option A or C based on REQ_111's first-analyzer timing.
- **Whether `required_weights` joins `required_hooks` as a separate declaration.** Default: no. `required_hooks` is the load-bearing surface; weight reads `KeyError` cleanly when a weight is missing, which is enough for the small number of analyzers that read weights without reading activations.
- **Whether `library/activations.py` and `library/weights.py` fold into `HookedModel` or rewrite to take it as an arg.** Default: fold. If implementation surfaces an analysis-side concern that doesn't belong on `HookedModel`, push that concern back into a re-purposed library module.
- **Migration order across analyzers.** Default: by file, in alphabetical order. Alternative: by complexity (start with read-only analyzers; finish with cross-epoch analyzers that compose multiple sites). Choice is a discoverability question, not a correctness one.

---

## Architecture Notes

### What the canonical-name pipeline filter looks like

```python
# In the pipeline, before invoking analyzers per epoch:
runnable = []
for analyzer in family.analyzers:
    missing = [h for h in analyzer.required_hooks if h not in model.hook_names()]
    if missing:
        log.skip(analyzer=analyzer.name, family=family.name, missing_hooks=missing)
        continue
    runnable.append(analyzer)
for analyzer in runnable:
    analyzer.analyze(model, cache, epoch)
```

The skip log is the structured record of "which analyzers don't run on which families and why." Existing skips today (via `architecture_support`) are silent; this is a strict improvement.

### How architecture-conditional analyzers split into subclasses

Today (legacy path):

```python
class FrequencyAnalyzer:
    architecture_support = {"transformer", "mlp_one_hot", "mlp_embedding"}
    def analyze(self, model, cache, epoch):
        if isinstance(model, HookedTransformer):
            we = model.W_E  # transformer's shared embedding
        elif isinstance(model, HookedEmbeddingMLP):
            we = model.embed_a + model.embed_b  # forced summation
        else:  # one-hot
            we = identity_matrix
        # ... compute spectrum ...
```

After REQ_114:

```python
class FrequencyFromSharedEmbedding:
    required_hooks = ["embed.hook_out"]
    required_weights = ["embed.W_E"]  # if we adopt this declaration
    def analyze(self, model, cache, epoch):
        we = model.get_weight("embed.W_E")
        # ... compute spectrum ...

class FrequencyFromPerInputEmbeddings:
    required_hooks = ["embed.hook_out"]
    required_weights = ["embed.embed_a", "embed.embed_b"]
    def analyze(self, model, cache, epoch):
        ea = model.get_weight("embed.embed_a")
        eb = model.get_weight("embed.embed_b")
        # ... compute spectrum, distinguishing per-input contributions ...
```

The pipeline picks which one runs based on what the family's model publishes. Both produce per-epoch artifacts under different `analyzer_name` directories, so cross-variant comparison can pull from either.

The transformer dispatch is structurally clean (one analyzer publishes its support; the pipeline picks it). The embedding-MLP dispatch surfaces a real semantic question — *what does "frequency from per-input embeddings" mean when the two are summed at inference time?* — that the legacy path glossed over with a forced summation. This is a *feature*: the boundary makes domain ambiguity visible.

### Why `groups.py` and `sites.py` stay analysis-side

The original REQ_105 (pre-rewrite) listed:

> `analysis/core/groups.py` folds into `ArchitectureAdapter`
> `analysis/core/sites.py` folds into `ArchitectureAdapter`

Reversed here. Both modules carry domain knowledge:

- `groups.py` defines `COMPONENT_GROUPS`, `WEIGHT_MATRIX_NAMES`, `ATTENTION_MATRICES`, `ARCH_WEIGHT_NAMES` — *how the analysis layer organizes weights into conceptual groups* (e.g., MLP weights as a group, attention Q/K/V/O as a group). The model itself doesn't have an opinion on what counts as a "group"; that's an analysis decision.
- `sites.py` defines `WeightSite` and `ActivationSite` enums — *which canonical hook paths the analysis layer treats as reportable measurement sites*. Again, the model publishes hooks; the analysis layer chooses which ones are sites.

Folding them into `HookedModel` would either:

- Push frequency-group / site knowledge into model code (where it doesn't belong; `HookedTransformer` shouldn't know what a "frequency group" is).
- Force the model to depend on the analysis layer's vocabulary (cycles).
- Make `HookedModel` a god-class.

The right shape: `groups.py` and `sites.py` are *consumers* of `miscope.core.architecture` and `miscope.core.weights`. They map canonical names into analysis-domain abstractions. They take canonical names as input; they do not extend the model surface.

### Coordination diagram with REQ_111 / REQ_102

```
                          REQ_111 (parallel build)
                       ┌──────────────────────────┐
   REQ_109 primitives  │  build new analyzer X    │
   (foundation)        │       parallel with      │
        │              │       old analyzer X     │
        │              │                          │
        ▼              │  validate parity on      │
   REQ_106 layering    │      reference variants  │
                       │           │              │
                       │           ▼              │
                       │  record outcome in Notes │
                       └──────────────┬───────────┘
                                      │
                              REQ_102 (retirement)
                       ┌──────────────────────────┐
                       │  retire old analyzer X   │
                       │  per validation outcome  │
                       └──────────────┬───────────┘
                                      │
                       REQ_114 (HookedModel migration)
                       ┌──────────────────────────┐
                       │  surviving analyzer (new │
                       │  if Option B; old if old │
                       │  survived parity)        │
                       │                          │
                       │  on HookedModel surface  │
                       └──────────────────────────┘
```

The flow is single-threaded per analyzer; multiple analyzers can be at different stages simultaneously. Option B compresses this by birthing the new analyzer on `HookedModel` directly, skipping the "old code stays on legacy, new code on `HookedModel`, then sweep" step.

---

## Notes

- **REQ_114 is the cleanup REQ.** REQ_105/112/113 are the load-bearing structural work. By the time REQ_114 starts, the boundary has been proven (canary) and exercised across all three architectures. This REQ scales the pattern to the rest of the analyzers — incremental work, low conceptual risk, but high migration volume.
- **The "fold groups.py / sites.py into adapter" instruction from the original REQ_105 is intentionally reversed here.** Recording the reversal so future readers understand it was a deliberate change, not an oversight.
- **REQ_111 may produce findings that change REQ_114's scope.** A common pattern would be "REQ_111 found that old analyzer X had bug Y; X gets retired immediately; X is removed from REQ_114's migration list." That's the expected workflow; REQ_114's manifest is recomputed against REQ_102's retirement list at the start of execution.
- **Architecture-conditional split as a research-quality improvement.** The forced summation in the legacy `FrequencyAnalyzer` is the kind of thing that silently encodes a methodological choice. Splitting into two `required_weights`-distinguished analyzers makes the choice explicit and reviewable. Worth flagging during migration when a forced branch is uncovered — those are publication-level documentation moments (and likely fieldnotes prompts).
- **CHANGELOG entries** describe the migration outcomes: which analyzers split into multiple subclasses; which architecture-conditional behaviors were uncovered; any artifact-schema changes (none expected — this REQ is byte-identical-preserving).
- **Branching note.** Suggested: a fresh `feature/analyzer-migration` branch from `develop` after REQ_113 merges, since REQ_114's diff is large and dispersed across `analysis/analyzers/`.

---

## Implementation outcome (2026-05-06)

Landed on `feature/architecture-adapter` continuing the REQ_105 → REQ_112 → REQ_113 → REQ_114 sequence.

**Infrastructure:**
- `ActivationContext` gains `model: HookedModel | None`, `cache: ActivationCache | None`, `logits: torch.Tensor | None` fields. The legacy `bundle` field is removed entirely.
- `ActivationBundle` Protocol deleted from `analysis/protocols.py`.
- `TransformerLensBundle` (`src/miscope/analysis/bundle.py`) and `MLPBundle` (`src/miscope/analysis/mlp_bundle.py`) deleted.
- `family.run_forward_pass` removed from `BaseModelFamily`, `ModelFamily` Protocol, and all three concrete family implementations. The pipeline now calls `model.run_with_cache(probe)` directly.
- `analysis/library/activations.py` rewritten to consume canonical-name `ActivationCache` and `HookedModel`. TL imports cleared. Helpers: `extract_mlp_activations`, `extract_attention_patterns`, `extract_residual_stream`, `get_embedding_weights`, `reshape_to_grid`, `compute_grid_size_from_dataset`. The legacy `run_with_cache` convenience wrapper is gone (callers use `HookedModel.run_with_cache` directly).
- `analysis/library/weights.py`: `extract_parameter_snapshot` and `compute_weight_singular_values` now take `HookedModel`, read via `model.get_weight(canonical_name)`. The legacy short-name dict (W_E, W_in, embed_a, ...) is preserved as the consumer-facing return shape so artifact schemas don't drift.
- `analysis/library/landscape.py`: type relaxed from `HookedTransformer` to `nn.Module`. TL imports cleared.

**Primary analyzer migration (13 analyzers):**
Each replaces `architecture_support` with `required_hooks` and reads via `ctx.cache[canonical]` / `ctx.model.get_weight(canonical)` / `ctx.logits`:
- attention_patterns, attention_freq (read `blocks.0.attn.hook_pattern`)
- attention_fourier (reads embed.W_E + per-head Q/K/V)
- neuron_activations, coarseness, neuron_freq_clusters (read `blocks.0.mlp.hook_out`)
- input_trace (reads `ctx.logits`; handles both transformer (3D) and MLP (2D) shapes)
- dominant_frequencies (reads embed.W_E)
- fourier_nucleation (reads embed.W_E + blocks.0.mlp.in.W)
- parameter_snapshot, effective_dimensionality (use `extract_parameter_snapshot` / `compute_weight_singular_values` taking HookedModel)
- landscape_flatness (reads `ctx.model` directly via nn.Module surface)
- gradient_site (cross-epoch; uses `model.get_weight(...).grad` instead of `model.embed.W_E.grad`)
- repr_geometry was already migrated under REQ_112.

**Non-primary analyzer cleanup:**
`architecture_support` flag dropped from 11 cross-epoch + secondary analyzers (centroid_dmd, fourier_frequency_quality, freq_group_weight_geometry, global_centroid_pca, input_trace_graduation, intragroup_manifold, neuron_dynamics, neuron_fourier, neuron_group_pca, parameter_trajectory_pca, transient_frequency). These analyzers process saved artifacts, not live model state — `architecture_support` was vestigial.

**Pipeline filter:**
`_run_single_epoch` checks `required_hooks` against `model.hook_names()` before invoking each analyzer. Analyzers whose required hooks are absent are skipped with a structured info-level log entry (analyzer name, missing hooks, epoch). Empty `required_hooks` (e.g., for analyzers reading only weights or logits) means "runs on any architecture."

**Test migration:**
Every test file constructing `TransformerLensBundle` or TL-tuple-keyed mock caches migrated to canonical-name `ActivationCache` and `HookedModel`-shaped mock models. The legacy bundle test classes were removed (`TestMLPBundleOnOneHotMLP`, `TestMLPBundleOnEmbeddingMLP`, `TestModuloAddition2LMLPActivationBundle`, etc.) — those tested the bundle adapter; with the bundle gone, the canonical surface is exercised directly.

**Quarantine — load-bearing:**
`grep -rn 'transformer_lens' src/miscope/` returns hits only inside the canonical surface: `architectures/hooked_transformer.py` and `architectures/hooks.py` (the `HookPoint` alias). Every entry on `LEGACY_TL_IMPORTERS` was cleared during this REQ. The strict version of the quarantine test from REQ_112 is now load-bearing — adding any TL import outside the canonical surface fails the test suite.

**Validation (2026-05-06):**
- 1462 tests passing, 0 regressions on the local suite.
- **REQ_086 byte-identity passed: 3027/3027 artifacts byte-identical to develop** on canon (`p113/s999/ds598`) against `reference_checksums_req112.json`. Validated the prediction that the canonical-cache read path is bit-equivalent to the legacy bundle reads — analyzer compute paths are unchanged; only the input source moved (zero-copy translation between TL's cache and the canonical-keyed cache).
- **Smoke test passed:** a fresh variant created post-REQ_114 ran the full analysis pipeline end-to-end without issue, exercising paths the regression scaffold doesn't cover (model construction, training loop, full pipeline run, all analyzers).

**Process notes:**
- `required_hooks: list[str] = []` for analyzers that read only weights or logits (no cache reads). The pipeline's filter trivially passes for these — they run on any architecture. Family-level registration (in family.json) is the primary mechanism for restricting analyzers to compatible architectures; runtime `KeyError` on `model.get_weight` is the safety net for misregistration.
- `required_weights` was considered but not adopted (REQ_114's flexibility note). The two-axis declaration (hooks + weights) adds complexity for the small number of analyzers that read weights without reading activations; family registration + KeyError is sufficient.
- The forced-summation antipattern called out in the original REQ_114 spec did not surface during migration — none of the migrated analyzers had architecture-conditional branching that needed splitting into two `required_hooks`-distinguished subclasses. If such branching is added in the future (e.g., a frequency analyzer that wants both transformer's `embed.W_E` and embedding-MLP's `embed.embed_a`+`embed.embed_b`), the two-subclass pattern is the right resolution.
- Coordination with REQ_111: Option A from the original ordering discussion — REQ_114 migrated the surviving analyzers en masse, after REQ_111 ships any new ones. REQ_111's first-analyzer-pair decision can now be Option B (born on `HookedModel`) without follow-up sweep, since the migration is complete.
