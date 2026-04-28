# REQ_105: Architecture Adapter Layer (Model Construction + Hook Standardization)

**Status:** Draft
**Priority:** High
**Branch:** TBD
**Dependencies:**
- REQ_106 (layering principle — `ArchitectureAdapter` is the data-plane edge for live model state; derivations consume adapter outputs as pure tensors, not the adapter itself).
- REQ_087 (ActivationBundle) — predecessor. `ActivationBundle` was a
  partial step that backed into the architecture-abstraction problem
  from the analyzer side without addressing construction. This REQ
  steps back and asks whether a single `ArchitectureAdapter` covers
  both construction and run-time access, potentially making the
  separate Bundle layer unnecessary. **Whether `ActivationBundle`
  survives as a distinct protocol is a discovery question resolved
  during this REQ's implementation** — not a precondition.
- Blocks REQ_103 (PyPI Publication Hardening). Adapter shape affects the
  public API surface and the dependency story; v1.0 should mint after
  this lands.
**Attribution:** Engineering Claude

---

## Problem Statement
There are two major motivators for this work:
1. We need clean support for multiple model architectures trained against the same task
2. We need to isolate dependencies on the TransformerLens library due a recent release and upcoming plans to deprecate HookedTransformer functionality.
3. The Analyzer pipeline should be able to process models with different 
architectures agnostically.

An audit uncovered the following areas where there is tight coupling with TransformerLens, beyond the analyzer-side coupling addressed by REQ_087:

1. **Model construction.** `BaseModelFamily.create_model` returns a
   concrete `HookedTransformer` ([families/protocols.py:105](src/miscope/families/protocols.py#L105),
   [families/implementations/modulo_addition_1layer.py:55-71](src/miscope/families/implementations/modulo_addition_1layer.py#L55-L71)).
   The platform also supports two non-TL families that each return their
   own concrete `nn.Module` subclass and invent parallel abstractions
   (`*ActivationBundle` classes + the `architecture_support` flag on
   analyzers) to bridge to the analyzer layer:
   - [`ModuloAddition2LMLPFamily`](src/miscope/families/implementations/modulo_addition_2l_mlp.py)
     — one-hot 2L MLP from REQ_088 (no learned representation, no attention).
   - [`ModuloAdditionEmbedMLPFamily`](src/miscope/families/implementations/modulo_addition_embed_mlp.py)
     — learned-embedding 2L MLP (rung 2 of the architecture ladder:
     learned `embed_a` + `embed_b` summed into a shared d_embed
     representation, then a single hidden layer; no attention, no
     residual stream).
2. **Hook naming.** Analyzers reference TL's hook strings directly
   (e.g., `hook_attn_out`, `hook_pre`) and read from the cache by string
   key. There is no platform-owned vocabulary for "the MLP pre-activation
   hook" independent of TL's spelling.
3. **Forward-pass orchestration.** `analysis/library/activations.py`
   wraps `model.run_with_cache()` with TL-specific assumptions baked in.

This coupling is a strategic risk for v1.0 publication. TransformerLens
3.0 (released 2026-04-17) deprecates `HookedTransformer` in favor of
`TransformerBridge`, which is positioned for HuggingFace pretrained
models. The migration guide is pretrained-centric; long-term support
for from-scratch toy-model authorship is unclear.

The fix is structural, not tactical: introduce an `ArchitectureAdapter`
layer that quarantines all TL-specific code behind a single boundary,
and adopt TL 3.0's standardized component / hook naming as the platform's
canonical vocabulary. This delivers three wins independent of the TL
version question:

- **Three-architecture unification.** All three modular addition
  families currently in this platform — the 1-layer transformer
  (`ModuloAddition1LayerFamily`), the one-hot 2L MLP
  (`ModuloAddition2LMLPFamily`), and the learned-embedding 2L MLP
  (`ModuloAdditionEmbedMLPFamily`) — sit behind one contract. The
  `architecture_support` flag on analyzers retires; per-family
  `*ActivationBundle` classes fold into their adapters.
- **Naming win without migration risk.** TL 3.0's component hierarchy
  (`embed`, `blocks.{i}.attn.q.hook_out`, etc.) is a thoughtful design.
  Adopt the names internally; a thin translation layer handles whichever
  TL version is mounted underneath.
- **Publication-ready dependency story.** TL becomes a swappable backend.
  Researchers extending miscope with a novel architecture provide an
  adapter, not a fork.

This REQ does *not* commit to migrating to TL 3.0. It commits to making
that decision a leaf change rather than a structural one.

---

## Conditions of Satisfaction

### Canonical naming (`miscope.core.architecture`)

- [ ] New module `miscope/core/architecture.py` defines the platform's
  canonical component names and hook point names, mirroring the TL 3.0
  Model Structure spec.
- [ ] **Top-level components:** `embed`, `pos_embed`, `blocks`,
  `ln_final`, `unembed`.
- [ ] **Block-level:** `ln1`, `attn`, `ln2`, `mlp`.
- [ ] **Attention subcomponents:** `q`, `k`, `v`, `o`, `qkv`.
- [ ] **MLP subcomponents:** `in`, `pre`, `out`.
- [ ] **Standard hook points** (per-component, where applicable):
  `hook_in`, `hook_out`, `hook_pre`, `hook_pattern`,
  `hook_attn_scores`, `hook_hidden_states`.
- [ ] Names are exposed as enums or string constants in
  `miscope.core.architecture`. Analyzer code references the constants,
  not raw strings.
- [ ] A short docstring on the module cites the TL 3.0 Model Structure
  page as the source spec, with rationale for adopting it verbatim
  rather than inventing a parallel vocabulary.

### `ArchitectureAdapter` protocol (`miscope.architectures` or
`miscope.analysis`)

- [ ] Protocol `ArchitectureAdapter` defines:
  - `construct(config: dict) -> Self` — class-method or factory; builds
    the underlying model from a config dict.
  - `forward(probe: torch.Tensor) -> AdapterRunContext` — runs the
    model with hooks. The return type exposes canonical-name reads
    of activations captured during the pass. **Implementation choice:**
    either the adapter itself acts as its own post-forward run
    context (Option A, simpler), or `forward()` returns a separate
    object that may or may not be the existing `ActivationBundle`
    (Option B). See Architecture Notes.
  - `get_activation(canonical_name: str) -> torch.Tensor` —
    canonical-name read of an activation captured during the most
    recent forward pass.
  - `get_weight(canonical_name: str) -> torch.Tensor` — named weight
    matrix access (e.g., `W_E`, `W_in`).
  - `supports(canonical_name: str) -> bool` — explicit support check
    that replaces the per-analyzer `architecture_support` flag.
- [ ] The protocol does not expose any TL types. `from transformer_lens
  import ...` outside an adapter implementation is a regression.

### `TransformerLensAdapter` implementation

- [ ] `TransformerLensAdapter` wraps the existing
  `HookedTransformer` + `HookedTransformerConfig` construction path.
- [ ] Holds an internal mapping from canonical names (TL 3.0 spec) to
  TL 2.x legacy hook names. The mapping is the entire surface where
  TL's spelling is allowed to leak.
- [ ] Implements `forward()` by delegating to `model.run_with_cache()`.
  The cache is held internally; canonical-name reads translate to TL
  hook strings via the adapter's mapping table. Whether the cache is
  exposed via a separate Bundle object or via methods on the adapter
  follows the implementation decision in the protocol section above.
- [ ] `supports()` returns `True` for transformer-only canonical names
  (e.g., `blocks.{i}.attn.hook_pattern`) and falls through for
  unsupported names.

### `OneHotMLPArchitectureAdapter` implementation

- [ ] `OneHotMLPArchitectureAdapter` wraps the one-hot 2L MLP
  construction path introduced in REQ_088
  (`ModuloAddition2LMLPFamily`). Input is one-hot concatenation of
  `(a, b)`; no learned embedding, no attention, no residual stream.
- [ ] Implements canonical names where they apply (`mlp.hook_pre`,
  `mlp.hook_out`, weights `W_in` / `W_out` via `get_weight`); returns
  `False` from `supports()` for transformer-only names (e.g., attention
  hooks) **and** for embedding-bearing names (`embed.hook_out`, `W_E`).
- [ ] `ModuloAddition2LMLPActivationBundle` is retired. Captured
  activations are reachable through canonical-name accessors on the
  adapter (or its run-context return value, per the protocol decision
  above).

### `EmbeddingMLPArchitectureAdapter` implementation

- [ ] `EmbeddingMLPArchitectureAdapter` wraps the learned-embedding
  2L MLP construction path (`ModuloAdditionEmbedMLPFamily`). Two learned
  embedding matrices (`embed_a`, `embed_b`) are summed into a shared
  d_embed representation; no attention, no residual stream.
- [ ] Implements canonical names for both the embedding and MLP
  components: `embed.hook_out` (post-sum representation),
  `mlp.hook_pre`, `mlp.hook_out`. `get_weight` exposes `embed_a`,
  `embed_b`, `W_in`, `W_out`. Returns `False` from `supports()` for
  attention-bearing canonical names.
- [ ] **Embedding identity.** The current
  `ModuloAdditionEmbedMLPActivationBundle` deliberately exposes the
  two embeddings under `embed_a` / `embed_b` keys (not `W_E`) to
  prevent transformer-class dispatch from misfiring on this MLP
  architecture
  ([modulo_addition_embed_mlp.py:81-104](src/miscope/families/implementations/modulo_addition_embed_mlp.py#L81-L104)).
  The adapter must preserve this distinction: presence of `W_E` is
  reserved for the transformer adapter; the embedding-MLP exposes
  per-input embeddings under their own canonical names.
- [ ] `ModuloAdditionEmbedMLPActivationBundle` is retired. Captured
  activations are reachable through canonical-name accessors on the
  adapter (or its run-context return value).

> The two MLP adapters may share a base class or composition helper
> if implementation surfaces a clean factoring (e.g., a shared MLP
> hook-capture mixin). This is a **flexibility**, not a requirement —
> see Constraints. Don't fold them prematurely; the embedding presence
> changes both `supports()` and `get_weight` semantics.

### Family integration

- [ ] `BaseModelFamily.create_model` return type changes to
  `ArchitectureAdapter`. The concrete adapter class is family-specific.
- [ ] `ModuloAddition1LayerFamily.create_model` returns a
  `TransformerLensAdapter`.
- [ ] `ModuloAddition2LMLPFamily.create_model` returns a
  `OneHotMLPArchitectureAdapter`.
- [ ] `ModuloAdditionEmbedMLPFamily.create_model` returns an
  `EmbeddingMLPArchitectureAdapter`.
- [ ] No analyzer or library function imports from `transformer_lens`.
  The TL dependency is contained to `TransformerLensAdapter`.

### Analyzer migration

- [ ] All analyzers in `analysis/analyzers/` migrate from raw cache
  string keys (`cache["blocks.0.mlp.hook_pre"]`,
  `cache["post", layer, "mlp"]`) to canonical-name lookups via the
  adapter (or its run-context return value).
- [ ] The `architecture_support` class attribute on analyzers is
  retired. Architecture-conditional behavior is expressed by checking
  `adapter.supports(canonical_name)` at call sites, or by the analyzer
  declaring required hooks and the pipeline filtering analyzers per
  family.
- [ ] `analysis/library/activations.py` either folds into the adapter (preferred)
  or its functions are rewritten to take `ArchitectureAdapter`
  instead of `(model, cache)` tuples.
- [ ] `analysis/library/weights.py` either folds into the adapter (preferred)
  or its functions are rewritten to take `ArchitectureAdapter`
  instead of taking ActivationBundle as an argument.
- [ ] `analysis/core/groups.py` folds into `ArchitectureAdapter`
- [ ] `analysis/core/sites.py` folds into `ArchitectureAdapter`
  

### Validation

- [ ] REQ_086 regression scaffold passes on all reference variants
  with the refactored code path. Numerical outputs are byte-identical
  (or differ only within documented floating-point tolerance) before
  and after migration.
- [ ] All existing tests pass.
- [ ] **Quarantine smoke test:** `grep -rn "transformer_lens" src/miscope/`
  returns hits only inside the `TransformerLensAdapter` module. Any
  other location is a regression.
- [ ] Stub-adapter test: a no-op `StubAdapter` can be substituted for
  `TransformerLensAdapter` and analyzer code fails at the adapter
  boundary (predictable error), not deeper in the analyzer.

---

## Constraints

**Must:**
- TL imports occur in exactly one module: `TransformerLensAdapter`.
  This is the load-bearing quarantine.
- Canonical names match the TL 3.0 Model Structure spec exactly.
  We are adopting the standard, not extending or inventing one.
- Adapter is a zero-cost wrapper. No tensor copies, no recomputation.
  Canonical-name reads pass through to the underlying cache / model
  attribute.
- Adapter must be compatible with TL 2.x at the time of this REQ
  landing. TL 3.x migration is a *separate* decision, gated on the
  maintainer-question outcomes in
  [Roadmap_TL_Critical.md](../Roadmap_TL_Critical.md).

**Must avoid:**
- Premature abstraction for hypothetical *future* architectures. The
  three adapters specified above (transformer, one-hot MLP,
  embedding-MLP) cover every architecture currently trained against
  this platform's tasks and are sufficient to validate the protocol.
  A *fourth* adapter (e.g., for a future non-periodic toy task or an
  external researcher's architecture) is a follow-up under the
  requirement that introduces it.
- Collapsing the two MLP adapters too eagerly. They share an MLP
  block but differ at the input boundary: one-hot MLP has no
  embedding component; embedding-MLP exposes `embed.hook_out` and
  per-input embedding weights. Sharing implementation via a base
  class is fine if it falls out cleanly; forcing a single adapter
  to handle both via branching is not.
- Making the adapter framework-aware on the consumer side. An analyzer
  must not be able to tell whether it's running against TL or against
  raw PyTorch hooks. If it can, the abstraction has leaked.
- Coupling this REQ to a specific TL version. The adapter design must
  permit the underlying TL pin to move (2.x → 3.x compat layer → 4.x
  Bridge if from-scratch authorship lands → off-TL entirely) without
  touching analyzer code.
- Coupling adoption of canonical names to migration of TL itself.
  Internal canonical names + a translation table is the entire change;
  the underlying TL version is independent.

**Flexible:**
- Whether `ArchitectureAdapter` is a `Protocol` (structural) or an
  ABC. Default: Protocol, matching REQ_087's `ActivationBundle`.
- Module placement: `miscope/architectures/` (new top-level submodule)
  vs `miscope/analysis/architectures/` vs co-locating with families.
  Default: a new top-level `miscope/architectures/` submodule, since
  adapters cross-cut analysis and family concerns.
- Whether the canonical-to-legacy hook name mapping is a static dict
  or generated programmatically from a TL introspection helper.
  Default: static dict, explicitness > cleverness.
- **Whether `ActivationBundle` survives as a distinct protocol.**
  Default: Option A — adapter exposes canonical-name reads directly;
  Bundle is absorbed and removed. Option B (Bundle preserved as a
  per-probe run-context object) is admissible only if a concrete need
  surfaces during migration. See Architecture Notes for the decision
  framing.
- Whether `MLPActivationBundle` is deleted outright or kept as an
  internal implementation detail of `MLPArchitectureAdapter`. Default:
  fold inside the adapter; remove the standalone export.

---

## Architecture Notes

**Revisiting `ActivationBundle` — this is a pivot moment.**
REQ_087 introduced `ActivationBundle` as an analyzer-facing read
interface, but it was — in the user's framing — "somewhat backed into."
It abstracted the read side without addressing construction, hook
naming, or the underlying architecture asymmetry between the
transformer and MLP families. With a proper `ArchitectureAdapter` in
place, the bundle's role is open. Two options to evaluate during
implementation:

- **Option A — Bundle absorbed (default).** The adapter holds the
  most recent forward-pass cache internally; canonical-name reads
  are methods on the adapter. No separate Bundle type. Simpler
  surface; analyzers receive a single object.

  ```
  Family.create_model()
         │
         ▼
  ArchitectureAdapter            ◄── construction + naming + run-time
         │ .forward(probe)
         │ .get_activation(canonical_name)
         │ .get_weight(canonical_name)
         ▼
  Analyzer                       ◄── architecture-agnostic
  ```

- **Option B — Bundle preserved.** The adapter's `forward()` returns
  a Bundle, which is the analyzer-facing read interface with its own
  lifecycle (per-probe). Worth keeping only if a concrete need
  surfaces — e.g., parallel probes with parallel caches, or a hard
  insistence that analyzers never receive a construction-capable
  object.

  ```
  ArchitectureAdapter
         │ .forward(probe)
         ▼
  ActivationBundle               ◄── per-probe read interface
         │ .get_activation(canonical_name)
         ▼
  Analyzer
  ```

Default to Option A unless implementation surfaces a real need for
Option B. Either way, REQ_087's contribution is preserved as the
*shape* of the read interface (canonical-name access, weight access,
explicit support checks) — what changes is whether that interface
lives on a separate object or on the adapter itself. This is a pivot
moment: REQ_087 may end up superseded entirely.

**Why adopt TL 3.0's naming verbatim.**
The naming hierarchy on the TL 3.0
[Model Structure page](https://transformerlensorg.github.io/TransformerLens/content/model_structure.html)
is a clean conceptual design — uniform `hook_in` / `hook_out`
suffixes, explicit Q/K/V/O sub-paths, separate `ln1` / `ln2` / `attn` /
`mlp` block decomposition. Inventing a parallel vocabulary would be
churn for no gain and would make researcher onboarding harder
(they already know TL's names). Adoption is the right call regardless
of whether we ever migrate to TL 3.x at the runtime level.

**Why quarantine TL even if we stay on 2.x.**
- The MLP family already proves the abstraction has more than one
  consumer; its current half-solution (`MLPActivationBundle` + the
  `architecture_support` flag) is technical debt.
- For PyPI v1.0, dependency surface matters. Researchers running miscope
  notebooks shouldn't be forced into a specific TL version range; a
  declared adapter can pin TL conservatively in its own metadata.
- Strategic optionality: if TL 3.0/4.0 close the door on from-scratch
  authorship, miscope retains analyzer code that doesn't care.

**Critical-question decoupling.**
[Roadmap_TL_Critical.md](../Roadmap_TL_Critical.md) lists open questions
to ask the TL maintainers about from-scratch model support in 4.0. Those
questions are *informational* — they affect the priority of TL version
migration but do not gate this REQ. The adapter design is sound either
way:
- Maintainers confirm continued from-scratch support → adapter is still
  the right home for the construction path; just easier to migrate.
- Maintainers say "compat layer until 4.0, then Bridge-only" → adapter
  contains the blast radius; we pin TL accordingly.
- Maintainers say "off-ramp from-scratch users" → adapter swaps to a
  different backend (raw PyTorch + custom hooks, nnsight, etc.) without
  touching analyzers.

**Retiring the `architecture_support` flag.**
The current pattern (each analyzer declaring which architectures it
supports as a class attribute) is a code smell: analyzers know about
architectures. The right pattern is the inverse — adapters declare what
hooks they support; the pipeline filters. This REQ's `supports()`
method is the replacement.

**The coupling points after REQ_105 (Option A default):**

| Coupling site | Current | After REQ_087 (in flight) | After REQ_105 |
|---|---|---|---|
| Analyzer reads cache | `cache[...]` | `bundle.mlp_post(...)` | `adapter.get_activation(canonical_name)` |
| Analyzer reads weights | `model.W_E` | `bundle.weight("W_E")` | `adapter.get_weight(canonical_name)` |
| Family creates model | returns `HookedTransformer` | returns `HookedTransformer` | returns `ArchitectureAdapter` |
| Pipeline runs forward | `model.run_with_cache(probe)` | `model.run_with_cache(probe)` (wrapped) | `adapter.forward(probe)` |
| TL imports | scattered | scattered | confined to `TransformerLensAdapter` |

Under Option B, the read columns would route through a Bundle object
returned by `adapter.forward()`. Same canonical-name surface; different
object owning it.

---

## Notes

- **This is a pivot moment for the platform.** Earlier work
  (REQ_087, the `architecture_support` flag, `MLPActivationBundle`)
  approached the architecture-abstraction problem incrementally from
  the analyzer side. This REQ steps back and treats architecture as a
  first-class platform concept. The user's framing: "ActivationBundle
  was somewhat backed into; a full revisit is warranted." Implementers
  should not assume any specific piece of REQ_087 survives — judge it
  on whether it earns its keep against the new abstraction.
- **Reference for canonical names:** the
  [TransformerLens 3.0 Model Structure page](https://transformerlensorg.github.io/TransformerLens/content/model_structure.html).
  When the spec page changes, the canonical names module's docstring
  should call out the version of the spec adopted and the date.
- **Sequencing relative to publication.** This REQ should land before
  REQ_103 (PyPI gate), because the public API surface includes the
  adapter protocol and naming module. Researchers building on miscope
  v1.0 will be writing against `miscope.core.architecture` and
  (potentially) implementing custom adapters.
- **Not in scope for this REQ:** TL 3.x migration of the underlying
  runtime. That is a follow-up decision once maintainer questions
  resolve. Per the constraints above, the adapter is shaped to make
  that decision low-cost.
- **In scope:** all three architectures currently trained against
  this platform — the 1-layer transformer, the one-hot 2L MLP, and
  the learned-embedding 2L MLP. Implementing all three under the
  protocol up front avoids a guaranteed follow-up refactor and
  exercises three meaningfully different shapes (with attention,
  with embedding but no attention, with neither).
- **Not in scope:** a fourth architecture adapter. If a future
  requirement introduces a new architecture (e.g., for an upcoming
  non-periodic toy task), its adapter lands under that requirement.
- **Branching note:** discovery on the current `refactor-dataview`
  branch should not block on this REQ; the adapter refactor is large
  enough to warrant its own branch when sequenced.
