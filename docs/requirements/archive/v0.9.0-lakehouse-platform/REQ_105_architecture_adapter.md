# REQ_105: HookedModel Base Class and Canonical Vocabulary

**Status:** Draft (revised after the original adapter framing surfaced as wrong-shaped during initial implementation; see Architecture Notes — *Why this REQ was rewritten*).
**Priority:** High — load-bearing for v1.0 publication. The model boundary affects the public API surface and the dependency story.
**Branch:** `feature/architecture-adapter` (current — pre-rewrite work on canonical vocabulary lives here; see commit b3681ac).
**Dependencies:**
- REQ_106 (layering principle — `HookedModel` is the data-plane edge for live model state; analysis layer consumes its outputs as pure tensors, not the model object).
- REQ_087 (ActivationBundle) — predecessor. `ActivationBundle` was a partial step that approached the problem from the analyzer side. Under this REQ's framing, the bundle is **superseded entirely**: cache reads happen through `cache[canonical_name]`; weight reads through `model.get_weight(canonical_name)`. REQ_087's contribution survives as the *idea* of canonical-name access; the object goes away.
- Blocks REQ_112 (HookedTransformer implementation — the first concrete subclass).
- Blocks REQ_113 (HookedMLP implementations — the second and third concrete subclasses).
- Blocks REQ_114 (analyzer migration onto `HookedModel`).
- Blocks REQ_103 (PyPI Publication Hardening). The `HookedModel` interface is part of the v1.0 public API surface.

**Attribution:** Engineering Claude (revised under user direction after the original adapter framing was retired).

---

## Problem Statement

There are three motivators for this work:

1. We need clean support for multiple model architectures trained against the same task. The platform currently has three: a 1-layer transformer (`ModuloAddition1LayerFamily`), a one-hot 2L MLP (`ModuloAddition2LMLPFamily`), and a learned-embedding 2L MLP (`ModuloAdditionEmbedMLPFamily`).
2. We need to isolate dependencies on the TransformerLens library. TransformerLens 3.0 (released 2026-04-17) deprecates `HookedTransformer` in favor of `TransformerBridge`, which is positioned for HuggingFace pretrained models. The migration guide is pretrained-centric; long-term support for from-scratch toy-model authorship is unclear. miscope's needs go in the opposite direction — we train models, we don't wrap pretrained ones.
3. The Analyzer pipeline must process models with different architectures agnostically. Today the architecture leaks: each non-transformer family invents a parallel abstraction (`*ActivationBundle` classes plus the `architecture_support` flag on analyzers) to bridge to the analyzer layer.

An audit identified the coupling sites:

1. **Model construction.** `BaseModelFamily.create_model` returns a concrete `HookedTransformer` ([families/protocols.py:105](src/miscope/families/protocols.py#L105), [families/implementations/modulo_addition_1layer.py:55-71](src/miscope/families/implementations/modulo_addition_1layer.py#L55-L71)). The two MLP families return their own `nn.Module` subclasses with parallel `*ActivationBundle` abstractions.
2. **Hook naming.** Analyzers reference TL's hook strings directly (e.g., `hook_attn_out`, `hook_pre`) and read from the cache by string key. There is no platform-owned vocabulary.
3. **Forward-pass orchestration.** `analysis/library/activations.py` wraps `model.run_with_cache()` with TL-specific assumptions baked in.

**The fix is to make `HookedModel` the boundary.** A platform-owned base class that the trained model itself implements — not a wrapper around it. Architecture subclasses (`HookedTransformer`, `HookedOneHotMLP`, `HookedEmbeddingMLP`) each implement `setup_hooks()` to publish the canonical surface natively. Analyzers receive a `HookedModel` and a cache keyed by canonical names; they cannot tell whether the underlying model uses TransformerLens, raw PyTorch, or anything else.

This delivers three wins:

- **Three-architecture unification.** All three modular addition families sit behind one contract. The `architecture_support` flag retires; per-family `*ActivationBundle` classes fold inside their `HookedModel` subclasses.
- **Naming win without migration risk.** TL 3.0's component hierarchy (`embed`, `blocks.{i}.attn.q.hook_out`, etc.) is a thoughtful design. Adopt the names internally; the actual TL runtime version is independent.
- **Publication-ready dependency story.** TL becomes a swappable backend confined to one module. Researchers extending miscope with a novel architecture provide a `HookedModel` subclass, not a fork.

This REQ does **not** commit to migrating to TL 3.0. It commits to making that decision a leaf change rather than a structural one.

---

## Conditions of Satisfaction

This REQ delivers the abstract surface only. Concrete subclasses, family integration, and analyzer migration are scoped to follow-up REQs (REQ_112, REQ_113, REQ_114) so the boundary survives contact with one real architecture before the broad migration starts.

### Canonical naming (`miscope.core.architecture`)

- [x] Module `miscope/core/architecture.py` defines canonical component names and hook point names (already in place from commit b3681ac).
- [ ] Module docstring updated to remove `TransformerLensAdapter` references and describe the `HookedModel` framing.
- [ ] Module exports a frozen list of all canonical hook-point paths a model implementation may publish (consumers iterate this list to validate `required_hooks` declarations against an architecture).

### Canonical weight-name vocabulary (`miscope.core.weights`)

- [ ] New module `miscope/core/weights.py` defines canonical weight names, distinct from hook names (hook names refer to activation points; weight names refer to learned parameters).
- [ ] **Transformer weights:** `embed.W_E`, `pos_embed.W_pos`, `blocks.{i}.attn.q.W`, `blocks.{i}.attn.k.W`, `blocks.{i}.attn.v.W`, `blocks.{i}.attn.o.W`, `blocks.{i}.mlp.in.W`, `blocks.{i}.mlp.out.W`, `unembed.W_U`. Bias names follow the same paths with `.b` suffix.
- [ ] **MLP-only weights:** `blocks.{i}.mlp.in.W`, `blocks.{i}.mlp.out.W` (one-hot input case has no embedding; embedding-MLP case adds `embed.embed_a`, `embed.embed_b` — see REQ_113 for the embedding-identity rationale).
- [ ] Helpers mirror `architecture.py`: `weight(layer, component, sub)` etc.

### `HookPoint` (`miscope.architectures.hooks`)

- [ ] `HookPoint(nn.Module)` class with:
  - `name: str` — canonical path identifying the hook point.
  - `forward(x) -> x` — identity by default.
  - `add_hook(fn)` — register a forward hook.
  - `remove_hooks()` — clear registered hooks.
- [ ] When TL 2.x is the runtime, `HookPoint` may be a thin alias / subclass of `transformer_lens.hook_points.HookPoint`. Implementation choice — the canonical interface is what matters.

### `HookedModel` abstract base (`miscope.architectures.hooked_model`)

- [ ] `HookedModel(nn.Module)` abstract base class with:
  - `config` — concrete subclasses define their own config dataclass; the base does not prescribe a shape.
  - `hook_points: dict[str, HookPoint]` — populated by `setup_hooks()`.
  - `setup_hooks() -> None` — **abstract**; subclasses override to register their canonical hook points. Called from `__init__` after submodules are constructed.
  - `hook_names() -> list[str]` — return all canonical hook paths this model exposes. Default implementation reads `hook_points.keys()`.
  - `weight_names() -> list[str]` — return all canonical weight paths this model exposes. Concrete subclasses implement.
  - `get_weight(canonical_name: str) -> torch.Tensor` — canonical-name access to a learned weight matrix or bias. Concrete subclasses implement.
  - `run_with_cache(input, fwd_hooks: list | None = None) -> tuple[torch.Tensor, ActivationCache]` — forward pass with full activation capture. Returns `(logits, cache)`.
  - `run_with_hooks(input, fwd_hooks: list) -> torch.Tensor` — forward pass with caller-supplied hooks (no full cache).
- [ ] `HookedModel` declares **no TL types** in its signatures. `from transformer_lens import ...` outside a concrete subclass module is a regression.
- [ ] Calling an unknown canonical name raises `KeyError(name)` with a message listing the model's published hook / weight names. No silent `None` returns.

### `ActivationCache` (`miscope.architectures.activation_cache`)

- [ ] `ActivationCache` class — minimal wrapper over a `dict[str, torch.Tensor]` keyed by canonical hook names.
- [ ] `cache[name]`, `cache.keys()`, `cache.get(name, default)` — standard dict semantics.
- [ ] No analyzer-facing computation methods. The cache is a read interface, not a primitive layer.
- [ ] Optional: bracket-key sugar matching TL 2.x conventions (e.g., `cache["post", 0, "mlp"]` translating to `blocks.0.mlp.hook_out`) — **deferred to REQ_112** if the canary analyzer needs it; this REQ does not commit.

### Stub model + smoke tests

- [ ] `tests/architectures/test_hooked_model_stub.py` defines a minimal `StubHookedModel` subclass (one trivial linear layer, a single `mlp.hook_out` hook point) and asserts:
  - Construction succeeds; `setup_hooks()` is called.
  - `hook_names()` returns the published name.
  - `get_weight(unknown_name)` raises `KeyError` with helpful message.
  - `run_with_cache(probe)` returns `(logits, cache)` with the canonical name populated.
  - Substituting `StubHookedModel` for a real model in a downstream consumer (a placeholder analyzer that reads `cache["mlp.hook_out"]`) succeeds with no architecture-aware branching.
- [ ] Test suite imports nothing from `transformer_lens`.

### Module placement

- [ ] New top-level submodule `miscope/architectures/` holds `hooked_model.py`, `activation_cache.py`, `hooks.py`, and (under follow-up REQs) the concrete subclasses. This submodule cross-cuts analysis and family concerns; co-locating with families would imply families own architectures, which they don't.

### Validation

- [ ] All existing tests pass; this REQ is purely additive. No analyzer, family, or pipeline code changes under this REQ.
- [ ] Quarantine smoke test (preview): `grep -rn "transformer_lens" src/miscope/architectures/hooked_model.py src/miscope/architectures/activation_cache.py` returns zero hits. The base class and cache are TL-free.

---

## Constraints

**Must:**

- `HookedModel` is the contract. There is no wrapper, no adapter, no translation layer at this level. Concrete subclasses *are* the model.
- Canonical names match the TL 3.0 Model Structure spec exactly. We are adopting the standard, not extending or inventing one.
- `HookedModel` is a zero-cost interface. No tensor copies, no recomputation. Canonical-name reads pass through to the underlying cache / model attribute.
- `KeyError` on unknown names. Silent `None` is forbidden.

**Must avoid:**

- **Re-introducing the adapter pattern.** If a future requirement surfaces a need to wrap an external pretrained model, that's a `PretrainedModelAdapter(HookedModel)` subclass — wrapping happens *inside* the subclass, not above the abstraction. The base class never mediates.
- **Premature concrete subclasses.** This REQ ships the base class and stub. The first real subclass (`HookedTransformer`) lands under REQ_112 with the canary analyzer migration; this REQ doesn't commit to that work.
- **Folding analysis-side helpers into `HookedModel`.** The original adapter REQ proposed folding `analysis/core/groups.py` and `analysis/core/sites.py` into the adapter. Don't. Those modules carry analysis-side domain knowledge (frequency groups, what counts as a "site") that takes canonical-name inputs but is not part of the model's published surface. Folding them would either make `HookedModel` a god-class or push frequency-group knowledge into model code.
- **Coupling adoption of canonical names to migration of TL itself.** Internal canonical names + per-subclass translation tables are the entire change; the underlying TL version is independent.
- **Coupling this REQ to a specific TL version.** `HookedModel`'s contract permits the underlying TL pin to move (2.x → 3.x compat layer → 4.x Bridge if from-scratch authorship lands → off-TL entirely) without touching analyzer or base-class code. Concrete subclasses absorb the version question.

**Flexible:**

- Whether `HookedModel` is a `Protocol` (structural) or an ABC. Default: ABC, since `setup_hooks()` is genuinely abstract and `nn.Module` inheritance is structural anyway.
- Whether `HookPoint` is aliased from `transformer_lens.hook_points.HookPoint` or re-implemented in `miscope.architectures.hooks`. Default: alias for now (TL 2.x is in the dep tree regardless via `HookedTransformer`); re-implement if the TL pin is dropped under a future REQ.
- Whether `ActivationCache` exposes TL 2.x-style bracket-key sugar (`cache["post", 0, "mlp"]`). Default: no; canonical-name string lookup is the surface. Re-evaluate if migration ergonomics under REQ_112/114 surface a real need.
- Module placement: `miscope/architectures/` (new top-level submodule) is the default. `miscope/core/architectures/` is acceptable if the tree settles that way.

---

## Architecture Notes

### Why this REQ was rewritten

The original REQ_105 proposed an `ArchitectureAdapter` layer that would wrap concrete models, expose canonical-name reads, and translate TL hook names internally. During initial implementation on `feature/architecture-adapter`, the adapter pattern surfaced as wrong-shaped:

- The adapter started as a *model wrapper* (right shape for TransformerLens 3.0, which wraps HF pretrained models) but miscope owns every model it trains. There is no upstream object to "adapt."
- Interfaces leaked from TransformerLens (`run_with_cache`) onto the adapter because the adapter forwarded to the underlying model and inherited its idiom.
- The `model(input)` signature got overloaded in forced ways to accommodate both the wrapped model's call convention and the adapter's run-context return.
- The adapter ended up holding both construction (`construct(config) -> Self`) and runtime (`forward`, `get_activation`, `get_weight`) responsibilities — two different lifecycles glued together.
- The Option A vs Option B framing on `ActivationBundle` (does the bundle survive as a separate object?) was a symptom of unresolved identity: if the adapter is a wrapper, the bundle is its per-probe handle; if the adapter is the model, the bundle vanishes.

The reframe: **make `HookedModel` the contract.** The class your training loop instantiates *is* a `HookedModel`. Subclasses (`HookedTransformer`, `HookedOneHotMLP`, `HookedEmbeddingMLP`) implement `setup_hooks()` to publish the canonical surface natively. There is no wrapper. Construction and runtime are the standard PyTorch shape: `__init__` builds, instance methods run.

This collapses the awkward decisions in one move:

| Original REQ_105 | Revised REQ_105 |
|---|---|
| `ArchitectureAdapter.construct(config) -> Self` | Subclass `__init__(config)` (standard PyTorch) |
| `forward(probe) -> AdapterRunContext` | `run_with_cache(input) -> (logits, ActivationCache)` |
| `adapter.get_activation(name)` | `cache[canonical_name]` |
| `adapter.get_weight(name)` | `model.get_weight(canonical_name)` (kept) |
| `adapter.supports(name)` | `name in model.hook_names()` (queried by pipeline) |
| Option A / Option B Bundle question | Resolved: no Bundle. Cache is dict-keyed; per-probe lifetime falls out of `run_with_cache` returning a fresh cache per call. |
| Adapter holds construction + runtime | Subclass holds them via `nn.Module` inheritance |

The `run_with_cache` and `run_with_hooks` method names match TL's spelling **by choice** — they're sensible names, and matching reduces researcher onboarding cost. The fact that TL's underlying implementation may be reused is incidental; the contract is ours.

### The boundary

```
                    Training loop / experiments
                              │
                  Family.create_model(variant_config)
                              │
                              ▼
                       HookedModel  ◄── canonical interface
                       (concrete subclass)
                              │
                  .run_with_cache(probe)
                              │
                              ▼
              (logits, ActivationCache[canonical_name])
                              │
                              ▼
                         Analyzer  ◄── architecture-agnostic;
                                       reads cache[canonical_name]
                                       and model.get_weight(canonical_name)
```

Above the line: experiment knowledge (training, variants, families, tasks).
Below the line: canonical hook names + tensors. Nothing more.

The base class enforces this structurally rather than by convention. An analyzer that imports a concrete subclass type fails review; an analyzer that calls `cache["raw_TL_string"]` fails review. The grep test (REQ_114) makes this load-bearing.

### Why `HookedModel` is the right abstraction (not a `Protocol`)

`HookedModel` extends `nn.Module`. Subclasses are real PyTorch models — they have parameters, gradients, optimizer-friendly state dicts, and can be checkpointed via the existing safetensors path. A `Protocol` would require all of this to be re-asserted by every subclass and would not give us `nn.Module`'s machinery for free. The user's confirmation that miscope owns every model that runs on the platform makes ABC the right default.

### Quarantine principle

The `HookedModel` base class, `ActivationCache`, and `HookPoint` (if not aliased) contain **zero TL imports**. The grep test is the structural guarantee:

```
grep -rn "transformer_lens" src/miscope/architectures/hooked_model.py
grep -rn "transformer_lens" src/miscope/architectures/activation_cache.py
```

returns zero hits after this REQ lands. TL imports are confined to concrete subclasses (`HookedTransformer` under REQ_112). If TL is ever dropped from the dep tree, only that one module changes.

### Sequencing

This REQ is a small, contained change. It ships:
- The abstract base class.
- Canonical hook + weight name vocabularies.
- A stub model + smoke tests proving the surface is well-formed.

It does **not** ship any concrete subclass, family integration, or analyzer migration. Those land under:

- **REQ_112** — `HookedTransformer` for the 1-layer family. End-to-end proof: train + analyze p113/s999/ds598 (canon); regression scaffold byte-identical against current artifacts; **one canary analyzer migrated end-to-end**. This is where the boundary survives contact with reality before the broad migration starts.
- **REQ_113** — `HookedOneHotMLP` and `HookedEmbeddingMLP` for the two MLP families. Retires the per-family `*ActivationBundle` classes.
- **REQ_114** — Bulk analyzer migration onto `HookedModel`. Coordinates with REQ_111 (parallel analyzer build-out) on ordering.

This sequence honors the user direction: *lock the interface, prove it on one architecture end-to-end, then expand.*

### Reference for canonical names

The [TransformerLens 3.0 Model Structure page](https://transformerlensorg.github.io/TransformerLens/content/model_structure.html). When the spec page changes, the canonical names module's docstring should call out the version of the spec adopted and the date.

### Critical-question decoupling

[Roadmap_TL_Critical.md](../Roadmap_TL_Critical.md) lists open questions about TL's from-scratch model support in 4.0. Those questions are *informational* — they affect the priority of TL version migration but do not gate this REQ. The base class is sound either way:

- Maintainers confirm continued from-scratch support → `HookedTransformer` keeps subclassing TL 2.x for the foreseeable future.
- Maintainers say "compat layer until 4.0, then Bridge-only" → `HookedTransformer` pins TL 2.x or vendors the minimal subset; either change is contained to that one module.
- Maintainers say "off-ramp from-scratch users" → `HookedTransformer` swaps to a different backend (raw PyTorch + custom hooks, nnsight, etc.) without touching the base class or analyzers.

---

## Notes

- **This REQ is the rewrite of the original REQ_105 (Architecture Adapter Layer).** The original framing — `ArchitectureAdapter` as a wrapper layer — was retired after surfacing as wrong-shaped during initial implementation. The diagnosis and reframe are documented in `notes/infra_discovery/miscope_hooked_model_class_diagram.md` and the accompanying SVG. The adapter pattern is the right shape when you don't own the model; miscope owns every model it trains.
- **Phase 1 work already in place.** Commit b3681ac added `src/miscope/core/architecture.py` with the canonical hook-name vocabulary. That work survives the rewrite — only the module docstring needs to drop `TransformerLensAdapter` references.
- **Variant runtime status preserved.** The class diagram's `VariantConfig` should be read as the persisted-on-disk schema portion of the existing `Variant` runtime. The `Variant` façade with `.at(epoch).view(name)` semantics, intervention discovery, and the view catalog stay unchanged. This REQ does not touch `Variant`.
- **`ActivationBundle` is superseded.** REQ_087's contribution survives as the *idea* of canonical-name access + weight access + explicit support checks. The object goes away. `MLPActivationBundle` and `EmbeddingMLPActivationBundle` are deleted under REQ_113 when the corresponding `HookedModel` subclasses land.
- **Views are out of scope.** The view catalog and `views/` module are universal instruments and remain untouched here. They consume canonical-name reads via the same surface analyzers use; no view-side changes are needed for this REQ.
- **Sequencing relative to publication.** This REQ + REQ_112 + REQ_113 + REQ_114 should land before REQ_103 (PyPI gate). The public API surface includes the `HookedModel` interface and the canonical name modules; researchers building on miscope v1.0 will be writing against `miscope.architectures.HookedModel` and `miscope.core.architecture` / `miscope.core.weights`.
- **Branching.** Continue on `feature/architecture-adapter`. Branch name preserved for git history continuity even though the conceptual framing has changed; rename at merge time if helpful.
