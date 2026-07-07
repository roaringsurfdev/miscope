# REQ_113: HookedMLP Implementations (One-Hot and Embedding 2L MLP)

**Status:** Implemented
**Priority:** High — second concrete proof of the `HookedModel` boundary; exercises the abstraction with no TL involvement and a structurally different shape (no attention, optionally no embedding).
**Branch:** TBD (suggested: continuation of the architecture-adapter branch sequence after REQ_112).
**Dependencies:**
- REQ_105 (HookedModel base class + canonical vocabulary).
- REQ_112 (HookedTransformer end-to-end proof). Sequencing 112 → 113 lets the TL-bearing subclass shake out first; the MLP subclasses are simpler and benefit from following the established pattern.
- Blocks REQ_114 (bulk analyzer migration — once both MLP families also conform to `HookedModel`, the analyzer migration covers all three architectures uniformly).

**Attribution:** Engineering Claude (under user direction).

---

## Problem Statement

The platform trains three architectures against the modular addition task:

1. **1-layer transformer** (`ModuloAddition1LayerFamily`) — addressed by REQ_112.
2. **One-hot 2L MLP** (`ModuloAddition2LMLPFamily`) — input is one-hot concatenation of `(a, b)`; no learned embedding, no attention, no residual stream.
3. **Learned-embedding 2L MLP** (`ModuloAdditionEmbedMLPFamily`) — two learned embedding matrices (`embed_a`, `embed_b`) summed into a shared `d_embed` representation, then a single hidden layer; no attention, no residual stream.

These two MLP families currently each return their own `nn.Module` subclass and invent parallel abstractions to bridge to the analyzer layer:

- `ModuloAddition2LMLPActivationBundle` ([modulo_addition_2l_mlp.py](src/miscope/families/implementations/modulo_addition_2l_mlp.py)).
- `ModuloAdditionEmbedMLPActivationBundle` ([modulo_addition_embed_mlp.py:81-104](src/miscope/families/implementations/modulo_addition_embed_mlp.py#L81-L104)).

These bundles, plus the `architecture_support` flag on analyzers, are the existing half-solution. With `HookedModel` in place, both bundles can be retired and the model surface unified.

Two distinct subclasses (not one shared with branching) are required:

- The one-hot MLP has no embedding — `embed.hook_out` and `W_E` are not part of its surface.
- The embedding-MLP has two per-input embedding matrices, deliberately exposed as `embed_a` / `embed_b` rather than as a single `W_E`. This **embedding-identity preservation** prevents transformer-class dispatch from misfiring on this MLP architecture (per the original REQ_105's note).

Sharing implementation via a base class or composition helper is acceptable if it falls out cleanly; forcing a single subclass to handle both via runtime branching is not. The user's framing in REQ_105: *"don't fold them prematurely; the embedding presence changes both `supports()` and `get_weight` semantics."*

---

## Conditions of Satisfaction

### `miscope.architectures.HookedOneHotMLP`

- [ ] New module `miscope/architectures/hooked_one_hot_mlp.py`.
- [ ] `HookedOneHotMLP` extends `miscope.architectures.HookedModel`. **No TransformerLens involvement.**
- [ ] **Construction** takes a config dataclass (n_inputs, d_hidden, p — the modulus, etc.) and assembles the underlying MLP — two `nn.Linear` layers with the activation function in between. Existing `ModuloAddition2LMLPFamily` config maps through.
- [ ] **`setup_hooks()`** registers canonical hook points appropriate to a no-embedding, no-attention MLP. At minimum: `blocks.0.mlp.hook_pre`, `blocks.0.mlp.hook_out`, `blocks.0.hook_in`, `blocks.0.hook_out`, `unembed.hook_in`, `unembed.hook_out`. **Does not** publish `embed.hook_out` or any attention hooks.
- [ ] **`hook_names()`** returns the published list. An analyzer with `required_hooks = ["embed.hook_out"]` (or any attention path) is filtered out by the pipeline before it sees this model.
- [ ] **`weight_names()`** returns `blocks.0.mlp.in.W`, `blocks.0.mlp.out.W`, plus biases. **Does not** include `embed.W_E`.
- [ ] **`get_weight(canonical_name)`** maps to `self.mlp.in.weight`, `self.mlp.out.weight`, etc. Asking for `embed.W_E` raises `KeyError(name)` with a message naming the model's published weights.
- [ ] **`run_with_cache(input, fwd_hooks=None) -> (logits, ActivationCache)`** runs the forward pass and captures activations into a canonical-keyed cache.
- [ ] **`run_with_hooks(input, fwd_hooks)`** runs with caller-supplied canonical-keyed hooks.

### `miscope.architectures.HookedEmbeddingMLP`

- [ ] New module `miscope/architectures/hooked_embedding_mlp.py`.
- [ ] `HookedEmbeddingMLP` extends `miscope.architectures.HookedModel`. **No TransformerLens involvement.**
- [ ] **Construction** takes a config dataclass (p — the modulus, d_embed, d_hidden) and assembles `embed_a: nn.Embedding`, `embed_b: nn.Embedding`, the MLP layers, and the unembed projection. Existing `ModuloAdditionEmbedMLPFamily` config maps through.
- [ ] **`setup_hooks()`** registers: `embed.hook_out` (capturing the post-sum representation, i.e. `embed_a(a) + embed_b(b)`), `blocks.0.mlp.hook_pre`, `blocks.0.mlp.hook_out`, `blocks.0.hook_in`, `blocks.0.hook_out`, `unembed.hook_in`, `unembed.hook_out`. Does not publish attention hooks.
- [ ] **`hook_names()`** returns the published list.
- [ ] **`weight_names()`** returns `embed.embed_a`, `embed.embed_b`, `blocks.0.mlp.in.W`, `blocks.0.mlp.out.W`, plus biases. **Does not** include `embed.W_E` — this is the load-bearing distinction.
- [ ] **`get_weight(canonical_name)`** maps:
  - `embed.embed_a` → `self.embed_a.weight`
  - `embed.embed_b` → `self.embed_b.weight`
  - MLP weights as above.
  - `embed.W_E` raises `KeyError`. Asking for it is a category error: this model has two per-input embedding matrices, not one shared embedding, and that distinction matters for downstream analysis (frequency-spectrum analyzers that summed `W_E` across inputs would silently produce wrong output if dispatched here).
- [ ] **Embedding identity preservation.** A test asserts that `embed.W_E` raises `KeyError` for `HookedEmbeddingMLP` instances. This guarantee is structural, not documentary.

### Optional shared base / mixin

- [ ] **Default: no shared base.** The two subclasses sit side by side. Each is small enough that the duplication is not a cost. Both have a single MLP block with nearly identical hook setup, but the input boundary differs enough that a base class would either branch on `has_embedding` or push embedding-shaped concepts into the no-embedding case. Don't force it.
- [ ] **Acceptable alternative if it falls out cleanly during implementation:** a `_register_mlp_hooks(layer: int)` helper function (not a base class) that both subclasses call from their own `setup_hooks()`. This is a function-shaped reuse; it does not change the type hierarchy.

### Family integration

- [ ] `ModuloAddition2LMLPFamily.create_model() -> miscope.architectures.HookedOneHotMLP`.
- [ ] `ModuloAdditionEmbedMLPFamily.create_model() -> miscope.architectures.HookedEmbeddingMLP`.
- [ ] Existing checkpoints (safetensors files for both families) load into the new subclasses without re-saving. State dict key compatibility is required.

### Bundle retirement

- [ ] `ModuloAddition2LMLPActivationBundle` is **deleted**, not deprecated. Activations and weights are now reachable through canonical-name accessors on `HookedOneHotMLP`.
- [ ] `ModuloAdditionEmbedMLPActivationBundle` is **deleted**. Same surface migration via `HookedEmbeddingMLP`.
- [ ] All references to either bundle class in family modules, dashboard pages, notebooks (in active use), and tests are migrated. Notebooks in `notebooks/` that are research artifacts (not in active use) may be left unmigrated — flag them in this REQ's Notes section.
- [ ] **`MLPActivationBundle`** (the parent abstraction) — also removed if no consumers remain after the two concrete bundles are retired. If a consumer survives (e.g., a future MLP architecture that doesn't go through `HookedModel` yet), document the consumer and defer.

### Validation

- [ ] **REQ_086 regression scaffold byte-identical** on at least one canon variant per MLP family. Reference variants: pick one variant from `ModuloAddition2LMLPFamily`'s training set and one from `ModuloAdditionEmbedMLPFamily`'s training set with existing per-epoch artifacts. Outputs match before and after migration within documented tolerance.
- [ ] **All existing tests pass** for both MLP families. Family-specific analyzer tests continue to pass; analyzers either use the legacy path (architecture_support flag — handled by REQ_114) or the new canonical-name path (the canary from REQ_112, if it applies to MLP families).
- [ ] **Training reproduces.** A short training run from scratch on each MLP family produces checkpoints and per-epoch artifacts identical (within tolerance) to a corresponding pre-REQ run.
- [ ] **Embedding-identity test** (load-bearing): with a `HookedEmbeddingMLP` instance, `model.get_weight("embed.W_E")` raises `KeyError`; `model.get_weight("embed.embed_a")` and `model.get_weight("embed.embed_b")` both return tensors of the expected shape.
- [ ] **Quarantine test from REQ_112 still passes.** Specifically: `grep -rn "transformer_lens" src/miscope/architectures/hooked_one_hot_mlp.py src/miscope/architectures/hooked_embedding_mlp.py` returns zero hits. The MLP subclasses must not import from TL.

---

## Constraints

**Must:**

- Two distinct subclasses, not one with branching. The embedding presence is a structural difference, not a runtime configuration.
- Embedding-identity preservation. `HookedEmbeddingMLP` does not publish `embed.W_E`; it publishes `embed.embed_a` and `embed.embed_b`. The structural test enforces this.
- No TransformerLens involvement in either subclass. The MLP subclasses are pure miscope.
- Existing checkpoints continue to load. State dict key compatibility is required; no migration script.

**Must avoid:**

- **Premature consolidation into a single subclass.** The user's REQ_105 framing: *"forcing a single adapter to handle both via branching is not [acceptable]."* Same applies under `HookedModel`.
- **Republishing `embed.W_E` for `HookedEmbeddingMLP` "for compatibility."** The point of the canonical name is its precision. If a downstream analyzer relies on `embed.W_E` and breaks here, it was wrong-shaped — the right answer is the analyzer declares `required_hooks = ["embed.embed_a", "embed.embed_b"]` and computes against the per-input embeddings, not the (nonexistent) shared one.
- **Migrating notebooks under this REQ.** Active-use notebooks (the ones imported by dashboard or fieldnotes) migrate here. Research-artifact notebooks (one-off explorations preserved for archaeology) get listed in Notes for later cleanup, not retrofitted.
- **Touching the analyzer layer beyond what REQ_112's canary established.** Bulk analyzer migration is REQ_114. This REQ retires the bundles and integrates the families; it does not migrate analyzer code.

**Flexible:**

- **Whether to share a `_register_mlp_hooks` helper.** Default: no. Implement each `setup_hooks()` independently and refactor only if a third MLP architecture surfaces under a future REQ.
- **Module placement.** Default: alongside `hooked_transformer.py` in `miscope/architectures/`. Alternative: a `miscope/architectures/mlp/` subdirectory if more MLP variants land later.
- **Whether `MLPActivationBundle` (the parent abstraction) is deleted in this REQ or carried briefly.** Default: delete now if no consumers remain.

---

## Architecture Notes

### Why the two MLP subclasses do *not* share a base class by default

Both subclasses register a `blocks.0.mlp.hook_pre` and `blocks.0.mlp.hook_out`. Both have a single hidden layer and an activation function. Tempting to extract a `HookedMLPBase`.

The asymmetry that pushes back:

| Aspect | `HookedOneHotMLP` | `HookedEmbeddingMLP` |
|---|---|---|
| Input | one-hot concatenation, no learned params before MLP | two `nn.Embedding` matrices, summed |
| Published `embed.hook_out` | no | yes (post-sum representation) |
| Published `embed.*` weights | none | `embed.embed_a`, `embed.embed_b` |
| Forward signature | `forward(one_hot_input)` | `forward(a_indices, b_indices)` |

A base class would either:
- Branch on `has_embedding` inside `setup_hooks()` and `forward()` — introduces conditional logic in the type hierarchy.
- Define an empty `_setup_embed_hooks()` in the base that the embedding subclass overrides — leaks an abstraction (the *concept* of an embedding) into the no-embedding case.

Neither pays for itself with two subclasses. If a third MLP architecture surfaces (e.g., a tied-embedding MLP, or a depth-2 MLP), revisit. Per CLAUDE.md: *"three similar lines is better than a premature abstraction."*

### Why `embed.W_E` must raise `KeyError` on the embedding-MLP

The original `ModuloAdditionEmbedMLPActivationBundle` deliberately exposed embeddings under `embed_a` / `embed_b` keys to prevent transformer-class dispatch from misfiring (REQ_105's note, sourced from [modulo_addition_embed_mlp.py:81-104](src/miscope/families/implementations/modulo_addition_embed_mlp.py#L81-L104)).

The risk: an analyzer writes `model.get_weight("embed.W_E")` expecting the transformer's shared embedding. On the embedding-MLP, that string could plausibly be silently aliased to either `embed_a` or `embed_a + embed_b` or `concat([embed_a, embed_b])` — and any of those would silently produce wrong output. The analyzer would not crash; it would compute against the wrong tensor and write wrong artifacts.

`KeyError` is the right behavior: it converts a category error into a loud failure at the boundary. The fix is on the analyzer side, not on the model.

### How analyzers handle architecture-conditional logic in this regime

Today: `architecture_support = {"transformer", "mlp_one_hot", "mlp_embedding"}` — analyzer declares which architectures it supports.

Under `HookedModel`: `required_hooks = [...]` — analyzer declares which canonical hooks it needs. The pipeline filters before invocation:

```
runnable = [a for a in analyzers if all(h in model.hook_names() for h in a.required_hooks)]
```

For an analyzer that needs to handle "transformer or one-hot MLP, but not embedding-MLP," the inversion is awkward. Two responses:

1. **Most analyzers don't need this.** Almost all analyzers care about a specific subset of canonical names; the architecture filter falls out for free.
2. **For the few that need explicit branching** (e.g., a frequency analyzer that wants to use either `embed.W_E` *or* `embed.embed_a + embed.embed_b`), the right pattern is two `required_hooks` declarations on two analyzer subclasses, registered separately. Family / pipeline picks the runnable one. **Don't introduce a `model.architecture_kind` accessor** — that re-introduces the architecture-aware antipattern by another name.

This pattern is exercised concretely in REQ_114; flagging here so the MLP subclass design doesn't pre-empt it.

### Preserving the embedding sum semantics in `embed.hook_out`

The embedding-MLP computes `repr = embed_a(a) + embed_b(b)` and feeds `repr` to the MLP. The canonical hook `embed.hook_out` should capture `repr` (post-sum), not the individual embeddings. This matches the transformer's `embed.hook_out` semantics — "the representation entering the MLP / attention block" — and lets analyzers that work in representation space treat the two architectures uniformly.

The two embedding matrices are accessible via `get_weight("embed.embed_a")` and `get_weight("embed.embed_b")` for analyzers that need the underlying parameters.

---

## Notes

- **This REQ completes the architecture coverage.** After REQ_112 (transformer) and REQ_113 (both MLPs), all three currently-trained architectures sit behind the `HookedModel` boundary. Family integration is uniform; the `architecture_support` flag and per-family bundles are gone (modulo any analyzers awaiting REQ_114).
- **Reference variants for byte-identity validation.** TBD per family — pick variants with existing per-epoch artifacts that exercise both happy-path training and at least one analyzer of interest. Recommend coordinating with REQ_112's reference set if there's overlap.
- **Out of scope: a fourth architecture.** If a future requirement introduces a new architecture (e.g., a tied-embedding MLP, an attention-only model, an external researcher's architecture), its `HookedModel` subclass lands under that requirement.
- **Notebook migration triage.** During implementation, list any `notebooks/` files that import the retired bundles in this REQ's Notes section. Active-use ones get migrated as part of this REQ; research-artifact ones are flagged for later cleanup (likely under REQ_103's repo-presentation work).
- **Branching note.** Suggested: continue on the architecture-adapter branch sequence after REQ_112 merges, or fork from `develop` at REQ_112's merge commit.

---

## Implementation outcome (2026-05-06)

Landed on `feature/architecture-adapter`. All three trained architectures now sit behind the `HookedModel` boundary.

**Components:**
- `src/miscope/architectures/hooked_one_hot_mlp.py` — `HookedOneHotMLP` + `HookedOneHotMLPConfig`. No TL involvement. Publishes `blocks.0.{hook_in,hook_out,mlp.hook_pre,mlp.hook_out}`, `unembed.{hook_in,hook_out}`. Weights: `blocks.0.mlp.{in,out}.W` only.
- `src/miscope/architectures/hooked_embedding_mlp.py` — `HookedEmbeddingMLP` + `HookedEmbeddingMLPConfig`. No TL. Adds `embed.hook_out` (post-sum representation). Weights: `embed.embed_a`, `embed.embed_b`, `blocks.0.mlp.{in,out}.W`. **`embed.W_E` raises `KeyError` — load-bearing for the embedding-identity invariant.**
- `src/miscope/analysis/mlp_bundle.py` — shared `MLPBundle` adapter exposing the legacy `ActivationBundle` protocol over a `HookedModel` + canonical `ActivationCache`. Used by both family `run_forward_pass` methods so non-migrated analyzers keep working until REQ_114.
- `src/miscope/families/implementations/modulo_addition_2l_mlp.py` and `modulo_addition_embed_mlp.py` rewritten to: (a) `create_model` returns the new `HookedModel` subclass, (b) `run_forward_pass` wraps the canonical cache in `MLPBundle`. Standalone `nn.Module` classes (`ModuloAddition2LMLP`, `ModuloAdditionEmbedMLP`) and per-family `*ActivationBundle` classes are deleted.
- Test files migrated: `tests/test_modulo_addition_2l_mlp.py` and `tests/test_modulo_addition_embed_mlp.py` now exercise `HookedOneHotMLP`/`HookedEmbeddingMLP` and `MLPBundle` directly. Embedding-identity invariant test added (`test_embedding_identity_invariant_W_E_raises`).
- State-dict compatibility preserved: submodule names (`W_in`, `W_out`, `embed_a`, `embed_b`) match the retired classes, so existing checkpoints load unchanged.

**Validation:**
- 1487 tests passing (was 1478 before REQ_113; +9 net new tests on the architecture surface).
- 0 regressions on non-MLP tests.
- Quarantine smoke test still passes — neither MLP subclass imports TransformerLens.
- **Byte-identity regression skipped per user direction:** the families are slated for analytical-surface rework, so faithful reproduction of pre-REQ_113 outputs is not required.

**Process notes:**
- The user pre-pruned `regression/reference_checksums_req112.json` to canon + healthy-reference variants only; the MLP families have no reference artifacts in that file. If/when MLP regression baselines are needed, they get generated separately under the rework work.
- `MLPBundle` is intentionally architecture-agnostic-within-MLP-class (works for both one-hot and embedding-MLP). After REQ_114 retires bundles entirely, this file goes away.
- The `forward(inputs: Tensor)` signature on `HookedEmbeddingMLP` differs from the retired `ModuloAdditionEmbedMLP.forward(a, b)` — the new model takes a single `(batch, 2)` tensor and unbinds internally. This matches the rest of the platform's single-input convention and aligns with `HookedModel.run_with_cache(inputs)`.
