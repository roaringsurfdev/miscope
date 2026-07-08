# REQ_087: ActivationBundle Abstraction

**Status:** Active
**Priority:** High
**Branch:** feature/req-087-activation-bundle-abstraction
**Attribution:** Engineering Claude

---

## Problem Statement

The analysis pipeline is currently coupled to TransformerLens (`HookedTransformer` and
`ActivationCache`) at the protocol boundary. The `Analyzer.analyze()` signature in
`protocols.py` takes these as concrete types, and `analysis/library/activations.py` calls
into TL-specific string key lookups (`cache["post", layer, "mlp"]`) and model attribute
access (`model.embed.W_E`).

This coupling makes it impossible to run the existing analyzers against non-transformer
architectures (e.g., a 2L MLP) without forking the analyzer code. The fix is to introduce
a thin `ActivationBundle` protocol that abstracts the activation extraction interface,
implemented by the TL family as a wrapper and later by the MLP family via plain PyTorch hooks.

The regression scaffold (REQ_086) must pass before this work begins.

---

## Conditions of Satisfaction

### Protocol Change
- [ ] `ActivationBundle` protocol defined in `miscope/analysis/protocols.py`:
  - `mlp_post(layer, position) -> torch.Tensor` — post-activation MLP neurons
  - `residual_stream(layer, position, location) -> torch.Tensor` — resid_pre/post/attn_out
  - `attention_pattern(layer) -> torch.Tensor` — raises `NotImplementedError` for non-transformer architectures
  - `weight(name) -> torch.Tensor` — named weight matrices (W_in, W_out, W_E, W_pos, W_Q, W_K, W_V, W_O, W_U)
  - `logits(position) -> torch.Tensor`
- [ ] `Analyzer.analyze()` signature changes from `(model: HookedTransformer, probe, cache: ActivationCache, context)` to `(bundle: ActivationBundle, probe, context)`
- [ ] `TransformerLensBundle` concrete class in `miscope/families/` (or `analysis/`) wraps the existing `(HookedTransformer, ActivationCache)` pair and implements the protocol

### Analyzer Migration
- [ ] All analyzers in `analysis/analyzers/` updated to use `bundle.*` calls instead of direct
  `cache[...]` and `model.*` access
- [ ] `analysis/library/activations.py` updated: functions that took `(cache, model)` now take
  `bundle`, or are replaced by direct bundle calls in the analyzers
- [ ] Transformer-only analyzers (`attention_freq`, `attention_fourier`, `attention_patterns`)
  still work correctly via the bundle — they call `bundle.attention_pattern(layer)` which is
  implemented by `TransformerLensBundle`

### Pipeline Wiring
- [ ] `AnalysisPipeline` constructs a `TransformerLensBundle` from `(model, cache)` before
  calling each analyzer — no analyzer receives raw TL objects
- [ ] `ModelFamily.prepare_analysis_context()` unchanged — context dict is still family-provided,
  bundle is pipeline-constructed

### Validation
- [ ] REQ_086 regression check passes on all four reference variants with the refactored code
- [ ] All existing tests pass

---

## Constraints

**Must:**
- The `TransformerLensBundle` must be a zero-cost wrapper — no copies of tensors, no
  re-computation. It delegates directly to the underlying TL objects.
- Analyzer behavior must be numerically identical to the pre-refactor code (verified by REQ_086)
- Transformer-only analyzers must not require changes to support MLP architectures —
  they simply raise `NotImplementedError` through the bundle, and the MLP family excludes
  them from its analyzer list

**Must not:**
- Change the artifact format or field names produced by any analyzer
- Introduce any new dependencies

**Flexible:**
- Whether `ActivationBundle` is a `Protocol` (structural subtyping) or an ABC
- Exact location of `TransformerLensBundle` class (families/ or analysis/)
- Whether `analysis/library/activations.py` is refactored in-place or its functions are
  deprecated in favor of bundle calls

---

## Architecture Notes

**The two coupling points:**

1. `Analyzer.analyze(model: HookedTransformer, probe, cache: ActivationCache, context)` —
   change to `analyze(bundle: ActivationBundle, probe, context)`

2. `analysis/library/activations.py` functions — each calls TL-specific APIs.
   `extract_mlp_activations(cache, layer, position)` becomes `bundle.mlp_post(layer, position)`.
   `get_embedding_weights(model)` becomes `bundle.weight("W_E")`.

**Cross-epoch and secondary analyzers are unaffected** — they consume `.npz` artifacts, not
models or caches.

**Weight name convention for `bundle.weight(name)`:**
Use the same names as `parameter_snapshot`: `W_E`, `W_pos`, `W_Q`, `W_K`, `W_V`, `W_O`,
`W_in`, `W_out`, `W_U`. The bundle implementation looks these up from TL's attribute tree.

---

## Notes

- The `attention_pattern()` method raising `NotImplementedError` for MLPs is intentional —
  it makes the failure explicit rather than silent. Analyzers that call it will crash loudly
  if accidentally run against an MLP family, which is preferable to silently returning zeros.
- The bundle pattern is the middle path: analyzers don't need to know about architectures,
  and architectures don't need to know about analyzers. The bundle is the seam.
- If `run_with_cache` needs to change (e.g., MLP families capture activations differently),
  that stays inside the family — the pipeline still calls `family.run_forward_pass(model, probe)`
  or equivalent and receives a bundle back. This may require a small addition to the
  `ModelFamily` protocol: a `run_forward_pass()` method that returns `ActivationBundle`.
