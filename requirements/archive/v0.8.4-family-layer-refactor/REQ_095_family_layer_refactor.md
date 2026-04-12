# REQ_095: Family Layer Refactor

**Status:** Active
**Priority:** High
**Branch:** feature/req-095-family-layer-refactor
**Attribution:** Engineering Claude

---

## Problem Statement

The family and analysis layers have accumulated several structural issues as the platform
has grown from a single transformer family to three architectures:

1. **`ActivationBundle` + `probe` + `context` as three separate `analyze()` arguments** — many
   analyzers don't use all three and suppress linter warnings with `# noqa: ARG002`. These
   three objects are all bound to the same forward pass at the same checkpoint and belong
   together.

2. **Optimizer and loss function hardcoded in `Variant`** — `Variant.train()` constructs
   `AdamW` directly and owns `_loss_function()`, both of which are architecture-specific
   decisions that belong to the family.

3. **`JsonModelFamily` name encodes the loading mechanism, not the role** — this class is
   always subclassed; it is never used as a standalone family. Its name should reflect that
   it is a base class.

4. **`ModelFamily` protocol exposes two methods that don't belong there** — `architecture`
   is used internally by implementation subclasses but not by any external caller.
   `get_variant_directory_name` has a single external caller (`Variant.name`) and its logic
   is a one-line format string that belongs inlined there.

5. **Two MLP family files use generic names** — `two_layer_mlp.py` / `TwoLayerMLP` and
   `learned_emb_mlp.py` / `LearnedEmbeddingMLP` don't signal the task. All other families
   use task-prefixed names. The classes should follow the same convention.

---

## Conditions of Satisfaction

### 1. File and Class Renames

- [ ] `src/miscope/families/implementations/two_layer_mlp.py` renamed to
  `modulo_addition_2l_mlp.py`
- [ ] `TwoLayerMLP` renamed to `ModuloAddition2LMLP`
- [ ] `TwoLayerMLPFamily` renamed to `ModuloAddition2LMLPFamily`
- [ ] `src/miscope/families/implementations/learned_emb_mlp.py` renamed to
  `modulo_addition_embed_mlp.py`
- [ ] `LearnedEmbMLP` (and any `LearnedEmbeddingMLP` aliases) renamed to
  `ModuloAdditionEmbedMLP`
- [ ] `LearnedEmbMLPFamily` renamed to `ModuloAdditionEmbedMLPFamily`
- [ ] All import sites, registry registrations, and references updated

### 2. `JsonModelFamily` → `BaseModelFamily`

- [ ] `json_family.py` renamed to `base_model_family.py`
- [ ] Class renamed to `BaseModelFamily`
- [ ] `from_json` classmethod retained — it is the canonical construction path
- [ ] `__repr__` updated to reflect new name
- [ ] All import sites updated

### 3. Protocol Cleanup

- [ ] `architecture` property removed from `ModelFamily` protocol in `protocols.py`
  - Rationale: used only internally by subclasses, not by any pipeline or dashboard caller
  - Property stays on `BaseModelFamily` (it is a legitimate JSON config field)
- [ ] `get_variant_directory_name` removed from `ModelFamily` protocol
  - Implementation inlined in `Variant.name`:
    `return self._family.variant_pattern.format(**self._params)`
  - Method removed from `BaseModelFamily`
  - `variant_pattern` stays in the protocol — the registry uses it for discovery

### 4. `ActivationContext` — Replaces Three `analyze()` Arguments

- [ ] `ActivationContext` dataclass (or attrs class) defined in
  `miscope/analysis/protocols.py` with fields:
  - `bundle: ActivationBundle` — activations and weights for this checkpoint
  - `probe: torch.Tensor` — the analysis dataset (full input grid)
  - `analysis_params: dict[str, Any]` — family-provided domain context
    (`params`, `fourier_basis`, `loss_fn`, `labels`, etc.)
- [ ] `Analyzer.analyze()` signature changes from
  `(bundle, probe, context) -> dict` to `(ctx: ActivationContext) -> dict`
- [ ] `ActivationBundle` protocol retained — `ActivationContext` wraps it, does not replace it
- [ ] Pipeline constructs `ActivationContext` in `_run_single_epoch()`:
  ```python
  ctx = ActivationContext(bundle=bundle, probe=probe, analysis_params=context)
  ```
  Families are not responsible for constructing `ActivationContext`.
- [ ] All ~24 analyzers in `analysis/analyzers/` migrated to the new signature
- [ ] All `# noqa: ARG002` suppressions removed from analyzer `analyze()` methods
- [ ] `SecondaryAnalyzer.analyze(artifact, context)` signature **unchanged** — it does not
  receive a bundle or probe, so the three-argument problem does not apply

### 5. Optimizer and Loss Function on `ModelFamily`

- [ ] `create_optimizer(model) -> torch.optim.Optimizer` added to `ModelFamily` protocol
- [ ] `compute_loss(logits, labels) -> torch.Tensor` added to `ModelFamily` protocol
- [ ] `BaseModelFamily` provides a default `create_optimizer` implementation using
  `get_training_config()` values (`lr`, `weight_decay`, `betas`) and `AdamW`
- [ ] `BaseModelFamily` raises `NotImplementedError` for `compute_loss` — shape differences
  between architectures make no universal default safe
- [ ] All three family implementations override `compute_loss`
- [ ] `Variant.train()` delegates both calls to the family:
  - `optimizer = self._family.create_optimizer(model)`
  - `loss = self._family.compute_loss(logits, labels)`
- [ ] `Variant._loss_function()` removed

### Validation

- [ ] All existing tests pass
- [ ] REQ_086 regression check passes on reference transformer variants
- [ ] A training run completes successfully for at least one variant of each family
- [ ] Analysis pipeline runs successfully on at least one variant of each MLP family

---

## Constraints

**Must:**
- Artifact format and field names produced by analyzers must be unchanged
- `ActivationContext` construction must remain in the pipeline — families must not need a
  new method to build it
- `from_json` classmethod must stay on `BaseModelFamily`

**Must not:**
- Change the on-disk storage structure (checkpoint paths, artifact paths)
- Introduce new external dependencies

**Flexible:**
- Whether `ActivationContext` is a `dataclass`, `attrs` class, or plain class
- Exact field ordering in `ActivationContext`
- Whether `BaseModelFamily.create_optimizer` is the default implementation or each
  family overrides it explicitly

---

## Architecture Notes

**Why `ActivationContext` and not just keyword arguments:**
The goal is a single coherent object that carries everything needed for one checkpoint's
analysis pass. Adding keyword args to `analyze()` would reduce the `# noqa` noise but
leave the conceptual fragmentation. A named type makes it possible to type-check, document,
and extend without changing every analyzer signature again.

**`analysis_params` naming:**
The dict previously called `context` in `analyze()` is renamed `analysis_params` on the
context object. "Context" is already used as a class name elsewhere in the system
(`EpochContext`, `ActivationContext`). `analysis_params` signals "parameters needed for
analysis" and is intentionally a dict — it accommodates family-specific precomputed values
beyond the named fields without requiring schema changes.

**`get_variant_directory_name` removal:**
The registry does not call this method — it uses `variant_pattern` directly for regex-based
variant discovery. The only caller was `Variant.name`. Inlining the one-line format string
there removes the method without behavioral change.

**`compute_loss` not on `BaseModelFamily`:**
Transformer logits are `(batch, seq_len, vocab)` requiring `logits[:, -1]` slice.
MLP logits are `(batch, vocab)` — no sequence dimension. A shared default would require
a shape branch that belongs in family code, not base class code.

**`repr_geometry`, `centroid_dmd`, `global_centroid_pca` runtime gap (out of scope):**
These analyzers declare `architecture_support = ["transformer", "mlp"]` but call
`bundle.residual_stream()`, which raises `NotImplementedError` on MLP bundles. They are
listed in the MLP `family.json` files but will fail at runtime. Fixing these requires
either MLP-compatible residual-stream equivalents or removing them from the MLP family
configs. This is tracked as follow-on work.

---

## Notes

- The rename in CoS item 1 should be done before the other changes — it keeps the diff
  readable and makes the subsequent changes easier to review.
- `modulo_addition_2l_mlp` and `modulo_addition_embed_mlp` follow the existing convention
  established by `modulo_addition_1layer` — task prefix, then architecture descriptor.
- The `architecture` property staying on `BaseModelFamily` is not inconsistent — it is a
  legitimate part of the JSON config schema. Removing it from the *protocol* only means
  external callers (pipeline, dashboard) don't depend on it as a contract.
