# REQ_112: HookedTransformer Implementation (1-Layer Family + End-to-End Canary)

**Status:** Implemented; validation passed
**Priority:** High — first concrete proof that the `HookedModel` boundary survives contact with a real architecture and a real analyzer end-to-end.
**Branch:** TBD (likely a continuation of `feature/architecture-adapter` after REQ_105 lands).
**Dependencies:**
- REQ_105 (HookedModel base class + canonical vocabulary). REQ_112 cannot start until REQ_105's abstract surface is in place.
- REQ_086 (regression snapshot scaffold). The byte-identical guarantee on canon variants is the load-bearing validation gate for this REQ.
- Blocks REQ_113 (HookedMLP implementations — they share the `HookedModel` base but exercise it without TL involvement; sequencing 112 → 113 lets the TL subclass shake out first).
- Blocks REQ_114 (bulk analyzer migration — once one analyzer is migrated end-to-end here, the rest can proceed in parallel).

**Attribution:** Engineering Claude (under user direction).

---

## Problem Statement

REQ_105 ships the `HookedModel` abstract base. To prove the boundary holds, one concrete subclass must exist — chosen to exercise:

1. The TransformerLens quarantine (since TL is the most invasive existing dep).
2. A real training + analysis cycle on a canon variant.
3. At least one analyzer migrated end-to-end through the new interface.

The 1-layer transformer family is the right canary for all three:

- It is the architecture with the deepest TL coupling. If the quarantine works here, it works everywhere.
- Canon variant `p113/s999/ds598` has byte-identical regression artifacts under REQ_086, giving us a hard validation anchor.
- The existing analyzers were written against this family first, so migrating one is the smallest unit of analyzer-side proof.

**Implementation strategy (confirmed with user):** subclass `transformer_lens.HookedTransformer` internally. miscope's `HookedTransformer` *is* a TL `HookedTransformer` underneath; it adds canonical-name aliasing on top. TL imports are confined to this one module — the load-bearing quarantine — so the rest of the platform sees only the `HookedModel` interface.

The user's framing on the rewrite question:

> "I have never really wanted to be in the business of rewriting TransformerLens, even more so now since they are going in a different direction. That said, I wince at the idea of recreating the open-source standard for managing HookedTransformers. I want to make sure that the miscope API does not surface anything TL-specific, so we're going to need to wrap it/quarantine it until we're properly forced to make another decision."

So: subclass + quarantine, not vendor + own. If TL 4.0 forces our hand, that decision lives entirely inside this one module.

---

## Conditions of Satisfaction

### `miscope.architectures.HookedTransformer`

- [ ] New module `miscope/architectures/hooked_transformer.py` containing `HookedTransformer`, the **only** module in `src/miscope/` allowed to import from `transformer_lens`.
- [ ] `HookedTransformer` extends both `miscope.architectures.HookedModel` and `transformer_lens.HookedTransformer`. Multiple inheritance is acceptable here because the platform-side interface (`HookedModel`) is structurally a refinement of the TL interface; method-resolution conflicts are documented in the module docstring if any arise.
- [ ] **Construction** takes a config dataclass (shape determined during implementation; can wrap or compose `HookedTransformerConfig`) and assembles the underlying TL model. Existing `ModuloAddition1LayerFamily` config knobs (n_heads, d_model, d_mlp, n_ctx, etc.) map cleanly through.
- [ ] **`setup_hooks()`** registers the canonical hook-point names (per `miscope.core.architecture`) on top of TL's existing hook structure. Implementation choice: either a translation table looking up TL's legacy hook strings on read, or aliasing at the `nn.Module` level so canonical names appear directly in TL's hook dict. Default: translation table — explicitness over cleverness.
- [ ] **`hook_names()`** returns the canonical paths this 1-layer transformer publishes (`embed.hook_out`, `pos_embed.hook_out`, `blocks.0.attn.q.hook_out`, `blocks.0.attn.k.hook_out`, `blocks.0.attn.v.hook_out`, `blocks.0.attn.o.hook_out`, `blocks.0.attn.hook_pattern`, `blocks.0.attn.hook_out`, `blocks.0.mlp.hook_pre`, `blocks.0.mlp.hook_out`, `blocks.0.hook_in`, `blocks.0.hook_out`, `unembed.hook_in`, `unembed.hook_out`). Final list determined during implementation against the canonical name module's published vocabulary.
- [ ] **`weight_names()`** returns the canonical weight paths (`embed.W_E`, `pos_embed.W_pos`, `blocks.0.attn.q.W`, `blocks.0.attn.k.W`, `blocks.0.attn.v.W`, `blocks.0.attn.o.W`, `blocks.0.mlp.in.W`, `blocks.0.mlp.out.W`, `unembed.W_U`, plus biases as `.b`).
- [ ] **`get_weight(canonical_name)`** maps canonical weight names to TL's parameter attributes (`self.embed.W_E`, `self.blocks[0].attn.W_Q`, etc.). Mapping table is the single point where TL's spelling is allowed to leak into this module.
- [ ] **`run_with_cache(input, fwd_hooks=None) -> (logits, ActivationCache)`** delegates to TL's `run_with_cache`; the returned cache is wrapped in `miscope.architectures.ActivationCache` with canonical-name keys. The translation from TL's cache keys to canonical names happens in this method's epilogue.
- [ ] **`run_with_hooks(input, fwd_hooks)`** delegates to TL's `run_with_hooks`. `fwd_hooks` may be specified by caller using canonical names; the wrapper translates to TL hooks before delegation.
- [ ] **Calling unknown canonical names** raises `KeyError(name)` with a message listing the model's published names.

### Family integration

- [ ] `BaseModelFamily.create_model` return type widens from `nn.Module` to `miscope.architectures.HookedModel`.
- [ ] `ModuloAddition1LayerFamily.create_model` returns a `miscope.architectures.HookedTransformer`. Existing config plumbing maps through; no caller-side changes beyond the imported type.
- [ ] Existing checkpoints (safetensors files in `results/modulo_addition_1layer/{variant_dir}/checkpoints/`) load into the new `HookedTransformer` without re-saving. State dict key compatibility is required.

### Canary analyzer migration

- [ ] **One analyzer migrated end-to-end** through the new interface as proof. Recommended choice: `repr_geometry` (small, well-understood, exercises both `cache[canonical_name]` reads and `model.get_weight(canonical_name)` reads). Alternative acceptable: `dominant_frequencies` if its REQ_111 successor is already in flight.
- [ ] The migrated analyzer drops its `architecture_support` class attribute in favor of a `required_hooks: list[str]` declaration matching the canonical names it reads.
- [ ] The pipeline gains a thin filter step: an analyzer with `required_hooks` runs only if `all(h in model.hook_names() for h in required_hooks)`. If skipped, the pipeline logs a structured skip record rather than raising.
- [ ] All other analyzers continue to work via the legacy path (their `architecture_support` flag and bundle-style reads). REQ_114 handles the bulk migration.

### TransformerLens quarantine

- [ ] **Quarantine smoke test (load-bearing).** A test under `tests/architectures/test_quarantine.py` runs:
  ```bash
  grep -rn "transformer_lens" src/miscope/
  ```
  and asserts that every hit is inside `src/miscope/architectures/hooked_transformer.py`. Any other location is a regression and fails the test.
- [ ] **Stub-model substitution test.** A `StubHookedTransformer(HookedModel)` (no TL) can be substituted into the canary analyzer's call site and the analyzer code fails predictably at the model boundary (`KeyError` on a canonical name the stub doesn't publish), not deeper in the analyzer. This proves the analyzer is architecture-agnostic in practice, not just in convention.

### Validation

- [ ] **REQ_086 regression scaffold byte-identical** on canon variant `p113/s999/ds598` (and at least one other reference variant — recommend `p109/s485/ds598` per the user's "reference healthy model" framing). Numerical outputs are byte-identical or differ only within documented floating-point tolerance before and after migration of the canary analyzer.
- [ ] **All existing tests pass.** Including all non-canary analyzer tests against the new `HookedTransformer` (they use the legacy path; the path must remain functional during the parallel period).
- [ ] **Training reproduces.** A short training run from scratch (e.g., 1000 epochs on `p113/s999/ds598`) produces checkpoints and per-epoch artifacts identical (within tolerance) to a corresponding run before this REQ. Loss curves overlay.

---

## Constraints

**Must:**

- TL imports occur in exactly one module: `miscope/architectures/hooked_transformer.py`. The grep test enforces this.
- `HookedTransformer` exposes only the `HookedModel` interface to consumers. Concrete subclass methods must not surface TL types in their signatures.
- Subclass approach (not vendor, not rewrite). Per user direction, miscope is not in the business of rewriting TransformerLens.
- Canary analyzer migration must complete in this REQ. Without one analyzer end-to-end, the boundary has not been proven.

**Must avoid:**

- **Migrating more than one analyzer in this REQ.** Bulk migration is REQ_114's scope; doing it here couples the proof to a large refactor and obscures whether the boundary actually held. One canary, end-to-end, with regression validation. That's the deliverable.
- **Pinning TL 2.x in `pyproject.toml` as a hard constraint here.** TL 2.x is the runtime today; if a future TL update breaks `HookedTransformer` internally, that's a contained fix in this one module, not a platform-wide pin decision. The pin question lives under REQ_103 (PyPI hardening).
- **Leaking TL types into the canary analyzer.** If the canary analyzer's migrated form requires a TL-flavored helper, that helper is wrong-shaped. Push back into the canonical interface.
- **Changing the `Variant` runtime.** The view catalog, intervention discovery, and `.at(epoch)` access continue to work unchanged — they consume per-epoch artifacts on disk, not live model state.

**Flexible:**

- **Translation strategy** (table vs `nn.Module` aliasing). Default: static dict, explicitness > cleverness.
- **Canary analyzer choice.** Default: `repr_geometry`. Alternative: `dominant_frequencies` if its REQ_111 successor is already in flight (in which case the new analyzer is born on `HookedModel` and the old one stays untouched per REQ_111's parallel-construction rule).
- **`HookedTransformer` config shape** — wrap, compose, or mirror `HookedTransformerConfig`. Default: a thin miscope-side dataclass that constructs `HookedTransformerConfig` internally; this keeps caller code free of TL-shaped configs.
- **Whether `ActivationCache` exposes TL-style bracket-key sugar** (`cache["post", 0, "mlp"]`). Default: no. Re-evaluate only if canary migration surfaces a real ergonomic gap.

---

## Architecture Notes

### Why subclass rather than wrap

The wrapper-vs-subclass debate is what derailed the original REQ_105. With the `HookedModel` base class in place, the question resolves cleanly:

- A wrapper would be a `HookedTransformer` that *contains* a TL `HookedTransformer` and delegates calls. This has the same identity confusion as the original adapter — is it a model, or is it wrapping a model?
- A subclass *is* both: it satisfies `HookedModel`'s contract and is a real TL `HookedTransformer` underneath. Construction, parameters, state dicts, optimizer integration — all work the same as today.

Multiple inheritance (`class HookedTransformer(HookedModel, TLHookedTransformer)`) is fine here because `HookedModel` is structurally a refinement of `nn.Module` with an added contract. MRO conflicts, if any, get documented and resolved during implementation.

### Translation table

The mapping from canonical names to TL's legacy hook strings is the entire surface where TL's spelling is allowed to leak. A static dict is the explicit form:

```python
_CANONICAL_TO_TL_HOOK = {
    "blocks.0.mlp.hook_pre": "blocks.0.mlp.hook_pre",   # already match
    "blocks.0.mlp.hook_out": "blocks.0.mlp.hook_post",  # TL spells it differently
    "blocks.0.attn.hook_out": "blocks.0.hook_attn_out",
    # ...
}
```

The dict captures the asymmetry. Once written, it's reviewable: anyone reading the module can see exactly which hooks miscope publishes, what TL calls them, and where the differences are. This is the **load-bearing audit surface** for the quarantine.

A `_TL_TO_CANONICAL` reverse dict is also needed for cache-key translation in `run_with_cache`'s epilogue.

### Why one canary, not all analyzers

The user's direction is "build one end-to-end path through the Analyzer as proof." Migrating all analyzers in this REQ would mean:

- Discovery friction across 24+ analyzers, each with its own quirks.
- The quarantine test would fail repeatedly during a long branch's life.
- A regression in the boundary contract would surface late and require multiple analyzer rollbacks.

One canary, validated against `p113/s999/ds598` with REQ_086 byte-identity, says "the boundary works." Then REQ_114 inherits that proof and scales it.

### Coordination with REQ_111

REQ_111 builds new analyzers in parallel against the legacy interface (since old analyzers are the parity anchor). This REQ migrates *one* analyzer onto the `HookedModel` interface. There is no immediate conflict:

- If the canary analyzer is one REQ_111 is rewriting in parallel, the old form stays on the legacy path (per REQ_111's "old code is not modified" rule), and the new REQ_111 form is born on `HookedModel`. Both work.
- If the canary is not in REQ_111's modernization scope, it migrates here cleanly.

The shared decision point ("do new REQ_111 analyzers consume `HookedModel` from day one?") is recorded once REQ_111 ships its first analyzer pair. Default: yes — there is no benefit to building new analyzers against the legacy interface they will then need to migrate off. See REQ_114 for the full ordering discussion.

### After this REQ lands

The coupling-points table (from the original REQ_105) updates as follows after REQ_112 lands for the canary analyzer:

| Coupling site | Pre-REQ_112 | Canary path post-REQ_112 | Other analyzers |
|---|---|---|---|
| Analyzer reads cache | `cache[raw_TL_string]` or `bundle.mlp_post(...)` | `cache[canonical_name]` | unchanged (legacy path) |
| Analyzer reads weights | `model.W_E` or `bundle.weight("W_E")` | `model.get_weight(canonical_name)` | unchanged |
| Family creates model | returns concrete `HookedTransformer` (TL) | returns `miscope.HookedTransformer` | same returned type, same legacy reads work |
| Pipeline runs forward | `model.run_with_cache(probe)` | `model.run_with_cache(probe)` (canonical-keyed cache) | TL's run_with_cache still works on the same model |
| TL imports | scattered | confined to `architectures/hooked_transformer.py` | same |

REQ_113 brings the two MLP families through the same boundary. REQ_114 finishes the analyzer side.

---

## Notes

- **This REQ is the proof-of-life for REQ_105.** If the boundary doesn't hold here, REQ_113 and REQ_114 are blocked. Conversely, if it holds for this canary against canon `p113/s999/ds598`, the rest of the migration is incremental work.
- **Reference variant for byte-identity validation.** `p113/s999/ds598` is the canon model per `MEMORY.md`. Recommend also validating against `p109/s485/ds598` (reference healthy model) to exercise a structurally different but well-understood variant. REQ_086's scaffold supports both.
- **TL version question is decoupled.** This REQ assumes TL 2.x as the runtime (the version in the dep tree today). If the TL 4.x decision arrives during REQ_112's implementation, the scope question is whether the subclass keeps working under the new TL — answer comes from running the test suite. The decision to *migrate* TL versions is REQ_103's territory, not this REQ's.
- **Canary analyzer artifact compatibility.** The canary analyzer must produce per-epoch artifacts at the same path and with the same schema as before (`artifacts/{analyzer_name}/epoch_{NNNNN}.npz`). REQ_086's regression scaffold validates this by file content, not just by analyzer return value.
- **Branching note.** Suggested branch name: `feature/hooked-transformer` if forking from `develop`, or continue on `feature/architecture-adapter` if REQ_105 is merging through that branch. Naming choice driven by git-history continuity rather than process.

---

## Validation outcome (2026-05-06)

**REQ_086 byte-identity passed: 6054/6054 artifacts byte-identical to develop** across both reference variants (`p113/s999/ds598` canon, `p109/s485/ds598` healthy reference). 0 mismatches, 0 missing, 0 extras.

The canary `repr_geometry` is byte-identical despite migrating from `bundle.mlp_post(0, -1)` / `bundle.residual_stream(0, -1, location)` to `cache[canonical_name][:, -1, :]`. The earlier f64-noise concern was an artifact of comparing against a stale checksums file (predating REQ_109's `_pca_var_pc{1,2,3}` schema additions). Against a fresh develop-side baseline produced from the same analyzer set, the canonical-cache slice path produces bit-exact results.

**Process notes for future migrations:**
- The reference checksums must be regenerated whenever the analyzer set or output schema changes — the comparison is sha256-of-file, so any schema drift between baseline and current produces noise in the comparison.
- The user's `notebooks/run_analysis_regression.py` is the canonical refresh script; `scripts/run_regression_check.py` is updated under this REQ to register the same analyzer set so byte-identity comparisons are apples-to-apples.
- Bundle dual-mode (REQ_112 introduction): the legacy `TransformerLensBundle` path produces byte-identical output to the pre-REQ_112 codebase for all non-migrated analyzers. The dual-mode rewire is invisible to legacy consumers.
