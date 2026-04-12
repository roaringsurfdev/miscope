# v0.8.4 — Family Layer Refactor

**Date:** 2026-04-11
**Branch:** feature/req-095-family-layer-refactor → develop

## Requirements

- **REQ_095** — Family and analysis layer structural cleanup

## Key Changes

### Naming
- `JsonModelFamily` → `BaseModelFamily` (`json_family.py` → `base_model_family.py`)
- `two_layer_mlp.py` / `TwoLayerMLPFamily` → `modulo_addition_2l_mlp.py` / `ModuloAddition2LMLPFamily`
- `learned_emb_mlp.py` / `LearnedEmbMLPFamily` → `modulo_addition_embed_mlp.py` / `ModuloAdditionEmbedMLPFamily`

### Protocol cleanup
- Removed `architecture` property from `ModelFamily` protocol
- Removed `get_variant_directory_name` from `ModelFamily` protocol; inlined as `variant_pattern.format(**params)`

### ActivationContext
- Added `ActivationContext` dataclass to `analysis/protocols.py` — bundles `bundle`, `probe`, `analysis_params`
- All 13 primary analyzers migrated from `analyze(bundle, probe, context)` to `analyze(ctx: ActivationContext)`
- Pipeline constructs `ActivationContext`; families and analyzers are decoupled

### Optimizer and loss ownership
- `create_optimizer(model)` added to `ModelFamily` protocol and `BaseModelFamily` (default AdamW)
- `compute_loss(logits, labels)` added to `ModelFamily` protocol; `BaseModelFamily` raises `NotImplementedError`
- All 3 family implementations override `compute_loss` with architecture-appropriate logic
- `Variant._loss_function()` removed; `Variant.train()` delegates to family

## Files
- `src/miscope/families/base_model_family.py`
- `src/miscope/families/protocols.py`
- `src/miscope/families/variant.py`
- `src/miscope/families/implementations/modulo_addition_{1layer,2l_mlp,embed_mlp}.py`
- `src/miscope/analysis/protocols.py`
- `src/miscope/analysis/pipeline.py`
- `src/miscope/analysis/analyzers/*.py` (13 primary analyzers)
