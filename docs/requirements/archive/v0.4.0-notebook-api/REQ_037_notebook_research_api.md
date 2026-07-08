# REQ_037: Notebook Research API

**Status:** Draft
**Priority:** High (unblocks ad-hoc research workflow)
**Dependencies:** REQ_036 (Application Configuration)
**Last Updated:** 2026-02-13

## Problem Statement

As a researcher, I know which model family and variant I want to work with. I may want to inspect the activation cache in one notebook, explore parameter trajectories in another, and iterate through checkpoints in a third. In each case, I want to get to the variant quickly and then reach whatever I need from there — without knowing file paths, directory conventions, or which modules to import.

The middle tier already has the right abstractions (FamilyRegistry, Variant, ArtifactLoader, renderers). The problem is that using them from a notebook requires knowing how to wire them together: which classes to import, what paths to pass, how to get from a Variant to its artifacts or metadata. This is plumbing, not research.

## Design

### Entry Point

A single top-level function that gets the researcher to a family:

```python
from tdw import load_family

family = load_family("modulo_addition_1layer")
```

`load_family` uses application configuration (REQ_036) for path resolution. No paths required.

### Variant Access by Domain Parameters

From a family, get a variant by the parameters you know:

```python
variant = family.get_variant(prime=113, seed=999)
```

This looks up the variant by domain parameter values, verifies it exists (is trained), and returns the Variant object. Raises a clear error if the variant doesn't exist or isn't trained.

Discovering what's available:

```python
family.list_variants()           # All available variants
family.list_variant_parameters() # Parameter combinations as list of dicts
```

### Variant as Hub

The Variant object is the central access point. From it, the researcher reaches whatever they need for their current notebook:

**Checkpoints and models** (already exists on Variant):
```python
variant.get_available_checkpoints()       # → list[int]
variant.load_model_at_checkpoint(epoch)   # → HookedTransformer
```

**Analysis artifacts** (new convenience):
```python
artifacts = variant.artifacts              # → ArtifactLoader (lazy, no loading)
artifacts.load_epoch("dominant_frequencies", 26400)
artifacts.load_summary("coarseness")
artifacts.get_available_analyzers()
```

**Training metadata** (new convenience):
```python
variant.metadata        # → dict with train_losses, test_losses, etc.
variant.config          # → dict with model config (d_mlp, n_heads, etc.)
variant.train_losses    # → list[float] (shortcut)
variant.test_losses     # → list[float] (shortcut)
```

**Forward pass with cache** (new convenience):
```python
probe = variant.make_probe([[3, 29]])     # Family-aware probe construction
logits, cache = variant.run_with_cache(probe, epoch=26400)
```

Or with the full analysis dataset:
```python
probe = variant.analysis_dataset()        # Full p x p grid
logits, cache = variant.run_with_cache(probe, epoch=26400)
```

**Analysis context** (for custom analysis):
```python
context = variant.analysis_context()      # Family-specific context (e.g., fourier_basis)
```

### What the API Does NOT Do

- **No new analysis capabilities** — this is access, not computation
- **No caching or state management** — each call loads fresh; the researcher manages their own variables
- **No dashboard coupling** — this API is independent of Gradio/Dash
- **No new artifacts or storage** — reads what the pipeline already produces

### Import Design

The top-level `tdw` package provides the entry point. Renderers and library functions remain importable from their existing locations:

```python
from tdw import load_family                              # Entry point
from visualization import render_dominant_frequencies     # Renderers (unchanged)
from analysis.library import compute_neuron_coarseness   # Library (unchanged)
```

The `tdw` module is intentionally thin — it's a front door, not a parallel API.

## Example Workflows

### Workflow 1: Inspect activations at a checkpoint

```python
from tdw import load_family

family = load_family("modulo_addition_1layer")
variant = family.get_variant(prime=113, seed=999)

probe = variant.make_probe([[3, 29]])
logits, cache = variant.run_with_cache(probe, epoch=26400)

# Now work with the cache directly
mlp_out = cache["post", 0, "mlp"][:, -1, :]
```

### Workflow 2: Iterate checkpoints

```python
from tdw import load_family

family = load_family("modulo_addition_1layer")
variant = family.get_variant(prime=113, seed=999)

for epoch in variant.get_available_checkpoints():
    model = variant.load_model_at_checkpoint(epoch)
    # ... custom per-checkpoint analysis
```

### Workflow 3: Render existing artifacts

```python
from tdw import load_family
from visualization import render_dominant_frequencies

family = load_family("modulo_addition_1layer")
variant = family.get_variant(prime=113, seed=999)

epoch_data = variant.artifacts.load_epoch("dominant_frequencies", 26400)
fig = render_dominant_frequencies(epoch_data, 26400)
fig.show()
```

### Workflow 4: Compare variants

```python
from tdw import load_family

family = load_family("modulo_addition_1layer")

for variant in family.list_variants():
    losses = variant.test_losses
    # ... compare across variants
```

## Scope

**This requirement covers:**
1. `tdw` module with `load_family()` entry point
2. `family.get_variant(**params)` — variant lookup by domain parameters
3. `family.list_variants()` and `family.list_variant_parameters()`
4. Variant convenience properties: `artifacts`, `metadata`, `config`, `train_losses`, `test_losses`
5. `variant.run_with_cache(probe, epoch)` — forward pass convenience
6. `variant.make_probe(inputs)` and `variant.analysis_dataset()` — probe construction
7. `variant.analysis_context()` — family-specific analysis context
8. Tests

**This requirement does not cover:**
- New analysis capabilities or analyzers
- Visualization changes (renderers are already notebook-friendly)
- Dashboard integration (dashboard continues using its own path)
- Artifact caching or session state
- Custom probe persistence or management

## Conditions of Satisfaction

### Entry Point
- [ ] `from tdw import load_family` works from a notebook in `notebooks/`
- [ ] `load_family("modulo_addition_1layer")` returns a family object without path arguments
- [ ] Clear error message when family name doesn't exist

### Variant Access
- [ ] `family.get_variant(prime=113, seed=999)` returns a Variant
- [ ] Clear error when variant doesn't exist or isn't trained
- [ ] `family.list_variants()` returns all available variants
- [ ] `family.list_variant_parameters()` returns parameter dicts

### Variant Hub
- [ ] `variant.artifacts` returns an ArtifactLoader (no path knowledge needed)
- [ ] `variant.metadata` returns parsed metadata dict
- [ ] `variant.config` returns model configuration dict
- [ ] `variant.train_losses` and `variant.test_losses` return loss arrays
- [ ] `variant.run_with_cache(probe, epoch)` loads model and runs forward pass
- [ ] `variant.make_probe(inputs)` constructs a family-appropriate probe tensor
- [ ] `variant.analysis_dataset()` returns the full analysis probe
- [ ] `variant.analysis_context()` returns family-specific context dict

### Tests
- [ ] Entry point resolves paths correctly via config
- [ ] Variant lookup by parameters matches expected variant
- [ ] Variant lookup with invalid parameters raises informative error
- [ ] Metadata and config properties load correctly
- [ ] run_with_cache returns logits and cache for a valid checkpoint
- [ ] list_variants discovers all trained variants

## Constraints

**Must have:**
- Works from notebooks without path manipulation or sys.path hacking
- Variant lookup by domain parameter names (not variant directory names)
- No breaking changes to existing Variant, FamilyRegistry, or ArtifactLoader APIs

**Must avoid:**
- Duplicating functionality that already exists (thin wrappers only)
- Adding state or caching (researcher manages their own variables)
- Coupling to dashboard framework
- Requiring new dependencies

**Flexible:**
- Whether `tdw` is a module, package, or just a top-level `__init__.py`
- Whether convenience methods live on Variant directly or on a wrapper
- Exact signature for `make_probe` (depends on what family-specific probe construction looks like)
- Whether `run_with_cache` accepts epoch as argument or requires a pre-loaded model
- Additional convenience methods discovered during implementation

## Decision Log

| Date | Question | Decision | Rationale |
|------|----------|----------|-----------|
| 2026-02-13 | Monolithic session object vs thin entry point? | Thin entry point (`load_family`) + variant as hub | Researcher may want different things in different notebooks; don't impose a session model |
| 2026-02-13 | New wrapper vs extend existing Variant? | TBD during implementation | Either approach works; prefer extending Variant if it doesn't bloat the class |

## Notes

**Relationship to dashboard:** The dashboard currently wires FamilyRegistry, ArtifactLoader, and renderers together in `app.py` and `state.py`. Once this API exists, the dashboard could optionally be simplified to use it — but that's not in scope. The APIs coexist.

**Relationship to app config (REQ_036):** The `load_family()` function depends on knowing where `model_families/` and `results/` live. REQ_036 provides this. Without it, `load_family` would need path arguments, defeating the purpose.

**Naming: `tdw`**: Short for Training Dynamics Workbench. Keeps the import concise. Open to alternatives if this conflicts or feels wrong.
