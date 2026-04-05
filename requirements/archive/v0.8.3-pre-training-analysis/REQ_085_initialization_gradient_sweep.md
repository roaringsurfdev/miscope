# REQ_085: Initialization Gradient Sweep

**Status:** Complete
**Priority:** High
**Branch:** feature/req-085-initialization-gradient-sweep
**Menu:** Pre-Training Analysis (new menu item)
**Attribution:** Engineering Claude

---

## Problem Statement

The first gradient step encodes a model's frequency bias at initialization — before any learning has occurred. The (prime, model_seed, data_seed) combination determines which frequencies receive the strongest initial push, and different combinations produce meaningfully different gradient energy profiles across all three sites (embedding, attention, MLP).

Researchers currently have no interactive way to explore initialization space. Understanding the gradient energy landscape at epoch 0 allows intentional seed selection — including deliberately choosing challenging initializations for intervention studies, not just "safe" ones. The platform is about understanding, not finding working models.

The prototype analysis (`early_gradient_analysis.py`, `mseed_gradient_comparison.py`) validated the core computation and surfaced two key findings:
- Model seed controls epoch-0 weight structure; data seed controls which training pairs apply gradient pressure
- The two effects are separable: at epoch 0, all variants sharing a model seed have identical weights, so running multiple data seeds against the same checkpoint isolates the pure data-selection effect

This tool generalizes those experiments into an interactive sweep across hypothetical (prime, model_seed, data_seed) combinations, with no training required.

---

## Conditions of Satisfaction

### Core Functionality
- [x] User can specify a prime p and one or more (model_seed, data_seed) pairs as sweep candidates
- [x] Tool initializes fresh `HookedTransformer` models from scratch using the specified seeds — no existing trained variant required
- [x] Per-frequency gradient energy is computed at all three sites (embedding, attention, MLP) using the existing `_fourier_gradient_by_site()` implementation from `gradient_site.py`
- [x] Results are computed on-demand when the user triggers a sweep; no artifacts written to disk

### Visualization
- [x] **Overlaid profiles**: per-site frequency energy curves for all specified candidates on a shared axis — one panel per site
- [x] **Difference view**: pairwise gradient energy difference between any two candidates (e.g., data_seed=42 − data_seed=598); positive = candidate pushes harder on that frequency
- [x] **Site convergence**: pairwise cosine similarity between the three site spectra for each candidate — shows whether embedding, attention, and MLP agree at initialization
- [x] Key frequencies (from canonical set for the chosen prime, if known) marked on all plots as vertical reference lines

### Dashboard Integration
- [x] Page is accessible under a new **"Pre-Training Analysis"** menu item in the top navigation
- [x] Prime, model_seed(s), and data_seed(s) are configurable from the UI — no code changes required to run a new sweep
- [x] Output is descriptive — no good/bad scoring; the tool surfaces what's happening and lets the researcher interpret it

---

## Constraints

**Must:**
- Reuse `_fourier_gradient_by_site()` from `src/miscope/analysis/analyzers/gradient_site.py` — the computation is already validated and correct
- Work for any valid prime in the family's parameter schema, not hardcoded to specific values
- Support deliberately "bad" seeds as valid inputs — the tool must not filter or warn against any seed choice

**Must not:**
- Create Variant objects or write any training artifacts
- Involve the analysis pipeline
- Score or classify seeds as good/bad

**Flexible:**
- Whether multiple model_seed values sharing a prime reuse one Fourier basis computation (optimization, not CoS)
- Exact layout of the three-panel site view
- Whether session-level result caching is implemented in v1

---

## Architecture Notes

**Computation path (no pipeline):**
```
User input → fresh HookedTransformer(seed=model_seed) → generate_training_dataset(data_seed) →
fourier_gradient_by_site() → numpy arrays → Plotly figures
```

**Key reuse point:** `_fourier_gradient_by_site()` in `gradient_site.py` is already tested and handles all three sites correctly. Extract it (or expose it from a shared library location) rather than re-implementing.

**Fourier basis:** computed once per prime, shared across all candidates with the same prime.

**Model initialization:** `ModuloAddition1LayerFamily.create_model(params)` handles seed-based initialization — use the family to create the model rather than instantiating `HookedTransformer` directly, so the architecture stays consistent.

**Data generation:** `Variant.generate_training_dataset()` won't work here (requires an existing variant). Use `family.generate_training_dataset(params, data_seed=data_seed)` directly.

**No variant creation:** Do not call `FamilyRegistry.create_variant()` — this registers the variant in the results directory. The sweep operates entirely in memory.

---

## Notes

- The epoch-0 data-seed isolation insight (all variants with the same model_seed have identical epoch-0 weights) is load-bearing for interpreting sweep results. Consider surfacing this as a UI note or tooltip.
- The intentional-challenge use case is a first-class use case: a researcher may sweep data seeds specifically to find one that creates initialization conflict on the canonical frequencies, then train that variant for intervention study. The UI should not suggest this is an error.
- Multiple model seeds for the same prime can be loaded and swept in one run — this is the `mseed_gradient_comparison.py` pattern generalized.
- If a trained variant already exists for a given (prime, model_seed, data_seed), the tool could optionally load its epoch-0 checkpoint instead of re-initializing. This is a stretch goal — consistent initialization is fine for v1.
- REQ_086 (Viability Certificate) will follow on the same "Pre-Training Analysis" page. Keep the page structure extensible.
