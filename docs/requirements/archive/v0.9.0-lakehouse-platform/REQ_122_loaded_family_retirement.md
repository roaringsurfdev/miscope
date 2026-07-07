# REQ_122: Retire LoadedFamily — Family Owns Its Variant Lookup

**Status:** Completed; merged to develop 2026-05-26.
**Priority:** Medium — the wrapper is vestigial and the `.family` unwrap escape hatch broadcasts it, but nothing is broken. The value of this REQ is removing a layer that future readers will keep asking about. Step 3 of 5 in the ETL exploration's forward plan.
**Branch:** `feature/req-122-loaded-family-retirement`
**Supersedes:** None.
**Dependencies:** None blocking. Independent of REQ_119 / REQ_120 / REQ_121 — the analyzer protocol work and the family/variant API live on different axes.
**Coordinates with:**
- REQ_123 (Unified Data Root) — REQ_123 *hard depends* on this REQ. The path-layout migration in REQ_123 is much cleaner once the family object owns its data root; doing it the other way around means rewriting the family API twice.
- REQ_111 (parallel analyzer build-out) — no overlap. REQ_111 touches analyzer code; this REQ touches family/variant code. They can land independently.
**Downstream consumers:**
- *REQ_123 — Unified Data Root.* The Family object becomes the natural place to anchor the `data/{family}/` root resolution. Once Family owns variant lookup, swapping the on-disk layout becomes one constructor change instead of touching every `(model_families_dir, results_dir)` pair.
- *Future cross-family / cross-variant work.* If a future REQ introduces cross-family analyzers, the family object is the symmetric counterpart to the variant object — both own their own lookup.
**Attribution:** Engineering Claude (under user direction). Outcome of the design dialogue captured in `feature/generic_analyzer` [BRANCH_NOTES.md](../../../../BRANCH_NOTES.md) and the follow-up dialogue documented in [docs/notes/thoughts.md](../../notes/thoughts.md) (if logged there).

---

## Problem Statement

[`LoadedFamily`](../../../packages/miscope/src/miscope/loaded_family.py) is a wrapper around `(ModelFamily, FamilyRegistry)`. Its sole job is letting notebooks call `family.get_variant(prime=...)` without having to construct a `FamilyRegistry` first. It exposes a `.family` property to hand back the unwrapped `ModelFamily` — which gives away that the wrapper is vestigial. The escape hatch is a smell.

[`FamilyRegistry`](../../../packages/miscope/src/miscope/families/registry.py) does two distinguishable jobs:

1. **Family discovery.** Scan `model_families/` for `family.json` files; instantiate the right class via `_FAMILY_IMPLEMENTATIONS` mapping.
2. **Variant discovery.** Given a family, scan `results/{family}/` for variant directories matching `family.variant_pattern`; construct `Variant` instances.

The second job is what `LoadedFamily.get_variant()` proxies. But variant discovery is properly a *property of the family* — the family knows its own `variant_pattern`, its own results location, its own implementation. The registry holding it is an accident of how the code was structured, not a structural requirement.

The result: ~20 caller files (`scripts/*.py`, `apps/dashboard/`, `apps/research/sketches/`, tests, cross-variant renderers) import `load_family` and consume the wrapper. A few sites have already started reaching past it (`scripts/fill_checkpoints.py:51` does `loaded_family.family` to get the unwrapped instance). The pattern is broadcasting that the layer doesn't earn its keep.

### What this REQ does

Retires `LoadedFamily` entirely. The family object absorbs variant lookup methods. `FamilyRegistry` collapses to a thin filesystem helper used internally by `load_family`. `load_family(name)` returns a `ModelFamily` directly. ~20 call sites updated mechanically in the same PR — no compatibility shim.

This is a **mechanical API refactor**. No analyzer protocol changes. No on-disk layout changes. No new behavior — the same actions are taken against the same data, just through a cleaner object.

The exploratory `get_model_family()` and the family-config stubs left over in [`packages/miscope/src/miscope/__init__.py`](../../../packages/miscope/src/miscope/__init__.py#L90) and `results/modulo_addition_1layer/{family.json, family_config.json, model_config.json}` are deleted as part of this REQ — their value was the reconnaissance ("family.json belongs *with* the variants"), which is preserved as input to REQ_123 not as code.

---

## Conditions of Satisfaction

### `ModelFamily` grows variant-lookup methods

- [ ] **`ModelFamily.get_variant(**params) -> Variant`** — looks up a variant by domain parameter values. Raises `ValueError` with the available-variant list if the variant doesn't exist or is untrained. Absorbs the current `LoadedFamily.get_variant` implementation verbatim.
- [ ] **`ModelFamily.variants -> list[Variant]`** — property returning all discovered variants for this family. Replaces `LoadedFamily.list_variants()`. Property rather than method because there are no arguments and notebook-style access reads more naturally (`family.variants[0]`).
- [ ] **`ModelFamily.variant_parameters -> list[dict[str, Any]]`** — property returning parameter combinations for all variants. Replaces `LoadedFamily.list_variant_parameters()`.
- [ ] **`ModelFamily.create_intervention_variant(parent_params: dict, intervention_config: dict) -> Variant`** — looks up the parent variant and delegates to `parent.create_intervention_variant(intervention_config)`. Absorbs `LoadedFamily.create_intervention_variant` with the unused `results_dir` parameter dropped.
- [ ] **`ModelFamily` knows its `results_dir`.** Currently passed externally through `Variant.__init__`; under this REQ the family is constructed with its results_dir and propagates it to variants it constructs. Discovery cost: the family becomes filesystem-aware. Benefit: variant lookup is a property of the family, not a property of the wrapper.

### `FamilyRegistry` collapses to a thin filesystem helper

- [ ] **`FamilyRegistry` renamed / shrunk to `FamilyDiscovery`** (name TBD at implementation time; the intent is "this is no longer a registry, it's a directory scanner"). Its responsibilities reduce to:
  - `list_family_dirs(model_families_dir) -> list[Path]` — return the subdirectories of `model_families_dir` containing a `family.json`.
  - `load_family_from_dir(family_dir, results_dir) -> ModelFamily` — read `family.json`, look up the implementation class via the import-time registration, instantiate the family with both directories.
- [ ] **`FamilyRegistry.get_variants()`, `.create_variant()`, `.get_family()` deleted.** These responsibilities move to the family (variant lookup) and to `load_family` (family lookup by name).
- [ ] **`FamilyRegistry.get_family_names()`, `.list_families()` either deleted or replaced** by module-level `list_families()` (which already exists in `miscope/__init__.py` and is kept).

### Family implementations self-register on import

- [ ] **Each implementation file declares its own registration.** Replace the centralized `_register_default_implementations()` block in `families/registry.py` with a module-bottom `register_family_implementation(NAME, ThisClass)` call in each implementation module (`modulo_addition_1layer.py`, `modulo_addition_2l_mlp.py`, `modulo_addition_embed_mlp.py`).
- [ ] **`packages/miscope/src/miscope/families/__init__.py`** imports all implementation modules at the bottom (after the protocol/base imports) so registration happens as a side effect of `import miscope`. Same outcome as today, but the registration call lives next to the class definition — adding a new family no longer requires editing a central function.
- [ ] **`_FAMILY_IMPLEMENTATIONS` map moves** to wherever the new `FamilyDiscovery` helper lives. It's still a dict; just no longer hidden inside the registry module.

### `load_family` returns a `ModelFamily` directly

- [ ] **`miscope/__init__.py::load_family(name) -> ModelFamily`** — return-type change from `LoadedFamily` to `ModelFamily`. Internally: resolve config, locate the family directory, call `FamilyDiscovery.load_family_from_dir(family_dir, results_dir)`, return the family. No wrapper.
- [ ] **`miscope/__init__.py` exports update.** `LoadedFamily` removed from `__all__`. The class itself is deleted.
- [ ] **The broken `get_model_family()` function is deleted.** It depended on a `family_class_type` field that doesn't exist on any tracked `family.json` and had a stale debug `print`. Its motivation ("can the loader work off pure data?") is preserved by family implementations self-registering — the same outcome via a less brittle path.
- [ ] **The duplicate `family.json` + `family_config.json` + `model_config.json` under `results/modulo_addition_1layer/`** are deleted. They were reconnaissance from the parked ETL exploration; the placement insight (family.json belongs *with* the variants) is captured in REQ_123 as the target layout for `data/{family}/family.json`.

### Caller migration (~20 sites)

- [ ] **`scripts/run_analysis.py`** — `family = load_family(FAMILY_NAME)` continues to work; downstream calls (e.g. `family.get_variant(**)`) work unchanged.
- [ ] **`scripts/run_analysis_regression.py`** — same.
- [ ] **`scripts/fill_checkpoints.py`** — `loaded_family.family` unwrap deleted; the variable becomes just `family = load_family(FAMILY_NAME)`.
- [ ] **`scripts/refresh_variant_summaries.py`** — same.
- [ ] **`scripts/migrate_dseed.py`** — same.
- [ ] **`scripts/create_animation.py`** — same.
- [ ] **`apps/research/sketches/`** — `sketch_per_group_kinks.py`, `weight_space_dmd.py`, `neuron_fourier_poc.py`, `mseed_gradient_comparison.py`, `early_gradient_analysis.py`, `sketch_lissajous_fit.py`, `sketch_lissajous_v2_common_basis.py`, `generate_manifold_stats.py` — all update if needed (mostly already use `family.get_variant(...)` which works either way).
- [ ] **`packages/miscope/src/miscope/views/cross_variant.py`, `visualization/renderers/cross_variant.py`, `visualization/renderers/band_concentration.py`** — `LoadedFamily` type hints become `ModelFamily`. Function bodies likely unchanged.
- [ ] **`packages/miscope/tests/test_notebook_api.py`, `test_cross_variant_comparison.py`, `test_second_descent_diagnostics.py`** — `LoadedFamily` type assertions become `ModelFamily`.
- [ ] **`apps/dashboard/src/dashboard/pages/variant_table.py`, `peer_comparison.py`, `initialization_sweep.py`** — these construct `FamilyRegistry` directly. Updated to use either `load_family` or the new `FamilyDiscovery` helper depending on what they're doing (listing all families vs. loading one).
- [ ] **`scripts/run_regression_check.py`** — constructs `FamilyRegistry` directly with two different `results_dir` values (`cfg.results_dir` and `args.output_dir`). The pattern needs a small adaptation: either two `load_family` calls with different results-dir overrides, or expose `FamilyDiscovery.load_family_from_dir(family_dir, results_dir)` directly for this case. Flag at implementation time.

### Tests / validation

- [ ] **API parity test:** for each method on the old `LoadedFamily`, verify the equivalent method on `ModelFamily` returns the same value on a reference family. Implemented as a snapshot test (record current `LoadedFamily.list_variants()` output, etc.) so the parity bar is mechanical.
- [ ] **Import-time side-effect test:** importing `miscope` registers all three family implementations. Verify `FamilyDiscovery` (or whatever the helper's name is) reports the same three family names as today.
- [ ] **No remaining `LoadedFamily` references** anywhere in the repo after the PR (`grep -rn 'LoadedFamily' .` returns nothing). Same for `loaded_family.py`, `loaded_family.` imports.
- [ ] **`scripts/run_analysis.py`, `scripts/run_analysis_regression.py`, dashboard variant_table** smoke-tested end-to-end on a canon variant (p113/s999/ds598) — load family, get variant, run a small analysis or render a view. The behavior surface is preserved.

---

## Constraints

**Must:**
- Hard cutover. No `LoadedFamily` shim, no deprecated alias, no compat property. The 20 call sites are updated in the same PR.
- No on-disk layout changes. `model_families/` and `results/` paths stay where they are. REQ_123 covers that migration; this REQ explicitly does not.
- `ModelFamily` protocol/abstract interface stays minimal. Variant lookup methods become *required* on the protocol — `BaseModelFamily` gets the default implementation, but the methods are part of the family interface, not a separate "convenience" trait.
- `Variant` API unchanged. The change is "where variants are constructed from," not "what a variant is."
- All three current family implementations continue to work without per-implementation changes beyond the self-registration line.

**May:**
- The `FamilyDiscovery` (or successor) helper's exact API is open at implementation time. The constraint is *not* a specific interface; it's that variant discovery is no longer one of its responsibilities.
- Naming of new methods (`variants` vs `list_variants`, etc.) can be tuned during implementation. The constraint is "family owns variant lookup," not specific method names.

**Must Not:**
- Introduce a new abstract class layer (e.g., the `AbstractModelFamily(ABC)` left over on `feature/generic_analyzer` in [base_model_family.py](../../../packages/miscope/src/miscope/families/base_model_family.py)). The existing `ModelFamily` protocol + `BaseModelFamily` concrete base are sufficient.
- Restructure `model_families/` or `results/` directories. That's REQ_123.
- Introduce a `family_class_type` JSON field. The self-registration-on-import approach replaces the broken `get_model_family()` exploration; no JSON-driven dynamic import.
- Modify any analyzer, view, or visualization code beyond updating type hints. The family API change is the *only* substantive change.

---

## Notes

- The escape hatch `LoadedFamily.family` is the loudest signal that the wrapper isn't earning its keep. After this REQ, calling `.family` on a result of `load_family()` is meaningless because the result *is* the family.
- The `feature/generic_analyzer` exploration left two pieces of evidence behind that this REQ tidies up:
  1. The broken `get_model_family()` in `__init__.py:90` (relies on a non-existent JSON field; commented-out type check that would always be true).
  2. The duplicate `family.json` (+ `family_config.json` + `model_config.json`) under `results/modulo_addition_1layer/`. The user added these "while exploring refactoring … to get a feel for where the files might rightfully belong" — the *placement insight* is preserved as input to REQ_123, the *files* are not.
- The third abstract layer `AbstractModelFamily(ABC)` added on `feature/generic_analyzer` in [`base_model_family.py`](../../../packages/miscope/src/miscope/families/base_model_family.py) is **explicitly not adopted.** Three abstraction layers for one concrete is over-architected; we have `ModelFamily` (protocol) and `BaseModelFamily` (concrete base) and that's plenty.
- The 20 caller count is the upper bound — many of them already only use `family.get_variant(...)`, which has the same signature pre- and post-REQ. The mechanical churn is mostly in import statements and type annotations.
- The cross-variant test surface (`test_cross_variant_comparison.py`) is the largest single fixture; coordinate test updates there carefully.

---

## Out of Scope

These belong to subsequent steps and explicitly do not land here:

- **REQ_123 — Unified Data Root.** `model_families/` and `results/` collapse into `data/`; `variant_pattern` shortens (drops family prefix); `MISCOPE_DATA_DIR` env var replaces `MISCOPE_RESULTS_DIR` + `MISCOPE_MODEL_FAMILIES_DIR`. Lands second; hard depends on this REQ.
- **Cross-family / cross-variant API.** Symmetric "list all variants across all families" patterns might emerge later. Not designed here.
- **Family-level convenience APIs beyond variant lookup.** E.g., "load all variants matching a parameter range" or "load only trained variants." Future-feature scope; this REQ stays surgical.
- **`AbstractStore` abstraction.** Not introduced here. Triggered by REQ_100 / REQ_101.
