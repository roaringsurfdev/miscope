# REQ_125: Research Notebook Template & ArtifactLoader Privacy

**Status:** Implemented on `feature/req-125-research-retrofit`; pending merge to `develop`. Bulk notebook retrofit remained deferred per follow-up discussion.
**Priority:** Low–Medium — propagation lever (notebook template + `ArtifactLoader` privacy signal) matters more than mass-retrofitting old notebooks. With the template in place and the privacy signal active, future notebook composition lands on the right patterns without per-notebook touch-ups.
**Branch:** `feature/req-125-research-retrofit` (to be created at start of implementation).
**Dependencies:** REQ_123 (Unified Data Root) and REQ_124 (Per-App Config + Storage Encapsulation) — both landed on `develop`. REQ_124 added the API accessors (`variant.summary`, `family.variant_registry`) this REQ documents in the template.
**Coordinates with:** none active. Adjacent: REQ_103 / REQ_108 — publication hardening cleans up the same surface from a different angle (PyPI metadata, public-API contract).
**Attribution:** Engineering Claude (drafted as a follow-up on 2026-05-26 during REQ_122 review; CoS filled in 2026-05-26 after REQ_124 wrap-up).

---

## Problem Statement

Two narrowly-scoped gaps remain after REQ_124 closed the app-side enforcement pass:

1. **`ArtifactLoader` is publicly importable from `miscope.analysis`**, which signals to future readers (Claude or human) that direct instantiation is a supported pattern. The audit shows research sketches (`apps/research/sketches/neuron_fourier_poc.py`, `sketch_per_group_kinks.py`) do exactly that — `ArtifactLoader(str(variant.variant_dir / "artifacts"))`. The Variant API already provides `variant.artifacts` for the same purpose; the alternative path exists because the loader is public surface. The pipeline and cross-epoch analyzers legitimately use `ArtifactLoader` (they are the storage layer); consumers do not.

2. **`apps/research/notebooks/` has no canonical starting point.** A template was attempted but lost in earlier refactoring. Today, future notebook composition copies patterns from existing notebooks — some of which (`parameter_space_pca.ipynb`, `parameter_trajectory_pca.ipynb`, `regression_intragroup_manifold.ipynb`) demonstrate the right shape, others do not. The fix is to land a curated `_template.ipynb` that demonstrates the post-REQ_123/REQ_124 patterns and to make it the explicit place to copy from.

### What this REQ does

- Drops `ArtifactLoader` from `miscope.analysis`'s public exports (`__all__` and re-export). Internal callers (analyzers, pipeline) continue to import it from the full path `miscope.analysis.artifact_loader`; consumers reach a configured loader through `variant.artifacts`. The class is not renamed, moved, or restructured — only the signal that it's public surface changes.
- Adds `apps/research/notebooks/_template.ipynb` demonstrating `load_family`, `family.get_variant`, `variant.artifacts.load_*`, `variant.summary`, and `family.variant_registry`. Calls out the variants-under-study pattern. No path literals; no `ArtifactLoader(…)` instantiation.
- Audits `scripts/` and library code for any remaining direct `ArtifactLoader(…)` calls outside the storage layer. The `saddle_transport.py` library module instantiates the loader from a Variant — that's mechanical cleanup using `variant.artifacts`.

Bulk retrofit of `apps/research/notebooks/*.ipynb` and `apps/research/sketches/*.py` is **out of scope** — research code is exploratory and not mission-critical; the propagation lever is the template and the privacy signal, not file-by-file edits.

---

## Conditions of Satisfaction

### `ArtifactLoader` privacy signal

- [ ] **`miscope.analysis.__init__.py`** — remove `ArtifactLoader` from the import list and from `__all__`. Internal callers continue to import via the full path (`from miscope.analysis.artifact_loader import ArtifactLoader`).
- [ ] **`artifact_loader.py` module docstring** — leading paragraph states: *"Internal storage primitive. Consumers should access a configured loader via `variant.artifacts` rather than importing this class directly. This class is used inside the analysis pipeline (writers, cross-epoch analyzers) and remains part of the storage layer; only the public-API signal changes."*
- [ ] **`packages/miscope/src/miscope/analysis/README.md`** — example `from miscope.analysis import ArtifactLoader` block updated to use `variant.artifacts`; a callout near the top documents that `ArtifactLoader` is internal.
- [ ] **`Variant.artifacts` docstring** — confirms this is the canonical public surface for loading artifacts; calls out `load_epoch`, `load_summary`, `load_cross_epoch`, `load_epochs` as the read methods.

### `saddle_transport.py` retrofit

- [ ] Two sites at `packages/miscope/src/miscope/analysis/saddle_transport.py:50` and `:392` — `ArtifactLoader(str(variant.variant_dir / "artifacts"))` replaced with `variant.artifacts`. The local `loader` variable retains the same downstream usage.
- [ ] No other behavior changes in this module.

### Notebook template

- [ ] **`apps/research/notebooks/_template.ipynb`** — new file. Underscore-prefixed so it sorts to the top of the directory listing and visually marks itself as the template, not a research artifact.
- [ ] **Template content** (rough outline; final layout chosen during implementation):
  1. **Header markdown cell** — what this template is, the storage encapsulation invariant ("no path literals, no `ArtifactLoader(...)`"), how to copy it.
  2. **Imports** — `miscope`, `miscope.config.get_config`, family helpers; no `Path("results/...")`, no `ArtifactLoader`.
  3. **Variants under study** — curated list of (prime, seed, data_seed) tuples following the `parameter_space_pca.ipynb` / `parameter_trajectory_pca.ipynb` pattern; labels for legend / titles.
  4. **Family + variants resolution** — `family = load_family("modulo_addition_1layer")`; `variants = [family.get_variant(**p) for p in study]`.
  5. **Artifact access** — at least one example of `variant.artifacts.load_epoch(...)`, `load_summary(...)`, `load_cross_epoch(...)` each, with comments on which is appropriate when.
  6. **Variant-level summary / registry** — `variant.summary`, `family.variant_registry`. Note these read on access; assign to a variable for repeated use.
  7. **Plotting placeholder** — a one-cell minimal Plotly figure to anchor visualization patterns. No specific analysis.
  8. **Footer markdown** — "Why this shape: the storage layout is internal to the API. See `docs/PROJECT.md` invariant #3."
- [ ] Template is gitignored only at the *artifact* level (run output / exported figures) — the source `.ipynb` is tracked.

### Scripts / library audit

- [ ] `grep -rn 'ArtifactLoader(' scripts/` returns no results (sweeping confirms REQ_122 already migrated these).
- [ ] `grep -rn 'Path("results\|Path("model_families' scripts/` returns no results.
- [ ] In `packages/miscope/src/`, `ArtifactLoader(` instantiation is allowed only in: the pipeline (`pipeline.py`), the cross-epoch analyzer modules under `analyzers/`, the visualization export utility (`visualization/export.py` — takes `artifacts_dir` directly, no Variant available), the accessor implementations themselves (`families/variant.py`). Sweep confirms no other instantiations remain after the saddle_transport edit.
- [ ] In `apps/dashboard/src/`, `ArtifactLoader(` instantiation count remains 0.

### Tests

- [ ] **`packages/miscope/tests/test_analysis_public_api.py`** (new, ~10 lines) — `assert "ArtifactLoader" not in miscope.analysis.__all__`; `import miscope.analysis as a; assert not hasattr(a, "ArtifactLoader")`. Catches accidental re-exposure.
- [ ] Existing tests still pass — the import-path change to `miscope.analysis.artifact_loader.ArtifactLoader` is internal-only; no test currently imports `ArtifactLoader` from `miscope.analysis` at the public surface. (Verify during implementation; if a test does, route it to the full path.)

---

## Constraints

**Must:**
- `ArtifactLoader` class itself stays where it is (`miscope.analysis.artifact_loader`). Internal callers keep using it. The privacy signal is about *re-export*, not relocation.
- Template `.ipynb` is JSON-formatted; cells are runnable end-to-end against the live `data/` tree (the canon variant exists, so the template can use it without preamble).
- Template demonstrates only patterns codified in the API. No exotic helpers, no plotting-library showcases, no analyzer-specific magic.

**May:**
- Other notebook-shaped templates can appear later (e.g. a cross-variant template) if a strong second pattern emerges. Not required now.
- The audit can surface other small accessor needs (e.g. a `family.variant_summary_paths()` helper). Land them only if the template needs them; don't speculatively widen the API.

**Must Not:**
- Touch `apps/research/notebooks/*.ipynb` (other than adding `_template.ipynb`) or `apps/research/sketches/*.py`. Bulk retrofit is explicitly deferred.
- Rename, move, or restructure `ArtifactLoader`'s class definition or methods.
- Add a runtime check or deprecation warning for `from miscope.analysis import ArtifactLoader`. Privacy is a signal, not a fence; runtime warnings would noisily affect internal callers that legitimately use it via this path.
- Land a CI / lint check enforcing the policy. Useful, but a separate REQ if desired.

---

## Notes

- The CLAUDE-as-first-composer dynamic justifies the template's priority. Future notebook composition (a fresh session, no in-memory context of REQ_123/124 patterns) will copy from existing notebooks. The template provides a curated copy source that demonstrates the right shape from line one.
- `parameter_space_pca.ipynb` and `parameter_trajectory_pca.ipynb` are the cleanest reference points for the variants-under-study pattern. `regression_intragroup_manifold.ipynb` demonstrates clean artifact access via `variant.artifacts` even though it's not polished as a research narrative. The template synthesizes from these.
- The bulk retrofit being deferred is deliberate: old notebooks describe past explorations, and their internal idioms (including occasional path literals) are not load-bearing for current work. They stay frozen the way archived requirement docs stay frozen.
- Library code that genuinely needs `ArtifactLoader` (cross-epoch analyzers, the pipeline itself) is part of the storage layer. The invariant binds *consumers*, not the storage primitive. The privacy signal is correctly scoped: the re-export goes away, the class stays.
- `visualization/export.py` takes `artifacts_dir` as a parameter (no Variant available in its calling contract) — its `ArtifactLoader(...)` instantiations are not in scope for retrofit. If a future REQ wants to unify export.py around Variant-based input, that's a separate scope.

---

## Out of Scope

- Bulk retrofit of `apps/research/notebooks/*.ipynb` (~14 files) and `apps/research/sketches/*.py` (~5 files). Deferred per follow-up discussion; old notebooks are exploratory and not mission-critical.
- Restructuring `ArtifactLoader` internals or its method signatures.
- A second notebook template (cross-variant, intervention-focused, etc.) — only `_template.ipynb` for the standard variants-under-study pattern.
- CI / lint enforcement of "no `ArtifactLoader` outside the storage layer".
- Rewriting any research analyses — this REQ is mechanical privacy-signal + template scaffolding, not a re-derivation of any finding.
- Additional API accessors beyond what already exists (`variant.artifacts`, `variant.summary`, `family.variant_registry`). Add more only if the template surfaces a concrete need.
