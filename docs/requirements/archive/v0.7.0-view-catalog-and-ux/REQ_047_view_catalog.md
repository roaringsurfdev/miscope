# REQ_047: View Catalog — Universal Presentation Layer

**Status:** Active
**Priority:** High
**Related:** REQ_037 (Notebook Research API), REQ_017 (Multi-Model Support, future)

## Problem Statement

The `miscope` library has no unified presentation layer between data sources (analysis artifacts, training metadata) and rendering (Plotly figures). When a researcher wants to visualize an analytical view, they must either:

1. Know which analyzer produced the relevant artifact, how to load it (which loader method, which data pattern), and which renderer to call with which arguments, or
2. Use `export_variant_visualization()` from `export.py`, which handles this wiring but only produces files — not inline notebook figures or dashboard widgets.

The consequence is that the dashboard, notebooks, and export scripts each implement their own data→renderer wiring independently. Views drift. Small inconsistencies between the notebook and dashboard versions of "the same" visualization introduce noise that is difficult to distinguish from real signal. Repeatability — a scientific requirement for valid cross-variant comparison — is not enforced at the library level.

Additionally, the current architecture encourages asking "which views does this family own?" when new families are introduced. This is the wrong question. Views are analytical instruments; they apply to any transformer. The library should enforce this.

There is also no library-level concept of a **shared epoch cursor** — the fundamental research operation of pinning a training moment and examining all lenses simultaneously. A single well-chosen epoch allows a researcher to compare the onset of grokking across every available view at once. Currently, each view is called independently, making it easy to inadvertently examine different training moments across views. The epoch is not a parameter to a view; it is the research instrument.

## Conditions of Satisfaction

### Core Interface

- [ ] `variant.at(epoch)` returns an `EpochContext` with the training moment locked in
- [ ] `EpochContext.view(name)` returns a `BoundView` — epoch is fixed, not passed per-view
- [ ] `BoundView.show()` renders the view inline in a Jupyter notebook
- [ ] `BoundView.figure()` returns the raw Plotly `Figure` (for dashboard embedding)
- [ ] `BoundView.export(format, path)` writes a static file (delegates to existing export machinery)
- [ ] `variant.view(name)` remains available as a convenience shortcut, equivalent to `variant.at(epoch=None).view(name)` — uses the first available epoch (0) for per-epoch views
- [ ] Cross-epoch views accessed through `variant.at(epoch)` use the epoch as a highlight cursor (e.g., current training position marked on a trajectory plot)
- [ ] `variant.at(epoch).view("unknown_name")` raises a clear error listing available view names

### View Catalog

- [ ] A `ViewCatalog` registry exists as a first-class concept in `miscope.views`
- [ ] All views currently in `export.py`'s `_VISUALIZATION_REGISTRY` are available through the catalog
- [ ] `ViewCatalog.names()` returns the full sorted list of registered view names
- [ ] Loss curve is registered as a universal view, loading from `metadata.json` rather than analysis artifacts — validates that the catalog handles multiple data source types

### Notebook Validation

- [ ] A notebook demonstrates `variant.at(epoch)` showing at least two views pinned to the same epoch — establishing the shared cursor pattern
- [ ] The notebook demonstrates at least three distinct views spanning different data sources: metadata-based (loss curve), per-epoch artifact (e.g., dominant frequencies), and cross-epoch artifact (e.g., parameter trajectory)
- [ ] The notebook accesses all views through `variant.at(epoch).view()` or `variant.view()` — no direct imports of `ArtifactLoader`, renderer functions, or `export_variant_visualization()`

## Constraints

**Must:**
- `Variant` grows two entry points: `at(epoch)` and `view(name)` — no rendering or data loading logic on `Variant` itself
- `EpochContext` is the primary research interface; `BoundView` is the Presenter — each knows only what it needs: context knows the variant + epoch, bound view knows the context + view definition
- All views registered in this requirement are universal — no family-scoped views

**Must not:**
- Break existing `export_variant_visualization()` behavior
- Move or restructure existing renderers — they are consumed by the catalog, not replaced

**Explicitly deferred to future requirements:**
- Mechanism for families to contribute task-specific views (e.g., accuracy curves against ground truth)
- Mechanism for families to enrich universal views with interpretive context (e.g., prime-based Fourier axis labels)
- Dashboard refactor to consume the catalog (separate requirement, after UX design stabilizes)
- Animated GIF output target on `BoundView`

## Context & Assumptions

The `_VISUALIZATION_REGISTRY` in `export.py` is already doing the right conceptual work — it maps view names to (analyzer, renderer, data pattern). This requirement formalizes and promotes that concept from a private utility dict to a first-class module. The work is closer to relocation, restructuring, and interface addition than a greenfield build.

The loss curve is the most important first case because it uses a different data source than artifact-based views (`metadata.json` vs. `.npz` files). Handling it cleanly validates that the catalog's data loading abstraction is general enough.

The pattern is Separated Presentation. `miscope.views` is the connective tissue between data loading and rendering. Neither callers (notebooks, dashboard) nor renderers know about each other. Both are consumed through the catalog.

See `PROJECT.md` for the full architectural invariant motivating this requirement.

## Notes

**Module structure:**
- `miscope/views/catalog.py` — `ViewDefinition` dataclass, `ViewCatalog` class, `BoundView` class
- `miscope/views/universal.py` — registration of all universal views (migrated from `export.py`)
- `miscope/views/__init__.py` — public exports

**ViewDefinition** should carry a `load_data` callable and a `renderer` callable rather than the current string-typed data pattern. This removes the large if/elif dispatch in `export_variant_visualization()` and makes each view self-contained. Signature for `load_data`: `(variant: Variant, epoch: int | None) -> Any`. Signature for `renderer`: `(data: Any, epoch: int | None, **kwargs) -> go.Figure`.

**Loss curve data source:** `variant.train_losses` and `variant.test_losses` already exist on `Variant` (REQ_037). The loss curve `ViewDefinition.load_data` reads these directly — no `ArtifactLoader` involved. This is the simplest possible view definition and should be implemented first.

**EpochContext:** The object returned by `variant.at(epoch)`. It holds a reference to the variant and the resolved epoch. `EpochContext.view(name)` performs the catalog lookup and returns a `BoundView` with epoch already bound — callers never pass epoch to individual views. `variant.view(name)` is syntactic sugar for `variant.at(epoch=None).view(name)`.

**Epoch resolution:** `EpochContext` resolves a `None` epoch to the first available epoch for the requested view — this resolution happens inside `EpochContext.view()`, after the catalog lookup tells it whether the view is epoch-parameterized. Cross-epoch views ignore the epoch entirely for data loading but receive it as `current_epoch` for highlighting. View definitions remain stateless.

**Cross-epoch views and the cursor:** When a cross-epoch view (e.g., parameter trajectory) is accessed via `variant.at(epoch)`, the epoch becomes the `current_epoch` highlight — the cursor position on the trajectory. This is consistent with how the dashboard slider works and makes `variant.at(epoch)` meaningful for all view types, not just per-epoch snapshots.

**2026-02-21:** Requirement drafted following architectural discussion that established the View Catalog / Separated Presentation pattern as the missing middle tier. The `_VISUALIZATION_REGISTRY` in `export.py` is the validated prototype.
