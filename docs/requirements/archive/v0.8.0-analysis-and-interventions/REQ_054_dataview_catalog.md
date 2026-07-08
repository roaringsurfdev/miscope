# REQ_054: Data View Catalog — Universal Data View Layer

**Status:** Active
**Priority:** High
**Related:** REQ_047 (View Catalog)

## Problem Statement

The `miscope` library has no unified data view layer between data sources (analysis artifacts, training metadata). When a researcher wants to view raw analysis data, they must either:

1. Know which analyzer produced the relevant artifact, how to load it (which loader method, which data pattern), and which renderer to call with which arguments, or
2. Load raw npz files and review code to determine the structure of each file.

The consequence is that the data is largely obscured and inaccessible to the researcher. It can be hard to determine whether to create a new analyzer or extend one, or whether a simple transform of the existing data would suffice. Without clear understanding of the raw data, it may also lead to confusion about what data is being analyzed, which could lead to inconsistent or incorrect interpretations.

Additionally, when designing new views on the data during exploratory phases, there is an over-reliance on creating new views before or instead of exploring raw data.

## Conditions of Satisfaction

### Core Interface

- [ ] `EpochContext.dataview(name)` returns a `BoundDataView` — epoch is fixed, not passed per-view
- [ ] `BoundDataView.data()` returns a `DataView` container with named fields (DataFrames for tabular/scalar data, ndarrays for tensor data)
- [ ] `BoundDataView.schema` returns a `DataViewSchema` describing all fields without loading data (no IO triggered)
- [ ] `variant.dataview(name)` provides convenience shortcut, equivalent to `variant.at(epoch=None).dataview(name)` — uses the first available artifact epoch for per-epoch views
- [ ] `variant.at(epoch).dataview("unknown_name")` raises a clear error listing available dataview names

### View Catalog

- [ ] A `DataViewCatalog` registry exists as a first-class concept in `miscope.views`
- [ ] `DataViewCatalog.names()` returns the full sorted list of registered dataview names
- [ ] Loss curve is registered as a universal dataview, loading from `metadata.json` rather than analysis artifacts — validates that the catalog handles multiple data source types

### Notebook Validation

- [ ] A notebook demonstrates `variant.at(epoch)` showing at least two dataviews pinned to the same epoch — establishing the shared cursor pattern
- [ ] The notebook demonstrates at least three distinct dataviews spanning different data sources: metadata-based (loss curve), per-epoch artifact (e.g., dominant frequencies), and cross-epoch artifact (e.g., parameter trajectory)
- [ ] The notebook accesses all dataviews through `variant.at(epoch).dataview()` or `variant.dataview()` — no direct imports of `ArtifactLoader`, renderer functions

## Constraints

**Must:**
- `Variant` grows one entry point: `dataview(name)` — no data loading logic on `Variant` itself
- `EpochContext` grows one entry point: `dataview(name)` — parallel to the existing `view(name)` method
- `EpochContext` is the primary research interface; `BoundDataView` is the Data Presenter — each knows only what it needs: context knows the variant + epoch, bound data view knows the context + dataview definition
- All dataviews registered in this requirement are universal — no family-scoped dataviews

**Must not:**
- Move or restructure existing analyzers — they are consumed by the catalog, not replaced

**Explicitly deferred to future requirements:**
- Mechanism for families to contribute task-specific dataviews
- `BoundDataView.export()` — when this becomes a need, it should follow the patterns established for `BoundView.export()`: multiple format targets and canonical default file name construction from variant, dataview name, and epoch

## Context & Assumptions

The creation of the ViewCatalog was a major step forward in the usefulness of the analyzers and renderers. It would be incredibly helpful to extend this usefulness to an ability to surface the raw data created by the analyzers. The output from the DataViewCatalog should provide human-readable schema information about the data itself — the equivalent of named groupings for subsets and field descriptions for a given subset.

The loss curve is the most important first case because it uses a different data source than artifact-based views (`metadata.json` vs. `.npz` files). Handling it cleanly validates that the catalog's data loading abstraction is general enough.

The pattern is Separated Presentation. `miscope.views` is the connective tissue between data loading and raw data analysis. Neither callers (notebooks, dashboard) nor renderers know about each other. Both are consumed through the catalog.

See `PROJECT.md` for the full architectural invariant motivating this requirement.

## Notes

**Module structure:**
- `miscope/views/dataview_catalog.py` — `DataViewDefinition` dataclass, `DataViewCatalog` class, `BoundDataView` class, `DataView` container class, `DataViewSchema` class
- `miscope/views/dataview_universal.py` — registration of all universal dataviews (initially populated by data consumed by renderers)
- `miscope/views/__init__.py` — public exports

**DataViewDefinition** carries a `load_data` callable and a static `schema` declaration. `load_data` signature: `(variant: Variant, epoch: int | None) -> DataView`. The `schema` is a `DataViewSchema` instance — declared at registration time, available before any data is loaded.

**DataView container:** A lightweight class with named fields accessible by attribute. Each field is either a `pd.DataFrame` (for tabular or scalar-summary data) or a `np.ndarray` (for tensor data), depending on what fits the data's natural shape. There is no forced flattening of multi-dimensional data. The `DataView` exposes its schema via `.schema`.

**DataViewSchema:** Describes all fields in a `DataView` — field name, type (DataFrame or ndarray), shape or column descriptions, and a short human-readable description. Accessible from both `DataViewDefinition.schema` (before loading) and `BoundDataView.schema` (convenience delegation). No IO required.

**Loss curve data source:** `variant.train_losses` and `variant.test_losses` already exist on `Variant` (REQ_037). The loss curve `DataViewDefinition.load_data` reads these directly — no `ArtifactLoader` involved. Returns a `DataView` with a single `losses` DataFrame (columns: `epoch`, `train_loss`, `test_loss`). This is the simplest possible dataview definition and should be implemented first.

**EpochContext:** `EpochContext.dataview(name)` performs the dataview catalog lookup and returns a `BoundDataView` with epoch already bound — callers never pass epoch to individual dataviews. `variant.dataview(name)` is syntactic sugar for `variant.at(epoch=None).dataview(name)`.

**Epoch resolution:** `EpochContext` resolves a `None` epoch to the first available artifact epoch for per-epoch dataviews — this resolution happens inside `EpochContext.dataview()`, after the catalog lookup tells it whether the dataview is epoch-parameterized. Cross-epoch dataviews ignore the epoch entirely; the cursor concept is a rendering concern and does not apply to raw data access. DataViewDefinitions remain stateless.
