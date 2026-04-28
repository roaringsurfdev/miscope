# REQ_101: DataFrame Support (Researcher-Facing Tabular Surface)

**Status:** Superseded by REQ_110 (Lakehouse Surface)
**Priority:** Medium
**Branch:** TBD
**Dependencies:** REQ_106 (defines the DataView contract that this REQ implements concretely — versioning, declared dependencies, schema declaration); REQ_097 (frequency primitives), REQ_098 (PCA primitives).
**Attribution:** Engineering Claude

> **Superseded by REQ_110 (Lakehouse Surface).** This REQ's scope (long-format canonical convention, the `to_wide()` consumer pivot, DataView population against new primitives, cross-variant `pd.concat`) was consolidated with REQ_108 (Publication Surface) into a single tabular-surface REQ. The long-format-with-explicit-dimension-columns framing carries forward into REQ_110 verbatim and is the foundation that REQ_110's `group_type` / `operation_type` discriminator design builds on. Implementation tracks under REQ_110.

---

## Problem Statement

Researchers consuming the published analysis (via notebooks pulling from HF)
need a tabular surface — DataFrames are the lingua franca of exploratory
analysis in the Python data ecosystem. They are filterable, joinable across
variants, and pivot naturally to wide-format for plotting.

The existing `DataView` infrastructure (`src/miscope/views/dataview_catalog.py`)
provides the right shape for this but the per-view dataframes are inconsistent:
some are wide (`projection_all_pc1`, `projection_mlp_pc1`, etc.), some are
narrow ndarray bundles, some are mixed. There is no canonical convention for
when wide vs long is appropriate, and no helpers for converting between.

The opportunity: **long-format canonical, with `.pivot_wide(...)` convenience.**
Long format is filterable and groupable across the natural coordinate system
that emerges from the consolidation work — `(variant, epoch, site, frequency)`
or `(variant, epoch, group, component)`. Researchers can pivot for plotting.

---

## Conditions of Satisfaction

### Convention

- [ ] Documented convention: DataView dataframes are long-format with
  explicit dimension columns (`variant`, `epoch`, `site`, `frequency`,
  `pc_index`, etc. as relevant) and value columns (`magnitude`,
  `projection`, `explained_variance_ratio`, etc.).
- [ ] `BoundDataView` (or a sibling helper) provides `to_wide(index, columns,
  values)` that wraps `pandas.pivot`. Researchers pivot for plotting.

### Dataview population

- [ ] `views/dataviews/weight_pca.py` populated against the new PCA library
  primitives (REQ_098). Inventory items become long-format dataframes:
  - Global trajectory PCA (per group: all/embedding/attention/mlp)
  - Per-matrix PCA (one dataframe with `matrix_name` column)
  - Per-frequency-group centroid trajectory PCA
- [ ] `views/dataviews/representation_pca.py` populated similarly for the
  activation-side PCA inventory.
- [ ] Frequency dataviews:
  - `frequency.spectrum.long` — `(variant, epoch, site, frequency, magnitude)`
  - `frequency.learned.long` — `(variant, epoch, site, frequency, threshold,
    method)` for committed sets
- [ ] Each dataview's schema documents which columns are the natural index
  for `to_wide()` (helps researchers).

### Cross-variant joins

- [ ] DataViews for canonical metrics support concatenation across variants
  via `pd.concat([variant.dataview(name).data().df for variant in ...])`.
  The `variant` column makes this trivial; the convention requires it.

### Migration

- [ ] Existing wide-format consumers (e.g., `get_parameter_trajectory_data_frame`
  in notebook prototypes) updated or deprecated. Where wide format is needed
  for an existing renderer, the loader uses `to_wide()` rather than
  re-implementing the pivot.

---

## Constraints

**Must:**
- Long-format canonical for new dataviews. The wide dataframes in current
  notebook prototypes (e.g., `projection_all_pc1` columns) are not the
  long-term shape.
- DataView schemas remain accurate and up-to-date when the underlying data
  changes. The schema is part of the published API.
- Variant column always present. Cross-variant analysis is a primary use case.

**Must avoid:**
- Loading every variant's dataframe into memory pre-emptively. Long format
  can scale poorly without care; consider chunking or lazy loaders if a
  cross-variant dataview becomes slow.
- Hidden conversions inside renderers. Renderers receive long-format and
  pivot explicitly if they need wide.

**Flexible:**
- Whether dimension columns use enum values or string values. (Discovery
  notes: enums for type safety, but enum.value is the storage form.)
- Whether `to_wide()` lives on `BoundDataView` or as a free function in a
  helpers module. Prefer the method for ergonomics.

---

## Architecture Notes

**Why long-format canonical:**
- Filterable: `df[df.site == 'mlp']` is natural.
- Joinable: cross-variant analysis is `pd.concat`, no schema reconciliation.
- Pivot is one call away when wide is needed.
- The natural coordinate system from the consolidation is multi-dimensional
  `(variant, epoch, site, frequency, ...)`; long format makes that explicit.

**Why not wide-format canonical:**
- Wide forms force a single hierarchy (e.g., is `site` the column or the
  row?) — pivoting it later is awkward.
- Cross-variant joins on wide form require schema reconciliation.
- Adding a new dimension (e.g., a new site) is a schema change in wide form,
  but a new row in long form.

**The DataView already supports this.** The infrastructure in `DataViewCatalog`
/ `DataViewSchema` is the right shape; this REQ populates it consistently
against the new library primitives.

---

## Notes

- The existing notebook prototype `get_parameter_trajectory_data_frame` in
  `dataview_analysis.ipynb` uses wide format. That's exploratory and fine;
  the canonical published shape is long.
- `to_wide()` should accept enum values for `columns` so type safety is
  preserved end-to-end.
- This REQ leans on REQ_098 (PCA returns `PCAResult`) — the dataframe
  is a flattening of `PCAResult` over the relevant dimension columns.
