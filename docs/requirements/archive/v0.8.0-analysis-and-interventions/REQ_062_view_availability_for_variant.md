# REQ_062: View Availability for Variant

**Status:** Active
**Priority:** High
**Related:** REQ_047 (View Catalog), REQ_054 (DataView Catalog)

## Problem Statement

`ViewCatalog` and `DataViewCatalog` register views universally — they apply to any transformer. But a given variant may not have all analyzers run against it (partially-trained models, new data seeds, work-in-progress variants). When a view's required artifacts don't exist, `load_data` raises `FileNotFoundError` at render time.

Currently there is no way to ask, before rendering, whether a view is available for a specific variant. The dashboard cannot hide or disable unavailable views proactively, and the notebook has no clean way to guard against missing data. As we begin training more variants against different data seeds, partially-analyzed variants will become routine rather than exceptional.

The root cause: artifact dependencies for each view are encoded inside `load_data` closures. The system knows *how* to load, but not *whether* the required data exists. Availability is currently implicit and invisible.

## Conditions of Satisfaction

### ArtifactKind enum

- [ ] An `ArtifactKind` enum exists with members: `EPOCH`, `SUMMARY`, `CROSS_EPOCH`, `CROSS_VARIANT`
- [ ] `CROSS_VARIANT` is defined but not used in this requirement — reserved for future cross-variant view support (see `cross_variant.py` / `load_family_comparison`)
- [ ] `ArtifactKind` is exported from `miscope.views`
- [ ] The implemented kinds map to existing `ArtifactLoader` checks:
  - `EPOCH`: `analyzer in variant.artifacts.get_available_analyzers()` (directory has per-epoch `.npz` files)
  - `SUMMARY`: `variant.artifacts.has_summary(analyzer)` (`summary.npz` exists)
  - `CROSS_EPOCH`: `variant.artifacts.has_cross_epoch(analyzer)` (`cross_epoch.npz` exists)
  - `CROSS_VARIANT`: raises `NotImplementedError` — not checkable against a single variant

### AnalyzerRequirement type

- [ ] An `AnalyzerRequirement` dataclass exists with fields `name: str` and `kind: ArtifactKind`
- [ ] `AnalyzerRequirement` is exported from `miscope.views`

### ViewDefinition

- [ ] `ViewDefinition` gains `required_analyzers: list[AnalyzerRequirement]` (default empty list)
- [ ] Empty `required_analyzers` means always available (metadata-based views like `training.metadata.loss_curves`)
- [ ] `ViewDefinition.is_available_for(variant) -> bool` returns `True` if all requirements are satisfied, `False` if any are missing

### DataViewDefinition

- [ ] `DataViewDefinition` gains `required_analyzers: list[AnalyzerRequirement]` (default empty list)
- [ ] `DataViewDefinition.is_available_for(variant) -> bool` follows the same semantics as for `ViewDefinition`

### ViewCatalog

- [ ] `ViewCatalog.available_names_for(variant) -> list[str]` returns sorted list of view names for which `is_available_for(variant)` is `True`

### DataViewCatalog

- [ ] `DataViewCatalog.available_names_for(variant) -> list[str]` returns sorted list of dataview names for which `is_available_for(variant)` is `True`

### EpochContext

- [ ] `EpochContext.available_views() -> list[str]` delegates to `catalog.available_names_for(variant)`
- [ ] `EpochContext.available_dataviews() -> list[str]` delegates to `dataview_catalog.available_names_for(variant)`

### Registration updates — universal.py

- [ ] `_make_per_epoch` factory auto-populates `required_analyzers = [AnalyzerRequirement(analyzer, ArtifactKind.EPOCH)]`
- [ ] `_make_summary` factory auto-populates `required_analyzers = [AnalyzerRequirement(analyzer, ArtifactKind.SUMMARY)]`
- [ ] All cross-epoch views in `universal.py` declare `required_analyzers` explicitly with kind `ArtifactKind.CROSS_EPOCH`
  - `parameters.pca.*` and `parameters.pca.velocity/component_velocity` → `AnalyzerRequirement("parameter_trajectory", ArtifactKind.CROSS_EPOCH)`
  - `activations.mlp.neuron_freq_trajectory`, `switch_count_distribution`, `commitment_timeline`, `per_band_specialization`, `neuron_frequency_range`, `neuron_frequency_specialization` → `AnalyzerRequirement("neuron_dynamics", ArtifactKind.CROSS_EPOCH)`
  - `geometry.global_centroid_pca` → `AnalyzerRequirement("global_centroid_pca", ArtifactKind.CROSS_EPOCH)`
  - `geometry.dmd_*` → `AnalyzerRequirement("centroid_dmd", ArtifactKind.CROSS_EPOCH)`
  - `parameters.attention.head_alignment_trajectory` → `AnalyzerRequirement("attention_fourier", ArtifactKind.EPOCH)` (uses `load_epochs`, same check as per-epoch)
  - `activations.mlp.dominant_frequencies_over_time` → `AnalyzerRequirement("dominant_frequencies", ArtifactKind.EPOCH)` (uses `load_epochs`, same check as per-epoch)
- [ ] Multi-artifact views declare all requirements:
  - `analysis.band_concentration.rank_alignment` → `[AnalyzerRequirement("neuron_dynamics", ArtifactKind.CROSS_EPOCH), AnalyzerRequirement("dominant_frequencies", ArtifactKind.EPOCH)]`
  - `analysis.band_concentration.trajectory` → `[AnalyzerRequirement("neuron_dynamics", ArtifactKind.CROSS_EPOCH)]`
- [ ] `geometry.centroid_pca_variance` and `geometry.timeseries` → `AnalyzerRequirement("repr_geometry", ArtifactKind.SUMMARY)`
- [ ] `geometry.centroid_pca`, `geometry.centroid_distances`, `geometry.fisher_heatmap` → `AnalyzerRequirement("repr_geometry", ArtifactKind.EPOCH)` (already have `epoch_source_analyzer="repr_geometry"`; `required_analyzers` is separate)
- [ ] `parameters.pca.variance_explained` → `AnalyzerRequirement("parameter_trajectory", ArtifactKind.CROSS_EPOCH)`
- [ ] `training.metadata.loss_curves` → empty `required_analyzers` (always available)

### Registration updates — dataview_universal.py

- [ ] All dataview registrations in `dataview_universal.py` declare `required_analyzers` consistent with their loading pattern
- [ ] Loss curve dataview keeps empty `required_analyzers`

### Tests

- [ ] `ViewDefinition.is_available_for(variant)` returns `True` when all required analyzers are present
- [ ] `ViewDefinition.is_available_for(variant)` returns `False` when any required analyzer is absent
- [ ] `ViewDefinition.is_available_for(variant)` returns `True` for empty `required_analyzers` regardless of what artifacts exist
- [ ] `ViewCatalog.available_names_for(variant)` excludes views whose requirements are not met
- [ ] `EpochContext.available_views()` returns a subset of `catalog.names()`

### Dashboard integration

- [ ] `VariantState` in `dashboard/state.py` is renamed to `VariantServerState` — it carries server-side Dash infrastructure (artifact loader, threading context, `EpochContext`) and is distinct from any `VariantState` concept in `miscope` itself; the name collision is misleading
- [ ] `VariantServerState.available_views` is computed once per variant load in `load_variant()` using `catalog.available_names_for(variant)` rather than `catalog.names()`
- [ ] The global instance is renamed from `variant_state` to `variant_server_state`
- [ ] All dashboard import sites updated to use the new name

## Constraints

**Must:**
- Use existing `ArtifactLoader` check methods — no new IO methods required
- `AnalyzerRequirement` must be a dataclass (not a namedtuple or plain tuple) — it will appear in repr and tests
- `required_analyzers` field must be independent of `epoch_source_analyzer` — they serve different purposes (epoch resolution vs. availability)
- All currently registered views must declare `required_analyzers` (no silently-skipped views)

**Must not:**
- Change `load_data` or `renderer` callable signatures
- Change `epoch_source_analyzer` semantics or usage
- Add new methods to `ArtifactLoader`
- Move or restructure any existing analyzer

**Explicitly out of scope:**
- Dashboard UI wiring to use `available_views` (e.g., disabling/hiding view controls) — separate UI requirement
- `ArtifactKind.CROSS_VARIANT` implementation — reserved enum member only
- Family-scoped availability overrides
- Partial availability (e.g., a view available only for some epochs but not others)

## Context & Assumptions

Four loading patterns appear in the current view registrations, each with a distinct availability check:

| Pattern | ArtifactKind | ArtifactLoader check |
|---|---|---|
| Per-epoch (`_make_per_epoch`) | `EPOCH` | `get_available_analyzers()` |
| Summary (`_make_summary`) | `SUMMARY` | `has_summary()` |
| Cross-epoch (`load_cross_epoch`) | `CROSS_EPOCH` | `has_cross_epoch()` |
| Metadata (`variant.metadata`) | (none — empty list) | always available |
| Cross-variant (`load_family_comparison`) | `CROSS_VARIANT` | reserved — not implemented here |

Stacked-epoch views (`load_epochs`) use the same check as per-epoch views — they require per-epoch `.npz` files to exist.

`required_analyzers` is purely declarative metadata — it does not alter how `load_data` works. A view that passes `is_available_for` can still fail at load time if data is corrupt or partially written. The goal is to handle the routine case (analyzer simply not run yet), not to replace error handling entirely.

## Notes

**`ArtifactKind` and `AnalyzerRequirement` placement:** Define both in `catalog.py` alongside `ViewDefinition`. Import into `dataview_catalog.py`. Export both from `miscope/views/__init__.py`.

**`is_available_for` placement:** Method on `ViewDefinition` / `DataViewDefinition`. The catalog delegates to it. `EpochContext` delegates to the catalog. Each layer does only its part.

**No `AnalyzerRequirement` for `parameters.pca.explained_variance`:** This view reuses `_load_parameter_trajectory` — requires `AnalyzerRequirement("parameter_trajectory", "cross_epoch")`.

**Test approach:** Use a mock `Variant` with a mock `ArtifactLoader` — no filesystem needed. Tests should cover `True`, `False`, and empty-list cases explicitly.
