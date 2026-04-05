# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.3] - 2026-04-05

### Added

- **Viability Certificate** (REQ_086)
  - `src/miscope/analysis/viability_certificate.py`: pure-analytical geometry module; no model weights required
  - Metrics: separation under compression (SVD of centroid cloud), aliasing risk per frequency, gap from ideal set, compression margin
  - Regime classification: `viable` / `aliasing_failure` / `coverage_concern` / `compression_risk`
  - Thresholds calibrated against three known cases (p59/s999 healthy, p59/s485 late grokker, p101/s999 aliasing failure)
  - Pre-computed ideal frequency sets for all corpus (prime, size) pairs in `model_families/modulo_addition_1layer/ideal_frequency_sets.json`; loaded at module init to survive app restarts
  - `variant_summary`: new `effective_dimensionality_crossover_W_E_pr` field — W_E participation ratio at the dimensionality crossover epoch
  - Dashboard page under Pre-Training Analysis: Separation Profile, Aliasing Risk, Ideal Set, and Summary tabs; variant loader populates inputs from registry

- **Initialization Gradient Sweep** (REQ_085)
  - Dashboard page under Pre-Training Analysis: epoch-0 gradient energy per frequency at embedding, attention, and MLP sites
  - Supports multiple model seeds and data seeds; Site Profiles (overlaid), Difference (A−B bar), and Site Convergence (cosine similarity) tabs

- **Frequency Quality vs Accuracy view** (REQ_053 — final gap)
  - `input_trace.frequency_quality_vs_accuracy`: cross-epoch overlay of Fourier frequency quality score against test accuracy on a shared [0,1] y-axis
  - Placed side-by-side with `residue_class_timeline` on the Input Trace page

- **Variant Table page** (REQ_082)
  - Sortable/filterable table of all variants with key metrics; row click selects variant globally via `variant-selector-store`

- **Dashboard plot export** (REQ_079)
  - Global graph registry in `analysis_page.py`; per-graph Export button; batch export panel in left nav sidebar
  - Server-side PNG via Kaleido with canonical filenames; global toast notification (8s auto-dismiss)

### Fixed

- Variant Table: dropdown options not populated on fresh row click (variant would load but left nav showed blank dropdowns)
- Variant context bar: committed frequencies displayed as k+1 due to double-increment; now shows correct 1-indexed k values

### Architecture

- Pre-Training Analysis top-nav menu grouping Initialization Sweep and Viability Certificate
- `scripts/precompute_ideal_sets.py`: one-time exhaustive search covering primes 59–127, sizes 2–5; incremental save for recovery from long runs

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.2] - 2026-04-04

### Added

- **Transient Frequency Analyzer** (REQ_084)
  - `TransientFrequencyAnalyzer`: cross-epoch analyzer tracking frequencies that appear and disappear during training
  - Ragged storage for `peak_members` (variable neuron counts per frequency); `TRANSIENT_CANONICAL_THRESHOLD=0.05` (5%) for detection, `FINAL_CANONICAL_THRESHOLD=0.10` (10%) for stable assignment
  - Views: `transient.committed_counts`, `transient.peak_scatter`, `transient.pc1_cohesion`
  - Dashboard Transient Frequencies page: summary badges, committed counts, scatter + PC1 cohesion side-by-side
  - Key finding: PC1 cohesion asymptote distinguishes recovery (~0.70 plateau) from attrition (~1.0 approach); homeless neurons persist after transient frequencies dissolve
  - 13/30 variants have transient frequencies; data seed 999 dominates high-homeless cases

- **Checkpoint Schedule Manager** (REQ_083)
  - Dashboard page for retraining existing variants with a denser checkpoint schedule
  - Visual range-based schedule builder integrated with the global variant selector
  - Supersedes REQ_015 (which addressed checkpoint selection, not retraining)

- **Artifact Freshness** (REQ_080)
  - `FreshnessReport` and `check_freshness()`: three staleness types — epoch-incomplete, epoch-stale, summary-stale
  - Cross-epoch staleness checked via metadata only (no full artifact load)
  - Incremental analysis is now the default; `force=True` for full rerun
  - `VariantAnalysisSummary` and `build_variant_registry` called automatically at end of analysis thread

- **Variant Summary Consolidation**
  - All per-variant metrics consolidated into `VariantAnalysisSummary`; `variant_registry.json` built from it

### Changed

- **API ergonomics**
  - `variant.dir` — alias property for `variant_dir` (shorter, more intuitive)
  - `variant.view("name", **kwargs)` and `EpochContext.view("name", **kwargs)` — kwargs forwarded to renderer; call-site kwargs take precedence over view-level kwargs

- **CI — Node24 migration**
  - `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true` added ahead of June 2026 deadline

- **Pre-push quality gate**
  - `.git/hooks/pre-push`: ruff check, ruff format, and pyright must pass locally before push

- **Documentation**
  - README fully updated: v0.8.x status, Dash, uv install, src layout, view catalog API
  - DOMAIN_MODEL updated: EpochContext, BoundView, ViewCatalog, InterventionVariant, variant.dir

### References

- Archived requirements: `requirements/archive/v0.8.2-checkpoint-schedule-and-transient-analysis/`
- Milestone summary: `requirements/archive/v0.8.2-checkpoint-schedule-and-transient-analysis/MILESTONE_SUMMARY.md`

## [0.8.1] - 2026-03-26

### Added

- **Per-Input Prediction Trace** (REQ_075)
  - `InputTraceAnalyzer`: per-checkpoint predictions, correctness, confidence, and train/test split for all p² pairs
  - `InputTraceGraduationAnalyzer`: cross-epoch summary — `graduation_epochs` (first stable-correct epoch per pair), `residue_class_accuracy` (per-class accuracy trajectory), `overall_accuracy_by_epoch`
  - Views: `input_trace.accuracy_grid` (per-epoch p×p correctness heatmap), `input_trace.residue_class_timeline` (per-class accuracy over training), `input_trace.graduation_heatmap` (graduation epoch by pair with anti-diagonal structure)
  - Full test suite: unit tests for shapes, graduation stability logic, residue-class counting; integration tests for artifact round-trip and all three views

- **Neuron Group PCA**
  - `NeuronGroupPCAAnalyzer`: within-frequency-group coordination in W_in; tracks top-3 PC variance explained per group across training
  - Dashboard Neuron Groups page: scatter, trajectory, phase, and graduation views
  - Off-by-one fix: neuron group frequency labels now match system convention

- **Phase Diagram Notebook**
  - Research notebook exploring variant classification in frequency × timing space
  - Early prediction cell: validates that early-epoch frequency commitments predict final grokking structure

- **Dashboard**
  - Variant context bar: displays active variant metadata inline with page content

### References

- Archived requirements: `requirements/archive/v0.8.1-prediction-trace-and-neuron-groups/`
- Milestone summary: `requirements/archive/v0.8.1-prediction-trace-and-neuron-groups/MILESTONE_SUMMARY.md`

## [0.8.0] - 2026-03-23

### Added

- **Site Gradient Convergence** (REQ_077)
  - `GradientSiteAnalyzer`: post-hoc per-site per-frequency gradient energy from saved checkpoints
  - Three sites: embedding (direct), attention (Q/K/V RMS projected through W_E[:p]), MLP (W_E[:p] @ grad_W_in)
  - Single variant-level `.npz` artifact: direction-normalized energy, raw magnitude, pairwise cosine similarity
  - Views: `site_gradient_convergence` (similarity + magnitude panels) and `site_gradient_heatmap` (3-panel frequency heatmap)
  - Window boundaries from `variant_summary.json` marked on all plots; NaN similarity values render as gaps

- **Multi-Stream Specialization** (REQ_066)
  - `multi_stream_specialization` view: 4-panel trajectory covering MLP neurons, attention aggregate, embedding dims, effective dimensionality
  - Attention panel uses `attn_active` floor (mean QK^T fraction); `fields=` selective loading in ArtifactLoader

- **Group Trajectory Overlay & Proximity** (REQ_072)
  - `parameters.pca.group_overlay` / `group_overlay_pc2_pc3`: normalized overlay of embedding/attention/MLP parameter-space trajectories on shared PC axes
  - `parameters.pca.proximity` / `proximity_pc2_pc3`: pairwise L2 distance over training with epoch cursor; sign-flip correction for PCA orientation ambiguity
  - Weight trajectory divergence views: per-group and per-weight-matrix PCA, component velocity comparison

- **Intervention Architecture** (REQ_067, REQ_068, REQ_070, REQ_071)
  - `InterventionVariant` subclass nested under parent: `results/{family}/{parent}/interventions/{label}/`
  - `FrequencyGainHook`: frequency-selective hook on `hook_attn_out` using W_E-based frequency directions
  - Intervention Check dashboard page: family→variant→intervention→epoch hierarchy with hook verification
  - `Variant.interventions` discovers sub-variants from filesystem; `create_intervention_variant()` factory

- **Variant Registry & Peer Comparison** (REQ_074, REQ_076, REQ_057, REQ_065)
  - `variant_summary.json` per variant: loss metrics, grokking window boundaries, frequency gains/losses, performance classification
  - `variant_registry.json`: compiled cross-variant aggregate
  - Peer comparison dashboard page with shared-axis cross-variant views

- **Analytical Views** (REQ_052, REQ_056, REQ_058, REQ_060, REQ_062, REQ_063, REQ_064)
  - Fourier frequency quality scoring, frequency specialization sequencing, neuron band concentration health
  - Neuron dynamics runtime threshold, view availability enforcement, Fourier nucleation predictor, data compatibility analyzer

- **Infrastructure** (REQ_054, REQ_061)
  - DataView catalog: `variant.at(epoch).dataview(name)` for tabular/structured data views
  - Data seed as domain parameter: `data_seed` in variant params, multi-seed variants supported throughout

- **Dashboard Navigation** (post-v0.7.0)
  - Site navigation converted to drop-down menus
  - Visualization and Summary pages preserved at direct URLs; analytical views promoted to dedicated pages

### Architecture Notes

- Non-epoch-keyed artifacts (gradient_site) stored as `artifacts/{name}/{name}.npz`; loaded via `ArtifactLoader.load_variant_artifact()`
- InterventionVariant is a sub-variant nested under its parent, not a separate family member
- Sign-flip correction in proximity renderer: `min(dist(A,B), dist(A,-B))` handles PCA orientation ambiguity

### References

- Archived requirements: `requirements/archive/v0.8.0-analysis-and-interventions/`
- Milestone summary: `requirements/archive/v0.8.0-analysis-and-interventions/MILESTONE_SUMMARY.md`

## [0.7.0] - 2026-03-03

### Added

- **View Catalog — Universal Presentation Layer** (REQ_047)
  - `miscope/views/` module: `catalog.py` (registry) + `universal.py` (all registered views)
  - Primary interface: `variant.at(epoch)` → `EpochContext` → `.view(name)` → `BoundView`
  - `BoundView`: `.show()` (notebook inline), `.figure()` (raw Plotly), `.export(format, path)` (file)
  - All views are universal instruments — families are context providers, not view owners
  - Canonical export path derivation on `BoundView`; animation kwargs bug fixed

- **Dashboard Navigation UX** (REQ_046)
  - Encapsulated `variant_selector` component with embedded store
  - `AnalysisPageGraphManager`: shared page logic driving `_VIEW_LIST` dispatch, prevents functionality drift
  - Pattern IDs (`{'view_type': ..., 'index': graph_id}`) enable ALL-pattern callbacks for click-to-navigate
  - Store-centric: all state flows through `variant-selector-store`; cross-page coordination via store
  - Removed unused pages and components; consolidated dashboard from `dashboard_v2` → `dashboard`

- **Summary Lens** (REQ_041)
  - New `/summary` page: dense grid of 12 cross-epoch visualizations answering "what's the shape of this model's training story?"
  - Loss curve, embedding Fourier, neuron specialization, attention specialization, PCA trajectories, component velocity, effective dimensionality
  - Temporal cursor: epoch slider synchronizes vertical indicator line across all time-axis plots without reloading data

- **Neuron Dynamics Page** (REQ_042)
  - New `/neuron-dynamics` page with `neuron_freq_trajectory` heatmap: neuron × epoch colored by dominant frequency
  - Natural order / sorted-by-final-frequency toggle reveals cluster structure
  - Switch count distribution and commitment timeline visualizations
  - Extended `neuron_freq_norm` summary: `switch_count` and `commitment_epoch` fields per neuron

- **Secondary Analysis Tier** (REQ_048)
  - Pipeline extension: secondary analyzers run after per-epoch analysis, consuming existing artifacts
  - `NeuronDynamicsAnalyzer`: computes switch counts and commitment epochs from `neuron_freq_norm` artifacts without re-running the primary pipeline

- **Neuron Fourier Decomposition** (REQ_049)
  - New `NeuronFourierAnalyzer`: per-neuron Fourier decomposition of MLP weights following He et al. (2026)
  - Proof-of-concept notebook: `notebooks/neuron_fourier_poc.py` with margin vs. switching analysis

- **Representational Geometry** (REQ_044)
  - New `RepresentationalGeometryAnalyzer`: first activation-space analysis in the platform
  - Per-epoch class centroid geometry (PCA, Fisher discriminant, centroid distances) at four sites: embedding, post-attention, MLP output, residual stream
  - New `/repr-geometry` dashboard page

- **Fisher Minimum Pair Analysis** (REQ_045)
  - Fisher heatmap view: per-epoch Fisher discriminant across class pairs
  - Extends representational geometry with targeted pairwise separability analysis

- **Global Centroid PCA** (REQ_050)
  - New `GlobalCentroidPCAAnalyzer`: PCA fit jointly across all training epochs — stable coordinate frame for centroid trajectory analysis
  - New `/dimensionality` page: parameter + centroid PCA, trajectory, SV spectrum, and variance timeseries views

- **Centroid DMD** (REQ_051)
  - New `CentroidDMDAnalyzer`: Dynamic Mode Decomposition of class centroid trajectories
  - Decomposes evolution of representational geometry into dynamic modes
  - New `/centroid-dmd` dashboard page with log-scale amplitude support

### Architecture

```
src/miscope/
  views/
    catalog.py                    # ViewDefinition protocol, ViewCatalog registry
    universal.py                  # All registered universal views
  analysis/
    pipeline.py                   # + secondary analysis tier
    analyzers/
      repr_geometry.py            # RepresentationalGeometryAnalyzer
      neuron_dynamics.py          # NeuronDynamicsAnalyzer (secondary tier)
      neuron_fourier.py           # NeuronFourierAnalyzer
      global_centroid_pca.py      # GlobalCentroidPCAAnalyzer
      centroid_dmd.py             # CentroidDMDAnalyzer
dashboard/                        # Consolidated (dashboard_v2 retired)
  components/
    analysis_page.py              # AnalysisPageGraphManager (shared logic)
    variant_selector.py           # Encapsulated variant selector + store
  pages/
    summary.py                    # Summary Lens (/summary)
    neuron_dynamics.py            # Neuron Dynamics (/neuron-dynamics)
    repr_geometry.py              # Representational Geometry (/repr-geometry)
    dimensionality.py             # Dimensionality (/dimensionality)
    centroid_dmd.py               # Centroid DMD (/centroid-dmd)
```

### References

- Archived requirements: `requirements/archive/v0.7.0-view-catalog-and-ux/`
- Milestone summary: `requirements/archive/v0.7.0-view-catalog-and-ux/MILESTONE_SUMMARY.md`
- Deferred: REQ_043 (Fourier Profile Expansion — W_U multi-matrix) → `requirements/future/`

## [0.6.0] - 2026-02-15

### Added

- **Cross-Epoch Analyzers** (REQ_038)
  - `CrossEpochAnalyzer` protocol: new analyzer type that runs after per-epoch analysis, consuming artifacts across all checkpoints
  - Two-phase pipeline: Phase 1 (per-epoch, unchanged) → Phase 2 (cross-epoch, new)
  - `ParameterTrajectoryPCA`: first cross-epoch analyzer — precomputes PCA projections and parameter velocity for all 4 component groups (all, embedding, attention, mlp)
  - Storage: `artifacts/{analyzer_name}/cross_epoch.npz` with group-prefixed keys
  - `ArtifactLoader.load_cross_epoch()` / `has_cross_epoch()` for loading precomputed results
  - Skip-if-exists logic with `force=True` override for recomputation

- **Dash Job Management UI** (REQ_040)
  - Training page: family selection, domain parameters, training config, checkpoint scheduling
  - Analysis Run page: variant selection, analyzer selection, run triggering
  - Site-level navigation via `create_navbar()` with multi-page routing
  - `ServerState` singleton for server-side training/analysis job management

### Changed

- **Project renamed to MIScope** — namespace `tdw` → `miscope` throughout (REQ_039)
- **Source layout restructuring**: core packages moved to `src/miscope/`, standard Python src-layout
- **Gradio dashboard decommissioned** — `dashboard/` removed, shared components migrated to `dashboard_v2/`
- **Package name** updated to `miscope` in pyproject.toml
- **Trajectory renderers** accept precomputed PCA data instead of raw weight snapshots — no computation at render time
- **Dashboard v2** loads trajectory data from cross-epoch artifacts, significantly faster epoch navigation
- **Export module** supports new data patterns (`cross_epoch_pca`, `cross_epoch_velocity`, etc.)
- **Family protocol** extended with `cross_epoch_analyzers` property

### Architecture

```
src/miscope/                          # Installable API (import miscope.*)
  analysis/
    protocols.py                      # + CrossEpochAnalyzer protocol
    pipeline.py                       # Two-phase execution
    analyzers/
      parameter_trajectory_pca.py     # First cross-epoch analyzer
    artifact_loader.py                # + load_cross_epoch, has_cross_epoch
  visualization/renderers/
    parameter_trajectory.py           # Refactored: precomputed data input
dashboard_v2/
  pages/training.py                   # New: Training page
  pages/analysis_run.py               # New: Analysis Run page
  navigation.py                       # New: site-level navigation
  state.py                            # + ServerState
```

### References

- Archived requirements: `requirements/archive/v0.6.0-miscope/`
- Milestone summary: `requirements/archive/v0.6.0-miscope/MILESTONE_SUMMARY.md`

## [0.5.0] - 2026-02-13

### Added

- **Dash Dashboard Migration** (REQ_035)
  - New `dashboard_v2/` built on Dash + Plotly for improved interactivity
  - Sidebar layout: variant selector, epoch slider, neuron index, and all visualization-specific controls in a persistent collapsible left panel
  - Click-to-navigate: click any data point on summary/trajectory plots to jump to that epoch
  - Selective rendering: epoch changes only re-render affected plots, not all 18
  - `Patch()` for epoch marker updates — summary plots update markers without full re-render
  - Neuron click-to-navigate from frequency clusters heatmap
  - All 18 Analysis tab visualizations migrated
  - Dependencies: `dash`, `dash-bootstrap-components`

### Architecture

```
dashboard_v2/           # Dash-based dashboard (new)
  app.py                # Application factory
  layout.py             # Sidebar + main content layout
  callbacks.py          # Per-visualization callbacks with click-to-navigate
  state.py              # DashboardState (variant/epoch/artifact management)
dashboard/              # Gradio dashboard (frozen, still functional)
```

### References

- Archived requirements: `requirements/archive/v0.5.0-dash-migration/`
- Milestone summary: `requirements/archive/v0.5.0-dash-migration/MILESTONE_SUMMARY.md`

## [0.4.0] - 2026-02-13

### Added

- **Application Configuration** (REQ_036)
  - `tdw.config` module with `get_config()` for project path resolution
  - Environment variable overrides: `TDW_RESULTS_DIR`, `TDW_MODEL_FAMILIES_DIR`, `TDW_PROJECT_ROOT`
  - Frozen `AppConfig` dataclass — single source of truth for project paths

- **Notebook Research API** (REQ_037)
  - `tdw` package with `load_family()` entry point for notebook-based research
  - `LoadedFamily` with variant lookup by domain parameters (`family.get_variant(prime=113, seed=999)`)
  - Variant discovery: `list_variants()`, `list_variant_parameters()`
  - Variant convenience properties: `artifacts`, `metadata`, `model_config`, `train_losses`, `test_losses`
  - Forward pass helpers: `run_with_cache(probe, epoch)`, `make_probe(inputs)`, `analysis_dataset()`, `analysis_context()`
  - `make_probe()` added to `ModelFamily` protocol with Modulo Addition implementation
  - Build system (hatchling) for proper package installation from notebooks

### Architecture

```
tdw/                    # Notebook research API
  __init__.py           # load_family(), list_families()
  config.py             # AppConfig, get_config()
  loaded_family.py      # LoadedFamily (variant access by params)
families/variant.py     # +convenience properties (artifacts, metadata, etc.)
families/protocols.py   # +make_probe() on ModelFamily protocol
```

### References

- Archived requirements: `requirements/archive/v0.4.0-notebook-api/`
- Milestone summary: `requirements/archive/v0.4.0-notebook-api/MILESTONE_SUMMARY.md`

## [0.3.1] - 2026-02-10

### Added

- **Parameter Space Trajectory Projections** (REQ_029)
  - PCA-based trajectory visualization of model weight evolution through training
  - 2D trajectory (PC1 vs PC2) with epoch-colored points and current epoch highlight
  - Explained variance scree plot (individual + cumulative)
  - Parameter velocity plot (L2 norm of change per epoch, normalized by epoch gap)
  - Per-component group velocity comparison (Embedding, Attention, MLP)
  - `compute_pca_trajectory()` and `compute_parameter_velocity()` library functions

- **Weight Matrix Effective Dimensionality** (REQ_030)
  - New `EffectiveDimensionalityAnalyzer` computes SVD-based participation ratio per checkpoint
  - Dimensionality trajectory renderer (one line per weight matrix across epochs)
  - Singular value spectrum renderer (per-epoch bar chart)
  - Summary statistics: participation ratio per matrix

- **Loss Landscape Flatness** (REQ_031)
  - New `LandscapeFlatnessAnalyzer` measures sensitivity to random perturbations
  - `compute_landscape_flatness()` library function with configurable perturbation scale/samples
  - Flatness trajectory renderer (selectable metric: mean/max/std delta loss)
  - Perturbation distribution renderer (per-epoch histogram of delta losses)
  - Summary statistics: mean, max, std delta loss per epoch

- **Parameter Trajectory PC3 Visualization** (REQ_032)
  - 3D interactive PCA trajectory (PC1 vs PC2 vs PC3) with rotation/zoom/pan
  - PC1 vs PC3 and PC2 vs PC3 2D projection panels
  - Shared `_render_trajectory_2d()` helper eliminates rendering code duplication
  - All projections respond to component group selector and epoch slider
  - Confirms trajectory "dip" structure is genuine 3D geometry, not a projection artifact

- **Visualization Export** (REQ_033)
  - `export_figure()`: Static export of any Plotly Figure to PNG, SVG, PDF, or HTML
  - `export_animation()`: Animated GIF from per-epoch renderers
  - `export_cross_epoch_animation()`: Animated GIF sweeping current_epoch across cross-epoch renderers
  - `export_variant_visualization()`: Name-based convenience function (25 visualizations mapped)
  - Works without a running dashboard — designed for notebook/CLI/Claude programmatic use
  - Dependencies added: kaleido, Pillow

### Changed

- Parameter velocity now normalized by epoch gap for non-uniform checkpoint schedules

### References

- Archived requirements: `requirements/archive/v0.3.1-trajectory-export/`
- Milestone summary: `requirements/archive/v0.3.1-trajectory-export/MILESTONE_SUMMARY.md`

## [0.3.0] - 2026-02-08

### Added

- **Attention Head Pattern Visualization** (REQ_025)
  - New `AttentionPatternsAnalyzer` extracts full attention pattern tensor per checkpoint
  - 2x2 grid renderer showing all 4 attention heads as heatmaps
  - Position pair dropdown: view different attention relationships (e.g., `= attending to a`)
  - `extract_attention_patterns()` library function

- **Attention Head Frequency Specialization** (REQ_026)
  - New `AttentionFreqAnalyzer` computes Fourier frequency decomposition of attention patterns
  - Per-epoch frequency heatmap (frequencies x heads)
  - Cross-epoch specialization trajectory (one line per head)
  - Cross-epoch dominant frequency step plot
  - Summary statistics: dominant frequency, max fraction, mean specialization per head

- **Neuron Frequency Specialization Summary Statistics** (REQ_027)
  - Extended `NeuronFreqClustersAnalyzer` with `get_summary_keys()` and `compute_summary()`
  - Tracks specialized neuron counts per frequency with configurable threshold (default 0.9)
  - Low/mid/high frequency range bucket counts
  - Cross-epoch specialization trajectory renderer (total + low/mid/high lines)
  - Per-frequency specialization heatmap renderer (frequencies x epochs)

### Fixed

- **Variant dropdown UX** (REQ_028)
  - Variants now sorted alphabetically in dropdown and table
  - Default selection is `None` instead of first variant, so users can select the first item directly
- **Attention plot clipping** — full-width layout for attention head visualization

### References

- Archived requirements: `requirements/archive/v0.3.0-attention-specialization/`
- Milestone summary: `requirements/archive/v0.3.0-attention-specialization/MILESTONE_SUMMARY.md`

## [0.2.1] - 2026-02-08

### Added

- **Family-Specific Summary Statistics** (REQ_022)
  - Optional `get_summary_keys()` and `compute_summary()` methods on Analyzer protocol
  - Pipeline collects summary statistics inline and writes single `summary.npz` per analyzer
  - ArtifactLoader `load_summary()` and `has_summary()` for cross-epoch data access
  - Gap-filling support for incremental summary updates

- **Coarseness Analyzer** (REQ_023)
  - New `CoarsenessAnalyzer` quantifies blob vs plaid neuron patterns
  - Per-epoch artifact: per-neuron coarseness values `(d_mlp,)`
  - Summary statistics: mean, std, median, percentiles, blob count, histogram
  - Library function: `compute_neuron_coarseness()` in `analysis/library/fourier.py`
  - Registered for Modulo Addition 1-Layer family

- **Coarseness Visualizations** (REQ_024)
  - Coarseness trajectory line plot with percentile band and epoch indicator
  - Per-epoch coarseness distribution histogram with blob/plaid/transitional coloring
  - Blob count trajectory renderer (notebook-focused)
  - Per-neuron coarseness bar chart (notebook-focused)
  - Conditional dashboard panels: appear only when coarseness artifacts exist
  - All eight trained variants analyzed with coarseness data

### References

- Archived requirements: `requirements/archive/v0.2.1-coarseness/`
- Milestone summary: `requirements/archive/v0.2.1-coarseness/MILESTONE_SUMMARY.md`

## [0.2.0] - 2026-02-06

### Added

- **Model Family Abstraction** (REQ_021)
  - `ModelFamily` protocol with `Variant`, `FamilyRegistry`, `ArchitectureSpec`
  - JSON-driven family definitions in `model_families/`
  - Generic analysis library (`analysis/library/`) + family-bound analyzers (`analysis/analyzers/`)
  - Modulo Addition 1-Layer family implementation with five trained variants

- **Per-Epoch Artifact Storage** (REQ_021f)
  - Artifacts stored as `epoch_{NNNNN}.npz` per (analyzer, epoch) — eliminates memory exhaustion
  - `ArtifactLoader` with on-demand `load_epoch()`, stacked `load_epochs()`, and discovery methods
  - Dashboard loads single epochs per slider interaction

- **Dashboard Integration** (REQ_021d, REQ_021e)
  - Family-aware Analysis tab with variant selection
  - Training tab with family selection and domain parameter inputs
  - Dynamic visualization tabs generated from family configuration

- **Checkpoint Epoch-Index Display** (REQ_020)
  - Checkpoint markers on loss curve show "Epoch: X (Index: Y)" in tooltip
  - Current Epoch display shows "Epoch X (Index Y)" near slider

### Architecture

```
model_families/          # Family definitions (family.json)
families/                # Protocols, registry, implementations
analysis/
  library/               # Generic analysis functions (Fourier, activations)
  analyzers/             # Family-bound analyzers
  pipeline.py            # Per-epoch artifact persistence
  artifact_loader.py     # On-demand loading
results/                 # Per-variant checkpoints + artifacts
```

### References

- Archived requirements: `requirements/archive/v0.2.0-foundations/`
- Milestone summary: `requirements/archive/v0.2.0-foundations/MILESTONE_SUMMARY.md`

## [0.1.3] - 2026-02-03

### Added

- **Checkpoint Epoch-Index Display** (REQ_020)
  - Checkpoint markers on loss curve show "Epoch: X (Index: Y)" in tooltip
  - Current Epoch display shows "Epoch X (Index Y)" near slider
  - Enables quick navigation to specific epochs

### Fixed

- **Slider None Guard** (BUG_003)
  - Fixed TypeError when clearing slider text input before typing new value
  - Gracefully handles None values by falling back to current state

## [0.1.2] - 2026-02-01

### Changed

- **Python 3.11+ Compatibility** (REQ_013)
  - Broadened support from Python 3.13-only to 3.11, 3.12, and 3.13
  - Relaxed numpy constraint (>=1.26) for TransformerLens compatibility
  - CI now tests against all three Python versions

- **Removed neel-plotly Dependency** (REQ_012)
  - Replaced git dependency with native Plotly visualization utilities
  - New `visualization.line()` function provides equivalent functionality
  - Zero external git dependencies improves installation reliability

- **Code Quality Enforcement** (REQ_011)
  - Ruff linting and formatting checks now blocking in CI
  - Pyright type checking now blocking in CI
  - Fixed all linting violations and type errors across codebase

### References

- Archived requirements: `requirements/archive/v0.1.2-quality/`
- Milestone summary: `requirements/archive/v0.1.2-quality/MILESTONE_SUMMARY.md`

## [0.1.1] - 2026-02-01

### Fixed

- Dashboard now uses GPU for training and analysis when CUDA is available (REQ_016, REQ_018)
  - Training previously required `CUDA_VISIBLE_DEVICES` env var to detect GPU
  - Analysis was hardcoded to CPU only

### References

- Archived requirements: `requirements/archive/v0.1.1-cuda/`
- Milestone summary: `requirements/archive/v0.1.1-cuda/MILESTONE_SUMMARY.md`

## [0.1.0] - 2026-02-01

**MVP Release** - First functional version of the Training Dynamics Workbench.

### Added

- **Training Runner** (REQ_001, REQ_002)
  - Configurable checkpoint epochs with optimized default schedule
  - Safetensors persistence with metadata.json and config.json
  - Backward-compatible loading for legacy pickle format

- **Analysis Pipeline** (REQ_003)
  - `AnalysisPipeline` orchestrator with resumable artifact generation
  - `ArtifactLoader` for visualization-layer independence
  - Progress callbacks for UI integration
  - Three analyzers: dominant frequencies, neuron activations, frequency clusters

- **Visualizations** (REQ_004, REQ_005, REQ_006)
  - Dominant embedding frequencies bar plot with threshold highlighting
  - Neuron activation heatmaps (single neuron + grid views)
  - Neuron-frequency cluster heatmap

- **Dashboard** (REQ_007, REQ_008, REQ_009, REQ_010)
  - Gradio web interface with Training and Analysis tabs
  - Synchronized epoch slider across all visualizations
  - Loss curves with epoch indicator line
  - Application versioning displayed in header

### Architecture

```
Training (ModuloAdditionSpecification)
    ↓ checkpoints/*.safetensors
Analysis (AnalysisPipeline + Analyzers)
    ↓ artifacts/*.npz
Visualization (Renderers)
    ↓ Plotly figures
Dashboard (Gradio)
```

### Technical Notes

- Python 3.13 with PyTorch + CUDA support
- TransformerLens for model architecture
- Artifact-based design separates expensive computation from cheap visualization
- 156 tests across 8 test files

### References

- Archived requirements: `requirements/archive/v0.1.0-mvp/`
- Milestone summary: `requirements/archive/v0.1.0-mvp/MILESTONE_SUMMARY.md`
