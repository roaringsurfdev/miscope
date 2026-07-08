# Milestone Summary: v0.7.0 — View Catalog and UX

**Released:** 2026-03-03
**Branch:** develop → main
**Requirements:** REQ_041–042, REQ_044–051 (10 requirements; REQ_043 deferred to future/)

## Theme

This milestone delivers two interlocking upgrades:

1. **View Catalog** (REQ_047) — a universal presentation layer that decouples analysis views from the dashboard framework. Any view registered in the catalog works in notebooks, the dashboard, and export scripts without change.

2. **Dashboard UX** (REQ_046) — architectural consolidation of the Dash dashboard: encapsulated components, store-centric state, `AnalysisPageGraphManager` shared logic, and pattern-ID callbacks for click-to-navigate. These changes make adding new pages fast and safe.

Together they reduced the cost of adding a new analysis from "wire up a page from scratch" to "register a view in the catalog; page wires itself."

## New Analytical Capabilities

### Representational Geometry (REQ_044, REQ_045)
First activation-space analysis in the platform. `RepresentationalGeometryAnalyzer` tracks per-epoch class centroid geometry (PCA, Fisher discriminant, centroid distances) at four sites: embedding, post-attention, MLP output, residual stream. Fisher heatmap enables targeted pairwise class-pair analysis.

**Key files:**
- `src/miscope/analysis/analyzers/repr_geometry.py`
- `src/miscope/visualization/renderers/repr_geometry.py`
- `dashboard/pages/repr_geometry.py`

### Neuron Dynamics (REQ_042, REQ_048)
Trajectory heatmap (neuron × epoch, colored by dominant frequency) reveals per-neuron frequency-switching behavior invisible in aggregate statistics. ~90% of neurons switch at least once; commitment cascades near grokking. Secondary analysis tier (REQ_048) enables `NeuronDynamicsAnalyzer` to compute switch counts and commitment epochs from existing per-epoch artifacts without re-running the primary pipeline.

**Key files:**
- `src/miscope/analysis/analyzers/neuron_dynamics.py`
- `src/miscope/analysis/pipeline.py` (secondary tier)
- `dashboard/pages/neuron_dynamics.py`

### Neuron Fourier Decomposition (REQ_049)
`NeuronFourierAnalyzer` implements per-neuron Fourier decomposition of MLP weights following He et al. (2026). Proof-of-concept notebook at `notebooks/neuron_fourier_poc.py`.

**Key file:** `src/miscope/analysis/analyzers/neuron_fourier.py`

### Global Centroid PCA (REQ_050)
`GlobalCentroidPCAAnalyzer` fits PCA across all training epochs jointly (not per-epoch), providing a stable coordinate frame for centroid trajectory analysis. Dashboard page shows parameter + centroid PCA, variance timeseries, and SV spectrum.

**Key files:**
- `src/miscope/analysis/analyzers/global_centroid_pca.py`
- `dashboard/pages/dimensionality.py`

### Centroid DMD (REQ_051)
`CentroidDMDAnalyzer` applies Dynamic Mode Decomposition to class centroid trajectories, decomposing the evolution of representational geometry into dynamic modes. Log-scale support for amplitude spectra.

**Key files:**
- `src/miscope/analysis/analyzers/centroid_dmd.py`
- `dashboard/pages/centroid_dmd.py`

### Summary Lens (REQ_041)
New `/summary` page: dense grid of 12 cross-epoch visualizations answering "what's the shape of this model's training story?" at a glance. Temporal cursor (epoch slider) synchronizes a vertical indicator line across all time-axis plots without reloading data.

**Key file:** `dashboard/pages/summary.py`

## Architecture Changes

### View Catalog (`src/miscope/views/`)
- `catalog.py` — `ViewDefinition` protocol, `ViewCatalog` registry
- `universal.py` — all registered views (universal instruments; families are context providers)
- Primary interface: `variant.at(epoch)` → `EpochContext` → `.view(name)` → `BoundView`
- `BoundView`: `.show()` (notebook inline), `.figure()` (raw Plotly), `.export(format, path)`
- Canonical export path derivation on `BoundView`

### Dashboard Architecture (`dashboard/`)
- `components/analysis_page.py` — `AnalysisPageGraphManager`: shared page logic, drives `_VIEW_LIST` dispatch
- `components/variant_selector.py` — encapsulated variant selector component
- `sitenav.py` — routing; coordinates `page_left_nav` + `page-content` outputs
- Pattern IDs (`{'view_type': ..., 'index': graph_id}`) — ALL-pattern callbacks for click-to-navigate
- Store-centric: all state changes flow through `variant-selector-store`

### Dashboard Pages Added
| Page | Route | Requirement |
|------|-------|-------------|
| Summary Lens | `/summary` | REQ_041 |
| Neuron Dynamics | `/neuron-dynamics` | REQ_042 |
| Representational Geometry | `/repr-geometry` | REQ_044 |
| Dimensionality | `/dimensionality` | REQ_050 |
| Centroid DMD | `/centroid-dmd` | REQ_051 |

## Deferred

**REQ_043 — Fourier Profile Expansion (W_U multi-matrix):** Moved to `requirements/future/`. Prerequisite for embedding/unembedding co-evolution analysis. No blocking dependencies; can be added when the research question becomes pressing.
