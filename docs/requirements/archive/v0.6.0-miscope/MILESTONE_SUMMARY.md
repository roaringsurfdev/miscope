# v0.6.0 — MIScope

Three requirements in this release: cross-epoch analyzers, Dash job management UI, and the source layout restructuring that renamed the project to MIScope.

## REQ_038: Cross-Epoch Analyzers

PCA trajectory analysis was previously computed at render time in the dashboard (every epoch slider change recomputed PCA from raw snapshots). REQ_038 introduced a `CrossEpochAnalyzer` protocol and two-phase pipeline so this computation happens once during analysis and is stored as an artifact.

- **New protocol**: `CrossEpochAnalyzer` — runs after per-epoch analysis, consumes artifacts across all checkpoints
- **Pipeline Phase 2**: Executes cross-epoch analyzers after per-epoch phase, with dependency validation and skip-if-exists logic
- **ParameterTrajectoryPCA**: First cross-epoch analyzer — precomputes PCA projections + velocity for 4 component groups
- **Renderer refactor**: All 7 trajectory renderers accept precomputed data instead of raw snapshots
- **Dashboard v2**: Loads from `cross_epoch.npz` instead of computing PCA at render time

## REQ_040: Dash Job Management UI

Migrated Training and Analysis Run management from the Gradio dashboard to Dash, completing the Dash migration started in v0.5.0.

- **Training page**: family selection, domain parameter inputs, training config, checkpoint scheduling
- **Analysis Run page**: variant selection, analyzer selection, run triggering
- **Site navigation**: `create_navbar()` with multi-page routing
- **Server-side state**: `ServerState` singleton for training/analysis job management

## REQ_039: Source Layout Restructuring & MIScope Rename

Namespace renamed from `tdw` to `miscope`. Core packages moved to `src/miscope/` (standard Python src-layout). Gradio dashboard removed.

- `analysis/`, `families/`, `visualization/` → `src/miscope/analysis/`, etc.
- `dashboard/` (Gradio) removed — shared components migrated to `dashboard_v2/`
- Package installs as `miscope==0.1.0`
- 135 import lines rewritten across 52 files

## Key Files

| File | Role |
|------|------|
| `src/miscope/analysis/protocols.py` | CrossEpochAnalyzer protocol |
| `src/miscope/analysis/pipeline.py` | Two-phase execution logic |
| `src/miscope/analysis/analyzers/parameter_trajectory_pca.py` | First cross-epoch analyzer |
| `src/miscope/analysis/artifact_loader.py` | load_cross_epoch / has_cross_epoch |
| `src/miscope/visualization/renderers/parameter_trajectory.py` | Refactored trajectory renderers |
| `dashboard_v2/pages/training.py` | Training page |
| `dashboard_v2/pages/analysis_run.py` | Analysis Run page |
| `dashboard_v2/navigation.py` | Site-level navigation |
| `dashboard_v2/state.py` | ServerState for job management |
| `tests/test_cross_epoch_analyzers.py` | Cross-epoch analyzer tests |
| `tests/test_req_040_dash_job_management.py` | Dash job management tests |

## Opens for Future

- Phase transition detection as a cross-epoch analyzer
- Per-input trace analysis (core research direction)
- `dashboard_v2/` rename to `dashboard/` once stable
- SQLite when async jobs or multi-probe support arrives
