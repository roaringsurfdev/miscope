# v0.3.1 Milestone: Parameter Trajectory & Visualization Export

**Released:** 2026-02-10
**Requirements:** REQ_029, REQ_030, REQ_031, REQ_032, REQ_033

## Summary

This milestone adds parameter space analysis tools and programmatic visualization export. The core theme is enabling deeper geometric analysis of training dynamics — how the model moves through weight space during grokking — and making visualizations accessible outside the dashboard.

## Key Decisions

- **PCA over UMAP/t-SNE** (REQ_029): PCA chosen for trajectory visualization because it preserves global structure and is deterministic. Non-linear methods would lose the meaningful distance relationships that make trajectory geometry interpretable.

- **PC3 motivated by research observation** (REQ_032): A characteristic "bend" in the MLP trajectory was observed across all 10 model variants. PC3 confirmed this is genuine 3D structure (not a 2D projection artifact) — the bend and endpoint separate by ~10 units in PC3.

- **Middle-tier export** (REQ_033): Export utilities designed as library functions (not dashboard features) so both notebooks and Claude can generate visualizations programmatically. Dashboard export UI deferred to a future requirement.

- **Velocity normalization** (REQ_029 bugfix): Parameter velocity normalized by epoch gap to handle non-uniform checkpoint schedules (e.g., 500-epoch gaps early, 100-epoch gaps later).

## File Locations

### Analysis
- `analysis/library/trajectory.py` — PCA trajectory and velocity computation
- `analysis/library/landscape.py` — Loss landscape flatness computation
- `analysis/library/dimensionality.py` — SVD-based effective dimensionality
- `analysis/analyzers/landscape_flatness.py` — Landscape flatness analyzer
- `analysis/analyzers/effective_dimensionality.py` — Effective dimensionality analyzer

### Visualization
- `visualization/renderers/parameter_trajectory.py` — 7 renderers (2D, 3D, velocity, variance)
- `visualization/renderers/landscape_flatness.py` — Flatness trajectory and perturbation distribution
- `visualization/renderers/effective_dimensionality.py` — Dimensionality trajectory and SVD spectrum
- `visualization/export.py` — Static and animated export utilities

### Tests
- `tests/test_parameter_trajectory.py` — 58 tests
- `tests/test_landscape_flatness.py` — 38 tests
- `tests/test_effective_dimensionality.py` — Tests for dimensionality analysis
- `tests/test_visualization_export.py` — 30 tests

## Dependencies Added
- `kaleido>=0.2.1` — Plotly static image export (PNG, SVG, PDF)
- `Pillow>=10.0` — GIF animation stitching
