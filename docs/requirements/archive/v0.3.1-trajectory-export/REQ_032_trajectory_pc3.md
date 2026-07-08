# REQ_032: Parameter Trajectory PC3 Visualization

**Status:** Draft
**Priority:** Medium (research-motivated enhancement to existing visualization)
**Dependencies:** REQ_029 (Parameter Space Trajectory Projections)
**Last Updated:** 2026-02-10

## Problem Statement

The existing parameter trajectory visualization (REQ_029) projects the model's path through weight space onto PC1 vs PC2. This captures 60-70% of variance for the Modulo Addition model and reveals the characteristic U-shaped grokking trajectory.

However, the 2D projection may hide important structure. During analysis of 10 model variants, a characteristic "bend" was observed in the MLP trajectory midway before the grokking turn. This bend appears to coincide spatially (in the 2D projection) with the model's final converged position, raising the question: does the model pass near the generalizing basin early in training, or is this a projection artifact where distinct regions of high-dimensional space happen to overlap in PC1-PC2?

PC3 captures the next largest variance direction and can disambiguate this. If the bend and endpoint overlap in PC1-PC2 but separate in PC3, it's a projection artifact. If they remain close, the "passing near the basin" interpretation gains support.

More generally, phase transitions and trajectory geometry are intrinsically 3D-or-higher phenomena. Making PC3 visible enables richer geometric analysis of training dynamics across all variants.

## Motivation

Analysis of p=101/seed=999 revealed anomalous grokking behavior consistent with the model settling on a sub-optimal manifold (see `notes/variant_analysis_p101_seed999.md`). The parameter trajectory is a key tool for studying this, but the 2D projection limits what can be observed. Specific observations motivating this requirement:

- A "bend" in the MLP trajectory before the grokking turn that may mark the onset of specialization
- The bend's apparent spatial coincidence with the final converged position in 2D
- The need to characterize "the turn" (grokking phase transition point) more precisely in parameter space

## Design

### Renderers

Three new renderer functions in `visualization/renderers/parameter_trajectory.py`:

```python
def render_trajectory_3d(
    snapshots: list[dict[str, np.ndarray]],
    epochs: list[int],
    current_epoch: int,
    components: list[str] | None = None,
    title: str | None = None,
    height: int = 550,
) -> go.Figure:
    """3D interactive PCA trajectory (PC1 vs PC2 vs PC3).

    Points colored by epoch (Bluered gradient), connected by path.
    Current epoch highlighted with marker. Plotly's interactive 3D
    supports rotation, zoom, and pan for exploratory analysis.
    """


def render_trajectory_pc1_pc3(
    snapshots: list[dict[str, np.ndarray]],
    epochs: list[int],
    current_epoch: int,
    components: list[str] | None = None,
    title: str | None = None,
    height: int = 450,
) -> go.Figure:
    """2D trajectory projection: PC1 vs PC3.

    Same visual style as existing PC1 vs PC2 trajectory.
    """


def render_trajectory_pc2_pc3(
    snapshots: list[dict[str, np.ndarray]],
    epochs: list[int],
    current_epoch: int,
    components: list[str] | None = None,
    title: str | None = None,
    height: int = 450,
) -> go.Figure:
    """2D trajectory projection: PC2 vs PC3.

    Same visual style as existing PC1 vs PC2 trajectory.
    """
```

All three renderers share the core PCA computation with the existing `render_parameter_trajectory`. Consider extracting the shared scatter/path/highlight logic into an internal helper to avoid duplication across the now four trajectory renderers.

### No Analyzer or Library Changes

`compute_pca_trajectory` already computes 3 components by default. The new renderers simply use the third column of the projections array that is currently discarded by `render_parameter_trajectory` (which passes `n_components=2`). The new renderers should request `n_components=3`.

### Dashboard Integration

Add the three new panels to the Parameter Trajectory section of the Analysis visualization tab:

- **3D Trajectory** — interactive 3D scatter, placed prominently
- **PC1 vs PC3** and **PC2 vs PC3** — 2D panels, placed in a row below the 3D view

All three panels should:
- Respond to the component group selector (All / Embedding / Attention / MLP)
- Sync with the epoch slider for the current epoch highlight
- Be conditional on `parameter_snapshot` artifacts, same as existing trajectory panels

### Layout Consideration

The Parameter Trajectory section currently has the 2D trajectory and velocity plots. Adding three more panels makes this section larger. Group the four trajectory projections (existing PC1-PC2, new 3D, new PC1-PC3, new PC2-PC3) together visually, separate from the velocity plots.

## Scope

**This requirement covers:**
1. Three new renderer functions (3D trajectory, PC1 vs PC3, PC2 vs PC3)
2. Dashboard integration of the three new panels
3. Refactoring shared rendering logic if beneficial
4. Tests for new renderers

**This requirement does not cover:**
- Changes to the analyzer or library (no changes needed)
- Alternative dimensionality reduction (UMAP, t-SNE)
- Cross-variant trajectory comparison
- Trajectory curvature or "turn detection" metrics (potential future requirement)

## Conditions of Satisfaction

### Renderers
- [ ] `render_trajectory_3d()` produces a Plotly 3D scatter with epoch-colored points, connecting path, and current epoch highlight
- [ ] 3D plot supports interactive rotation, zoom, and pan
- [ ] Axis labels include PC number and explained variance percentage for all three components
- [ ] `render_trajectory_pc1_pc3()` produces a 2D scatter of PC1 (x) vs PC3 (y)
- [ ] `render_trajectory_pc2_pc3()` produces a 2D scatter of PC2 (x) vs PC3 (y)
- [ ] 2D renderers visually consistent with existing `render_parameter_trajectory()` (same color scale, marker style, highlight)
- [ ] All renderers accept and respect the `components` parameter for component group filtering
- [ ] All renderers return `go.Figure` objects

### Dashboard
- [ ] 3D trajectory panel visible when `parameter_snapshot` artifacts exist
- [ ] PC1 vs PC3 panel visible when `parameter_snapshot` artifacts exist
- [ ] PC2 vs PC3 panel visible when `parameter_snapshot` artifacts exist
- [ ] All three panels respond to component group selector
- [ ] All three panels sync current epoch highlight with epoch slider
- [ ] Trajectory projection panels are grouped together visually

### Tests
- [ ] 3D renderer returns a Figure with Scatter3d traces
- [ ] 3D renderer projects onto 3 dimensions (not 2)
- [ ] PC1 vs PC3 renderer uses correct projection columns (0, 2)
- [ ] PC2 vs PC3 renderer uses correct projection columns (1, 2)
- [ ] All renderers highlight the current epoch when it exists in the epoch list
- [ ] All renderers handle component group filtering

## Constraints

**Must have:**
- Visual consistency with existing trajectory renderer (Bluered color scale, star marker for current epoch)
- Interactive 3D rotation in the dashboard (Plotly's default 3D behavior)
- Explained variance percentage on all axes

**Must avoid:**
- Duplicating PCA computation across renderers unnecessarily (compute once, render from shared result)
- Breaking the existing PC1 vs PC2 renderer or its dashboard integration

**Flexible:**
- Exact dashboard layout arrangement of the four trajectory panels
- Whether the 2D helper refactor happens (nice-to-have, not required)
- Default camera angle for the 3D plot
- Panel sizing relative to existing panels

## Notes

**Research context:** This enhancement is directly motivated by analysis of variant p=101/seed=999 showing anomalous grokking dynamics. The "bend" in the MLP trajectory and its relationship to the grokking phase transition is an active research question. See `notes/variant_analysis_p101_seed999.md` for full analysis.

**Future direction:** If the trajectory geometry proves useful for identifying phase transitions, a follow-on requirement could formalize "turn detection" — automatically identifying the point of maximum trajectory curvature as a grokking onset marker. This would complement the visual analysis enabled here with a quantitative metric.
