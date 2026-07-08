# REQ_024: Coarseness Visualizations

**Status:** Draft
**Priority:** High (directly enables grokking research hypotheses)
**Dependencies:** REQ_023 (Coarseness Analyzer), REQ_022 (Summary Statistics), REQ_008 (Dashboard)
**Last Updated:** 2026-02-08

## Problem Statement

The coarseness analyzer (REQ_023) produces per-epoch artifacts and summary statistics that quantify blob vs plaid neuron patterns across training. Analysis has been run across all trained variants. However, there is no way to visualize coarseness data in the dashboard, and no reusable renderers exist for notebook exploration.

The coarseness analysis recommendations document identifies five priority analyses. The top two — **coarseness distribution over time** and **mean coarseness trajectory** — require visualization infrastructure that doesn't exist yet. Currently, the researcher must write ad-hoc plotting code in the notebook for each variant.

### What This Enables

Dashboard coarseness visualizations allow the researcher to:

- Watch the neuron population shift from plaid-dominated to blob-dominated as grokking approaches (distribution histogram driven by epoch slider)
- See where the current epoch sits in the coarseness trajectory (line plot with epoch indicator, synchronized to slider)
- Correlate coarseness evolution with loss curves and frequency cluster changes at the same epoch (holistic dashboard exploration)

These are the minimum visualizations needed to test the core hypothesis: **grokking models develop blob neurons while non-grokking models don't**.

## Design

### Two Dashboard Visualizations

Both integrate with the existing epoch slider and follow established renderer patterns.

#### 1. Coarseness Trajectory (Cross-Epoch)

A line plot showing coarseness statistics across all analyzed checkpoints, with a vertical indicator line at the currently selected epoch. Analogous to the loss curves panel.

**Data source:** `summary.npz` via `ArtifactLoader.load_summary("coarseness")`

**Plot elements:**
- **Primary line:** Mean coarseness per epoch
- **Shaded band:** 25th–75th percentile range (p25 to p75)
- **Vertical indicator:** Current epoch (synchronized to slider), same pattern as loss curves
- **Horizontal reference line:** Blob threshold (0.7) as a dashed line with annotation
- **X-axis:** Epoch number
- **Y-axis:** Coarseness (0 to 1)

**Rationale:** This is recommendation #2 (mean coarseness vs epoch) from the coarseness research notes. The percentile band adds context about the full neuron distribution without requiring a separate plot. The blob threshold reference line makes it immediately visible when mean coarseness crosses into "blob territory."

**Update behavior:** Summary data is loaded once when a variant is selected (small file). Only the vertical indicator line updates when the epoch slider moves. No per-epoch artifact loading needed.

#### 2. Coarseness Distribution (Per-Epoch)

A histogram showing the distribution of coarseness values across all neurons at the currently selected epoch. Driven by the epoch slider.

**Data source:** Per-epoch artifact via `ArtifactLoader.load_epoch("coarseness", epoch)` → `{"coarseness": (d_mlp,)}`

**Plot elements:**
- **Histogram bars:** 20 bins over [0, 1] range
- **Vertical reference lines:** Blob threshold (0.7) and plaid threshold (0.5) as dashed lines with annotations
- **Color:** Bars colored by region — plaid (<0.5), transitional (0.5–0.7), blob (>=0.7) — using a discrete 3-color scheme
- **Annotation:** Blob count (neurons >= 0.7) and total neuron count in subtitle or corner text
- **X-axis:** Coarseness value (0 to 1)
- **Y-axis:** Neuron count

**Rationale:** This is recommendation #1 (coarseness distribution over time), the highest-priority analysis. As the researcher scrubs through epochs, they can directly watch the distribution shift from concentrated-at-low to spreading-toward-high — the signature of grokking. For non-grokking models (p=101, seed=999), the distribution should remain stuck at low values.

**Update behavior:** Loads per-epoch coarseness artifact on each slider change. The artifact is small — a single `(d_mlp,)` array — so loading is fast.

### Dashboard Layout

The coarseness visualizations should appear in the Analysis tab when coarseness artifacts are available. Suggested placement:

```
┌──────────────────────────────────────────────────────────┐
│ Loss Curves (existing, full width)                       │
├──────────────────────────────────────────────────────────┤
│ Coarseness Trajectory (full width, new)                  │
├────────────────────────────┬─────────────────────────────┤
│ Dominant Frequencies       │ Neuron Activation Heatmap   │
│ (existing)                 │ (existing)                  │
├────────────────────────────┴─────────────────────────────┤
│ Frequency Clusters (existing, full width)                │
├──────────────────────────────────────────────────────────┤
│ Coarseness Distribution (full width, new)                │
└──────────────────────────────────────────────────────────┘
```

The trajectory sits below loss curves because both are cross-epoch plots with epoch indicators — the researcher reads them together. The distribution sits at the bottom since it's a supporting detail view driven by the slider.

**Conditional rendering:** If coarseness artifacts are not available for the selected variant, the coarseness panels should not appear (no empty placeholders or error states). This keeps the dashboard clean for families or variants that haven't run the coarseness analyzer.

### Notebook/Exploratory Renderers

In addition to dashboard-integrated renderers, provide renderers usable in notebook contexts:

#### 3. Blob Count Trajectory (Cross-Epoch)

Line plot of blob neuron count over epochs. Uses `blob_count` from `summary.npz`.

```python
render_blob_count_trajectory(
    summary_data: dict[str, np.ndarray],
    current_epoch: int | None = None,
    title: str | None = None,
) -> go.Figure
```

Simple line plot. Optional epoch indicator for dashboard reuse. This is recommendation #5.

#### 4. Coarseness by Neuron Index (Per-Epoch)

Bar or strip plot showing coarseness value per neuron, colored by blob/plaid classification.

```python
render_coarseness_by_neuron(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    blob_threshold: float = 0.7,
    plaid_threshold: float = 0.5,
    title: str | None = None,
) -> go.Figure
```

Neurons on x-axis (0 to d_mlp-1), coarseness on y-axis. Color-coded by region. This is recommendation #3 (spatial pattern) in single-epoch form. Useful for identifying which specific neurons become blobs.

### Renderer API

All renderers follow established patterns:

```python
# Cross-epoch with indicator (dashboard)
render_coarseness_trajectory(
    summary_data: dict[str, np.ndarray],  # from load_summary("coarseness")
    current_epoch: int,                    # for vertical indicator
    blob_threshold: float = 0.7,
    title: str | None = None,
) -> go.Figure

# Per-epoch (dashboard, slider-driven)
render_coarseness_distribution(
    epoch_data: dict[str, np.ndarray],  # from load_epoch("coarseness", epoch)
    epoch: int,                          # for title
    blob_threshold: float = 0.7,
    plaid_threshold: float = 0.5,
    n_bins: int = 20,
    title: str | None = None,
) -> go.Figure
```

**Data format conventions:**
- `summary_data` contains `"epochs"`, `"mean_coarseness"`, `"p25_coarseness"`, `"p75_coarseness"`, `"blob_count"` arrays (all shape `(N,)`)
- `epoch_data` contains `"coarseness"` array of shape `(d_mlp,)`
- These match the existing outputs from `CoarsenessAnalyzer` and `ArtifactLoader`

### Visualization Registration

Add coarseness visualization identifiers to the Modulo Addition 1-Layer family's `visualizations` list in `family.json`:
- `"coarseness_trajectory"`
- `"coarseness_distribution"`

## Scope

This requirement covers:
1. Renderer: `render_coarseness_trajectory()` (cross-epoch line plot with indicator and percentile band)
2. Renderer: `render_coarseness_distribution()` (per-epoch histogram with region coloring)
3. Renderer: `render_blob_count_trajectory()` (cross-epoch line plot, notebook-focused)
4. Renderer: `render_coarseness_by_neuron()` (per-epoch bar/strip, notebook-focused)
5. Dashboard integration: conditional coarseness panels in Analysis tab
6. Dashboard wiring: epoch slider drives distribution updates; variant selection loads summary data
7. Export from `visualization/__init__.py`
8. Family registration: add visualization identifiers to `family.json`

This requirement does **not** cover:
- Cross-variant coarseness comparison (e.g., overlaying trajectories from multiple variants)
- Dual-axis overlay of coarseness trajectory on loss curves (future enhancement)
- Coarseness-grokking prediction scatter plot (recommendation #4 — requires cross-variant infrastructure)
- New coarseness metrics or analyzer changes

## Conditions of Satisfaction

### Renderers
- [ ] `render_coarseness_trajectory()` produces a Plotly figure with mean line, p25–p75 band, epoch indicator, and blob threshold reference
- [ ] `render_coarseness_distribution()` produces a Plotly histogram with 3-region coloring and threshold reference lines
- [ ] `render_blob_count_trajectory()` produces a Plotly line figure from summary data
- [ ] `render_coarseness_by_neuron()` produces a Plotly bar/strip figure with per-neuron coloring
- [ ] All renderers accept `title` override parameter
- [ ] All renderers return `plotly.graph_objects.Figure`
- [ ] Renderers are exported from `visualization/__init__.py`

### Dashboard Integration
- [ ] Coarseness trajectory panel appears in Analysis tab when coarseness artifacts exist
- [ ] Coarseness distribution panel appears in Analysis tab when coarseness artifacts exist
- [ ] Both panels are absent (not empty, not errored) when coarseness artifacts do not exist
- [ ] Epoch slider updates the distribution histogram and the trajectory indicator line
- [ ] Summary data loaded once on variant selection (not reloaded on slider change)
- [ ] Per-epoch coarseness data loaded on demand per slider interaction (same pattern as other per-epoch visualizations)
- [ ] Coarseness panels do not interfere with existing visualizations (loss, frequencies, activations, clusters)

### Visual Quality
- [ ] Trajectory percentile band is visually distinct but not overwhelming (semi-transparent fill)
- [ ] Distribution histogram clearly shows the blob/plaid/transitional regions
- [ ] Threshold reference lines are visible but secondary (dashed, muted color)
- [ ] Hover tooltips provide exact values
- [ ] Titles include epoch number where appropriate

### Family Registration
- [ ] Coarseness visualization identifiers added to Modulo Addition 1-Layer `family.json`

## Constraints

**Must have:**
- Follows established renderer API patterns (per-epoch and cross-epoch signatures)
- Uses `ArtifactLoader` for data access (no direct filesystem access in renderers)
- Conditional rendering based on artifact availability (not hardcoded to coarseness)
- Epoch slider synchronization for distribution histogram
- Summary data used for trajectory (not loading all per-epoch files)

**Must avoid:**
- Loading all per-epoch coarseness files to render the trajectory (use summary.npz)
- Interpolating between checkpoint epochs on the trajectory (discrete markers or connecting line through actual data points only — no synthetic interpolation)
- Adding coarseness panels for variants/families without coarseness artifacts
- Breaking existing visualization layout or event wiring

**Flexible:**
- Exact layout position of coarseness panels within the Analysis tab
- Whether distribution uses bars or area fill
- Color scheme for blob/plaid/transitional regions
- Whether blob count is shown as annotation on the distribution or as a separate display element
- Whether the trajectory and blob count are combined into one panel or kept separate in the dashboard

## Decision Log

| Date | Question | Decision | Rationale |
|------|----------|----------|-----------|
| 2026-02-08 | Which recommendations to prioritize for dashboard? | #1 (distribution) and #2 (trajectory) | Highest research value; directly test core hypothesis; natural fit with slider-driven and indicator-line patterns |
| 2026-02-08 | Include blob count as separate dashboard panel? | No — notebook renderer only | Dashboard already adds 2 new panels; blob count is derivable from the distribution; avoid clutter |
| 2026-02-08 | Trajectory: load per-epoch or summary? | Summary | Trajectory needs all epochs at once; summary.npz exists for exactly this access pattern (REQ_022) |
| 2026-02-08 | Conditional vs always-visible panels? | Conditional on artifact existence | Coarseness is family-specific; other families shouldn't see empty panels |

## Notes

**2026-02-08:** The coarseness analyzer has been run across all 8 trained variants (4 primes x 2 seeds). Summary and per-epoch artifacts exist in all variant directories. The notebook has been used for initial exploration (mean, std, p25 line plots via plotly express). This requirement formalizes those explorations into reusable renderers and dashboard integration.

**DC component:** The DC component (mean activation level) is already excluded from the coarseness computation by `compute_frequency_variance_fractions()`, which zeros out the `[0, 0]` entry of the Fourier-transformed activations before computing variance fractions. Coarseness measures the ratio of low-frequency to total non-DC energy. No additional DC removal option is needed.

**Relationship to recommendations document:** This requirement covers recommendations #1, #2, #3, and #5 from `requirements/drafts/coarseness_analysis_recommendations.md`. Recommendation #4 (early coarseness predicts grokking) requires cross-variant comparison infrastructure that doesn't exist yet and is deferred.
