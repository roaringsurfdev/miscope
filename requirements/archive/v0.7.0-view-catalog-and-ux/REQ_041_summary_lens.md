# REQ_041: Summary Lens

**Status:** Draft
**Priority:** High (primary single-variant analysis workflow improvement)
**Dependencies:** REQ_040 (Dash navigation structure — completed), REQ_035 (Dash migration — completed)
**Last Updated:** 2026-02-15

## Problem Statement

When examining a model variant for the first time, the current dashboard presents 18 visualizations in a single vertical scroll. This layout was designed when there were 4-5 plots; at 18, the workflow requires extensive scrolling to build a mental model of the variant's training dynamics.

The existing Visualization page is optimized for **deep-dive** analysis — per-epoch controls, neuron selection, configurable parameters. But the first-contact question is different: **"What's the shape of this model's training story?"** That question needs a dense, read-mostly layout showing cross-epoch summary views that reveal grokking timing, specialization trajectories, and weight-space structure at a glance.

### Current Workflow Pain

1. Load a variant → scroll through 18 plots, most of which show a single epoch snapshot
2. Mentally assemble the training narrative from scattered summary plots interspersed with per-epoch detail
3. Repeat for each variant when comparing

The summary information exists — it's just not arranged to answer the first-contact question efficiently.

### Framing: Lenses

A **lens** is an opinionated analytical perspective on the same underlying data. Each lens determines which visualizations to show, how they're arranged, and what's juxtaposed. The Summary Lens is the first lens — "show me the shape of this model." Future lenses (Cross-Variant, Per-Input Trace) would follow the same pattern: a dedicated page with a purpose-specific layout consuming shared renderers and data.

## Design

### New Dashboard Page: `/summary`

Add a Summary Lens page to the dashboard navigation. The page shows a single variant's training story in a dense, read-mostly layout. Every visualization is either cross-epoch or shows the full training trajectory. The only interactive control beyond variant selection is a **temporal cursor** — a lightweight epoch indicator that cross-references all time-axis plots simultaneously.

### Navigation Integration

Add "Summary" to the existing navbar (from REQ_040):

```
Visualization | Summary | Training | Analysis Run
```

The Summary page shares the family/variant selection pattern with the Visualization page. A minimal sidebar or top-bar selector for family + variant, plus an epoch slider for the temporal cursor. No neuron slider, attention position pair, trajectory group, or other deep-dive controls.

### Layout

The layout is organized by network component, with visualizations arranged to maximize information density while preserving readability. All heights are initial values — adjust during implementation if the content doesn't fit well.

```
┌─────────────────────────────────────────────────────┐
│ Train/Test Loss Curve                    (full width)│
├─────────────────────────────────────────────────────┤
│ Embedding Fourier Coefficients           (full width)│
├─────────────────────────────────────────────────────┤
│ Neuron Specialization   │ Attention Head             │
│ Over Training           │ Specialization             │
│ (wider, ~60%)           │ Over Training (~40%)       │
├─────────────────────────────────────────────────────┤
│ Specialized Neurons by Frequency         (full width)│
├─────────────────────────────────────────────────────┤
│ Attention Dominant Frequencies           (full width)│
├─────────────────────────────────────────────────────┤
│ Parameter Trajectory 3D                  (full width)│
├───────────────┬─────────────┬───────────────────────┤
│ PC1 vs PC2    │ PC1 vs PC3  │ PC2 vs PC3            │
│ (1/3)         │ (1/3)       │ (1/3)                 │
├─────────────────────────────────────────────────────┤
│ Component Velocity      │ Effective Dimensionality   │
│ (50%)                   │ (50%)                      │
└─────────────────────────────────────────────────────┘
```

### Visualization Inventory

| # | Visualization | Renderer | Analyzer | Data Pattern | Exists |
|---|---|---|---|---|---|
| 1 | Train/Test Loss Curve | `render_loss_curves_with_indicator` | N/A (metadata) | dashboard component | Yes |
| 2 | Embedding Fourier Coefficients by Epoch | `render_dominant_frequencies_over_time` | `dominant_frequencies` | cross_epoch | Yes |
| 3 | Neuron Specialization Over Training | `render_specialization_trajectory` | `neuron_freq_norm` | summary | Yes |
| 4 | Specialized Neurons by Frequency | `render_specialization_by_frequency` | `neuron_freq_norm` | summary | Yes |
| 5 | Attention Head Specialization Over Training | `render_attention_specialization_trajectory` | `attention_freq` | summary | Yes |
| 6 | Attention Dominant Frequencies | `render_attention_dominant_frequencies` | `attention_freq` | summary | Yes |
| 7 | Parameter Trajectory 3D | `render_trajectory_3d` | `parameter_trajectory` | cross_epoch_pca | Yes |
| 8 | Parameter Trajectory PC1/PC2 | `render_parameter_trajectory` | `parameter_trajectory` | cross_epoch_pca | Yes |
| 9 | Parameter Trajectory PC1/PC3 | `render_trajectory_pc1_pc3` | `parameter_trajectory` | cross_epoch_pca | Yes |
| 10 | Parameter Trajectory PC2/PC3 | `render_trajectory_pc2_pc3` | `parameter_trajectory` | cross_epoch_pca | Yes |
| 11 | Component Velocity | `render_component_velocity` | `parameter_trajectory` | cross_epoch_component_velocity | Yes |
| 12 | Effective Dimensionality | `render_dimensionality_trajectory` | `effective_dimensionality` | summary | Yes |

**All 12 visualizations use existing renderers and analyzers.** No new analyzers are needed for the initial implementation.

### Deferred Visualizations

The following were identified in the original Summary Lens sketch but require new analyzers:

- **Embedding Specialization Over Training** — needs a summary metric from `dominant_frequencies` (how many frequencies dominate in `W_E` over time)
- **Unembedding Specialization Over Training** — needs a new `W_U` Fourier analyzer
- **Unembedding Fourier Coefficients by Epoch** — needs the same `W_U` analyzer

These are separate requirements (new analyzers). The Summary Lens page should be designed so these can be added as slots when the analyzers exist, but they are not blockers for this requirement.

### Temporal Cursor (Epoch Indicator)

The Summary Lens includes a lightweight epoch slider that controls a **temporal cursor** — a vertical line annotation shown on every time-axis visualization simultaneously. This enables cross-referencing: "the specialization inflection happened here, and at that same moment, the dimensionality was here, and the velocity was here."

**What it does:**
- Draws a vertical line at the selected epoch on all time-axis plots (loss curve, specialization trajectories, embedding Fourier, attention specialization, attention dominant frequencies, component velocity, effective dimensionality)
- Updates the epoch marker on trajectory PCA plots (the existing renderers already accept a `current_epoch` parameter that highlights the corresponding point)

**What it does not do:**
- Load new data. All data is loaded once on variant selection.
- Re-render plots. The cursor update is a lightweight Plotly figure patch (add/move a `vline` shape), not a full re-render.
- Control per-epoch visualizations. There are no per-epoch plots on this page — no neuron heatmap, no frequency snapshot, no attention pattern.

**Implementation note:** The existing summary renderers (e.g., `render_specialization_trajectory`, `render_dimensionality_trajectory`) already accept a `current_epoch` parameter and draw a vertical indicator line. The trajectory PCA renderers accept `current_epoch` to highlight a point. This means the cursor behavior is already built into the renderers — the Summary page just needs to pass the slider value through.

### Data Loading

The Summary page needs to load data from multiple analyzers for a single variant. The existing patterns handle this:

- **Summary data**: `ArtifactLoader.load_summary(analyzer_name)` — for specialization trajectories, dimensionality
- **Cross-epoch data**: `ArtifactLoader.load_epochs(analyzer_name)` — for dominant frequencies over time
- **Cross-epoch precomputed**: `ArtifactLoader.load_cross_epoch(analyzer_name)` — for PCA trajectory
- **Loss data**: loaded directly from the variant's `metadata.json`

All data loads once when the variant is selected and doesn't change until a new variant is chosen. The epoch slider only updates the temporal cursor (vertical line position), not the underlying data. This makes the page inherently faster than the Visualization page (which re-renders per-epoch plots on every slider change).

### Rendering Strategy

All renderers already return `plotly.graph_objects.Figure`. The Summary page calls each renderer and places the resulting figure in the appropriate grid cell. No new rendering code is needed — just wiring.

For the trajectory plots that take a `group` parameter, use `"all"` (the default group). The Visualization page offers a group selector; the Summary page intentionally does not — it shows the all-parameters view.

The epoch slider value is passed as `current_epoch` to all renderers that accept it. On slider change, figures are re-rendered with the updated epoch (the renderers are fast since no data loading occurs). If Plotly patching (`figure.add_vline` / `Patch()`) proves simpler than full re-render for the cursor update, prefer that approach.

## Scope

**This requirement covers:**
1. New `/summary` page in dashboard_v2
2. Navigation integration (add "Summary" to navbar)
3. Layout implementation (12 visualizations in dense grid)
4. Family/variant selector for the Summary page
5. Data loading for all 12 visualizations on variant selection
6. Temporal cursor (epoch slider + synchronized vertical indicator across all time-axis plots)

**This requirement does not cover:**
- New analyzers (embedding specialization, unembedding Fourier)
- Click-to-navigate interactions (clicking a point on a plot to set the epoch cursor, or to jump to the Visualization page)
- Cross-variant comparison (separate requirement)
- Changes to the existing Visualization page
- Lens framework abstraction (this is the first lens; framework extraction happens when the second lens arrives)

## Conditions of Satisfaction

### Page Structure
- [ ] `/summary` route exists and is accessible from the navbar
- [ ] Page displays family and variant selection controls
- [ ] Page displays an epoch slider
- [ ] Selecting a variant loads and displays all 12 summary visualizations

### Layout
- [ ] Visualizations are arranged in a dense grid (not single-column vertical scroll)
- [ ] Neuron Specialization and Attention Head Specialization appear side-by-side
- [ ] Three secondary PCA trajectory plots appear in a single row
- [ ] Component Velocity and Effective Dimensionality appear side-by-side
- [ ] Layout is readable without horizontal scrolling at standard desktop widths (1280px+)

### Temporal Cursor
- [ ] Epoch slider controls a synchronized vertical indicator line across all time-axis plots
- [ ] Trajectory PCA plots highlight the corresponding epoch point when the slider changes
- [ ] Cursor update does not trigger new data loading (lightweight figure update only)

### Data
- [ ] All visualizations render correctly for any variant with complete artifacts
- [ ] Variant selection loads data once; epoch slider only updates the temporal cursor
- [ ] Page handles variants with missing analyzers gracefully (shows placeholder, not crash)

### Integration
- [ ] Existing Visualization page is unchanged
- [ ] Existing tests pass
- [ ] Navigation between Summary and other pages works correctly

## Constraints

**Must have:**
- Dense, read-mostly layout — this is the core value proposition
- All 12 existing visualizations rendering correctly
- Family/variant selection
- Temporal cursor (epoch slider with synchronized indicator across all time-axis plots)

**Must avoid:**
- Per-epoch data-loading controls (neuron slider, attention pair, trajectory group — those belong on the Visualization page)
- Premature abstraction of a "lens framework" (build the concrete page first)
- New analyzer work (use what exists)

**Flexible:**
- Exact proportions and heights of grid cells
- Whether family/variant selection is a sidebar, top bar, or inline dropdowns
- Whether the trajectory group defaults to "all" or provides a simple toggle
- Visual styling details

## Decision Log

| Date | Question | Decision | Rationale |
|------|----------|----------|-----------|
| 2026-02-15 | Include per-epoch visualizations? | No | Summary Lens answers "what's the shape?" — per-epoch detail belongs on the Visualization page |
| 2026-02-15 | Include epoch indicator? | Yes — temporal cursor | All summary renderers already accept `current_epoch`; lightweight slider enables cross-referencing events across plots without loading new data |
| 2026-02-15 | Block on new analyzers (unembedding)? | No | Ship with 12 existing visualizations; add slots for future analyzers |
| 2026-02-15 | Abstract a lens framework now? | No | Build the concrete page; extract the pattern when the second lens (Cross-Variant) arrives |
| 2026-02-15 | Include click-to-navigate? | No — future enhancement | Valuable interaction but adds callback complexity; get the layout right first |

## Notes

**Evolution path.** This page is designed to become the first instance of a lens framework. When the Cross-Variant Lens requirement lands, common patterns (variant selection, layout primitives, data loading) can be extracted. But premature abstraction before the second lens exists would be speculative.

**Static export companion.** The existing `export_variant_visualization()` infrastructure can generate a static HTML or image grid version of this lens with minimal work. This could be a lightweight follow-on task (not part of this requirement) that enables sharing Summary Lens views without running the dashboard.

**Deferred visualization slots.** The layout should accommodate future additions (embedding/unembedding rows) without requiring a layout redesign. Leaving a commented placeholder or an expandable section for the embedding/unembedding pair would make future integration smoother.
