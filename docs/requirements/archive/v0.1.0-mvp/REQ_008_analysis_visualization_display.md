# REQ_008: Analysis Execution and Visualization Display

## Problem Statement
After a training run completes, we need to trigger analysis and display the resulting visualizations in the dashboard. Currently, analysis requires running separate scripts manually.

The workbench should allow selecting a trained model, running analysis, and viewing the three MVP visualizations in one location with a synchronized checkpoint control to explore training dynamics holistically.

## Conditions of Satisfaction
- [x] Interface to select a trained model (path or dropdown)
- [x] Button to trigger analysis execution
- [x] Progress indication during analysis
- [x] Display area for three visualizations:
  - Dominant embedding frequencies (REQ_004)
  - Activation heatmaps (REQ_005)
  - Neuron frequency clusters (REQ_006)
- [x] **Global checkpoint slider** that synchronizes all three visualizations
- [x] Moving slider updates all visualizations to same checkpoint/epoch
- [x] Visualizations are interactive (Plotly features work)
- [x] Can re-run analysis or switch to different trained model
- [x] Neuron selector for activation heatmap (Stage 2 historical view from REQ_005)

## Constraints
**Must have:**
- Integrates with AnalysisPipeline (REQ_003)
- Displays artifacts from REQ_004, REQ_005, REQ_006
- Gradio framework for UI
- Synchronized checkpoint slider controlling all three visualizations
- All three visualizations visible without excessive scrolling (or easily accessible)
- Slider changes load from artifacts (fast, no recomputation)

**Must avoid:**
- Requiring manual file path entry (should be user-friendly)
- Losing visualization state when switching between models
- Unresponsive UI during analysis execution
- Recomputing artifacts when slider moves

**Flexible:**
- Layout of visualizations (tabs, accordion, grid, single column)
- Whether visualizations update live during analysis or appear when complete
- Controls for visualization parameters (color scales, thresholds, etc.)
- Exact UI controls for neuron selection

## Context & Assumptions
- Analysis runs after training completes (not real-time with training)
- Visualizations are Plotly figures that can be embedded in Gradio
- Analysis may take time depending on number of checkpoints
- Artifact-based architecture enables fast slider updates (just load different checkpoint data)
- Synchronized slider enables holistic exploration: see how frequencies, activations, and clusters evolve together
- User may want to compare visualizations from different training runs
- Assumption: Viewing one training run's analysis at a time is acceptable for MVP
- Assumption: Single synchronized slider is sufficient for MVP (per-viz sliders deferred)

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Select a completed training run from dashboard
- Trigger analysis
- Three visualizations appear in dashboard
- **Global checkpoint slider** is visible and responsive
- Moving slider updates all three visualizations to same epoch
- Slider movement is smooth and fast (loads from artifacts, no lag)
- Can observe correlations: "When frequencies emerge at epoch X, what happens to neuron clusters?"
- Visualizations are interactive (zoom, hover, etc.)
- Can select specific neurons in activation heatmap (REQ_005 Stage 2)
- Changing neuron selection doesn't affect other visualizations
- Can interpret dominant frequencies, activation patterns, and neuron clusters holistically
- Switching to different trained model updates visualizations correctly

---
## Notes

**Post-MVP enhancements:**
- Independent sliders per visualization with link/unlink toggle
- Side-by-side comparison of multiple training runs
- Bookmark/annotation of specific epochs of interest
- Export visualizations as static images or animations

**Implementation consideration:**
- Gradio supports single slider controlling multiple outputs via `.change()` events
- Artifact-based architecture makes synchronized updates fast and responsive

## Implementation Notes (Added by Claude)

**Implementation completed:** 2026-01-31

**Key code location:**
- `dashboard/app.py` - Main Gradio application (Analysis tab)
- `dashboard/state.py` - DashboardState for artifact caching
- `dashboard/utils.py` - Model discovery utilities

**Analysis Tab UI:**
- Model dropdown with "Refresh Models" button
- "Run Analysis" button triggers AnalysisPipeline
- Global epoch slider controls all 4 visualizations
- Neuron index slider for activation heatmap
- Layout: Loss curves (top), Freq + Activations (row), Clusters (wide bottom)

**Synchronized slider implementation:**
```python
epoch_slider.change(
    fn=update_visualizations,
    inputs=[epoch_slider, neuron_slider, state],
    outputs=[loss_plot, freq_plot, activation_plot, clusters_plot, epoch_display, state]
)
```

**State management:**
- Artifacts cached in DashboardState on model selection
- Slider updates read from cached numpy arrays (fast)
- Model config and losses loaded from metadata.json

**Integration:**
- Uses `ArtifactLoader` to load analysis artifacts
- Uses visualization renderers from REQ_004-006
- Triggers `AnalysisPipeline.run()` on analysis button
