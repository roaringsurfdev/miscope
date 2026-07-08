# REQ_005: Activation Heat Maps Visualization

## Problem Statement
To understand what computational patterns neurons learn during training, we need to visualize neuron activation patterns as heatmaps over the input space (a, b).

For a modulo addition task with inputs (a, b, =), each neuron's activation can be visualized as a 2D heatmap where position (a, b) shows the activation strength. This reveals whether neurons are learning frequency-based patterns, positional patterns, or other structures.

The model has 512 MLP neurons, which is too many to show individually. We need a two-stage workflow:
1. **Explore final trained state**: Browse all neurons in the trained model to identify interesting patterns
2. **Historical deep-dive**: Select specific neurons of interest, view their evolution across all checkpoints

Computing activations requires expensive forward passes, so we need to generate activation artifacts once and reuse them for visualization iteration.

## Conditions of Satisfaction
- [x] Analysis computation generates artifact: neuron activations for all checkpoints
- [x] Artifact saved to disk (e.g., `neuron_activations.npz`)
- [x] Artifact structure: shape (n_checkpoints, d_mlp, p, p)
- [x] **Stage 1 - Explore trained model**: Browse all 512 neurons in final checkpoint
- [x] **Stage 2 - Historical view**: Select 3-5 neurons, view evolution across checkpoints
- [x] Heatmap visualization of neuron activations with (a, b) as axes
- [x] Activation values shown as color intensity
- [x] Interactive slider for checkpoint selection (historical view)
- [x] Neuron selector (dropdown or similar) for choosing which neurons to display
- [x] Visualization loads from artifact (no recomputation)

## Constraints
**Must have:**
- Activations computed from model forward pass on full dataset (done once, saved to artifact)
- Artifact persisted to disk for reuse
- Heatmap dimensions match input space (p x p)
- Plotly format for interactivity
- Works for arbitrary model size (not hardcoded to 512 neurons)
- Two-stage workflow: explore final state, then historical deep-dive

**Must avoid:**
- Showing all 512 neurons × N checkpoints simultaneously (overwhelming)
- Recomputing activations when changing visualization parameters
- Running forward passes during visualization rendering

**Flexible:**
- Neuron selection UI (dropdown, slider, grid of thumbnails)
- Layout (single heatmap, faceted grid, side-by-side comparison)
- Color scale and normalization strategy
- Whether to show multiple neurons simultaneously or one at a time
- Default neuron selection for initial view (first N, highest variance, etc.)

## Context & Assumptions
- Neuron activations available via cache: `cache["post", 0, "mlp"][:, -1, :]`
- Shape: (p², d_mlp) needs to be reshaped to (d_mlp, p, p) for visualization
- Original notebook shows first 5 neurons using faceted heatmaps
- Forward passes are expensive (run once per checkpoint during analysis, not during visualization)
- Activation artifacts enable fast iteration on visualization parameters
- Two-stage workflow: user explores trained model to find interesting neurons, then examines their training history
- Assumption: Interesting neurons show clear patterns (not just noise)
- Assumption: Viewing 3-5 neurons historically is sufficient for MVP
- Assumption: Final trained state provides good signal for identifying interesting neurons

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Run analysis on trained model
- Artifact file created and persisted to disk
- **Stage 1**: Can browse all 512 neurons in final trained checkpoint
- **Stage 1**: Identify 3-5 neurons with interesting patterns
- **Stage 2**: Select specific neurons for historical view
- **Stage 2**: Interactive slider shows neuron evolution across checkpoints
- Heatmap shows clear activation patterns (e.g., frequency-based stripes)
- Changing neuron selection or checkpoint doesn't trigger recomputation (loads from artifact)
- Can identify when selected neurons develop structured responses
- Artifact loads in Gradio dashboard with full interactivity

---
## Notes

## Implementation Notes (Added by Claude)

**Implementation completed:** 2026-01-31

**Key code location:**
- `visualization/renderers/neuron_activations.py`

**Functions implemented:**
- `render_neuron_heatmap(artifact, epoch_idx, neuron_idx, title, colorscale)` → go.Figure
  - Single neuron heatmap with RdBu colorscale centered at 0
  - Hover template showing (a, b) position and activation value
- `render_neuron_grid(artifact, epoch_idx, neuron_indices, cols, title)` → go.Figure
  - Grid of multiple neurons using subplots
  - Shared colorscale across all neurons for comparison
  - Configurable column count
- `render_neuron_across_epochs(artifact, neuron_idx, epoch_indices, cols)` → go.Figure
  - Single neuron shown across multiple epochs
  - Useful for tracking pattern development during training
- `get_most_active_neurons(artifact, epoch_idx, top_k)` → list[int]
  - Returns neurons with highest activation variance
  - Useful for identifying "interesting" neurons to display

**Design decisions:**
- RdBu colorscale centered at 0 for activation values
- Global color range across subplots for consistent comparison
- Hover tooltips with exact activation values
- Uses plotly.subplots.make_subplots for grid layouts
- Square aspect ratio maintained for (a, b) axes
