# REQ_042: Neuron Frequency Trajectory Visualization

**Status:** Draft
**Priority:** Medium (research insight — reveals subnetwork competition dynamics)
**Dependencies:** Existing `neuron_freq_norm` analyzer and artifact pipeline
**Last Updated:** 2026-02-15

## Problem Statement

The current neuron frequency specialization visualizations show either a single-epoch snapshot (the freq_clusters heatmap) or aggregate summary statistics (specialized neuron counts over time). Neither reveals the **per-neuron temporal dynamics** — whether individual neurons switch frequency allegiance during training, when they commit to their final frequency, or whether commitment cascades during grokking.

Prototype analysis (notebooks/neuron_freq_trajectory.py) reveals that ~90% of neurons switch their dominant frequency at least once during training, with over half switching 3+ times. This "thrashing" is direct evidence of subnetwork competition and is invisible in current visualizations.

### Questions This Visualization Answers

1. **Do neurons thrash between frequencies or commit monotonically?** (They thrash — heavily.)
2. **Does commitment cascade at grokking?** (Appears to — collision visible at grokking epoch.)
3. **Do neurons that end up in the same frequency cluster commit at similar times?**
4. **Which neurons are the most contested — recruited and released by competing circuits?**

## Design

### 1. Trajectory Heatmap Renderer

A new renderer: `render_neuron_freq_trajectory()`

- **Y-axis:** Neuron index (0 to d_mlp)
- **X-axis:** Epoch
- **Color:** Dominant frequency (the frequency with highest frac_explained)
- **Grey/transparent:** Neurons below an uncommitted threshold (default: 3× uniform baseline, i.e., `3 / (p // 2)`)
- **Colorscale:** HSL hue rotation across frequency count, muted saturation (0.5) and moderate lightness (0.55) to avoid retinal assault

Two sort modes:
- **Natural order:** Neuron index as-is
- **Sorted by final frequency:** Group neurons by their final dominant frequency, secondary sort by frac_explained descending. This reveals cluster structure.

### 2. Summary Metrics (computed by existing analyzer at summary time)

Extend the `neuron_freq_norm` summary with additional fields:

| Field | Shape | Description |
|-------|-------|-------------|
| `switch_count` | `(d_mlp,)` | Times each neuron changes dominant frequency (above threshold) |
| `commitment_epoch` | `(d_mlp,)` | Epoch at which neuron locks into its final frequency (NaN if uncommitted) |

These are derived from per-epoch `norm_matrix` data that already exists — no new analyzer needed.

### 3. Neuron Activation Animation (for top thrashers)

A utility function that generates an animated GIF of a single neuron's activation heatmap across epochs, using the existing `render_neuron_heatmap` renderer. This shows whether frequency switches correspond to smooth transitions or abrupt phase changes in the activation pattern.

- Input: variant, neuron index, epoch list (or "all")
- Output: animated GIF or sequential figure list
- Leverages existing `_fig_to_pil` and `_save_gif` helpers in `export.py`

## Conditions of Satisfaction

1. `render_neuron_freq_trajectory()` produces a Plotly heatmap with neuron×epoch layout, colored by dominant frequency, with uncommitted neurons masked
2. Sorted view groups neurons by final dominant frequency
3. Uncommitted threshold defaults to `3 / (p // 2)` (3× uniform baseline), configurable
4. Colorscale uses muted HSL hue rotation (saturation ≤ 0.5)
5. Summary computation adds `switch_count` and `commitment_epoch` fields to `neuron_freq_norm` summary
6. Neuron activation animation function generates frame sequence for a given neuron across epochs
7. Renderer is registered in the visualization export registry
8. Dashboard integration: own page (`/neuron-dynamics`) following the lens pattern from REQ_041, with variant selector and no epoch slider (inherently cross-epoch)
9. Page layout: trajectory heatmap with natural/sorted toggle, commitment timeline histogram, switch count distribution

## Constraints

- No new analyzer — all data derives from existing `neuron_freq_norm` per-epoch artifacts
- Renderer follows existing API conventions (returns `go.Figure`)
- Animation export uses existing GIF infrastructure in `export.py`
- Summary field additions must be backward-compatible (old summaries without these fields should not break the dashboard)
- Page follows existing dashboard navigation patterns (sidebar link, shared family/variant selector)

## Decision Authority

- **Claude decides:** renderer implementation details, colorscale tuning, sort algorithm, page layout specifics
- **Discuss first:** any changes to analyzer compute

## Notes

- The prototype notebook lives at `notebooks/neuron_freq_trajectory.py` — use as reference implementation
- The color intensity difference between variants (e.g., p=101 neon vs. p=113 muted) reflects genuine structural differences: fewer dominant frequencies → larger blocks of maximally-contrasting colors. The muted colorscale mitigates this but doesn't eliminate the information.
- Non-visual readouts (switch counts, commitment stats, top thrashers) are deferred to a separate UI requirement. For now these are notebook-only outputs.
