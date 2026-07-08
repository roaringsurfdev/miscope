# REQ_004: Dominant Embedding Frequencies Visualization

## Problem Statement
To understand how the model learns to represent modular arithmetic during training, we need to visualize which Fourier frequency components become dominant in the embedding space over time.

The visualization should show how the norm of Fourier basis coefficients evolves across training checkpoints, helping identify when specific frequencies emerge as important features.

## Conditions of Satisfaction
- [x] Analysis computation generates artifact: Fourier coefficient norms per checkpoint
- [x] Artifact saved to disk (e.g., `fourier_coefficients.npz`)
- [x] Visualization loads from artifact (no recomputation)
- [x] Line graph showing `(fourier_bases @ W_E).norm(dim=-1)` for each checkpoint
- [x] X-axis: Fourier basis components (frequency indices or labels)
- [x] Y-axis: Normed coefficients
- [x] Interactive slider to scrub through checkpoints (epoch selector)
- [ ] Animation capability (auto-play through checkpoints) as additional feature
- [x] Uses FourierEvaluation utilities for computation
- [x] Visualization clearly shows dominant frequencies (threshold-based highlighting)
- [x] Threshold for dominance configurable in visualization (doesn't trigger recomputation)

## Constraints
**Must have:**
- Fourier bases defined by modulus (period = p)
- W_E is embedding weights (excluding the equals token)
- Plotly format for interactivity
- Interactive slider for checkpoint selection (primary interaction method)
- Generated as artifact consumable by dashboard

**Must avoid:**
- Hardcoded frequency indices (must work for arbitrary p)
- Hardcoded dominance threshold (must be configurable)
- Cluttered visualization (too many frequencies obscure the signal)

**Flexible:**
- Color scheme for highlighting dominant frequencies
- Whether to show all frequencies or filter to dominant ones
- Layout and styling details
- Whether animation auto-plays by default or requires user activation

## Context & Assumptions
- FourierEvaluation.get_fourier_bases() provides basis vectors
- FourierEvaluation.get_basis_coefficients() computes norms
- Current dominant basis threshold is 1.0 (will need more intelligent logic in future)
- Visualization should help identify key frequencies like those in key_freqs from original notebook
- Artifact structure: NumPy array shape (n_checkpoints, n_fourier_components)
- Changing visual threshold doesn't require regenerating artifact (computation vs. rendering separation)

**Research hypothesis to investigate:**
- Early training likely shows noise with no clear dominant frequencies
- Dominant frequencies emerge at some point during training
- Frequencies may shift/change before settling into final pattern
- Interactive slider enables manual exploration of this emergence pattern

**Assumptions:**
- Dominant frequencies emerge clearly enough to be visually identifiable
- Slider interaction provides better insight than pure animation for investigating emergence timing

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Run analysis on trained modulo addition model (p=113)
- Artifact file created and persisted to disk
- Visualization loads from artifact without running forward passes
- Interactive slider allows scrubbing through checkpoints smoothly
- Visualization updates responsively as slider moves
- Can identify when dominant frequencies first emerge
- Can observe if/when frequencies shift before settling
- Changing visualization threshold updates display immediately (no recomputation)
- Threshold highlighting clearly indicates which frequencies are dominant
- Changing p to different prime produces different but still interpretable visualization
- Artifact loads in Gradio dashboard with full interactivity

---
## Notes

**Post-MVP enhancements:**
- More intelligent threshold determination logic (instead of fixed threshold)
- Automatic grokking phase detection and visual indicators
- Ability to mark/annotate specific epochs of interest
- Comparative view across multiple training runs

## Implementation Notes (Added by Claude)

**Implementation completed:** 2026-01-31

**Key code location:**
- `visualization/renderers/dominant_frequencies.py`

**Functions implemented:**
- `render_dominant_frequencies(artifact, epoch_idx, threshold, highlight_dominant, title)` → go.Figure
  - Bar chart with Fourier component labels
  - Threshold line with annotation
  - Highlighting for dominant components
- `render_dominant_frequencies_over_time(artifact, component_indices, top_k, title)` → go.Figure
  - Line chart showing top components across epochs
- `get_dominant_indices(coefficients, threshold)` → list of indices above threshold
- `get_fourier_basis_names(n_components)` → list of component labels

**Design decisions:**
- Bar chart instead of line for cleaner discrete component display
- Threshold line with annotation for clear visual reference
- Opacity-based highlighting (full vs. faded) for dominant vs. non-dominant
- Hover templates showing component name and exact value
- Uses Plotly `go.Figure` for Gradio compatibility

**Animation note:** Animation capability deferred to Gradio dashboard integration (REQ_008)
