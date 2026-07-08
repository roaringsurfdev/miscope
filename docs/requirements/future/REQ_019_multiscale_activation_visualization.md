# REQ_019: Multi-Scale Activation Visualization

## Problem Statement

The existing neuron activation and attention pattern visualizations show fine-grained periodic structure (113×113 heatmaps), but this can obscure coarser spatial organization. When viewed at lower resolution (e.g., via bilinear interpolation to 28×28), distinct "blob" regions emerge that reveal where neurons/heads are responsive vs. quiet across the input space.

This multi-scale structure appears in both:
- **Neuron activations**: Some neurons show clear blob regions at low resolution, others retain grid patterns
- **Attention patterns**: Different heads specialize - some show coarse blob structure (heads 0, 3), others retain high-frequency patterns (heads 1, 2)

Understanding when this multi-scale organization emerges during training could reveal insights about how the model learns modular arithmetic.

## Origin

Discovered accidentally when viewing the "First 5 neuron activations" visualization in a squished VSCode window. The browser's bilinear interpolation revealed structure that wasn't apparent at full resolution. Validated experimentally in `ModuloAdditionRefactored.py` (experiment cells for downsampling).

## Conditions of Satisfaction

1. **Analyzer**: New `DownsampledActivationsAnalyzer` computes bilinear-interpolated versions of:
   - Neuron activations (first N neurons, configurable)
   - Attention patterns per head (to 'a' and 'b' positions)
   - At target resolution(s) (default: 28×28, optionally 56×56)

2. **Artifact**: `downsampled_activations.npz` containing:
   - `neuron_acts_downsampled`: shape (num_neurons, target_size, target_size)
   - `attn_to_a_downsampled`: shape (num_heads, target_size, target_size)
   - `attn_to_b_downsampled`: shape (num_heads, target_size, target_size)
   - Metadata: target_size, num_neurons, source resolution (p)

3. **Renderer**: Visualizations with faceted heatmaps showing:
   - Downsampled neuron activations (grid of neurons)
   - Downsampled attention patterns (grid of heads, separate for 'a' and 'b')

4. **Dashboard Integration**:
   - New visualization panel(s) accessible via epoch slider
   - Can observe emergence of multi-scale structure over training

## Constraints

- Must use existing dependencies (torch `F.interpolate` with bilinear mode)
- Should follow existing analyzer/renderer patterns for consistency
- Artifact size should be reasonable (downsampled data is much smaller than full resolution)

## Context & Assumptions

- Primary value is in training dynamics - seeing when coarse vs. fine structure emerges
- The bilinear interpolation acts as a low-pass filter revealing low-frequency organization
- Different neurons/heads may specialize at different scales - this visualization helps identify that
- This is exploratory - value will be validated through use

## Technical Notes

Working prototype in `ModuloAdditionRefactored.py`:
```python
# Bilinear interpolation approach (validated)
neuron_acts_2d = einops.rearrange(neuron_acts[:, :5], "(a b) neuron -> neuron a b", a=p, b=p)
neuron_acts_4d = neuron_acts_2d.unsqueeze(1).float()
interpolated = F.interpolate(neuron_acts_4d, size=(28, 28), mode="bilinear", align_corners=False)
```

Alternative approaches tested (for reference):
- Average pooling (einops reduce, torch avg_pool2d) - retains grid pattern
- Blur-then-sample - works but bilinear is cleaner

## Priority

Medium - Adds new analytical capability aligned with project goals

## Effort

Medium - Follows established patterns, main work is analyzer + renderer implementation

## References

- Experimental screenshots in `temp_artifacts/` (session 2026-02-02)
- Related to existing `NeuronActivationsAnalyzer` and activation heatmap renderer
