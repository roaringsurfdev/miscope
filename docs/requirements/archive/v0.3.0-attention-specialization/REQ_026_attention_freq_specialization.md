# REQ_026: Attention Head Frequency Specialization

## Summary

Compute Fourier frequency decomposition of attention patterns per head, analogous to `NeuronFreqClustersAnalyzer` for MLP neurons. Tests the hypothesis that attention heads become frequency specialists during grokking.

## Implementation

### Analyzer: `AttentionFreqAnalyzer`

- **File:** `analysis/analyzers/attention_freq.py`
- **Name:** `attention_freq`
- Extracts attention pattern for a configurable position pair (default: `= -> a`)
- Reshapes to `(n_heads, p, p)` grid, applies 2D Fourier transform
- Computes per-frequency variance fractions: `freq_matrix: (n_freq, n_heads)`
- Summary statistics: `dominant_freq_per_head`, `max_frac_per_head`, `mean_specialization`

### Renderers

- **File:** `visualization/renderers/attention_freq.py`
- `render_attention_freq_heatmap(epoch_data, epoch)` — per-epoch heatmap (frequencies x heads)
- `render_attention_specialization_trajectory(summary_data, current_epoch)` — cross-epoch line plot (one line per head)
- `render_attention_dominant_frequencies(summary_data, current_epoch)` — cross-epoch step plot of dominant frequency per head

### Dashboard Integration

- Two plots in a row: frequency heatmap (per-epoch) + specialization trajectory (cross-epoch)
- Both update on epoch slider changes

### Key Reused Functions

- `compute_2d_fourier_transform()`, `compute_frequency_variance_fractions()` from `analysis/library/fourier.py`
- `extract_attention_patterns()` from `analysis/library/activations.py`

## Status: Complete
