# REQ_027: Neuron Frequency Specialization Summary Statistics

## Summary

Extend the existing `NeuronFreqClustersAnalyzer` with summary statistics tracking how many neurons are "locked in" (specialized above a threshold) to each frequency over training. This enables cross-epoch visualization of specialization dynamics.

## Implementation

### Analyzer Extension

- **File:** `analysis/analyzers/neuron_freq_clusters.py`
- Added `__init__(specialization_threshold=0.9)` parameter
- Added `get_summary_keys()` and `compute_summary()` methods
- Summary statistics per epoch:
  - `specialized_count_per_freq`: `(n_freq,)` — neurons above threshold per frequency
  - `specialized_count_low/mid/high`: scalar — neurons in low/mid/high frequency ranges (thirds)
  - `specialized_count_total`: scalar — total specialized neurons
  - `mean_max_frac`: scalar — average max variance fraction across all neurons
  - `median_max_frac`: scalar — median max variance fraction

### Renderers

- **File:** `visualization/renderers/neuron_freq_clusters.py` (extended)
- `render_specialization_trajectory(summary_data, current_epoch)` — cross-epoch line plot with total/low/mid/high counts
- `render_specialization_by_frequency(summary_data, current_epoch)` — cross-epoch heatmap (frequencies x epochs) showing per-frequency specialist counts

### Dashboard Integration

- Two plots in a row between Frequency Clusters and Attention Patterns sections
- Both update on epoch slider changes with vertical epoch indicator

### Known Limitation

Previously-analyzed variants will not have `summary.npz` for `neuron_freq_norm` because the pipeline skips existing artifacts. Workaround: delete the analyzer's artifact directory and re-run analysis. A "recompute summaries" feature is tracked as a future requirement.

## Status: Complete
