# v0.3.0 — Attention Head Specialization

**Released:** 2026-02-08
**Version:** 0.3.0

## Summary

This release adds attention head analysis alongside neuron specialization tracking. Attention patterns and their Fourier frequency decomposition reveal whether attention heads, like MLP neurons, become frequency specialists during grokking. Neuron frequency specialization now includes cross-epoch summary statistics showing how many neurons "lock in" to each frequency over training.

## Requirements

| Requirement | Description | Key Files |
|-------------|-------------|-----------|
| [REQ_025](REQ_025_attention_head_visualization.md) | Attention head pattern visualization with position pair selector | `analysis/analyzers/attention_patterns.py`, `visualization/renderers/attention_patterns.py` |
| [REQ_026](REQ_026_attention_freq_specialization.md) | Attention head frequency specialization (Fourier decomposition) | `analysis/analyzers/attention_freq.py`, `visualization/renderers/attention_freq.py` |
| [REQ_027](REQ_027_neuron_specialization_summary.md) | Neuron frequency specialization summary statistics and renderers | `analysis/analyzers/neuron_freq_clusters.py`, `visualization/renderers/neuron_freq_clusters.py` |
| [REQ_028](REQ_028_ui_improvements.md) | Variant dropdown sorting and default selection fix | `dashboard/components/family_selector.py`, `dashboard/app.py` |

## Key Decisions

- **Attention patterns stored as full tensor:** Enables flexible position pair selection without re-running analysis
- **Position pair as dropdown:** Six meaningful position pairs (= attending to a/b, b attending to a/b, etc.) rather than free-form indices
- **Reuse of Fourier library:** `AttentionFreqAnalyzer` composes from the same `compute_2d_fourier_transform()` and `compute_frequency_variance_fractions()` as neuron frequency analysis
- **Summary statistics as optional protocol extension:** `NeuronFreqClustersAnalyzer` gains summary methods without breaking existing analyzers
- **Specialization threshold as constructor parameter:** Default 0.9, configurable per-instance
- **Frequency range buckets as thirds:** Low/mid/high ranges split evenly across frequency indices

## Research Capability Added

- Attention head pattern visualization across all 4 heads with position pair selection
- Attention head frequency specialization heatmap and cross-epoch trajectory
- Neuron specialization trajectory (total + low/mid/high frequency counts over training)
- Per-frequency specialization heatmap over training epochs
- Sorted variant dropdown with explicit no-selection default

## Test Coverage

- 348 tests total (31 new for REQ_027, 40 new for REQ_026)
- Analyzer protocol conformance, output shapes, value ranges, known inputs
- Renderer output types, trace counts, epoch indicators, custom titles
