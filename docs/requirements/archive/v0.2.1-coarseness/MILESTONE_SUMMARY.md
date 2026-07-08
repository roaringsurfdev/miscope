# v0.2.1 — Coarseness Analysis

**Released:** 2026-02-08
**Version:** 0.2.1

## Summary

This release adds the summary statistics infrastructure and the coarseness analysis pipeline — the first research-driven analysis added to the foundational architecture. Coarseness quantifies blob vs plaid neuron patterns, enabling researchers to track the emergence of low-frequency neuron specialization during grokking.

## Requirements

| Requirement | Description | Key Files |
|-------------|-------------|-----------|
| [REQ_022](REQ_022_family_summary_statistics.md) | Family-specific summary statistics infrastructure | `analysis/protocols.py`, `analysis/pipeline.py`, `analysis/artifact_loader.py` |
| [REQ_023](REQ_023_coarseness_analyzer.md) | Coarseness analyzer with per-epoch artifacts and summary stats | `analysis/analyzers/coarseness.py`, `analysis/library/fourier.py` |
| [REQ_024](REQ_024_coarseness_visualizations.md) | Coarseness visualizations (trajectory, distribution, by-neuron, blob count) | `visualization/renderers/coarseness.py`, `dashboard/app.py` |

## Key Decisions

- **Summary statistics as optional Analyzer protocol extension:** `get_summary_keys()` / `compute_summary()` — backward compatible, no changes to existing analyzers required
- **Single summary.npz per analyzer:** Cross-epoch access pattern needs all values at once; small enough for a single file
- **Inline computation:** Summary stats computed during analysis pass, not as post-processing
- **Modular Fourier approach for coarseness:** Consistent with existing analysis model; composes from tested library functions
- **Conditional dashboard panels:** Coarseness visualizations only appear when artifacts exist
- **Checkpoint-only resolution:** Summary stats plotted as discrete markers, not interpolated lines, to preserve scientific integrity

## Research Capability Added

- Mean coarseness trajectory with percentile bands across training
- Per-epoch coarseness distribution histograms (slider-driven)
- Blob neuron count tracking over training
- Per-neuron coarseness visualization
- All eight trained variants (4 primes x 2 seeds) analyzed with coarseness
