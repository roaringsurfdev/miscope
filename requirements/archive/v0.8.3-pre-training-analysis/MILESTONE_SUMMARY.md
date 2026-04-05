# v0.8.3 — Pre-Training Analysis

**Released:** 2026-04-05
**Branch:** develop → main

## Key Decisions

- **Viability Certificate is analytical, not empirical.** All metrics derive from the Fourier centroid structure. The only empirical input is the W_E participation ratio at the dimensionality crossover epoch, loaded from existing artifacts. No model weights required.
- **Aliasing risk is a ceiling, not just a signal.** A frequency set with max aliasing risk > 0.80 is fragile even when its min pairwise distance looks adequate. Calibrated against three known corpus cases.
- **Ideal set search is pre-computed.** Exhaustive search over subsets of {1,…,(p-1)/2} is tractable (worst case ~3.8M subsets for p=127, size=5) but takes minutes. All 29 corpus (prime, size) pairs are pre-computed in `ideal_frequency_sets.json` and loaded at module init to survive app restarts.
- **REQ_053 closed via input_trace.** The per-class accuracy infrastructure was implemented as part of REQ_075 (input_trace analyzer). The remaining gap was only `frequency_quality_vs_accuracy`; the single-epoch bar chart by residue was superseded by the more informative timeline view.
- **Context bar frequency off-by-one bug found.** The committed frequencies display was double-incrementing (the registry already stores 1-indexed k values; the context bar was adding +1 again). Found while investigating a Viability Certificate result that didn't match the multi-stream display.

## Files Added

- `src/miscope/analysis/viability_certificate.py` — core geometry module
- `dashboard/pages/viability_certificate.py` — dashboard page
- `dashboard/pages/initialization_sweep.py` — dashboard page (REQ_085, prior milestone)
- `notebooks/viability_certificate_calibration.py` — calibration notebook
- `scripts/precompute_ideal_sets.py` — one-time exhaustive search script
- `model_families/modulo_addition_1layer/ideal_frequency_sets.json` — 29 pre-computed entries
- `dashboard/pages/variant_table.py` — variant table page (REQ_082)
- `dashboard/components/export_panel.py` — plot export panel (REQ_079)

## Empirical Findings Enabled

- p101/s485/ds999: aliasing_failure (k=50 at Nyquist) — 7% accuracy regression post-grokking; loses mid-frequency k=28 during second descent
- p113/s485/ds999: viable but near warning threshold (k=40, risk=0.714) — 2% accuracy regression; k=36 and k=40 share identical hard pair periodicity (period=3), creating correlated aliasing pressure
- Regression severity tracks aliasing risk continuously, not just at the binary threshold

## Active Requirements (not in this release)

- REQ_055 — Attention head phase analysis
- REQ_073 — Weight-space DMD
- REQ_075 — Per-input prediction trace (complete but archived in v0.8.2)
- REQ_081 — Structural training diagnostics
