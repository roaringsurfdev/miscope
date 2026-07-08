# REQ_058: Neuron Band Concentration as a Grokking Health Predictor

**Status:** Active
**Priority:** High
**Area:** Analysis — Neuron Dynamics + Cross-Variant

---

## Problem Statement

Current cross-variant comparison (REQ_057) surfaces grokking onset, final loss, and failure mode labels. What it does not surface is *why* one variant groks cleanly and another fails — the structural difference during training that separates them.

Two observations motivate this requirement:

**Observation 1 (p101/999 — failure):** MLP neurons in higher frequency bands grew at disproportionate rates compared to frequency 5, which had early neuron commitment and initial attention head alignment. The volume of neurons in competing frequencies created a gradient imbalance that overwhelmed attention's ability to hold frequency 5. Embedding Fourier magnitudes told the opposite story — from embeddings, frequency 5 appeared dominant. The embedding-neuron disagreement is a diagnostic signal that current tooling doesn't surface.

**Observation 2 (p103/485 — healthy control):** Neuron counts across active frequency bands grew at similar slopes throughout training. The model gracefully lost one frequency (20), with QK^T updating ahead of V — the router adapted before the content encoder. The three remaining frequencies (6, 24, 40) absorbed the neurons cleanly. The result looks healthy from the outside and is healthy mechanistically.

**Hypothesis:** The distribution of committed neurons across frequency bands — specifically its concentration, growth balance across bands, and alignment with embedding Fourier structure — predicts grokking health better than any single scalar metric (grokking onset, final loss, or embedding dominant frequency) taken alone.

---

## Conditions of Satisfaction

### CoS 1: Per-Epoch Band Concentration Metric
Given a variant with a `neuron_dynamics` artifact, compute at each epoch:
- **Active band count**: number of distinct frequency bands with committed neurons above threshold
- **Concentration index**: normalized Herfindahl-Hirschman Index (HHI) over committed neuron counts across active bands. HHI = Σ(share_k²), where share_k = count_k / total_committed. Ranges from 1/n (uniform) to 1.0 (monopoly). Higher = more concentrated.
- **Max band share**: fraction of committed neurons in the single most-populated band.

Threshold is parameterizable (same convention as per_band_specialization).

### CoS 2: Embedding-Neuron Rank Alignment
Given a variant with both `neuron_dynamics` and `dominant_frequencies` artifacts, compute at each epoch:
- **Rank correlation**: Spearman rank correlation between (a) frequency band embedding Fourier magnitudes and (b) neuron counts per band. +1 = perfectly aligned, 0 = no relationship, negative = misaligned (as observed in p101/999).

Only bands with at least one committed neuron are included in the correlation.

### CoS 3: Cross-Variant Distribution Summary
Given a family, produce a per-variant summary table with these metrics extracted at two reference points:
- **Midpoint of training** (50% of total epochs)
- **Grokking onset** (using grokking_onset_epoch from REQ_057, if available)

Summary columns: `active_band_count`, `hhi`, `max_band_share`, `embedding_neuron_rank_correlation`, at each reference point.

### CoS 4: Band Growth Balance
Given a variant's neuron_dynamics data, estimate the slope of committed neuron growth for each active band in the pre-grokking window. Compute:
- **Slope CV (coefficient of variation)**: std dev of per-band growth slopes divided by mean. Low CV = balanced competition. High CV = runaway growth in one or more bands.

This discriminates the p101/999 pattern (high CV, imbalanced growth) from the p103/485 pattern (low CV, balanced growth).

### CoS 5: Critical Mass Snapshot

Given a variant's neuron_dynamics data and a neuron count threshold N (default: 100) and specialization threshold T (default: 0.75):
- Identify the **critical mass epoch**: the first epoch where total committed neurons ≥ N.
- At that epoch, extract: `active_band_count`, `hhi`, `max_band_share`, and `committed_per_freq` (count per frequency band).
- Return as a dict (single variant) or as additional columns in the cross-variant summary (CoS 3).

This snapshot captures the frequency distribution at the moment the gradient pull crosses the hypothesized tipping point. Healthy variants are expected to show balanced distributions at crossing; pathological variants concentrated ones.

### CoS 6: Visualizations
- **Concentration trajectory**: line chart of HHI over epochs for a single variant. Overlaid with grokking onset marker if available.
- **Rank alignment trajectory**: line chart of embedding-neuron rank correlation over epochs for a single variant.
- **Cross-variant scatter**: scatter plot of midpoint HHI vs. grokking_onset_epoch (one point per variant), colored by failure_mode from REQ_057.

### CoS 7: Tests
- HHI computed correctly for known distributions (uniform n bands = 1/n, monopoly = 1.0)
- Rank correlation returns +1 for identical rankings, -1 for reversed
- Slope CV returns 0 for equal slopes across all bands
- Cross-variant summary table has correct shape and column names
- Critical mass snapshot returns None when threshold N is never crossed; returns correct epoch and distribution when it is
- All visualizations return `go.Figure`

---

## Constraints

- **No new analyzer run required.** All metrics are derived from existing artifacts:
  - `neuron_dynamics` cross_epoch (dominant_freq, max_frac, epochs)
  - `dominant_frequencies` per-epoch (coefficients)
- Threshold for commitment must be parameterizable; default matches existing convention (3/n_freq)
- Metric functions must accept numpy arrays directly (usable from notebooks without variant objects)
- Cross-variant summary integrates with REQ_057 `load_family_comparison()` output — either as additional columns or a joinable DataFrame

---

## Decision Authority

- Which reference points to use for cross-variant summary: Claude's judgment
- Whether to extend `compute_variant_metrics()` or create a parallel function: Claude's judgment
- Visualization layout (combined or separate figures): Claude's judgment

---

## Notes

**On the QK-leads-V observation (p103/485):** QK^T concentration on {6, 24, 40} preceded V's loss of frequency 20 signal. This suggests QK is the active adaptor and V is the passive content carrier — a healthy sequence. Contrast with p101/999 where neuron volume (driving V gradient) outpaced and overwhelmed QK alignment. The *who leads* question (QK or V/neuron volume) may generalize as a failure mode discriminator. This is a stretch goal for this requirement — worth noting but not required for initial implementation.

**On the embedding-neuron misalignment in p101/999:** From embedding Fourier magnitudes, frequency 5 appeared dominant. From neuron specialization, higher frequencies dominated. This disconnect is a novel diagnostic. It suggests embedding-based analyses alone are insufficient for predicting grokking outcome — the computation graph (what neurons actually compute) matters more than the information available in the embeddings.

**Potential cross-family relevance:** If concentration and rank alignment generalize beyond p=113, these metrics could become universal health indicators applicable to any modular arithmetic family. Initial implementation is within the p=113 family; generalization is a future consideration.
