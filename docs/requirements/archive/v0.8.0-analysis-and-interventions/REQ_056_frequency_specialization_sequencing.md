# REQ_056: Frequency Specialization Sequencing and Threshold Analysis

**Status:** Active
**Priority:** High
**Related:** REQ_052 (Fourier Quality Scoring), REQ_055 (Attention Head Phase Analysis)

## Problem Statement

Neuron specialization is currently viewed at a binary threshold: a neuron is "specialized" if it meets a fixed commitment criterion (>90% of its Fourier norm concentrated in one frequency). This captures the endpoint of competition but obscures the process.

Two questions cannot be answered with the current views:

1. **What happens at intermediate thresholds?** At 30%, 50%, 80% commitment, are neurons already organizing into frequency bands? Does the transition have a soft onset or a sharp threshold crossing?

2. **In what order do frequency bands commit?** Does the model develop lower frequencies first (consistent with literature suggesting models learn low frequencies before high), or do all bands move together? For anomalous variants, does the missing frequency band (p59/485: no frequency 15) fail to emerge from the start, or does it appear briefly and lose the competition?

The data to answer both questions already exists in neuron_dynamics cross-epoch artifacts. This requirement is primarily about exposing and visualizing that data at the right granularity.

## Conditions of Satisfaction

### Adjustable Threshold Views

- [ ] Existing neuron specialization views support an adjustable commitment threshold parameter (e.g., 30%, 50%, 70%, 90%)
- [ ] At each threshold, the view shows: number of neurons meeting the threshold per frequency band, per epoch
- [ ] Threshold can be adjusted interactively (dashboard slider or notebook parameter) without re-running analysis

### Per-Band Onset Timeline

- [ ] For each frequency band present in a variant, extract from neuron_dynamics: the epoch at which the first neuron commits at threshold t%, and the epoch at which the band reaches 25%, 50%, and peak neuron count
- [ ] Visualize band onset timelines as a gantt-style or stacked timeline view — which bands emerge first, which lag, which never fully develop
- [ ] Align onset timeline against grokking onset (test loss drop) and parameter velocity spike

### Anomaly Diagnostics

- [ ] For p101/999: show the frequency 5 → frequency 13 transition at adjustable thresholds — at what threshold does frequency 5 appear and when does it disappear?
- [ ] For p59/485: show that frequency band 15 never reaches meaningful neuron count at any threshold across training
- [ ] Compare p101/999 and p59/485 onset timelines against a healthy reference variant

### Data Access

- [ ] Per-band onset timing data exposed as a dataview following REQ_054 pattern — consumers should not need to re-derive onset epochs from raw neuron_dynamics data

## Constraints

**Must:**
- Threshold adjustment must operate on existing stored neuron_dynamics data — no re-running of the analyzer
- Band onset dataview must be derivable at read time from existing cross-epoch data, not stored separately

**Must not:**
- Add a new analysis pass just to support threshold views — the data is already there

**Explicitly deferred:**
- Per-neuron trajectory tracing (which specific neurons switch and when) — this is per-input trace territory, a separate scope
- Attention head threshold analysis (covered in REQ_055)

## Context & Assumptions

The neuron_dynamics cross-epoch artifact tracks, for each neuron at each epoch, the dominant frequency and the concentration metric (how much of the neuron's Fourier norm is in that frequency). The specialization threshold currently used (>90%) is a display convention that collapses the full distribution to a binary.

The "low frequencies first" pattern is documented in the ML literature and appears to be visible in this dataset from visual inspection. Formalizing this into a threshold-parameterized view would make the pattern measurable rather than anecdotal.

For p101/999, the frequency 5 → frequency 13 transition is the key event to characterize. Adjustable thresholds would show: at what concentration level does frequency 5 first appear? At what level does it vanish? Does it fail slowly (declining concentration over many epochs) or abruptly (present at 80%, gone next epoch)? The answer constrains the mechanism — gradual decline suggests gradient competition, abrupt loss suggests amplitude lock-in.

## Notes

**Implementation approach:**
- The concentration metric per neuron per epoch is already stored in neuron_dynamics artifacts (or derivable from stored Fourier coefficients)
- Threshold parameter should default to current value (90%) so existing views are unchanged
- Per-band onset dataview: iterate over epochs in neuron_dynamics cross-epoch data, find first epoch where each band reaches threshold — pure aggregation, no new artifacts

**Key visualization targets:**
- Animated or slider-controlled specialization heatmap with adjustable threshold
- Band onset timeline chart (bands as rows, epochs as x-axis, opacity/color encoding commitment level)
- Side-by-side comparison of p101/999, p59/485, and a healthy reference on band onset timing
