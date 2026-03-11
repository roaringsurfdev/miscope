# REQ_066: Multi-Stream Frequency Specialization Trajectory

**Status:** Active
**Priority:** High
**Related:** REQ_056 (Frequency Specialization Sequencing), REQ_055 (Attention Head Phase Analysis), REQ_052 (Fourier Quality Scoring)
**Last Updated:** 2026-03-10

## Problem Statement

The platform can show how neurons specialize into frequency bands over training (Per-Band Neuron Specialization), and has several views into attention head behavior. What it cannot show is how the three computational streams — embeddings, attention heads, and MLPs — accumulate frequency specialization in parallel and in relation to each other.

Understanding the relative timing of specialization across streams is the core open question: does attention head commitment to a frequency precede, follow, or co-emerge with MLP neuron mass in that frequency? Does the embedding space organize before either? The answer bears directly on whether attention acts as a signal booster that enables MLP specialization, or whether MLP mass drives attention routing.

Three separate views already exist for pieces of this picture. No single view lets you compare the timing and trajectory of all three streams on the same axis.

## Conditions of Satisfaction

### Three-Panel View: `multi_stream_specialization`

- [ ] A new cross-epoch view `multi_stream_specialization` is registered in the view catalog
- [ ] The view renders three vertically stacked panels sharing a common x-axis (epochs) and consistent per-frequency color coding
- [ ] The set of frequencies shown is derived from the variant's dominant frequencies (via existing `dominant_frequencies` artifacts); frequencies absent from the model's final-state specialization are excluded by default
- [ ] The view is accessible via `variant.at(epoch).view("multi_stream_specialization")`

### Panel 1 — MLP Neuron Specialization

- [ ] Y-axis: committed neuron count per frequency (same metric as existing Per-Band Neuron Specialization)
- [ ] Commitment threshold is adjustable via a slider (default: 50%)
- [ ] Computation reuses existing neuron_dynamics cross-epoch data — no new analysis pass

### Panel 2 — Attention Head Aggregate Commitment

- [ ] Y-axis: aggregate head commitment per frequency, defined as the mean QK^T Fourier fraction across all 4 heads for that frequency — a continuous [0, 1] signal
- [ ] No threshold applied; the aggregate is plotted as a raw trajectory
- [ ] Computed at render time from `parameter_snapshot` artifacts (W_Q, W_K per head per epoch): construct QK^T in the Fourier basis, extract the per-frequency fraction per head, average across heads
- [ ] The continuous framing reflects that the 4-head bottleneck makes per-head counts less meaningful than aggregate commitment

### Panel 3 — Embedding Dimension Specialization

- [ ] Y-axis: count of d_model dimensions whose Fourier power (along the vocab axis) is concentrated above threshold in a given frequency
- [ ] For each d_model dimension j and frequency k: power fraction = |Σ_n W_E[n,j] · cos(2πkn/p)|² / Σ_{k'} |...|²
- [ ] Commitment threshold is adjustable via a separate slider (default: 50%)
- [ ] Computed at render time from `parameter_snapshot` artifacts (W_E per epoch) — no new analysis pass

### Panel 4 — Effective Dimensionality

- [ ] Y-axis: Effective Dimensionality curves for W_E, W_in, W_out, W_O (same computation as existing Effective Dimensionality view)
- [ ] Reuses existing Effective Dimensionality data from `parameter_snapshot` — no new computation
- [ ] Placed as the bottom panel to read "context" against the three specialization panels above

### Controls

- [ ] Shared epoch range slider applies to all four panels
- [ ] Shared epoch cursor: clicking or scrubbing on any panel sets a vertical cursor line across all four panels simultaneously (following existing dashboard cursor pattern)
- [ ] MLP threshold slider and Embedding threshold slider are independent — they do not affect each other or the Attention panel
- [ ] All four panels update together when epoch range changes

### Validation

- [ ] View renders correctly for p=113, seed=999 (healthy variant with known timing structure)
- [ ] For the healthy variant, the attention panel shows H2 beginning to accumulate in freq 9 before MLP neurons commit to freq 9 — consistent with the known observation from the QK^T Fourier heatmap sequence
- [ ] The Effective Dimensionality panel aligns the W_O/W_out crossover with the second wave in the MLP panel — the timing relationship should be visible without cross-referencing a separate view
- [ ] Notebook cell demonstrates the view for at least one healthy and one anomalous variant

## Constraints

**Must:**
- All three panels computed from existing artifacts (`neuron_dynamics`, `parameter_snapshot`, `dominant_frequencies`) — no new analyzer required
- Attention panel is continuous [0, 1], not count-based — the 4-head architecture makes threshold counting misleading
- Per-frequency colors must be consistent across all three panels so the eye can track a single frequency vertically

**Must not:**
- Lock the MLP and Embedding thresholds together — the ability to sweep them independently is research-critical for testing whether observed timing is threshold-stable
- Apply a threshold to the Attention panel — the aggregate fraction is already a meaningful continuous signal
- Use a single normalized "specialization currency" across streams — each stream has its own natural unit

**Decision authority:**
- **Resolved:** Attention metric is mean QK^T Fourier fraction across heads (not max, not count-over-threshold). Mean reflects group commitment regardless of which head carries it. Known tradeoff: H2 at 100% freq 9 + others at 0% reads identically to all four heads at 25%. The heads appear to distribute load rather than lock independently, so this conflation is acceptable for the timing question.
- **Resolved:** Embedding metric is per-d_model-dimension Fourier power fraction (the direct analog of per-neuron Fourier power in MLPs), not per-token alignment. The d_model dimension is the unit of specialization, not the input token.
- **Resolved:** A fourth panel showing Effective Dimensionality curves (W_E/W_in/W_out/W_O trajectories) is included. The primary research hypothesis links attention head commitment to a frequency with the first inflection in Effective Dimensionality. Placing this in the same renderer makes the timing relationship directly visible and makes the combined view shareable with Research Claude without cross-referencing separate visualization files. Loss curves are intentionally excluded — the existing loss curve graph serves that alignment role and the dashboard's multi-graph layout already supports showing them together.
- **Resolved:** All four panels share an epoch cursor (following existing dashboard cursor pattern) — clicking any panel sets the cursor position across all panels.
- **Claude decides:** Layout proportions across the four panels; exact color palette

## Context

The research motivation: observing from the QK^T Fourier heatmap sequence for p=113/seed=999 that Head 2 begins accumulating specialization in freq 9 around epoch 5000–8500, well before MLP neurons commit to freq 9 (which starts ~epoch 9000). By the time the MLP second wave begins (~epoch 12000), Head 2 is already near-fully committed to freq 9, and freq 55 then appears simultaneously in both the attention heads and the MLP second wave. This suggests attention head specialization may be a prerequisite for — or at minimum a leading indicator of — MLP mass accumulation in a frequency.

The embedding panel is exploratory. Whether embeddings lead, lag, or co-emerge with the other streams is unknown and is a primary research question this view is designed to answer.

The three-panel view intentionally uses independent thresholds because the structural robustness of the MLP threshold sweep (10%–90% shows the same winner frequencies) does not guarantee that the same stability holds for the embedding dimension metric, which has not yet been examined at any threshold.

## Notes

- The attention aggregate metric (mean QK^T fraction) naturally handles the case where specialization is spread across multiple heads vs. concentrated in one — a model where all 4 heads are at 25% commitment and one where a single head is at 100% will read the same (both = 25% aggregate). Whether that conflation matters for the timing question is an open research question.
- Freq 9 is observed as an early mover in the MLP view; the attention panel should make visible whether this is because attention committed to freq 9 first, or whether neuron mass and attention commitment arrive together
- The embedding panel may reveal very early organization (pre-epoch 5000) that neither the MLP nor attention views can show — this is the unexplored region in the current visualization suite
- If loss curve overlay (panel 4) is added, align it with the second descent to make timing comparisons unambiguous
