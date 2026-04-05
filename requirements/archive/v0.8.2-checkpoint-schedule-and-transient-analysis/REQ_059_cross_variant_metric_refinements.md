# REQ_059: Cross-Variant Metric Refinements

**Status:** Future — accumulating observations, not yet ready for active implementation
**Depends on:** REQ_057 (cross-variant comparison), REQ_058 (band concentration)
**Attribution:** Engineering Claude

---

## Problem

As cross-variant exploration matures (REQ_057 + REQ_058), several metric definitions have been found to be
incomplete or imprecise. This requirement tracks refinements — changes that are small in scope but
meaningful for research accuracy. Items here are candidates for a future sprint once the research
direction stabilizes enough to know which metrics are load-bearing.

---

## Backlog Items

### Item A: `max_circularity` alongside `final_circularity`

**Current behavior:** `compute_variant_metrics()` records `final_circularity` — the circularity value at
the last available epoch.

**Problem:** For pathological variants (e.g., p101/999), circularity peaks transiently then collapses.
`final_circularity` misses the peak entirely — and the peak is exactly what distinguishes "found the
algorithm but couldn't hold it" from "never found it."

**Proposed addition:** Add `max_circularity_resid_post` and `max_circularity_attn_out` as separate
columns (per-site). This avoids combining sites into a single value prematurely — the relative timing
of when each site peaks may itself be diagnostic.

**Open questions:**
- Is the site where max is achieved (resid_post vs attn_out) a meaningful classifier?
- Should we also record the epoch at which max occurs (`max_circularity_epoch`)?

---

### Item B: Derivative-based grokking onset

**Current behavior:** `grokking_onset_epoch` is the first epoch where `test_loss < threshold` (default 0.1).

**Problem:** This is a state-based definition. For anomalous variants that approach the threshold
gradually, the "onset" epoch is any of dozens of epochs where loss slowly drifts below 0.1. It doesn't
identify the *event* — the inflection point where loss is actively falling steeply.

**Proposed addition:** Add `grokking_event_epoch` — the epoch of maximum rate of test loss decrease,
i.e., `argmax(loss[t-1] - loss[t])` over a smoothing window. A candidate threshold: loss drop > 1.0
per epoch over a short window (e.g., 5 epochs). This is derivative-based, so it's not predictive —
it can only be measured post-hoc — but it is mechanistically cleaner than the threshold crossing.

**Relationship to `critical_mass_epoch`:** `critical_mass_epoch` (REQ_058) is a weight-space predictor
of grokking; `grokking_event_epoch` would be its activation-space confirmation. Comparing the two
across variants may validate the predictive power of `critical_mass_epoch`.

**Open questions:**
- What smoothing window best isolates the sharp event vs noise? (candidate: 5-epoch rolling mean)
- Should the threshold (>1.0 drop/epoch) be a `ClassificationRules` parameter?
- Keep both definitions (`grokking_onset_epoch` for backward compat, `grokking_event_epoch` as new)?

---

### Item C: Neuron specialization health metric (noted in ClassificationRules)

**From user notes in `ClassificationRules`:** "Do neuron specialization counts rise together or out of
balance?" — this is partially addressed by `slope_cv` in REQ_058. May be fully satisfied once
REQ_058 metrics are validated against the full variant set.

**Status:** Monitor — may not need separate treatment.

---

### Item D: Neuron specialization diversity metric (noted in ClassificationRules)

**From user notes in `ClassificationRules`:** "Is there enough frequency mix (low/mid/high)? This
probably boils down to: is there a low-frequency band?"

**Proposed:** A boolean flag or count for whether any committed neurons belong to the low-frequency
band (frequencies 0 to n_freq/3). Absence of a low-frequency band is a strong pathology signal
(p59/485 missing frequency 15 entirely).

**Status:** May be partially captured by `midpoint_active_band_count` and the per-band breakdown
in the critical mass snapshot. Evaluate after REQ_058 data is explored.

---

## Notes

- Items A and B are the highest priority — both affect the correctness of the primary classification
  signal (`failure_mode`).
- Items C and D may resolve naturally once REQ_058 metrics are explored against real data.
- All items are additive (new columns, no breaking changes to existing columns).
- When promoted to active, these should be bundled into a single small PR to avoid churn on
  `cross_variant.py` and its tests.
