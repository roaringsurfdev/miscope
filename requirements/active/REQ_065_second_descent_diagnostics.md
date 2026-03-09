# REQ_065: Second Descent Diagnostics

**Status:** Active
**Priority:** High
**Related:** REQ_057 (Cross-Variant Health Comparison), REQ_058 (Band Concentration), REQ_063 (Fourier Nucleation Predictor)
**Last Updated:** 2026-03-09

---

## Problem Statement

The current `compute_variant_metrics` and `classify_failure_mode` framework treats second descent as a binary milestone: either a model grokked (test loss crossed a threshold) or it didn't. This is insufficient for the failure modes now observed in the data.

A model can enter second descent — defined as the test loss falling rapidly from its plateau — and fail to survive it. 113/999/999 is a concrete example: test loss descends 80%+ from peak, then climbs back, ending with high test loss. This is not the same as "never grokked" nor as "late grokker." The current classifier has no category for it.

Additionally, the dynamics that determine whether second descent succeeds are visible before and during the descent:
- Which frequency achieved first-mover advantage (first significant neuron mass)?
- Was the frequency portfolio band-diverse at descent onset?
- Did the model sustain or lose Fourier alignment in the MLP/residual stream during descent?
- What was the maximum circularity achieved before descent?

These are pre-descent and in-descent indicators that the current metrics don't capture. Adding them enables a richer classification and surfaces the causal chain: init bias → first-mover frequency → portfolio composition → descent survivability.

---

## Conditions of Satisfaction

### 1. Second descent detection in `compute_variant_metrics`

Add the following fields to the dict returned by `compute_variant_metrics`:

**Test loss trajectory:**
- `peak_test_loss`: float — maximum test loss across all training epochs
- `peak_test_loss_epoch`: int — epoch at which peak test loss occurred
- `second_descent_onset_epoch`: int | None — first epoch where `descent_fraction >= 0.8`, where `descent_fraction = (peak_test_loss - test_loss) / peak_test_loss`
- `second_descent_survived`: bool | None — True if `final_test_loss <= rules.grokking_threshold` AND `second_descent_onset_epoch` is not None
- `post_descent_recovery`: bool | None — True if test loss climbed back by more than `rules.recovery_threshold` (default 0.2 × peak) after `second_descent_onset_epoch`

**Frequency portfolio at second descent onset:**
- `descent_onset_frequency_bands`: list[str] | None — band classification of each active frequency at `second_descent_onset_epoch`. Band thresholds: low = k ≤ p//4, high = k > 3*p//8, mid = between. Requires `neuron_dynamics` artifact.
- `descent_onset_has_low_band`: bool | None — True if at least one low-band frequency was active at descent onset
- `descent_onset_band_count`: int | None — number of distinct bands represented at descent onset

**First-mover frequency:**
- `first_mover_frequency`: int | None — first frequency to reach `rules.first_mover_neuron_threshold` neurons (default: 20% of d_mlp)
- `first_mover_epoch`: int | None — epoch at which first-mover threshold was first crossed
- `first_mover_band`: str | None — "low", "mid", or "high" classification of the first-mover frequency
- `first_mover_survived`: bool | None — True if the first-mover frequency was still active in the final trained model

### 2. `ClassificationRules` additions

Add the following fields to `ClassificationRules`:

```python
second_descent_threshold: float = 0.8    # descent_fraction to qualify as second descent
recovery_threshold: float = 0.2          # fraction of peak loss re-gained to flag recovery
first_mover_neuron_threshold: float = 0.2  # fraction of d_mlp to qualify as first-mover
```

### 3. New failure mode in `classify_failure_mode`

Add `degraded_recovery` category:

- **degraded_recovery**: `second_descent_onset_epoch` is not None AND `post_descent_recovery` is True AND `final_test_loss > rules.grokking_threshold`

The full classification order (earlier checks take priority):
1. `no_grokking`: never crossed `grokking_threshold`
2. `degraded_recovery`: entered second descent, test loss climbed back
3. `degraded`: high final test loss, never properly descended
4. `late_grokker`: grokked but past `late_grokking_epoch`
5. `healthy`: grokked on time, final loss acceptable

### 4. `load_family_comparison` includes new metrics

The DataFrame produced by `load_family_comparison` should include all new fields. No new columns should be silently dropped. NaN for missing artifacts is correct behavior.

### 5. Tests

- Unit: `second_descent_onset_epoch` is None when test loss never descends 80% from peak
- Unit: `post_descent_recovery` is True for a synthetic loss series that descends then climbs
- Unit: `post_descent_recovery` is False for a series that descends cleanly
- Unit: `first_mover_frequency` returns the correct frequency for a synthetic `dominant_freq`/`max_frac` array
- Unit: `classify_failure_mode` returns `degraded_recovery` for metrics with `second_descent_onset_epoch` set and `post_descent_recovery=True`
- Unit: `classify_failure_mode` ordering — `degraded_recovery` takes priority over `degraded`
- Integration: `load_family_comparison` runs without error on the real family and includes all new fields

---

## Constraints

- Computation must remain cheap: all new metrics derive from existing artifacts (`metadata["test_losses"]`, `neuron_dynamics` cross_epoch). No new analyzers required.
- `second_descent_onset_epoch` uses `peak_test_loss_epoch` as the reference, not epoch 0. This correctly handles models where training loss oscillates early.
- Band thresholds (low/mid/high) use the `prime` from the variant — they are relative to p, not absolute frequency values.
- `first_mover_band` and `descent_onset_*` fields are None when `neuron_dynamics` artifact is absent. Graceful degradation is required.
- Do not break the existing `load_family_comparison` output for variants that are missing `neuron_dynamics` artifacts.

---

## Notes

- **Why `descent_fraction >= 0.8` for second descent onset?** This captures the steep phase of the descent, not just the beginning of a slow drift. A model with 80% of its plateau-to-zero drop complete is unambiguously in second descent.
- **Why track `first_mover_survived`?** The anomalous models show a pattern where the first-mover frequency is high-band and eventually wins — but the model struggles. If `first_mover_survived = True` and `first_mover_band = "high"` correlates with `post_descent_recovery = True`, that is a testable prediction from the first-mover hypothesis.
- **`degraded_recovery` classification rationale:** 113/999/999 entered second descent, showing the model is capable of the generalization computation, but lost stability. This is mechanistically distinct from a model that never entered descent. The distinction matters for understanding what went wrong.
- **Circularity before descent** was identified as a potential predictor (`pre_descent_max_circularity`) but deferred — it requires the `repr_geometry` summary to be aligned with `second_descent_onset_epoch`, which adds complexity. Marked as a follow-on.
- **MLP vs. attention alignment drop timing** (alignment drop epoch vs. commitment timeline) is a related follow-on. The foundation for it is `second_descent_onset_epoch` as a reference point.
