# REQ_086: Viability Certificate

**Status:** Complete
**Priority:** High
**Branch:** feature/req-086-viability-certificate
**Menu:** Pre-Training Analysis (existing menu item, new tab)
**Attribution:** Drafted by Research Claude, written by Engineering Claude

---

## Problem Statement

When a model selects a frequency set during training, the quality of that selection determines whether generalization is achievable — but the link between frequency choice and geometric outcome is not obvious from the frequencies alone. A frequency set that looks reasonable may be aliasing-prone, geometrically fragile under the compression the model actually undergoes, or categorically worse than an ideal alternative that uses different frequencies.

The existing platform instruments *what happened* during training: which frequencies were selected, when commitment occurred, whether neurons specialized. This tool asks a complementary question: *given that the model selected this frequency set, is the destination geometrically viable?*

The question is analytical, not empirical. The centroid geometry for any (prime, frequency_set) combination can be computed exactly. The aliasing risk per frequency is closed-form. The only empirical input is the effective dimensionality at second descent — the compression constraint the model actually faced.

Research Claude developed the conceptual framework and identified failure modes in an earlier prototype; this requirement builds on those learnings. The calibration notebook (Phase 1) must validate the metric definitions against known outcomes before the tool goes onto the dashboard (Phase 2).

---

## Conditions of Satisfaction

### Phase 1: Calibration Notebook

- [x] A notebook at `notebooks/viability_certificate_calibration.py` implements the core geometric computation and runs it on three known cases:
  - `p59/s999/d598`: healthy generalizer, frequencies {5, 15, 21}
  - `p59/s485/d598`: late grokker, frequencies {5, 21}
  - `p101/s999/d598`: late grokker / aliasing failure, frequencies {35, 41, 43, 44}
- [x] For each case the notebook computes and displays:
  - **Separation under compression**: minimum pairwise centroid distance as a function of effective dimensionality, from d_model (128) down to 1, with the observed crossover PR marked
  - **Aliasing risk**: per frequency, k / ((p−1)/2); displayed as a bar chart
  - **Predicted hard pairs**: for each frequency k, residue class pairs separated by p/k steps — the pairs the geometry predicts will be hardest to separate
  - **Ideal frequency set**: minimum-cardinality subset of {1, …, (p−1)/2} that maximizes minimum pairwise centroid distance; computed by exhaustive search (tractable for p ≤ 127)
  - **Distance from ideal**: which frequencies in the actual set are not ideal, and which ideal frequencies are missing
- [x] The three cases produce clearly distinguishable metric profiles — if they don't, the metric definitions need revision before Phase 2
- [x] The notebook documents calibrated threshold recommendations for regime classification (compact torus / high-dim separation / aliasing failure) derived from the case data
- [x] The crossover PR values are loaded from actual variant artifacts — not approximated or hardcoded

### Phase 2: Dashboard Page

- [x] A new tab or section on the Pre-Training Analysis page provides the Viability Certificate tool
- [x] User can specify: prime, frequency set (comma-separated), d_model (defaults to 128), crossover PR (entered manually or loaded from a trained variant)
- [x] All five metrics from Phase 1 are displayed
- [x] Regime classification is shown with the calibrated thresholds from Phase 1
- [x] **Separation under compression** plot is the primary visualization: curve of min pairwise distance vs effective dimensionality, crossover PR marked, threshold band indicated
- [x] Key frequencies from the canonical set (if any trained variant exists for the prime) are available as a reference in the frequency input

---

## Constraints

**Must:**
- Calibrate against real observed outcomes before putting thresholds in the dashboard
- Load crossover PR from variant artifacts in Phase 1 (not from Research Claude's approximations — those need verification)
- Display aliasing risk per frequency, not just as a mean
- Treat aliasing risk as a ceiling on robustness: a set with high individual-frequency aliasing risk is not viable even if its separation margin looks adequate

**Must not:**
- Claim to predict whether a model will grok — the tool characterizes destination quality, not training dynamics
- Use hardcoded thresholds without calibration evidence
- Assume the ideal frequency set is arithmetically spaced (it must be computed)

**Flexible:**
- Whether comparison mode (two frequency sets side-by-side) is in v1 or deferred to a follow-on
- Whether the centroid geometry slider (compression applied visually) is in v1
- Exact regime classification labels and boundary values — the notebook determines these

---

## Architecture Notes

**Computation is purely analytical.** The centroid geometry for a given (prime, frequency_set) is exact:
```
c_r = [cos(2πkr/p), sin(2πkr/p) for k in F],  r = 0, …, p−1
```
No model weights are needed. The only empirical input is the crossover PR.

**Crossover PR source.** The effective dimensionality artifact (`artifacts/effective_dimensionality/`) contains per-epoch PR values. The crossover epoch is in `variant_registry.json` (`effective_dimensionality_cross_over_epoch`). Loading the PR value at that epoch from the artifact gives the compression constraint.

**Ideal set search.** Exhaustive search over subsets of {1, …, (p−1)/2}. For p=113, (p−1)/2=56; searching subsets of size ≤5 is tractable. Cache results per prime.

**Separation under compression.** SVD of the (p × 2|F|) centroid matrix embedded in d_model-dim space, then track minimum pairwise distance as singular values are zeroed from smallest to largest. The crossover PR maps to a specific number of retained dimensions via participation ratio formula.

**Aliasing risk.** For frequency k: `k / ((p−1)/2)`. Frequencies above 0.5 have aliasing period < 2 (pairs separated by 1 step become indistinguishable as frequency increases). Research Claude found this is a ceiling on robustness, not just a risk signal.

**Phase 1 → Phase 2 handoff.** The calibration notebook produces threshold values and notes documenting the regime classification rationale. Those values get hardcoded into the dashboard page, with a comment citing the notebook as the source.

---

## Notes

- Research Claude's prior prototype had a flaw: it placed idealized centroids in perfectly orthogonal subspaces, which caused participation ratio to mechanically equal 2|F|/d_model. This told you nothing. The real question is how the theoretical minimum (2|F| dimensions) compares to the *observed* crossover PR, and whether the frequency geometry survives compression to that PR.
- Research Claude labeled p59/s485 "partial failure." The registry labels it "late_grokker." This discrepancy should be resolved by the calibration notebook — the metric values will reveal whether p59/s485 is structurally closer to the healthy case or the aliasing failure case.
- The calibration notebook is publishable research in its own right. It makes a specific empirical claim: that these three geometric metrics predict outcome quality better than any single metric alone. The fieldnotes entry should follow once the results are in.
- REQ_085 (Initialization Gradient Sweep) surfaces *which frequencies get the first push*. REQ_086 surfaces *whether the destination those frequencies point toward is viable*. The two tools together cover the full pre-training analysis picture: gradient pressure at initialization → geometric quality of the selected destination.
