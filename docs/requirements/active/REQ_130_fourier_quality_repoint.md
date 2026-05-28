# REQ_130: Re-point `fourier_frequency_quality` off `dominant_frequencies` (unblock retirement)

**Status:** Stub — *problem statement; CoS to be developed.*
**Priority:** Medium — unblocks a REQ_102 deferral (the only surviving consumer of `dominant_frequencies`).
**Branch:** TBD
**Dependencies:**
- REQ_102 (Analyzer Deprecation — deferred the `dominant_frequencies` retirement specifically because this consumer blocks it).
- REQ_118 (neuron_grouping — the proposed new source).

**Attribution:** Engineering Claude (stubbed 2026-05-28, from the REQ_102 analyzer-dependency audit).

---

## Problem

`dominant_frequencies` is retired in spirit (REQ_126 absorbed its weight-side Fourier into `weight_basis_projection`) but **cannot be deleted** because `fourier_frequency_quality` still consumes it: `depends_on = "dominant_frequencies"` + `ArtifactInput("dominant_frequencies", scope="epoch")` (secondary analyzer). It is the sole remaining consumer.

The fix is to re-point `fourier_frequency_quality` from `dominant_frequencies` to **`neuron_grouping`** (per user). If feasible, that frees `dominant_frequencies` for retirement under REQ_102.

## Research context (shapes the approach)

Per the user's assessment (see [[frequency-choice-frame]]): `fourier_frequency_quality` is **not currently yielding valuable information** — itself a finding. It sits at the hard, unresolved question of whether model performance depends on *which* frequencies are chosen and *when*; the literature's "frequencies follow from the prime / group math" frame is not supported by the data here. Consequently:

- **Bit-wise parity is likely unnecessary.** This is not a mechanical port like `weight_spectra`; the analyzer will need close re-examination of what it should measure. Treat the re-point as an opportunity to reconsider the metric, not just swap the input source.

## Conditions of Satisfaction

*(Deferred — stub. Develop in a dedicated session: define what `neuron_grouping` provides vs. what `fourier_frequency_quality` needs; decide whether to preserve the current metric or redefine it; then re-point, validate, and hand the `dominant_frequencies` retirement back to REQ_102.)*

## Notes

- **Closure feeds REQ_102.** When `fourier_frequency_quality` no longer reads `dominant_frequencies` and no other consumer remains, REQ_102 retires `dominant_frequencies` (→ `weight_basis_projection`, embedding site).
- **Sibling, not bundled:** the `neuron_freq_clusters`/`neuron_freq_norm` consumer migration is a *separate* follow-up (candidate REQ_131), best sequenced after REQ_128 so its backbone consumers (`neuron_dynamics`, `freq_group_weight_geometry`, `neuron_group_pca`) re-point to the granular `activation_basis_projection` with selective/lazy loading rather than re-importing whole-artifact memory pressure.
