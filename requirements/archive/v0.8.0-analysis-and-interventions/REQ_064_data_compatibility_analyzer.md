# REQ_064: Fourier Data Compatibility Analyzer

**Status:** Active
**Priority:** Medium
**Related:** REQ_063 (Fourier Nucleation Predictor), REQ_061 (Data Seed as Domain Parameter)
**Drafted by:** Research Claude (fourier-nucleation-spec.md, Component 2). Scoped and edited by Engineering Claude.
**Last Updated:** 2026-03-08

---

## Problem Statement

REQ_063 established that MLP initialization has a structural bias toward certain Fourier frequencies — the initialization "wants" certain frequencies. But this is only half the question. Whether the model can follow through on that bias depends on what the training data can actually teach.

The training set is a 30% subsample of all p² (a, b) pairs, selected by a random permutation seeded by `data_seed`. Different data seeds produce structurally different training sets. For a given frequency k, the training set may provide excellent, uniform coverage of the phase `2πk·(a+b)/p` across all residues — or it may systematically over- or under-represent certain phase regions. A frequency that is poorly covered on the training set is harder to learn from gradient descent regardless of how strongly initialization is biased toward it.

This requirement implements a measure of **data compatibility**: for each Fourier frequency k, how well does the training set support learning that frequency? The key metric is derived from the 2×2 Gram matrix of the restricted Fourier basis on the training set — its condition number measures how balanced the cos and sin components of frequency k are represented in the training data.

Combined with REQ_063 output, this enables the overlap analysis: which frequencies are *both* loud at initialization *and* well-conditioned on the training data? Those are the predicted nucleation winners.

The splitting logic is deterministic and reproducible: `torch.manual_seed(data_seed)` followed by `torch.randperm(p*p)`, taking the first `floor(p² × 0.3)` as training. This matches the exact code in the training pipeline and requires no model weights.

---

## Conditions of Satisfaction

### 1. Computation module: `data_compatibility`

- A new module `src/miscope/analysis/data_compatibility.py` implements the per-frequency compatibility computation
- The module is self-contained: it takes only `prime`, `data_seed`, and optionally `training_fraction` (default 0.3) — no model, no checkpoint
- The splitting logic matches the training pipeline exactly: `torch.manual_seed(data_seed)` + `torch.randperm(p*p)`, first `int(p*p*training_fraction)` indices are training

**Per-frequency computation:**

For each frequency k in {1..floor(p/2)}, given the set of training (a, b) pairs:

1. Compute `s = (a + b) % p` for each training pair
2. Compute the 2×2 Gram matrix G of the restricted Fourier basis on the training set:
   ```
   G = [[Σ cos²(2πk·s/p), Σ cos·sin],
        [Σ cos·sin,        Σ sin²(2πk·s/p)]]
   ```
3. **Condition number**: ratio of max to min eigenvalue of G. A value of 1.0 means cos and sin are perfectly balanced (uniform phase coverage). High values mean the training set is phase-biased at this frequency.
4. **Phase uniformity**: entropy of the histogram of `(2πk·s/p) mod 2π` values across training pairs, normalized by maximum entropy (log of bin count). 1.0 = perfectly uniform. Default: 20 bins.
5. **Compatibility score**: `0.5 × condition_score + 0.5 × phase_uniformity`, where `condition_score = 1 / (1 + log10(max(1, condition_number)))`. Range [0, 1], higher = better supported.

**Output dict** (returned by `compute_data_compatibility(prime, data_seed, training_fraction)`):
- `frequencies`: int array shape (n_freqs,), values 1..p//2
- `condition_number`: float array shape (n_freqs,)
- `condition_score`: float array shape (n_freqs,)
- `phase_uniformity`: float array shape (n_freqs,)
- `compatibility_score`: float array shape (n_freqs,)
- Scalar fields: `prime`, `data_seed`, `training_fraction`, `n_training_pairs`

### 2. Views

**Architecture note:** Data compatibility is computed on demand from variant params — not from stored artifacts. The view's `load_data` calls `compute_data_compatibility(prime, data_seed)` directly. This is correct: the computation takes ~1ms and storing artifacts would add infrastructure for no benefit.

Two views registered in the universal view catalog:

**`analysis.data_compatibility.spectrum`**:
- Bar chart of compatibility score per frequency (x=k, y=compatibility)
- Color gradient: low compatibility = muted, high = highlighted
- Secondary axis or overlay showing phase_uniformity and condition_score separately, so the two components of the composite score are visible
- Title includes prime and data_seed so the view is self-describing

**`analysis.data_compatibility.overlap`**:
- Combined view overlaying data compatibility with nucleation init energy from REQ_063
- X-axis: frequencies
- Bars: init energy (from epoch-0 nucleation artifact, if available; hidden/empty if not)
- Line overlay: compatibility score
- Color coding: overlap score = init_energy × compatibility
- Annotation: flags frequencies in the top quartile of overlap score as predicted nucleation winners
- Degrades gracefully if the nucleation artifact is not present (shows compatibility only)

### 3. Dashboard integration

- Both views appear on the Neuron Dynamics page, below the existing nucleation views
- The overlap view is the primary one — it synthesizes REQ_063 and REQ_064
- Views render without any epoch slider interaction (they are epoch-independent)

### 4. Tests

- Unit: `compute_data_compatibility` returns correct shapes and field names
- Unit: condition number = 1.0 when training set is the full p×p grid (perfect coverage)
- Unit: compatibility score in [0, 1] for all frequencies
- Unit: splitting reproduces training pipeline exactly — given same prime and data_seed, the set of (a, b) pairs in the training split matches what the family's `generate_train_test_split` produces
- Integration: views render without errors against a real variant

---

## Constraints

- The splitting logic must match the training pipeline **exactly**. Use `torch.manual_seed(data_seed)` + `torch.randperm(p*p)` + `[:int(p*p*training_fraction)]`. Do not reimplement with numpy or a different PRNG.
- No artifact storage — compute on demand. If this becomes a performance issue (unlikely), caching can be added later.
- `training_fraction` defaults to 0.3 and does not need to be user-configurable in the dashboard for now.
- The overlap view should handle the missing-nucleation case gracefully. Don't error if `fourier_nucleation` epoch-0 artifact is absent.
- Compatibility score is a composite metric, not a ground truth. The components (condition number, phase uniformity) should remain visible in the view so the user can inspect what's driving the composite.

---

## Notes

- **Why Gram matrix condition number?** The 2×2 Gram matrix captures whether the training set can distinguish the cos and sin components of frequency k. If both are well-represented (condition number ≈ 1), gradient descent can independently tune both. If one component dominates (high condition number), the model effectively has only one degree of freedom at that frequency and may learn a misaligned phase.
- **Why phase uniformity as a second metric?** The condition number is algebraic (eigenvalue ratio) but doesn't capture non-uniformities that average out — e.g., two clusters of phases at opposite ends. The phase histogram entropy catches this.
- **The testable prediction from combining REQ_063 + REQ_064:** Frequencies with high overlap score (init energy × compatibility) should be the ones that specialize most neurons in the trained model. For 109/485: frequency 4 should have both high init energy and high compatibility. For 101/999: the init-predicted frequencies (30, 14, 25) should have low compatibility, and the actual specialization frequencies (35, 41, 43, 44) may have intermediate or also low compatibility.
- **Follow-on:** A cross-variant summary view — for each variant in a family, compute the top-3 overlap frequencies and display as a table alongside actual neuron specialization counts. This would allow rapid scanning across all 12+ variants.
