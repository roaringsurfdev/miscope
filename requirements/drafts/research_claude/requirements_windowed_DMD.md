# REQ_003: Windowed DMD Eigenvalue Analysis

## Problem Statement
The current DMD implementation computes a single set of eigenvalues across all checkpoints, which averages out local dynamical features — the very transitions and frequency commitments we're trying to detect. We need a windowed (time-resolved) DMD that computes local eigenvalue spectra at each phase of training, revealing how the dominant dynamical modes evolve.

This is motivated by a specific hypothesis: during grokking, the DMD eigenvalues should transition from real-dominated (pure growth/decay during memorization) to complex conjugate pairs at specific angles (oscillatory modes corresponding to learned Fourier frequencies). The timing and character of this transition in eigenvalue space should provide a principled, quantitative signature of the grokking window that complements the residual norm analysis from REQ_002.

## Conditions of Satisfaction

### Core computation
- [ ] Implement sliding-window DMD over checkpoint sequences for each model
- [ ] At each window position, compute local DMD eigenvalues, modes, and amplitudes
- [ ] Window size should be configurable (default: 10-15 checkpoints, but allow override)
- [ ] Window step size should be configurable (default: 1 checkpoint for maximum resolution)
- [ ] Store windowed eigenvalue trajectories as analysis artifacts (the full sequence of eigenvalue spectra across windows)

### Eigenvalue tracking
- [ ] Track individual eigenvalue trajectories across windows (eigenvalue correspondence across adjacent windows — match by nearest neighbor or Hungarian algorithm)
- [ ] For each eigenvalue at each window, record: magnitude |λ|, angle θ = arg(λ), real part, imaginary part
- [ ] Identify conjugate pairs and track them as paired modes

### Derived metrics (per window)
- [ ] Number of eigenvalues on or near the unit circle (|λ| within configurable tolerance of 1.0, default ±0.05)
- [ ] Angular spread: how distributed are the eigenvalue angles? (entropy or variance of angles)
- [ ] Angular stability: how much did eigenvalue angles change from the previous window?
- [ ] Radial stability: how much did eigenvalue magnitudes change from the previous window?
- [ ] Conjugate pair count: how many eigenvalues have clear conjugate partners (indicating oscillatory modes)?

### Visualizations
- [ ] **Eigenvalue migration plot**: animated or filmstrip showing eigenvalue positions on the complex plane evolving over training windows, with unit circle reference. Color by window index (training time).
- [ ] **Magnitude vs. epoch**: time series of |λ| for each tracked eigenvalue, with |λ|=1 reference line. Shows when modes reach the unit circle (sustained oscillation).
- [ ] **Angle vs. epoch**: time series of arg(λ) for each tracked eigenvalue. Shows frequency locking — angles stabilizing at fixed values.
- [ ] **Summary dashboard panel**: angular stability and radial stability metrics over training, overlaid with or adjacent to the residual norm plot from REQ_002 for direct comparison.
- [ ] Cross-model comparison: side-by-side eigenvalue migration plots or overlaid magnitude/angle trajectories for different models.

## Constraints
**Must have:**
- Depends on REQ_001 (global PCA) — windowed DMD operates on the same projected centroid data
- Eigenvalue tracking across windows (without this, individual plots per window are hard to interpret)
- Works across all model variants and all analyzed layers (Post-Embed, Attn Out, MLP Out, Resid Post)

**Must avoid:**
- Window sizes so small that eigenvalue estimates become unreliable (minimum ~5 checkpoints per window)
- Assuming evenly spaced checkpoints — window indexing should be by checkpoint position, with epoch labels for display

**Flexible:**
- Eigenvalue tracking/matching algorithm (nearest neighbor is fine for v1; Hungarian matching is more robust but more complex)
- Whether eigenvalue migration is animated vs. filmstrip vs. interactive slider
- Exact dashboard layout and integration with existing panels
- Whether to compute windowed DMD for all layers or start with a subset (Resid Post and Post-Embed are highest priority based on residual norm findings)

## Context & Assumptions
- 94 checkpoints per model; with window size 10 and step 1, this yields ~85 windows
- Checkpoint spacing may not be uniform in epoch-space — visualizations should use actual epoch values on time axes
- The number of DMD modes per window equals the number of retained PCA components (from REQ_001), which is typically 3-10 depending on variance threshold. With 3 PCA components and 113 centroids, the state vector is 339-dim but the snapshot matrix within a window of 10 checkpoints is only 339×9, so DMD will return at most 9 modes per window. This is expected — we're looking at the dominant dynamics within each window.
- Eigenvalue interpretation guide for the visualizations:
  - |λ| < 1: decaying mode (transient)
  - |λ| = 1: sustained mode (persistent oscillation or steady state)  
  - |λ| > 1: growing mode (expanding)
  - arg(λ) = 0: non-oscillatory (pure growth/decay)
  - arg(λ) ≠ 0: oscillatory at frequency proportional to the angle
  - Conjugate pairs (±θ): real-valued oscillation at frequency θ
- Research hypothesis to validate: successful grokking corresponds to eigenvalues migrating from the real axis (near (1,0)) to stable conjugate pairs on the unit circle. Failed grokking corresponds to eigenvalues that either never leave the real axis, never stabilize their angles, or never reach the unit circle.
- Secondary hypothesis: the angles at which eigenvalues stabilize post-grokking should correspond to Fourier frequencies that match the MLP neuron frequency commitments observed in existing analyses.

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- **Sanity check on p109/485 (fast, clean grokker)**: eigenvalues should transition quickly from real-axis-dominated to stable conjugate pairs on the unit circle, with the transition timing matching the known grokking window (~3k-6k epochs)
- **Contrast with p113/999 (canonical grokker)**: similar eigenvalue migration pattern but on a different timeline, with potentially more transient instability during the transition
- **Contrast with p101/999 and p59/485 (failure models)**: eigenvalue spectra should show qualitatively different behavior — wandering angles, failure to reach the unit circle, or inability to maintain stable conjugate pairs
- **Cross-validation with residual norms**: periods of high angular/radial instability in windowed eigenvalues should correlate with spikes in the DMD residual norm from REQ_002
- **Frequency correspondence**: for successful grokkers, post-grokking eigenvalue angles should be relatable to known Fourier frequencies from MLP neuron analysis

---
## Notes
[Claude Code adds implementation notes here]
