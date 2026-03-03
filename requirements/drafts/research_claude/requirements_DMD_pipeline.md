# REQ_001: Global PCA for Cross-Epoch Centroid Analysis

## Problem Statement
Current centroid class PCA is computed independently per epoch/checkpoint, meaning the PCA basis vectors rotate over training. This makes it impossible to track centroid trajectories in a consistent coordinate system across time, which is a prerequisite for any time-series analysis of representational dynamics (including DMD).

We need a single PCA basis computed across all checkpoints for a given model, so that centroid positions at every epoch are expressed in the same coordinate frame.

## Conditions of Satisfaction
- [ ] For a given model run, compute PCA over the concatenated centroid data from all checkpoints (i.e., stack all checkpoint centroids into one matrix, compute PCA on that)
- [ ] Project each checkpoint's centroids into this global basis
- [ ] Store the global PCA basis (eigenvectors, eigenvalues, explained variance) as a cross-epoch analysis artifact
- [ ] Store the per-checkpoint projected centroids as associated artifacts (or a single artifact indexed by checkpoint)
- [ ] Retain at minimum the top K components where K captures ≥95% of variance (but store enough to allow the user to choose dimensionality later)
- [ ] The existing per-epoch PCA remains unchanged — this is an additional analysis, not a replacement
- [ ] Visualizer support: animate or step through centroid positions in global PCA space across training (this enables visual validation that trajectories are smooth and coherent)

## Constraints
**Must have:**
- Consistent coordinate frame across all checkpoints for a single model
- Compatible with the existing cross-epoch analysis infrastructure
- Works for all current model variants (primes 59, 97, 101, 103, 107, 109, 113, 127)

**Must avoid:**
- Modifying or replacing existing per-epoch PCA analysis
- Assuming a fixed number of retained components (let the data determine this)

**Flexible:**
- Whether projected centroids are stored as one artifact per checkpoint or one combined artifact per model
- Visualization details (static snapshots vs. animation vs. slider) — whatever fits the existing dashboard pattern

## Context & Assumptions
- Each model has ~94 synchronized checkpoints
- Number of centroids per model = the prime p (e.g., 59 centroids for mod-59)
- Centroids live in the model's activation space (dimensionality depends on model architecture — Claude Code should verify from existing code)
- Current per-epoch PCA captures 60-90% of variance in top 3 components; global PCA may need more components since it must also account for between-epoch variance
- The existing analysis infrastructure has per-checkpoint analysis, cross-epoch analysis, and secondary analysis tiers

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- For a model where grokking occurs: visualizing centroid trajectories in global PCA space should show smooth, coherent paths that exhibit the Lissajous-like structures previously observed in per-epoch PCA
- Explained variance ratios should be documented — if top 3 components capture substantially less variance than per-epoch PCA, that itself is informative (it means between-epoch dynamics add significant new directions)
- Spot-check: projecting a single checkpoint's centroids into the global basis and into the per-epoch basis should yield similar *relative* arrangements (the centroids should look similar, just in a different orientation)

---
## Notes
[Claude Code adds implementation notes here]

---

# REQ_002: Standard DMD on Centroid Class Trajectories

## Problem Statement
We need to apply Dynamic Mode Decomposition to centroid class trajectories to extract the dominant linear dynamical modes (frequencies, growth/decay rates, spatial patterns) of representational change during training. This serves two purposes:

1. **Learning/validation**: DMD eigenvalues should reveal oscillatory modes consistent with the periodic structures already observed visually (Lissajous figures in centroid space). This provides a quantitative check on qualitative observations.
2. **Foundation for grokking window detection**: The DMD residual (where the linear approximation breaks down) is a candidate metric for principled identification of the grokking transition, and lays the groundwork for future LANDO analysis that will decompose dynamics into linear and nonlinear components.

## Conditions of Satisfaction
- [ ] Accepts global PCA-projected centroid trajectories (output of REQ_001) as input
- [ ] Constructs time-shifted snapshot matrices X and X' from the projected centroid data
- [ ] Computes standard DMD: eigenvalues, modes, and amplitudes
- [ ] Stores DMD results as analysis artifacts (eigenvalues, modes, amplitudes, reconstruction error per checkpoint)
- [ ] Computes and stores the DMD residual norm at each checkpoint: ||x'_actual - x'_predicted|| for each time step
- [ ] Configurable number of PCA components to use (default: components capturing ≥95% variance, but allow override)
- [ ] Works across all model variants

### Visualization requirements:
- [ ] Eigenvalue spectrum plotted on the complex plane (unit circle overlay for reference — eigenvalues on the circle indicate purely oscillatory modes, inside = decaying, outside = growing)
- [ ] DMD residual norm over training epochs (this is the key diagnostic — look for spikes or regime changes that correlate with grokking)
- [ ] Reconstruction quality: overlay of actual vs. DMD-reconstructed trajectories for the top PCA components
- [ ] Cross-model comparison of eigenvalue spectra (do grokking vs. non-grokking models have qualitatively different spectra?)

## Constraints
**Must have:**
- Depends on REQ_001 (global PCA) being complete
- Uses the economy-sized SVD for DMD computation (though at current data dimensions this is not strictly necessary, it's good practice)
- Eigenvalues, modes, and residuals must all be stored as artifacts for downstream analysis

**Must avoid:**
- Assuming continuous-time dynamics — this is discrete-time DMD (checkpoint-to-checkpoint)
- Hardcoding the number of DMD modes to retain (use singular value decay or energy threshold)

**Flexible:**
- Whether DMD is computed on all PCA components at once or on subsets
- Specific SVD truncation strategy (energy threshold vs. elbow method vs. user-specified rank)
- Dashboard layout for new visualizations

## Context & Assumptions
- Snapshot matrix dimensions (using mod-59 as example): if using top 3 global PCA components, state vector is 59 × 3 = 177 dimensions, with 93 time-shifted pairs from 94 checkpoints. This is well within direct DMD capability.
- For larger primes (127), state vectors are 127 × 3 = 381 dimensions — still manageable.
- DMD eigenvalues on the unit circle correspond to persistent oscillatory modes; their angles give frequencies. Eigenvalues inside the circle are decaying modes (transients). This is the key interpretive framework.
- The residual norm over time is our primary candidate for a principled grokking window metric. We expect it to be low during memorization (dynamics are relatively simple/linear), spike during the grokking transition (nonlinear restructuring), and potentially settle at a new level post-grokking.
- This analysis is a stepping stone toward LANDO (Baddoo et al.), which will decompose the dynamics into explicit linear and nonlinear components. For now, standard DMD + residual analysis gives us the linear baseline and identifies where it breaks down.

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- **Sanity check**: For a model that grokks, at least some DMD eigenvalues should land near the unit circle at frequencies consistent with the visually observed periodic structures
- **Grokking signal**: The residual norm time series should show visually distinct behavior during the grokking window compared to early/late training
- **Cross-model contrast**: Models that fail to grok should have qualitatively different eigenvalue spectra or residual profiles compared to those that succeed
- **Reconstruction**: DMD reconstruction of centroid trajectories should be reasonable (not perfect — we expect nonlinear residual — but capturing the dominant trends)

---
## Notes
[Claude Code adds implementation notes here]
