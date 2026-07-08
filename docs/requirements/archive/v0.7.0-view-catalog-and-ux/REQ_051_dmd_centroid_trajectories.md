# REQ_051: Standard DMD on Centroid Class Trajectories

**Status:** Active
**Priority:** High (primary candidate for principled grokking window detection)
**Dependencies:** REQ_050 (Global PCA — centroid trajectories in consistent coordinate frame)
**Last Updated:** 2026-03-02
**Attribution:** Drafted by Research Claude, edited by Engineering Claude

## Problem Statement

We have rich qualitative observations of how centroid class representations evolve during training — the Lissajous-like traversal, the apparent reorganization during grokking, the return to a stable geometry. What we lack is a quantitative characterization of the *dynamics* underlying these trajectories.

Dynamic Mode Decomposition (DMD) extracts the dominant linear dynamical modes from a time-series of system states — in this case, the trajectory of centroid positions in activation space across training. This serves two purposes:

1. **Validation of existing observations.** DMD eigenvalues should reveal oscillatory modes consistent with the periodic structures already observed (Lissajous figures in centroid space). This turns a visual observation into a quantitative claim.

2. **Principled grokking window detection.** The DMD residual — where the linear approximation breaks down — is the primary candidate for a data-driven, model-agnostic grokking transition marker. We expect the residual to be low during the memorization phase (dynamics are relatively smooth), spike during the grokking transition (nonlinear reorganization), and settle post-grokking. If this holds across variants, it provides the stable onset definition currently missing from the analysis.

This requirement is also the foundation for potential future LANDO analysis (Baddoo et al.), which decomposes dynamics into explicit linear and nonlinear components. Standard DMD first establishes the linear baseline and identifies where it fails.

## Conditions of Satisfaction

### Analysis
- [ ] Applies DMD to the globally-projected centroid trajectories from REQ_050, treating each checkpoint's centroid state as a time step
- [ ] Extracts DMD eigenvalues, spatial modes, and mode amplitudes
- [ ] Computes the DMD residual norm at each checkpoint — the magnitude of the difference between the DMD-predicted next state and the actual next state
- [ ] Stores eigenvalues, modes, amplitudes, and per-checkpoint residual norms as analysis artifacts
- [ ] The number of DMD modes retained is determined by the data (energy-based or spectral gap criterion), not hardcoded; the truncation strategy should be configurable
- [ ] Runs successfully across all model variants

### Visualization
- [ ] **Eigenvalue spectrum:** DMD eigenvalues plotted on the complex plane, with the unit circle shown as a reference. Eigenvalues on the circle are purely oscillatory; inside are decaying; outside are growing.
- [ ] **Residual norm over training:** Per-checkpoint residual norm plotted as a time series, with the grokking window (if defined) marked for reference. This is the key diagnostic output.
- [ ] **Reconstruction overlay:** Actual centroid trajectories (in global PCA space) vs. DMD-reconstructed trajectories, for the leading PCA components. Shows reconstruction quality and where the linear approximation breaks down.

## Constraints

**Must have:**
- Dependency on REQ_050 — DMD requires a consistent coordinate frame; per-epoch PCA projections are not a valid input
- Eigenvalues, modes, and residual norms stored as artifacts — these are needed for cross-variant comparison and downstream analyses
- Discrete-time DMD (checkpoint-to-checkpoint), not continuous-time

**Must avoid:**
- Hardcoding the number of DMD modes — use a data-driven truncation criterion
- Running DMD on per-epoch PCA projections (different bases per epoch make the time series meaningless)

**Flexible:**
- Specific truncation strategy (energy threshold, spectral gap, elbow detection) — flag the choice for review
- Whether DMD is computed separately per activation site or jointly
- Dashboard layout for the new views

## Context & Assumptions

- Input is the globally-projected centroid trajectories from REQ_050: for each variant and site, a matrix of shape (n_epochs, p × k) where p is the prime and k is the number of retained global PCA components. This is the "state" at each time step.
- With ~94 checkpoints, DMD operates on 93 consecutive time-step pairs. State vector dimensionality with 3 global PCA components: p × 3 (e.g., 113 × 3 = 339 for p=113). This is well within standard DMD capability without dimensionality concerns.
- Eigenvalues on the unit circle correspond to persistent oscillatory modes; their angles give the oscillation frequency in training-step units. Eigenvalues inside the circle are decaying (transients). Eigenvalues outside are growing (instabilities).
- The residual norm is the primary deliverable from a research standpoint. The reconstruction and eigenvalue plots are diagnostics and validation tools.
- This analysis is a stepping stone toward LANDO (Baddoo et al. 2022), which will decompose dynamics into explicit linear and nonlinear components. Standard DMD establishes the linear baseline.

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation

- **Eigenvalue sanity check:** For a cleanly grokking variant (e.g., p113/999), at least some DMD eigenvalues should land near the unit circle at frequencies consistent with the visually observed periodic structures in centroid space.
- **Residual as grokking signal:** The residual norm time series should show visually distinct behavior during the grokking window compared to early and late training — ideally a spike or sustained elevation during the transition. Absence of this signal would be informative and should be reported, not suppressed.
- **Cross-variant contrast:** Variants that fail to grok cleanly (p101/999, p59/485) should show qualitatively different residual profiles or eigenvalue spectra compared to clean grokkers. This is the primary cross-variant validation.
- **Reconstruction quality:** DMD reconstruction of centroid trajectories should be reasonable for the memorization and post-grokking phases, with larger errors during the transition. Perfect reconstruction is not expected.

---
## Notes

**2026-03-02 (Engineering Claude):** The residual norm is the highest-priority output. It's the candidate metric that could replace the current visual-inspection-based grokking window definition. Implementation should ensure the residual is computed and stored even if reconstruction visualization is deferred.

**2026-03-02:** The cross-variant residual comparison is what makes this analytically powerful. A view that overlays residual norms across all variants (normalized to their own training length or aligned to grokking onset) would directly support the validation criterion. This is a stretch goal for the visualization — flag for review whether it belongs here or in a subsequent cross-variant analysis requirement.

**2026-03-02:** LANDO reference: Baddoo, Herrmann, McKeon & Brunton (2022). *Physics-informed dynamic mode decomposition.* Proc. R. Soc. A. The key idea is replacing the linear DMD operator with a kernel-based operator that explicitly represents nonlinear contributions. The DMD residual from this requirement is the quantity LANDO will decompose into interpretable linear and nonlinear parts.

**2026-03-02:** The grokking window definition discussion (thoughts.md, 2026-03-02) identifies three candidate onset metrics: DMD residual, Fourier Alignment threshold, and test loss inflection point. Once this requirement is implemented, a notebook comparison of all three across variants would directly address the methodological gap in findings.md.
