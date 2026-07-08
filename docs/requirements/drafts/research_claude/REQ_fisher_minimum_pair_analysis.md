# REQ_XXX: Fisher Minimum Pair Analysis
*Drafted by Research Claude, edited by Engineering Claude*

## Problem Statement
We need to identify *which* class pairs are hardest to linearly separate at each training epoch, and understand why. The current representational geometry analyzer computes mean and min Fisher discriminant as scalar summaries, but these hide critical information: which specific residue pairs constitute the separation bottleneck, how those bottleneck pairs change over training, and whether the bottleneck pairs are predictable from the model's learned frequency spectrum.

This matters because the Fisher min tells us the model's worst-case vulnerability, but not *where* that vulnerability is. In a model that has learned specific Fourier frequencies, the hardest-to-separate class pairs should be those whose residue difference is poorly resolved by the learned frequencies. Confirming this connection would validate the Fourier interpretation of grokking at the representational level and provide a new diagnostic for understanding partial or failed grokking (e.g., 101/999's transient circle collapse, 59/485's near-zero Fisher min despite high Fisher mean).

**Origin**: This requirement emerged from analysis of representational geometry results across variants (conversation with Claude, 2026-02-16), specifically from observing that 59/485 achieves Fisher mean ~20 but Fisher min ~0, while 109/485 achieves Fisher mean ~23 with Fisher min ~2. The question of *which pairs* are vulnerable, and whether vulnerability is predictable from frequency allocation, is the next analytical step.

## Conditions of Satisfaction
- [ ] A Fisher heatmap visualization is available (p x p heatmap showing Fisher discriminant for all class pairs at a selected epoch), displayable alongside the existing centroid distance heatmap
- [ ] A time-series visualization tracks the identity of the argmin pair over training (showing whether the bottleneck pair is stable or shifts)
- [ ] (Stretch) The relationship between the argmin pair's residue difference and the model's learned frequencies can be visually inspected (e.g., annotating the argmin pair with |r* - s*| mod p and comparing to frequency-predicted blind spots)

## Constraints
**Must have:**
- Fisher discriminant formula: J(r, s) = ||mu_r - mu_s||^2 / (sigma_r^2 + sigma_s^2) where mu is class centroid, sigma^2 is mean squared distance from centroid (i.e., radius^2)
- Compute for Resid Post at minimum; other activation sites are desirable but lower priority
- The pairwise Fisher matrix should be symmetric; store only the upper triangle or full matrix as makes sense for the visualization code

**Must avoid:**

**Flexible:**
- Visualization layout and interactivity details — use judgment on what integrates well with the existing dashboard
- Whether the frequency-prediction comparison (CoS #3) is automated or left for manual visual inspection

## Context & Assumptions
**Domain context**: For a model that has learned Fourier frequencies k1, k2, ..., the centroid for residue class r is positioned in activation space based on (cos(2*pi*k_i*r/p), sin(2*pi*k_i*r/p)) for each frequency. Two residues r and s will have nearby centroids when k_i*(r-s) mod p ~ 0 for all learned frequencies simultaneously. This means the hardest-to-separate pairs should have residue differences near common multiples of p/k_i across the active frequencies.

**Dependencies**: This analysis depends on the representational geometry artifacts (centroids and per-class radii) already being computed. It should run after the representational geometry analyzer.

## Decision Authority
- [x] Make reasonable decisions and flag for review

Proceed with implementation using best judgment on integration with existing architecture.

## Success Validation
**Analytical validation**: For 109/485 (3 frequency bands, healthy Fisher min ~2), the Fisher heatmap should show a smooth circulant-like pattern with no extreme cold spots. For 59/485 (2 frequency bands, Fisher min ~0), the heatmap should show clear cold spots (low-Fisher pairs) at predictable positions based on frequencies 5/6 and 20. For 101/999 (pathological), the heatmap should show widespread low values with unstable structure across epochs.

**Visual validation**: The Fisher heatmap and the centroid distance heatmap, shown side by side for the same epoch, should tell complementary stories — distance heatmap shows raw geometric separation, Fisher heatmap shows separation *relative to within-class spread*.

**Practical test**: Load the artifacts for any analyzed variant, verify the stored argmin pair, and manually confirm by checking that the identified pair has the smallest J(r,s) value in the matrix.

---
## Notes
- Research Claude noted scale: for p=113, the pairwise matrix is 113x113 = 12,769 entries per epoch. This fits comfortably in existing artifact patterns.
- Engineering note: The existing repr_geometry analyzer already computes pairwise Fisher internally (vectorized broadcasting) and reduces to mean/min scalars. Extending to store the full matrix is straightforward — the computation is already done, we just need to keep the intermediate result.
- Engineering note: Existing `radii` arrays store RMS distance (not variance). The Fisher formula denominator uses radius^2, which equals variance. This is handled internally — no change to the formula definition above.
