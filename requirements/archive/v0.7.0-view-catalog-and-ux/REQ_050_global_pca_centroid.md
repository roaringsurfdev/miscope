# REQ_050: Global PCA for Cross-Epoch Centroid Analysis

**Status:** Active
**Priority:** High (prerequisite for REQ_051 DMD pipeline)
**Dependencies:** REQ_044 (Representational Geometry — centroid artifacts must exist)
**Last Updated:** 2026-03-02
**Attribution:** Drafted by Research Claude, edited by Engineering Claude

## Problem Statement

The centroid class PCA computed by REQ_044 (`repr_geometry` analyzer) is done independently per epoch. This means the PCA basis rotates over training — the same physical direction in activation space may project onto PC1 at one epoch and PC3 at another. This is appropriate for per-epoch visualization (each epoch gets the best-fit coordinate frame), but makes cross-epoch comparison meaningless: centroids can't be tracked as coherent trajectories in a consistent space.

A global coordinate frame — a single PCA basis computed from all epochs simultaneously — is a prerequisite for any time-series analysis of representational dynamics. Without it, we cannot answer questions like "does the model return to the same region of centroid space after grokking?" or "what trajectory did the centroids follow through representation space during the transition?"

This is also a prerequisite for the DMD pipeline (REQ_051), which requires centroid positions in a consistent coordinate frame to construct meaningful snapshot matrices.

## Conditions of Satisfaction

### Analysis
- [ ] For each variant and activation site, compute a single PCA basis across all available checkpoints by pooling centroid data from all epochs into a shared matrix before decomposing
- [ ] Each checkpoint's centroids are projected into this global basis, producing per-epoch projections in the shared coordinate frame
- [ ] The global basis and per-epoch projections are stored as analysis artifacts, following existing cross-epoch artifact conventions
- [ ] The number of retained components is determined by explained variance (retain components capturing ≥95% cumulative variance), with the full set stored to allow downstream dimensionality selection
- [ ] The existing per-epoch PCA (in `repr_geometry`) is not modified — this is an additive analysis

### Visualization
- [ ] A cross-epoch view allows stepping through centroid positions in global PCA space across training epochs, showing how centroids move in the shared coordinate frame
- [ ] Explained variance ratios for the global basis are accessible and reported (these may differ meaningfully from per-epoch ratios, and the difference is itself informative)

### Validation
- [ ] Projecting a single checkpoint's centroids into the global basis and into the per-epoch basis should yield similar *relative* arrangements — centroids should look similar, just in a potentially different orientation

## Constraints

**Must have:**
- A consistent coordinate frame across all checkpoints for a single variant — this is the entire purpose
- Compatibility with the existing cross-epoch artifact infrastructure

**Must avoid:**
- Modifying or replacing the existing per-epoch centroid PCA in `repr_geometry`
- Assuming a fixed number of retained components

**Flexible:**
- Whether global basis and per-epoch projections are stored as a single combined artifact or as separate files — follow the convention that best fits existing cross-epoch storage patterns
- Whether all four activation sites (resid_pre, attn_out, mlp_out, resid_post) are computed in a single pass or separately
- Visualization details (animation vs. slider-controlled stepping)

## Context & Assumptions

- Centroid artifacts are stored per-epoch under the `repr_geometry` analyzer. Each file contains centroids for all four activation sites (resid_pre, attn_out, mlp_out, resid_post).
- Each variant has ~94 checkpoints. Centroid shapes are (p, d_model) per epoch per site, where p is the prime (59–127) and d_model is the model's hidden dimension.
- The per-epoch PCA typically captures 60–90% of variance in the top 3 components. The global basis must also capture between-epoch variance, so more components may be needed to reach 95%.
- This analysis belongs to the cross-epoch tier of the analysis pipeline (not per-checkpoint, not secondary).

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation

- For a variant where grokking occurs: centroid positions in global PCA space should trace smooth, coherent paths over training. The Lissajous-like structures previously observed in per-epoch PCA should appear as trajectories rather than static snapshots, if the global basis captures the relevant structure.
- The explained variance ratios for the global basis should be documented. If top-3 components capture substantially less variance than per-epoch PCA, this is meaningful — it indicates between-epoch dynamics introduce new directions not present in any single snapshot.
- Cross-variant comparison of global basis geometry is a natural downstream use; the artifact structure should not preclude this.

---
## Notes

**2026-03-02 (Engineering Claude):** The existing `repr_geometry` cross-epoch summary artifacts already stack per-epoch data for time-series metrics. The global PCA is a different operation — it needs to *pool* centroids across epochs before decomposing, not decompose per epoch and then aggregate. This distinction should be clear in the implementation.

**2026-03-02:** The PC3 circularity/Fourier revisit (noted in thoughts.md, 2026-03-02) becomes more tractable once this analysis exists — a 3-component circularity metric computed in the global basis would be more meaningful than the current 2-component per-epoch version. This is a natural follow-on, not in scope here.

**2026-03-02:** The validation criterion (similar relative arrangements between global and per-epoch projections) is a sanity check, not a hard requirement. Some rotation and scaling is expected; the key is that the neighborhood structure of centroids (which centroids are near each other) is preserved.
