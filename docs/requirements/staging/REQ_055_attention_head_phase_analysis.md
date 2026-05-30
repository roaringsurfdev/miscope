# REQ_055: Attention Head Phase Relationship Analysis

**Status:** Completed
**Priority:** High
**Related:** REQ_052 (Fourier Quality Scoring), REQ_053 (Per-Class Error Analysis)

## Problem Statement

Current attention head analysis asks "which frequency does each head prefer" — a scalar summary that flattens the mechanistic question. In the Fourier algorithm for modular addition, attention heads compute phase relationships between input positions. The right question is: are Q and K for each head jointly aligned in the same Fourier frequency subspace, such that QK^T computes the phase difference cos(k(x±y)) between positions?

Without this, the attention head story is incomplete. We can see that heads lock onto frequencies, but we cannot see whether they are computing the correct phase transformation. This gap is especially significant for anomalous variants: a head that nominally "prefers frequency k" may be computing a degenerate or misaligned phase relationship that produces the wrong output.

## Conditions of Satisfaction

### Fourier Decomposition of Attention Weight Matrices

- [x] For each checkpoint epoch and each attention head, compute the Fourier decomposition of W_Q and W_K projected onto the model's prime-based Fourier basis
- [x] Compute the Fourier spectrum of QK^T per head — which frequency components does the head's similarity computation emphasize?
- [x] For W_V, compute its Fourier decomposition per head — what does the head output, and is that output structured in the same frequency subspace as the Q×K computation?

### Temporal View

- [x] For each head, track the Fourier spectrum of QK^T across epochs — when does alignment to the dominant frequency emerge, and how does it evolve relative to neuron specialization and grokking onset?
- [x] Surface whether different heads lock onto the same frequency simultaneously or sequentially

### Cross-Variant Comparison

- [x] Compare QK^T Fourier alignment between a healthy variant (e.g., p113/999) and an anomalous variant (e.g., p101/999) at matched epochs
- [x] For p101/999 specifically: does the QK^T spectrum reflect the same degenerate frequency concentration (high freq, imbalanced cos/sin) seen in the embeddings and neurons?

### Data Access

- [x] Attention head Fourier data exposed as dataviews following the DataViewCatalog pattern established in REQ_054 *(see Completion Note — exposed via the View Catalog, which superseded the DataViewCatalog pattern)*

## Constraints

**Must:**
- Fourier basis used must be the prime-specific basis — family provides this as context
- Analysis must be per-head, not aggregated across heads
- Temporal resolution must match existing checkpoint cadence

**Must not:**
- Collapse Q and K into a single dominant frequency — the joint alignment is the signal

**Explicitly deferred:**
- V-output tracing through the residual stream (requires activation analysis, not just weight analysis)
- Cross-head interaction analysis

## Context & Assumptions

The mechanistic interpretability literature (Nanda et al.) establishes that attention heads in grokking models compute terms of the form cos(k(x+y)) by attending across positions. This requires Q and K to be aligned in the same Fourier frequency subspace. A head aligned to frequency k will have QK^T dominated by the k-th Fourier component.

The current attention head views show which frequency each head "prefers" — likely computed from dominant embedding activations or attention pattern analysis. This is a first-order description. The second-order question is whether the head's Q×K inner product is computing the correct Fourier product for modular addition.

For anomalous variants, this matters because the degenerate solution in p101/999 (6:1 cos/sin imbalance at frequency 13) may be visible not just in the embeddings and neurons but in the attention head phase computation. If attention heads for p101/999 show a QK^T spectrum concentrated on frequency 13 with degraded structure at frequency 5, that would confirm the degenerate frequency selection propagated throughout the network.

The family (modulo_addition_1layer) provides the prime-based Fourier basis needed to project weight matrices. This analysis is family-assisted but the analyzer and view are universal — they apply to any attention head in any model with a prime-based Fourier structure.

## Notes

**Implementation sketch:**
- New analyzer: `AttentionFourierAnalyzer` — at each epoch, for each head h: load W_Q[h], W_K[h], W_V[h] from parameter snapshot; project onto Fourier basis; compute Fourier spectrum of Q·K^T; store per-head spectra
- Family provides Fourier basis construction (already exists via prime parameter)
- Store per-epoch artifacts (one file per epoch, matching existing checkpoint cadence)
- DataViewDefinitions for the resulting data following REQ_054 pattern

**Key visualization targets:**
- Per-head QK^T Fourier spectrum heatmap (frequency × head, per epoch)
- Temporal trajectory: dominant frequency alignment per head across epochs
- Side-by-side comparison of healthy vs. anomalous variant head alignment at matching epochs

## Completion Note (2026-05-29)

Marked **Completed** and moved to staging. All conditions of satisfaction are met
in substance; two implementation realities diverge from the original sketch and are
recorded here for archaeology, because the requirement was satisfied and *then* the
substrate beneath it was refactored.

**1. Analyzer absorbed into the universal `weight_basis_projection` analyzer.**
The sketch called for a standalone `AttentionFourierAnalyzer`. That analyzer was
absorbed into the universal `weight_basis_projection` analyzer (REQ_126/127), which
projects family-declared weight sites onto the family-supplied Fourier basis. The
family (`modulo_addition_1layer`) declares two sites that reproduce the intended
computation per head:
- `attn_qk` — composes `(W_E·W_Q)(W_E·W_K)ᵀ` per head, shape `(n_heads, p, p)`, 2D-projected
- `attn_v` — composes `(W_E·W_V)` per head, shape `(n_heads, p, d_head)`

This is *more* aligned with architectural constraint #1 (views are universal
instruments; analyzers are not family-owned) than the original per-family-analyzer
sketch. The legacy `qk_freq_norms` / `v_freq_norms` view shape is reconstructed from
the new analyzer's `attn_qk` / `attn_v` outputs via a small adapter in the view
loaders. Coverage: `packages/miscope/tests/test_weight_basis_projection.py`.

**2. Data access exposed via the View Catalog, not the REQ_054 DataViewCatalog.**
The Data Access CoS named the DataViewCatalog pattern from REQ_054. That experimental
pattern was superseded by the View Catalog (`variant.at(epoch).view(name)`, REQ_047),
which is now the canonical data-access surface. The attention-Fourier data is exposed
through it as three views:
- `parameters.attention.qk_fourier_heatmap` (per-epoch, head × frequency)
- `parameters.attention.v_fourier_heatmap` (per-epoch, head × frequency)
- `parameters.attention.head_alignment_trajectory` (temporal, dominant-freq + max-fraction per head)

The *intent* of the CoS — catalog-mediated access with no path literals or direct
storage construction — is fully satisfied; only the named mechanism evolved.

**Dashboard surface:** the QK and V Fourier alignment heatmaps were added to the
Activations page (commit `00a858c`). Per-variant artifacts exist under
`data/modulo_addition_1layer/variants/*/artifacts/weight_basis_projection/`.
