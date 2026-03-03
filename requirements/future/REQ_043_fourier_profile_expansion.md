# REQ_043: Fourier Profile Expansion — Multi-Matrix Frequency Analysis

**Status:** Draft
**Priority:** Medium (enables embedding/unembedding trajectory analysis)
**Dependencies:** Existing `dominant_frequencies` analyzer, Fourier library functions
**Last Updated:** 2026-02-15

## Problem Statement

The current `dominant_frequencies` analyzer only projects `W_E` (embedding weights) onto the Fourier basis. This captures which frequencies the model encodes in its input embeddings, but ignores `W_U` (unembedding weights) and other weight matrices that also carry frequency structure.

To understand how frequency structure co-evolves across the model — do embeddings and unembeddings converge on the same frequencies? Does `W_U` frequency structure stabilize before or after `W_E`? — we need per-frequency coefficient norms for multiple weight matrices, stored per-epoch.

This is also a prerequisite for extending the neuron dynamics page (REQ_042) with embedding/unembedding frequency trajectory panels.

## Design Paths

Three approaches, each with distinct tradeoffs:

### Path A: Expand `dominant_frequencies` analyzer

Add a configurable list of weight matrices to the existing analyzer. Each matrix gets its own coefficients array in the artifact.

- **Pro:** Minimal new code, single artifact directory
- **Con:** Analyzer name becomes misleading ("dominant_frequencies" → really "fourier_profile"). Artifact schema changes — existing artifacts would need migration or versioning.

### Path B: Separate analyzers per matrix

Create `embedding_freq_profile`, `unembedding_freq_profile`, etc. as independent analyzers.

- **Pro:** Clean separation, no schema migration, each can evolve independently
- **Con:** Multiplies artifact directories and config. Redundant Fourier basis computation. Cross-matrix comparison requires loading from multiple analyzers.

### Path C: New `fourier_profile` analyzer (configurable)

A new analyzer that takes a list of weight matrix names (like `parameter_snapshot` takes all 9 matrices). Stores per-matrix coefficient norms in a single artifact per epoch.

- **Pro:** Clean name, explicit design, follows `parameter_snapshot` precedent. Single artifact directory for all matrix profiles.
- **Con:** More upfront design work. Need to define matrix extraction interface cleanly.

### Recommendation

Path C is the most aligned with existing patterns (`parameter_snapshot` already solves the "multiple matrices, one analyzer" problem). It avoids the naming confusion of Path A and the fragmentation of Path B.

## Conditions of Satisfaction

1. Per-epoch artifacts contain Fourier coefficient norms for at least `W_E` and `W_U`
2. Coefficient norms are computed using the existing `project_onto_fourier_basis()` function
3. Artifacts follow the standard `epoch_{NNNNN}.npz` pattern
4. Summary includes per-matrix dominant frequency tracking over epochs
5. Backward-compatible: existing `dominant_frequencies` artifacts and renderers continue to work unchanged
6. At least one renderer visualizes the new data (cross-epoch frequency trajectory per matrix, analogous to the existing `render_dominant_frequencies` but over time)

## Constraints

- Must reuse existing Fourier library functions — no reimplementation
- Matrix extraction must handle varying model architectures gracefully (1-layer vs. multi-layer)
- Artifact size should remain reasonable (coefficient norms are small — O(p) per matrix per epoch)

## Decision Authority

- **Discuss first:** Which design path (A, B, or C). This affects artifact layout and downstream consumers.
- **Claude decides:** Implementation details within chosen path

## Notes

- The existing `dominant_frequencies` coefficients shape is `(n_fourier_components,)` which is `(p,)` for the full basis (constant + sin/cos pairs). The neuron freq norm uses `(p//2,)` (frequencies only, no constant, sin/cos combined into single norm). The new analyzer should decide which convention to follow — the per-frequency combined norm is more useful for trajectory analysis.
- This naturally feeds into a future "Embedding Dynamics" page that would sit alongside REQ_042's Neuron Dynamics page.
