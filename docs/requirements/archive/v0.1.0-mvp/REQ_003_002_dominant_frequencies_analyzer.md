# REQ_003_002: Dominant Frequencies Analyzer

## Problem Statement
The analysis pipeline needs its first concrete analyzer implementation. The Dominant Frequencies Analyzer computes the Fourier coefficient norms for embedding weights, identifying which frequencies dominate the learned representation at each checkpoint.

This is the simplest of the three MVP analyzers and validates the pipeline's analyzer interface.

## Conditions of Satisfaction
- [x] `DominantFrequenciesAnalyzer` class conforming to `Analyzer` protocol
- [x] Computes `(fourier_basis @ W_E).norm(dim=-1)` for embedding weights
- [x] Returns numpy array of shape `(n_fourier_components,)` for each checkpoint
- [x] Correctly excludes equals token from W_E (uses `W_E[:-1]`)
- [x] Uses FourierEvaluation utilities (reuses `get_fourier_bases`)
- [x] End-to-end test: pipeline produces correct artifact file
- [x] Artifact contains expected data structure (epochs + coefficients)

## Constraints
**Must have:**
- Conforms to Analyzer protocol (name property + analyze method)
- Uses existing FourierEvaluation utilities
- Returns CPU numpy arrays (moved from device)

**Must avoid:**
- Reimplementing Fourier basis computation
- Device-specific logic in analyzer (should work on any device)

**Flexible:**
- Whether to also compute dominant_bases_indices
- Additional metadata in returned dict

## Context & Assumptions
- Reference: `ModuloAdditionRefactored.py` line 137
- W_E has shape (vocab_size, d_model) = (p+1, 128)
- Equals token at index p should be excluded
- Fourier basis has shape (2*p//2 + 1, p) = (p+1, p) ≈ (114, 113) for p=113
- Result has shape (n_fourier_components,) = (114,) for p=113

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Analyzer passes protocol isinstance check
- Running with pipeline creates `dominant_frequencies.npz`
- Artifact epochs match checkpoint epochs
- Artifact coefficients shape is (n_epochs, n_components)
- Values are non-negative (they are norms)

---
## Notes

## Implementation Notes (Added by Claude)

**Implementation completed:** 2026-01-31

**Key code location:**
- `analysis/analyzers/dominant_frequencies.py` (51 lines)

**Approach taken:**
- Simple implementation following reference at `ModuloAdditionRefactored.py:137`
- Uses `model.embed.W_E[:-1]` to exclude equals token
- Computes `(fourier_basis @ W_E).norm(dim=-1)` for coefficient norms
- Returns `{"coefficients": array}` where array has shape (n_components,)
- Uses `.detach().cpu().numpy()` to handle gradient-enabled tensors

**Tests:** 14 tests in 3 classes
- `TestDominantFrequenciesAnalyzerProtocol` - Protocol conformance (3 tests)
- `TestDominantFrequenciesAnalyzerOutput` - Output shape and values (6 tests)
- `TestDominantFrequenciesAnalyzerIntegration` - Pipeline integration (5 tests)
