# REQ_003_003: Remaining Analyzers

## Problem Statement
Complete the MVP analyzer set with two additional analyzers:
1. **NeuronActivationsAnalyzer**: Extracts MLP neuron activations reshaped to input space
2. **NeuronFreqClustersAnalyzer**: Computes neuron-frequency specialization matrix

These enable visualization of activation patterns and neuron clustering by frequency.

## Conditions of Satisfaction
### NeuronActivationsAnalyzer
- [x] Conforms to Analyzer protocol
- [x] Extracts activations from `cache["post", 0, "mlp"][:, -1, :]`
- [x] Reshapes from `(p^2, d_mlp)` to `(d_mlp, p, p)` using einops
- [x] Returns numpy array with correct shape

### NeuronFreqClustersAnalyzer
- [x] Conforms to Analyzer protocol
- [x] Computes 2D Fourier transform of activations
- [x] Computes fraction of variance explained by each frequency for each neuron
- [x] Returns matrix of shape `(n_frequencies, d_mlp)` with values in [0, 1]

### Integration
- [x] Both analyzers work with pipeline
- [x] Artifacts saved with correct structure
- [x] End-to-end tests pass

## Constraints
**Must have:**
- Use einops for reshaping (consistent with codebase patterns)
- Extract from last token position (index -1)
- Normalize neuron_freq_norm by total variance

**Must avoid:**
- Hardcoding d_mlp or p values
- Keeping tensors on device after analysis

**Flexible:**
- Whether to include the DC component in neuron_freq_norm

## Context & Assumptions
- Reference: `ModuloAdditionRefactored.py` lines 88, 159-165, 185-198
- Neuron activations at last position: `cache["post", 0, "mlp"][:, -1, :]`
- 2D Fourier: `fourier_basis @ activations.reshape(p,p) @ fourier_basis.T`
- Frequency variance: Sum of squared coefficients at specific indices

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Both analyzers pass protocol checks
- Pipeline creates `neuron_activations.npz` and `neuron_freq_norm.npz`
- Shapes match expectations based on model config
- Values are reasonable (activations can be any value, freq_norm in [0, 1])

---
## Notes

## Implementation Notes (Added by Claude)

**Implementation completed:** 2026-01-31

**Key code locations:**
- `analysis/analyzers/neuron_activations.py` (52 lines)
- `analysis/analyzers/neuron_freq_clusters.py` (82 lines)

**NeuronActivationsAnalyzer:**
- Uses einops.rearrange for `(p^2, d_mlp)` → `(d_mlp, p, p)` reshaping
- Extracts from cache at `["post", 0, "mlp"][:, -1, :]` (last token)
- Returns `{"activations": array}` of shape (d_mlp, p, p)

**NeuronFreqClustersAnalyzer:**
- Computes 2D Fourier transform: `fourier_basis @ reshaped @ fourier_basis.T`
- Centers by setting DC component to 0
- For each frequency k, sums squared coefficients at indices (0, 2k-1), (0, 2k), etc.
- Normalizes by total variance with clamp to avoid division by zero
- Returns `{"norm_matrix": array}` of shape (p//2, d_mlp)

**Tests:** 16 tests in 5 classes
- Protocol conformance for both analyzers
- Output shape and value validation
- Integration with pipeline
