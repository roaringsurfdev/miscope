# REQ_016: CUDA Support for Analysis

**Status:** Future
**Priority:** High
**Estimated Effort:** Low
**Related:** REQ_018 (CUDA Support for Training)

## Problem Statement

Analysis computations in the dashboard are hardcoded to run on CPU (`device="cpu"` in `dashboard/app.py:227`). The architecture already supports device flexibility—the only barrier is this single hardcoded value. Enabling CUDA for analysis would improve performance for users with GPU hardware, especially as more analysis features are added.

## Conditions of Satisfaction

- [ ] Analysis runs on CUDA when available
- [ ] Graceful fallback to CPU when CUDA is unavailable
- [ ] No regression for CPU-only users
- [ ] Clear indication to user of which compute backend is active (optional enhancement)

## Technical Analysis

### Current State
- Analysis device is hardcoded: `device="cpu"` at [app.py:227](dashboard/app.py#L227)
- The rest of the architecture already handles device correctly:
  - Dataset created with `.to(device)`
  - Fourier basis accepts device parameter
  - Model loaded to specified device
  - Analyzers infer device from tensors they receive
- Final `.cpu().numpy()` calls in analyzers are correct (visualization needs numpy)

### Required Change

```python
# dashboard/app.py line 227
# Change from:
device="cpu"
# To:
device="cuda" if torch.cuda.is_available() else "cpu"
```

### GPU Benefit by Operation

| Operation | GPU Benefit |
|-----------|-------------|
| Model inference (`run_with_cache`) | **High** - transformer forward pass |
| NeuronFreqClustersAnalyzer | **Medium** - double matrix multiplications |
| DominantFrequenciesAnalyzer | **Low** - small matrix ops |
| NeuronActivationsAnalyzer | **Minimal** - reshape only |

## Constraints

**Must have:**
- Works without CUDA (CPU fallback)
- No breaking changes to existing workflows

**Must avoid:**
- Memory issues (OOM) - may need error handling for large models

**Flexible:**
- Whether to display device in UI
- Whether to add device selection toggle

## Implementation

This is a minimal change:
1. Add `import torch` if not present in app.py
2. Change device assignment to use `torch.cuda.is_available()`
3. Test with and without CUDA available

## Decision Authority

- [x] Make reasonable decisions and flag for review

## Success Validation

- Analysis runs on GPU when available
- Analysis still works on CPU-only machines
- No errors or regressions in analysis output

---

## Notes

**2026-02-01:** Requirement scoped down from original vision. Async architecture (background workers, job queues) deferred—synchronous CUDA is sufficient for current needs.

**Future consideration:** If analysis times grow significantly with new features, async processing could be revisited as a separate requirement.
