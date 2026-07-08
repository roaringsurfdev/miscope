# REQ_018: CUDA Support for Training

**Status:** Future
**Priority:** High
**Estimated Effort:** Low
**Related:** REQ_016 (CUDA Support for Analysis)

## Problem Statement

Training from the dashboard currently checks `os.environ.get("CUDA_VISIBLE_DEVICES")` to determine device selection ([app.py:96](dashboard/app.py#L96)). This is unreliable—it only uses CUDA if the user has explicitly set that environment variable, rather than detecting GPU availability directly.

Users can work around this by running `ModuloAdditionRefactored.py` directly (which does use CUDA properly), but training from the dashboard should also reliably use CUDA when available.

## Conditions of Satisfaction

- [ ] Training runs on CUDA when available
- [ ] Graceful fallback to CPU when CUDA is unavailable
- [ ] No regression for CPU-only users
- [ ] Consistent device detection with REQ_016 (analysis)

## Technical Analysis

### Current State
```python
# dashboard/app.py line 96
device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
```

This only selects CUDA if `CUDA_VISIBLE_DEVICES` is explicitly set in the environment.

### Required Change

```python
# dashboard/app.py line 96
# Change from:
device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
# To:
device="cuda" if torch.cuda.is_available() else "cpu"
```

### Why the Current Approach is Problematic

1. **Unreliable detection**: Most users with CUDA don't manually set `CUDA_VISIBLE_DEVICES`
2. **Inconsistent with analysis**: If REQ_016 is implemented, training and analysis would use different detection methods
3. **Inconsistent with direct script usage**: `ModuloAdditionRefactored.py` likely uses proper detection

## Constraints

**Must have:**
- Works without CUDA (CPU fallback)
- No breaking changes to existing workflows

**Must avoid:**
- Memory issues (OOM) - training should handle gracefully

**Flexible:**
- Whether to display device in training output
- Whether to add device selection toggle in UI

## Implementation

This is a minimal change:
1. Change device detection to use `torch.cuda.is_available()`
2. Ensure `torch` is imported (likely already is)
3. Test training with and without CUDA available

## Relationship to REQ_016

Both REQ_016 and REQ_018 address the same underlying issue: reliable CUDA detection in the dashboard. They could be implemented together as a single small PR, ensuring consistent device handling across training and analysis.

## Decision Authority

- [x] Make reasonable decisions and flag for review

## Success Validation

- Training runs on GPU when available (without needing env var)
- Training still works on CPU-only machines
- No errors or regressions in training output

---

## Notes

**2026-02-01:** Requirement created to address inconsistent device detection. Paired with REQ_016 for analysis CUDA support.
