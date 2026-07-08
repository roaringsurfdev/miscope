# BUG_001 Investigation

## Root Cause

In `dashboard/app.py`, the `load_model_data()` function returns values for updating the neuron slider. The issue was returning a raw integer value instead of a `gr.Slider()` object.

**Before (line 144):**
```python
return (
    ...
    512,  # This sets slider VALUE to 512, but max is 511
)
```

Gradio interprets a raw number as the slider's **value**, not its maximum. Since the slider was defined with `maximum=511`, returning `512` as the value caused the validation error.

## Fix

Changed both return statements to return `gr.Slider()` objects that properly update the slider's configuration:

```python
# No model selected case:
gr.Slider(minimum=0, maximum=511, value=0, step=1)

# Model selected case:
gr.Slider(minimum=0, maximum=state.n_neurons - 1, value=0, step=1)
```

## Files Modified

- `dashboard/app.py`: Lines 144 and 206

## Verification

- All 21 dashboard tests pass
- Manual testing needed to confirm error no longer appears on page load
