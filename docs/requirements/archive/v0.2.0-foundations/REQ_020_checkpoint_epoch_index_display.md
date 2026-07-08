# REQ_020: Checkpoint Epoch-Index Display

## Problem Statement

When navigating checkpoints using the Global Epoch Control slider, users cannot easily determine which slider index corresponds to which training epoch. The slider displays checkpoint indices (0, 1, 2...) but users think in terms of epochs (0, 100, 500...).

Currently:
- The Loss Curve tooltip shows checkpoint epochs when hovering over diamond markers
- The slider shows only the index number
- There's no way to quickly find "I want to jump to epoch 500" without trial and error

This creates friction when exploring training dynamics, especially when correlating observations across sessions or referencing specific epochs in notes.

## Conditions of Satisfaction

- [x] User can determine the epoch number for any checkpoint index
- [x] User can determine the checkpoint index for any visible epoch marker
- [x] Solution requires no additional clicks or mode changes (information is readily visible)

## Constraints

**Must have:**
- Works with existing Gradio components (no custom JS required)
- Information visible without extra interaction when possible

**Must avoid:**
- Breaking existing slider or loss curve functionality
- UI clutter that obscures primary visualizations
- Dependency on Plotly click events (known Gradio limitation per REQ_014)

**Flexible:**
- Exact UI placement and styling
- Whether to implement one or both approaches below
- Additional enhancements beyond core mapping display

## Proposed Solutions

Two complementary approaches (implement one or both):

### Option A: Enhanced Loss Curve Tooltip
Add checkpoint index to the existing tooltip on checkpoint markers.

Current: `Epoch: 500, Train Loss: 0.023, Test Loss: 0.019`
Enhanced: `Epoch: 500 (Index: 5), Train Loss: 0.023, Test Loss: 0.019`

**Pros:** Minimal change, uses existing hover behavior
**Cons:** Requires hovering over the specific marker

### Option B: Epoch Display Near Slider
Show the current epoch prominently near the slider control.

```
Global Epoch Control
[====|=============] Index: 5
Epoch: 500
```

**Pros:** Always visible, no hover required
**Cons:** Takes additional vertical space

### Recommendation

Implement both. Option A is near-zero effort and provides discovery. Option B provides persistent context while navigating.

## Context & Assumptions

- Checkpoint epochs are stored in training metadata (already loaded for loss curves)
- The slider's value is the index into the checkpoint list
- Users frequently want to jump to specific epochs they've identified as interesting
- This is a stepping stone; REQ_014 (click navigation) remains a future enhancement

**Relationship to other requirements:**
- Complements REQ_014 (Click-to-Navigate) - provides information even without click support
- Independent of REQ_015 (Checkpoint Editor)
- No architectural changes needed

## Decision Authority

- [x] Claude can make reasonable implementation decisions
- [ ] Flag if implementation reveals complications

## Success Validation

1. Start dashboard with a completed training run
2. Hover over a checkpoint marker on the loss curve
3. Tooltip shows both epoch number AND checkpoint index
4. Move the slider to a different position
5. Current epoch is displayed near the slider
6. User can now say: "I want epoch 500, that's index 5, let me set the slider to 5"

---

## Notes

**Implementation hints:**
- Loss curve tooltip: Modify `customdata` and `hovertemplate` in `dashboard/components/loss_curves.py`
- Epoch display: Add `gr.Markdown` or `gr.Textbox` below/beside slider that updates on slider change
- Checkpoint epochs available via `model_spec.get_available_checkpoints()` or metadata

**Post-implementation:**
- Consider if this obviates the need for REQ_014 or just complements it
- User feedback may reveal preference for one approach over the other

---

## Implementation Notes (Added by Claude)

**Implementation completed:** 2026-02-03

**Both options implemented:**

### Option A: Enhanced Loss Curve Tooltip
- **File:** `dashboard/components/loss_curves.py`
- **Change:** Added `customdata` with checkpoint indices to the checkpoint marker trace
- **Tooltip format:** `Epoch: X (Index: Y)`

### Option B: Epoch Display Near Slider
- **File:** `dashboard/app.py`
- **Change:** Added `format_epoch_display()` helper function
- **Display format:** `Epoch X (Index Y)`
- **Updates:** On slider change and on model load

**Key code locations:**
- `dashboard/components/loss_curves.py:88-108` - Checkpoint marker with index in tooltip
- `dashboard/app.py:297-300` - `format_epoch_display()` helper
- `dashboard/app.py:305-313` - `update_visualizations()` uses formatted display
- `dashboard/app.py:197-199` - `load_model_data()` returns initial epoch display

**Tests:** All 156 existing tests pass. No new tests added (UI display change).
