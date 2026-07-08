# REQ_009: Loss Curves with Epoch Indicator

## Problem Statement
When exploring training dynamics using the global checkpoint slider (REQ_008), users need to see where they are in the overall training timeline. The train/test loss curves provide critical context for interpreting other visualizations, but currently lack a visual indicator showing the current slider position.

The loss curve visualization should display a vertical line that moves in sync with the global slider, helping users correlate specific checkpoints with their position in the learning curve.

## Conditions of Satisfaction
- [x] Train/test loss curve chart displays both training and test loss over epochs
- [x] Vertical line indicator overlaid on loss curves shows current epoch position
- [x] Line indicator synchronizes with global checkpoint slider from REQ_008
- [x] Moving global slider updates the vertical line position in real-time
- [x] Line is visually distinct but doesn't obscure the loss curves
- [x] Loss curve data loaded from training artifacts/logs
- [x] Interactive Plotly features (zoom, hover, pan) remain functional
- [x] Indicator position updates smoothly without lag

## Constraints
**Must have:**
- Integration with global checkpoint slider (REQ_008)
- Plotly format for consistency with other visualizations
- Both train and test loss displayed on same chart
- Vertical line indicator synchronized with slider
- Fast updates (reads from artifact, no recomputation)

**Must avoid:**
- Obscuring loss curve data with indicator line
- Lag or jank when moving slider
- Requiring recomputation when indicator moves
- Independent control that could desync from global slider

**Flexible:**
- Color and style of vertical line indicator
- Whether loss curves use log scale or linear scale
- Additional annotations or markers on the curves
- Layout relative to other visualizations in dashboard

## Context & Assumptions
- Training logs or artifacts contain loss values per epoch/checkpoint
- Loss data artifact is separate from other analysis artifacts (available immediately after training)
- Global slider position is in epochs or checkpoint indices
- Slider updates are synchronous across all visualizations (REQ_008)
- Users want to see loss curve context while exploring other visualizations
- Assumption: Single vertical line is sufficient for MVP (no range/interval indicators)

**Research workflow this enables:**
- "At epoch X where frequencies emerge (REQ_004), what's happening to the loss?"
- "This interesting activation pattern (REQ_005) occurs at what point in the learning curve?"
- "Do neuron clusters (REQ_006) shift when loss plateaus?"

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Load dashboard with completed training run
- Loss curves are visible showing full training history
- Global checkpoint slider is at initial position (e.g., epoch 0)
- Vertical line indicator appears at corresponding position on loss curves
- Move slider to different epoch
- Vertical line moves to new position on loss curves in sync
- Can visually correlate: "This checkpoint is during the steep loss drop phase"
- Can identify: "This checkpoint is during the grokking plateau"
- Slider movement is smooth and responsive
- Loss curves remain readable and interactive
- Can zoom into loss curves and indicator line scales appropriately
- Other visualizations (REQ_004, REQ_005, REQ_006) update simultaneously with loss indicator

---
## Notes

**Post-MVP enhancements:**
- Additional vertical lines or shaded regions marking phase shifts (e.g., grokking boundaries)
- Automatic detection and annotation of phase transitions
- Color-coded regions indicating training phases (memorization, circuit formation, generalization)
- Clicking on loss curves to jump slider to that epoch
- Derivative or smoothed curves to highlight rate of change
- Confidence intervals or error bars if multiple runs are averaged

**Implementation considerations:**
- Plotly's `add_vline()` or shapes can create vertical line indicator
- Line position can be updated via Plotly relayout or by regenerating figure
- Consider performance: regenerating entire figure vs. updating shape coordinates
- Loss data likely stored separately from analysis artifacts (available sooner)

## Implementation Notes (Added by Claude)

**Implementation completed:** 2026-01-31

**Key code location:**
- `dashboard/components/loss_curves.py`

**Function implemented:**
```python
def render_loss_curves_with_indicator(
    train_losses: list[float] | None,
    test_losses: list[float] | None,
    current_epoch: int,
    checkpoint_epochs: list[int] | None = None,
    log_scale: bool = True,
) -> go.Figure
```

**Features:**
- Blue line for train loss, orange for test loss
- Red vertical line at current epoch via `fig.add_vline()`
- Annotation showing "Epoch {n}" at top right
- Green diamond markers for checkpoint epochs
- Log scale by default (configurable)
- Hover template showing exact loss values

**Design decisions:**
- Uses `add_vline()` for epoch indicator (regenerates figure on each update)
- Checkpoint markers help identify which epochs have saved states
- Compact height (300px) to fit above analysis visualizations
- Loss data loaded from metadata.json (available immediately after training)

**Tests:** 4 tests for loss curves in `test_dashboard.py`
