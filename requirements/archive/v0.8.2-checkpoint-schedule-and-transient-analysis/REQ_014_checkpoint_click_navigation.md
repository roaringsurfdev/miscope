# REQ_014: Click-to-Navigate Checkpoint Markers

**Status:** Deferred
**Priority:** Low
**Estimated Effort:** Medium
**Related:** REQ_015 (Checkpoint Editor)

## Problem Statement

The Train/Test Loss curve displays checkpoint epochs as green diamond markers. Currently, users navigate between checkpoints using the epoch slider. Adding the ability to click directly on a checkpoint marker would provide a more intuitive navigation method, especially when visually identifying interesting points in the loss curve.

## Conditions of Satisfaction

- [ ] Clicking on a green diamond checkpoint marker navigates to that checkpoint
- [ ] All synchronized visualizations update when a checkpoint is clicked
- [ ] Visual feedback indicates clickable markers (cursor change, hover effect)
- [ ] Existing slider navigation continues to work
- [ ] Click target is forgiving (doesn't require pixel-perfect clicks)

## Technical Analysis

### Current Implementation
- **Plotting library:** Plotly (v6.5.2)
- **Dashboard framework:** Gradio
- **Checkpoint rendering:** `dashboard/components/loss_curves.py:88-101`
- **State management:** `dashboard/state.py` via `DashboardState.current_epoch_idx`

### Blocking Issue

**Gradio's `gr.Plot()` component does not expose Plotly click events.**

Plotly natively supports click detection via the `plotly_click` JavaScript event. However, Gradio's Plot component is display-only and provides no mechanism to capture these events and route them to the Python backend.

### Implementation Options

| Option | Effort | Description |
|--------|--------|-------------|
| **Custom JavaScript injection** | Medium | Inject JS to capture `plotly_click` events and call Gradio backend via JavaScript interop |
| **Custom Gradio component** | Medium-High | Build a PlotWithClick component that wraps Plotly with event support |
| **Migrate to Plotly Dash** | High | Replace Gradio entirely; Dash has native `clickData` callback support |

### Recommended Approach

**Custom JavaScript injection** is likely the lowest-effort path:
1. Add custom JS to the Gradio app that listens for `plotly_click` on the loss curve
2. Filter clicks to only the checkpoint marker trace
3. Extract the epoch from the clicked point
4. Update a hidden Gradio component to trigger the Python callback
5. Update `current_epoch_idx` and refresh visualizations

## Constraints

**Must have:**
- No regression in existing slider navigation
- Works with current checkpoint data structure

**Must avoid:**
- Breaking changes to the dashboard architecture
- Dependency on additional frameworks beyond current stack

**Flexible:**
- Exact visual feedback style (cursor, highlight, etc.)
- Whether to support clicking on the loss line itself (not just markers)

## Context & Assumptions

- The Train/Test Loss curve is increasingly recognized as a key navigation tool
- This feature would complement REQ_015 (Checkpoint Editor), which also requires loss curve interaction
- If REQ_015 is implemented first, the interaction infrastructure could be reused
- Assumption: Users will still use the slider for fine-grained navigation; click-to-navigate is for quick jumps

## Relationship to REQ_015

REQ_015 (Checkpoint Editor) requires similar interaction capabilities:
- Selecting points/ranges on the loss curve
- Visual feedback for interactive elements
- Event routing from Plotly to Python

**Recommendation:** Consider implementing REQ_014 and REQ_015 together, as the underlying interaction infrastructure would be shared. The combined effort may be more efficient than implementing them separately.

## Decision Authority

- [ ] User approval required before implementation begins
- [ ] Technical approach should be validated with a prototype

## Success Validation

- User can click checkpoint markers to navigate
- Navigation is responsive (< 500ms to update visualizations)
- Feature works across supported browsers
- No accessibility regressions (keyboard navigation still works)

---

## Notes

**2026-02-01:** Requirement created as deferred. Technical analysis identified Gradio's Plot component limitation as the primary blocker. Recommend bundling with REQ_015 if that requirement is prioritized.
