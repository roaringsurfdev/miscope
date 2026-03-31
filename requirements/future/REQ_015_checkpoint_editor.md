# REQ_015: Checkpoint Editor Using Train/Test Loss Curve

**Status:** CANCELLED — Superseded by REQ_083
**Superseded by:** REQ_083 (Checkpoint Schedule Manager — Retrain Existing Variants)
**Cancelled:** 2026-03-30

## Why This Requirement Was Cancelled

REQ_015 was written during the Gradio era and addressed a different problem: visually selecting which *existing* checkpoints to *analyze*. The real need that emerged is different — retraining a variant with a denser checkpoint schedule to capture fast reorganization events (e.g., second descent in grokking models) that sparse checkpointing misses entirely.

REQ_083 supersedes this with the correct framing: a dedicated page for retraining existing variants with a visual range-based checkpoint schedule builder, integrated with the global variant selector to eliminate the error-prone manual parameter entry that motivated this requirement in the first place.

This requirement should not be resurrected without first reviewing REQ_083.

---

**Original Status:** Future
**Priority:** Medium
**Estimated Effort:** High (potential mini-release)
**Related:** REQ_014 (Click-to-Navigate)

## Problem Statement

Currently, checkpoint selection is implicit based on which epochs have saved model weights. Users analyzing training dynamics often want to focus on specific regions of interest in the loss curve (e.g., the grokking transition, early training, plateau regions).

The Train/Test Loss curve provides essential context for understanding where interesting phenomena occur. Users should be able to visually select regions on this curve to define which checkpoints to analyze, rather than relying on whatever checkpoints happen to exist.

## Conditions of Satisfaction

### Core Functionality
- [ ] User can select a start point on the loss curve (by clicking or dragging)
- [ ] User can select an end point on the loss curve
- [ ] User can specify a step size (e.g., every 100 epochs, every 500 epochs)
- [ ] System generates list of checkpoint epochs based on start/end/step
- [ ] Generated checkpoint list is displayed for user confirmation
- [ ] Checkpoint configuration is persisted per model

### Visual Interface
- [ ] Selected range is visually highlighted on the loss curve
- [ ] Start/end points are clearly marked (draggable handles)
- [ ] Preview of selected checkpoints shown as markers within the range
- [ ] Clear visual distinction between "available" and "selected" checkpoints

### Data Management
- [ ] Checkpoint configuration stored in model's metadata or dedicated config file
- [ ] Configuration includes: start_epoch, end_epoch, step_size, generated_epochs[]
- [ ] Dashboard loads checkpoint configuration on model selection
- [ ] Slider and navigation respect the configured checkpoint set

## Proposed User Workflow

1. **Open model in dashboard** - Loss curve displays with any existing checkpoints
2. **Enter checkpoint editor mode** - UI shifts to editing state
3. **Select start point** - Click or drag on loss curve to set start epoch
4. **Select end point** - Click or drag to set end epoch
5. **Set step size** - Input field or preset buttons (50, 100, 500, etc.)
6. **Preview checkpoints** - See generated epochs as markers on curve
7. **Confirm selection** - Save configuration
8. **Exit editor mode** - Return to analysis view with new checkpoint set

## Technical Considerations

### Interaction Infrastructure
This requirement shares interaction challenges with REQ_014:
- Plotly click/drag events not exposed through Gradio's `gr.Plot()`
- Will require one of: custom JS, custom Gradio component, or framework change

### Data Storage Options

**Option A: Explicit checkpoint list (Recommended for v1)**
```json
{
  "checkpoint_config": {
    "mode": "explicit",
    "epochs": [0, 500, 1000, 1500, 2000, 2500, 3000]
  }
}
```
- Simplest to implement
- Easy to understand and debug
- Can be manually edited if needed

**Option B: Range specification**
```json
{
  "checkpoint_config": {
    "mode": "range",
    "start": 0,
    "end": 3000,
    "step": 500
  }
}
```
- More compact
- Requires generation logic on load
- Harder to represent non-uniform intervals

**Option C: Multiple ranges**
```json
{
  "checkpoint_config": {
    "mode": "multi_range",
    "ranges": [
      {"start": 0, "end": 1000, "step": 100},
      {"start": 1000, "end": 3000, "step": 500}
    ]
  }
}
```
- Most flexible
- Supports dense sampling in interesting regions, sparse elsewhere
- More complex UI needed

### Relationship to Available Checkpoints

The editor defines which checkpoints the user *wants* to analyze. This is independent of which checkpoints *exist* (have saved weights). The system should:
1. Generate the desired checkpoint list from editor configuration
2. Intersect with available checkpoints (those with saved weights)
3. Warn user if desired checkpoints don't exist
4. Potentially support "run analysis on these epochs" workflow in future

## Constraints

**Must have:**
- Visual selection on the loss curve (not just form inputs)
- Persistence of checkpoint configuration
- Integration with existing slider/navigation

**Must avoid:**
- Breaking existing functionality for models without checkpoint config
- Requiring checkpoints that don't exist (graceful degradation)
- Over-complicated UI that obscures the loss curve context

**Flexible:**
- Exact storage format (Option A/B/C above)
- Whether to support multiple discontinuous ranges in v1
- Whether step size must be uniform

## Context & Assumptions

- The Train/Test Loss curve is the primary tool for understanding training dynamics
- Users often know "I want to look at epochs 1000-2000 in detail" from the loss shape
- Current checkpoint availability is often accidental (whatever was saved during training)
- This feature elevates the loss curve from passive display to active analysis tool

## Implementation Phases

### Phase 1: Basic Editor (MVP)
- Single contiguous range selection
- Uniform step size
- Explicit epoch list storage (Option A)
- Basic visual feedback

### Phase 2: Enhanced Selection
- Draggable start/end handles
- Multiple range support
- Non-uniform step sizes (dense in region of interest)

### Phase 3: Integration
- "Generate missing checkpoints" workflow (re-run training saves)
- Preset configurations (grokking region, early training, etc.)
- Share/export checkpoint configurations

## Decision Authority

- [ ] User approval required before implementation
- [ ] Phase 1 scope should be confirmed before starting
- [ ] Storage format decision needed (recommend Option A for simplicity)

## Success Validation

- User can define a checkpoint range visually on the loss curve
- Configuration persists across sessions
- Dashboard navigation respects configured checkpoints
- Feature works for new and existing models
- UI remains intuitive despite added complexity

---

## Notes

**2026-02-01:** Requirement created as future work. This is a substantial feature that may warrant its own mini-release. The interaction infrastructure challenges overlap with REQ_014, suggesting these could be tackled together.

**Key insight from user:** "The Train/Test Loss curve might need to be elevated in importance, because it provides a great deal of context for knowing where to dive into deeper analysis." This requirement directly addresses that insight by making the loss curve an active tool for defining analysis scope.
