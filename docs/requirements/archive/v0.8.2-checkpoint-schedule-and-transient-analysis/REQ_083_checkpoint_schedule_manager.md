# REQ_083: Checkpoint Schedule — Retrain Existing Variants

**Status:** Active
**Priority:** High
**Estimated Effort:** Medium
**Supersedes:** REQ_015 (cancelled 2026-03-30 — different problem, Gradio-era)

## Problem Statement

Identifying when and where to add checkpoint density requires seeing a model's loss curve first — you can't know where the second descent is until the model has run. Currently, retraining with a denser schedule requires:

1. Looking up the variant's parameters (p, seed, data_seed) manually
2. Writing a checkpoint range definition in `model_checkpoints.py`
3. Printing the generated list and copy-pasting it into the Training page's free-text field
4. Hoping you picked the right variant to paste into

This is cognitively expensive and error-prone. Mistakes have already been made selecting the wrong model. The core failure is that "retrain existing with new schedule" and "train new variant" are conflated into the same form.

A separate page for retraining existing variants should:
- Eliminate manual parameter entry by selecting from existing variants
- Show the loss curve with existing checkpoint density visible, so the second descent is immediately locatable
- Replace the free-text checkpoint field with a structured range builder

## Context

The checkpoint schedule is *additive only*. When a variant is retrained, only new checkpoint files are written; the directory is not wiped. Existing checkpoint files may be overwritten (same content, assuming deterministic training). The user maintains this invariant manually by ensuring each new schedule is a superset of the previous one. The new page must make this invariant automatic.

**Related finding:** Fast reorganization events (second descent in grokking variants) are easily missed by a sparse checkpoint schedule. The canonical default schedule has 100-epoch density in the expected grokking window, but individual variants have second descents at different epochs. Variants under extended training have no dense region at all beyond the default window.

## Conditions of Satisfaction

### Page Separation
- [ ] A new "Retrain" page is added to the dashboard, separate from the existing Training page
- [ ] The existing Training page ("New Variant") is unchanged
- [ ] Both pages are accessible from the side navigation

### Variant Selection
- [ ] The Retrain page reads from the global `variant-selector-store` — if a variant is already selected in the dashboard, it is pre-loaded on page arrival
- [ ] The user can change the selected variant from the Retrain page (via the existing variant selector component or an equivalent)
- [ ] Variant parameters (p, seed, data_seed, training_fraction) are displayed as read-only — they cannot be edited on this page
- [ ] Total Epochs is editable — extending training is a checkpoint schedule operation, not a regime change

### Loss Curve with Checkpoint Overlay
- [ ] The variant's train and test loss curves are displayed
- [ ] Existing checkpoint epochs are shown as vertical tick marks or a rug plot at the base of the loss curve
- [ ] Regions of dense vs. sparse coverage are visually apparent from the overlay

### Range-Based Schedule Builder
- [ ] The user defines additional checkpoint ranges via a table of rows: `[start epoch | end epoch | step | delete]`
- [ ] Rows can be added and removed
- [ ] The builder generates the union of all range rows
- [ ] The generated list is automatically merged with the existing checkpoint epochs (superset guaranteed — no manual union required)
- [ ] A preview of the resulting full checkpoint set is overlaid on the loss curve (alongside existing checkpoints, visually distinct)
- [ ] The total checkpoint count (existing + new) is displayed
- [ ] If any defined range extends beyond Total Epochs, the page warns before allowing retrain (guard against the forget-to-extend-epochs mistake)

### Retrain Execution
- [ ] A "Retrain" button launches training with the merged checkpoint epoch list
- [ ] Training parameters are sourced from the variant's existing config (not re-entered)
- [ ] Progress tracking reuses the existing training progress mechanism
- [ ] On completion, the variant registry is refreshed

### Safety
- [ ] If no new epochs would be added (the user's ranges are entirely covered by existing checkpoints), warn before allowing retrain
- [ ] The page does not expose any control that would change the training regime (fraction, architecture)

## Proposed User Workflow

1. User is analyzing a variant on the Visualization or Neuron Dynamics page
2. User notices sparse coverage around a region of interest on the loss curve
3. User navigates to Retrain page — selected variant is already loaded
4. Loss curve appears with existing checkpoint rug; the sparse region is immediately visible
5. User adds a range row covering that region (e.g., start: 14000, end: 18000, step: 100); if extending training, updates Total Epochs and adds a range in the new window
6. If any range extends past Total Epochs, the page warns immediately
7. Preview overlay shows new checkpoints filling the gap
8. User clicks Retrain — training runs with the merged schedule
8. New checkpoint files appear; existing files are untouched

## Constraints

**Must have:**
- Global variant selector integration (no manual p/s/ds entry)
- Loss curve + checkpoint overlay (this is the primary value over the current workflow)
- Automatic superset enforcement (existing epochs always included)
- Training params read-only on this page

**Must not:**
- Expose training regime controls (fraction, architecture)
- Wipe or overwrite the variant directory
- Require the user to construct the full epoch list manually

**Flexible:**
- Exact visual treatment of checkpoint overlay (rug plot, vertical lines, shaded regions)
- Whether ranges can overlap (union semantics handle this gracefully)
- Whether to support importing the current `model_checkpoints.py`-style range definitions

## Out of Scope (v1)

- Draggable handles on the loss curve for range selection (range table is sufficient for v1)
- Range builder for new variant training (user won't know where density is needed until after first run)
- Retroactive checkpoint extraction without retraining (not possible)
- Sharing or exporting checkpoint schedule definitions

## Decision Authority

- [x] Page name / navigation label — **"Checkpoint Schedule"** (captures both dense checkpointing and extended training)
- [ ] Visual treatment of checkpoint overlay — engineering judgment, user review
- [ ] Whether the range builder replaces or supplements the free-text input for power users — user decision

## Success Validation

- User can retrain a variant with a denser checkpoint schedule without typing p, seed, or data_seed
- Loss curve with checkpoint overlay makes second descent region locatable at a glance
- The generated schedule is always a superset of the existing schedule (no manual union required)
- The mistake scenario (wrong variant retrained due to manual entry error) is structurally prevented

---

## Notes

**2026-03-30:** Requirement written. Arose from need to capture second descent in variants where default schedule is too sparse, and from systematic errors in the current manual copy-paste workflow. Supersedes REQ_015 (analysis checkpoint selector, different problem, Gradio-era).

The `model_checkpoints.py` notebook file captures the current range-based mental model well (multi-range with density comments). The requirement formalizes this pattern into a structured UI rather than replacing it.

The training page's `parse_checkpoint_epochs()` utility and the thread-based training execution are reusable in the new page.
