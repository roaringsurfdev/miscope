# REQ_081: Structural Training Diagnostics

**Status:** Draft
**Branch:** TBD (post `feature/neuron-group-pca`)
**Priority:** High — findings from 2026-03-24 rebounder analysis directly motivate this

---

## Problem

Training decisions — when to stop, when to extend, when to declare a model done — are
currently made on loss thresholds alone. Loss thresholds are insufficient because:

1. **They miss models that need more time.** p101/s485/ds42 appeared plateau'd at 25K
   by naive criteria but grokked at epoch 34173. Loss was barely moving but structural
   crystallization was still in progress.

2. **They cannot distinguish "not yet" from "won't recover."** A model with rising test
   loss after a bounce looks similar to a model still descending slowly. Loss alone
   cannot tell them apart; structural geometry can.

3. **They don't predict trajectory.** A model that has bounced and frozen (p101/s485/ds999)
   and a model that is mid-crystallization (p101/s485/ds42 at epoch 20K) have similar
   loss profiles but completely different futures. Structural indicators distinguish them.

The long-term goal — predicting clean organization *before* second descent completes —
is not in scope here, but this requirement establishes the foundation.

---

## Goals

Provide a structural diagnostic toolkit that answers three questions:

1. **Stop?** — Has the model bounced and lost its attractor? Is further training futile?
2. **Continue?** — Is the model still actively crystallizing? Would more epochs help?
3. **Done?** — Has the model fully organized? Is the current state the stable end state?

All signals must be computable from existing artifacts without new analyzers.

---

## Conditions of Satisfaction

### CoS 1 — Stop signal: rebounder detection

Given a variant, compute `bounce_magnitude = test_loss_final - test_loss_min`.
Flag as a **stop candidate** when:
- `bounce_magnitude` exceeds a threshold (TBD from distribution, ~1e-03 based on observed cases)
- AND graduation activity has ceased: no new pair graduations after `test_loss_min_epoch`
  (detectable from `input_trace_graduation` `graduation_epochs` array)
- AND post-bounce geometry shows no crystallization: all groups have PC3 > threshold
  at final epoch (from `neuron_group_pca` `pc_var`)

All three conditions together constitute a reliable stop signal. Any single condition
alone is insufficient.

### CoS 2 — Continue signal: active crystallization detection

Flag as a **continue candidate** when any of the following hold at the final epoch:
- Eff_dim still descending: `repr_geometry` `resid_post_mean_dim` slope at final epoch
  is negative beyond noise threshold
- At least one neuron group has PC3 > 5% and PC3 decreasing over last N epochs
  (ring still closing)
- PC2/PC3 loop area (from `parameter_trajectory` `all__projections`) still shrinking
- Graduation activity ongoing: pairs still graduating in the final checkpoint window

### CoS 3 — Done signal: crystallization complete

Flag as **done** when all of the following hold:
- All neuron groups have PC3 < 5% (all rings closed, collapsed into PC1/PC2 plane)
- Polar histogram spoke structure stable across last N checkpoints
- `bounce_magnitude` below threshold (no rebound since minimum)
- Eff_dim flat at final epoch

### CoS 4 — Degenerate group detection

Flag neuron groups with PC1 > 70% as **degenerate 1D** — these are groups that
collapsed to a single spoke (insufficient neuron density to form a ring). Report:
- Group frequency and size
- Whether the group size is below an empirical minimum threshold for ring formation
  (~60–80 neurons based on observed cases)

Degenerate groups are structural, not recoverable by extended training.

### CoS 5 — Notebook utility

A notebook cell (or importable function) takes a variant and returns a diagnostic
summary dict:
```python
{
  "verdict": "stop" | "continue" | "done" | "uncertain",
  "stop_signals": [...],
  "continue_signals": [...],
  "degenerate_groups": [...],
  "bounce_magnitude": float,
  "graduation_frozen_at": int | None,
  "groups_pc3": {freq: float, ...},
}
```
Printable as a human-readable report. Batch-runnable across all variants.

### CoS 6 — Dashboard surface

The Analysis page surfaces the diagnostic verdict as a status badge on the variant
header or secondary nav bar. At minimum: a color-coded indicator (green=done,
yellow=continue, red=stop, grey=uncertain) with tooltip showing the signals.

---

## Constraints

- All signals computable from existing artifacts: `neuron_group_pca`, `repr_geometry`,
  `parameter_trajectory`, `input_trace_graduation`, `variant_summary`
- No new analyzers required for MVP
- Thresholds (PC3 cutoff, bounce threshold, minimum group size) should be configurable
  constants, not hardcoded — they will need empirical refinement as more models are examined
- Should not require full artifact loading for batch use — prefer summary-level signals
  where possible

---

## Out of Scope

- **Early prediction** (predicting clean organization before second descent): requires
  analysis of geometry at eff_dim crossover epoch to detect early spoke formation.
  This is the natural next requirement once diagnostic signals are validated.
- Automated stopping during training (would require online artifact computation)
- Per-checkpoint monitoring (diagnostic is post-hoc over existing checkpoints)

---

## Notes

**Motivation context (2026-03-24):** Rebounder analysis across p101/s485/ds999,
p113/s485/ds999, and p113/s999/ds999 revealed a consistent structural signature for
"attractor lost" failure: bounce magnitude, graduation activity frozen at bounce epoch,
and geometry showing no crystallization attempt across 10K+ post-bounce epochs. These
three models clearly belong in the "stop" category. Contrast with p101/s485/ds42 (no
bounce, still crystallizing, correctly extended) and p59/s485/ds598 (loop tightening,
correctly extended).

**Failure mode taxonomy informing this requirement:**

| failure mode | verdict | characteristic |
|---|---|---|
| Slow/supercritical | continue | gradual descent, loop still open |
| Needs more time | continue | onset late, still descending, no bounce |
| Partial crystallization | uncertain | small bounce, some groups organized |
| Degenerate + no recovery | stop | large bounce, geometry frozen, degenerate groups |
| Fully crystallized | done | all rings closed, stable spokes, no rebound |

**On early prediction (future direction):** The geometry at the eff_dim crossover epoch
may already contain signals about whether clean organization is likely. At that epoch,
if spoke formation is beginning in the polar histograms, the model is on track for clean
crystallization. If the phase distribution is still uniform, the model may be heading
for slow/contested organization. This hypothesis is untested but well-motivated by the
two-step model (compression precedes crystallization). See `notes/findings_ring_geometry.md`.
