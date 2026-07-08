# REQ_070: Intervention Effect Verification Page

**Status:** Active
**Priority:** High — blocks informed interpretation of intervention experiments
**Related:** REQ_067 (Intervention Family Spec), REQ_068 (Frequency-Selective Attention Tuning)
**Last Updated:** 2026-03-12

## Problem Statement

After running three frequency-gain hook interventions on the Thrasher (p=59/seed=485), the primary observable effect was a **rotation of the representational torus** in the Centroid Class PCA 3D Scatter — not the expected dampening or boosting of targeted frequencies. This raises a fundamental question: is the hook actually applying the intended gain to the intended frequencies?

Two possible explanations exist:

1. **Alignment gap**: W_E-based frequency directions (computed at epoch 1500) have diverged from the attention head's actual frequency directions by that point, causing the hook to act as a rotation rather than a frequency-selective gain.
2. **The intervention is working but gradient descent absorbs the signal**: The frequency modification is applied correctly but upstream weights adapt quickly enough that the effect on the residual stream is not preserved into the trained representation.

Before running further experiments or revising the hook design, we need a tool to directly verify what the hook is doing to the attention output signal at each checkpoint epoch.

## Conditions of Satisfaction

### Core Verification View

- [ ] A dashboard page that accepts an intervention variant and a checkpoint epoch as inputs
- [ ] At the selected epoch, loads the model from the checkpoint and runs the full analysis dataset through it with `run_with_cache`, capturing `hook_attn_out` activations
- [ ] Projects `hook_attn_out` onto the W_E-based frequency directions (D_sin, D_cos) to compute per-frequency amplitudes — both the baseline (no hook) and the hook-modified signal
- [ ] Displays per-frequency amplitude as a side-by-side bar chart: baseline vs. hook-modified
- [ ] Highlights the target frequencies from the intervention config
- [ ] Displays the ramp factor for the selected epoch, making clear whether the epoch is in the ramp-in zone, full-gain zone, or outside the window

### Frequency Direction Display

- [ ] Shows the W_E-based frequency directions used by the hook (i.e., the projection basis), so the user can reason about alignment between the hook's basis and the model's actual frequency structure
- [ ] Optionally: overlays the same amplitudes computed from a baseline (non-intervention) variant checkpoint at the same epoch, to show what the signal looked like before the intervention was applied during training

### Input Controls

- [ ] Epoch slider scoped to the intervention variant's available checkpoints
- [ ] Variant selector filtered to intervention variants (or all variants, with non-intervention ones showing only baseline signal)

### Scope

- [ ] This is a **read-only analytical tool** — it does not run training or apply the hook to modify training. It applies the hook forward-pass-only to verify the mechanism.
- [ ] Operates on `hook_attn_out`, not QK^T. The hook intercepts downstream of QK^T computation; the verification should match the hook's actual insertion point.

## Constraints

**Must:**
- Reconstruct D_sin, D_cos from the plateau checkpoint (the same epoch used during training, typically `intervention_config["epoch_start"]`) — not from the currently selected epoch — so the verification exactly matches the hook's projection basis
- Read the intervention config from the variant's config.json to determine target frequencies, gain values, epoch window, and ramp schedule

**Must not:**
- Apply the hook as a training modification — forward-pass only
- Require re-running any training

**Decision authority:**
- **Claude decides:** visualization layout (side-by-side bars vs. overlay), aggregation across batch/seq dimensions (mean amplitude per frequency), exact color scheme for target vs. non-target frequencies

## Context

The hook operates on `hook_attn_out` (post-softmax, post-V-multiplication), applying a gain in W_E-defined frequency directions. If W_E and the attention computation's actual frequency subspace have diverged by epoch 1500 (due to learned Q/K/V rotations), the projection would land in a misaligned basis, producing a rotation in residual stream space rather than a per-frequency gain.

The verification page is the diagnostic instrument to determine which of the two explanations above is correct — and therefore whether future interventions should revise the hook location, the projection basis, or both.

## Notes

- If the verification shows the hook is correctly applying gain but the geometric effect is still a rotation, the cause is the alignment gap hypothesis. That would motivate computing frequency directions from the attention heads' actual Q/K weights rather than from W_E.
- If the verification shows the hook is not producing the expected gain even in the immediate forward pass, that points to a bug or a misunderstanding of the hook insertion point.
- A no-hook baseline epoch (outside the intervention window) should show gain ratio = 1.0 for all frequencies — this is a useful sanity check built into the tool.
