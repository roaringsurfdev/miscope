# REQ_073: Weight-Space DMD Analysis

**Status:** Superseded by [REQ_117](REQ_117_dmd_reorganization.md) (DMD Reorganization — Activation and Parameter, Windowed and Per-Regime). Moved to staging 2026-05-27 to ride alongside REQ_117 into the next milestone archive at release time. Follows the supersession-staging precedent set by REQ_097, REQ_098, REQ_104 (all superseded by REQ_109 and parked in staging together).
**Branch:** TBD
**Attribution:** Drafted by Engineering Claude

---

## Supersession Note

This REQ specified standard DMD on weight trajectories as a parallel to the existing centroid (activation) DMD. REQ_117 absorbs and extends that scope into a unified, two-track treatment: activation DMD and parameter DMD ship together, both with windowed + per-regime structure. The parameter track in REQ_117 carries forward this REQ's CoS items — per-matrix DMD support, mode-Fourier projection, mode-spectrum and dominant-mode-trajectory views — and adds the windowed and per-regime layers that make the parameter side consistent with the activation side.

Sequencing change: REQ_117's parameter track has a hard dependency on [REQ_118](REQ_118_neuron_grouping.md) (the neuron grouping primitive). Per-group windowed DMD is the architecturally correct shape; this REQ's per-matrix framing remains as a supporting view inside REQ_117 rather than the primary one.

The REQ_073 content below is preserved for archaeology.

---

## Problem

The existing Centroid DMD (centroid_dmd page) captures how residue class representations evolve in activation space — the geometry of what the model outputs. This is a useful view of the downstream consequence of learning. But the driver of grokking is the weight matrices themselves. The Centroid DMD shows the manifold expanding and collapsing in representation space; it doesn't directly capture the parameter dynamics that cause that expansion and collapse.

Dynamic Mode Decomposition applied to the weight matrices would decompose the parameter evolution into its dominant spatial-temporal modes. Each DMD mode is a direction in weight space with an associated oscillation frequency and growth/decay rate. This would reveal:

- Which weight directions are actively evolving (growing modes) vs. converging (decaying modes) at any point in training
- Whether the critical first-descent window and the second-descent transition correspond to recognizable shifts in the dominant DMD modes
- Whether the fork between good-seed and bad-seed trajectories (from REQ_072) is reflected in different DMD mode structures early in training

The hypothesis is that weight-space DMD will capture the lottery-resolution dynamics more directly than centroid DMD, because the lottery is decided in weight space, not activation space. Centroid DMD is downstream of the decision; weight-space DMD is closer to the mechanism.

---

## Conditions of Satisfaction

1. **Weight-space DMD computation** — Given a sequence of weight snapshots (one per checkpoint), construct the data matrix by flattening each snapshot to a vector and stacking. Apply DMD to extract modes, eigenvalues (growth rates + frequencies), and mode energies. Support configurable checkpoint ranges (e.g., first 1000 epochs for early dynamics, full training run for global picture).

2. **Per-matrix DMD** — In addition to full-weight DMD, support computing DMD on individual weight matrices (W_E, W_in, W_out, W_Q/K/V) separately. This tests whether the critical dynamics are localized to specific matrices or are global.

3. **Mode spectrum view** — Plot DMD eigenvalues in the complex plane. Growing modes (|λ| > 1) vs. decaying modes (|λ| < 1) vs. oscillatory modes (|λ| ≈ 1). Color by mode energy. This gives an immediate read on whether training is in an active reorganization phase or a stable convergence phase.

4. **Dominant mode trajectory view** — For the top-N modes by energy, plot their energy over the training window. Transitions in which modes are dominant correspond to phase changes in the learning dynamics. Compare across variants to see whether good-seed vs. bad-seed trajectories show different mode structure.

5. **Mode-frequency Fourier projection** — For modes associated with W_E or W_in, project the mode vector back onto the Fourier basis to identify which Fourier frequencies the dominant DMD mode is activating. This connects the weight-space modal structure back to the Fourier geometry lens and the REQ_072 gradient analysis.

6. All views integrate with the existing view catalog and are accessible from the dashboard.

---

## Constraints

- **DMD requires a minimum number of snapshots** relative to the dimensionality of the weight space. Full-weight DMD on a 128-dim model may need truncated SVD (randomized SVD) to handle the parameter vector size. Per-matrix DMD on smaller matrices (W_E: 114×128, W_in: 128×512) is more tractable.
- **Checkpoint density determines temporal resolution.** Early-window DMD is only meaningful with early checkpoints (every 100 epochs from 0–1000 is the minimum; denser is better for detecting fast dynamics). This is a data-collection constraint, not a code constraint.
- **DMD assumes approximately linear dynamics** in the snapshot-to-snapshot transitions. Weight evolution during learning is nonlinear. DMD will capture the locally-linear approximation, which may be more meaningful in stable phases (plateau, post-grokking) than during rapid transitions. The mode spectrum view should be interpreted accordingly.
- **REQ_072 is a prerequisite** — the weight trajectory divergence analysis should be run first to identify which variant pairs show the most interesting early-epoch dynamics. REQ_073 should be targeted at those pairs.

---

## Notes

**Why weight-space DMD vs. centroid DMD:** Centroid DMD operates on (n_residues × d_model) activation snapshots — it captures the geometry of the model's output representation. Weight-space DMD operates on the parameter vectors directly — it captures the dynamics of the learning process itself. The centroid view shows *what* the model has learned at each epoch; the weight view shows *how* the learning is proceeding. For understanding critical windows and phase transitions, the parameter dynamics are more directly informative.

**Potential finding:** If the first-descent window corresponds to a shift from high-energy growing modes to lower-energy decaying modes in weight space (the model committing to a direction), this would be a mechanistic signature of the lottery resolving. The DMD mode spectrum at epoch 100 (the peak divergence epoch from REQ_072) vs. epoch 0 and epoch 500 would be the key comparison.

**Stretch goal:** If weight-space DMD modes can be reliably associated with specific Fourier frequencies (via the mode-frequency projection in CoS 5), then DMD mode energy trajectories become a per-frequency weight dynamics view — showing not just which Fourier frequency the model is using, but which frequencies are actively being built vs. consolidated at each training phase. This would be the closest thing to a mechanistic real-time view of the lottery playing out.
