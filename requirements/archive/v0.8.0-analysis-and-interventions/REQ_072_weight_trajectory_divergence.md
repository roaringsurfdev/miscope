# REQ_072: Weight-Trajectory Divergence Analysis

**Status:** Active
**Branch:** TBD
**Attribution:** Drafted by Engineering Claude

---

## Problem

The early_gradient_analysis.py notebook revealed a key finding: at epoch 0, gradient energy is nearly identical across data seeds sharing the same model seed. The trajectories diverge sharply during first descent (divergence peaks around epoch 100), then collapse as all models enter the memorization plateau. This tells us the critical fork is during first descent — but the gradient metric goes quiet at the plateau, masking whether the models have actually reconverged or have simply reached a state where all gradients are small.

To answer that question, we need to compare the weight matrices themselves. Weight-space trajectory divergence (L2 distance between variant weight matrices at each checkpoint) will show whether models are on genuinely different paths after first descent, or whether the gradient divergence was transient.

This analysis formalizes the exploration from early_gradient_analysis.py into the standard analyzer/renderer/view pipeline so it can be run systematically across any variant pair.

---

## Conditions of Satisfaction

1. **Epoch 0 gradient energy view** — For a set of variants sharing the same model seed, compute per-frequency gradient energy at epoch 0 (Fourier projection of ∂L/∂W_in through W_E). Rendered as overlaid line plots, one line per variant (differentiated by data seed or other parameter). Vertical markers at known key frequencies for the prime.

2. **Gradient energy difference view** — For a designated "reference" variant, plot the per-frequency difference (other − reference) as a bar chart at epoch 0. Shows which frequencies are pushed harder or softer by each data split at the very first step.

3. **Weight-space trajectory divergence view** — For each early checkpoint (epoch 0 through configurable limit), compute the L2 distance between each variant's flattened weight vector and the reference variant's weight vector. Plotted as a trajectory over epochs, one line per comparison pair. Answers: do the weights actually fork, and at what epoch?

4. **Per-weight-matrix divergence breakdown** — Decompose the weight-space L2 distance by matrix (W_E, W_Q, W_K, W_V, W_O, W_in, W_out, W_pos, W_U) to show which parts of the model diverge first and fastest.

5. **Key-frequency gradient trajectory view** — For a set of key frequencies (configurable, defaulting to the variant's known dominant frequencies), plot gradient energy at each of those frequencies over early epochs (e.g., 0–1000) for each variant. The four-panel per-frequency view from the notebook. Shows whether gradient signal at key frequencies diverges between seeds during first descent.

6. All views integrate with the existing view catalog (ViewDefinition pattern), load from existing checkpoints, and are accessible from the dashboard.

---

## Constraints

- **No new checkpoint infrastructure required.** Analysis operates on whatever checkpoints exist. The early epoch window (0–1000 every 100 epochs) is already present for existing variants; denser windows can be produced by retraining with a custom checkpoint schedule, which is already supported.
- **Epoch 0 gradient computation requires a forward+backward pass** on the training data against the epoch 0 weights. This is not a stored artifact — it must be computed. The analyzer should handle this without requiring the model to be in training mode (use `torch.no_grad()` for the forward pass for efficiency, then a scoped backward pass just for gradient capture).
- **Weight-space comparison requires all variants to share a model architecture.** The analyzer should validate this.
- **Views should accept a `reference_variant` and a list of `comparison_variants`** as inputs, rather than assuming a fixed good/bad seed structure. The data seed comparison is the motivating case but not the only valid one.

---

## Notes

**What the gradient analysis revealed:** The epoch 0 gradient profiles are near-identical across data seeds for the same model seed (as predicted by the flatness argument — random 30% sampling gives approximately uniform gradient signal). The divergence peaks at epoch 100 and collapses by epoch 300–400. This means either (a) the weights reconverge after first descent, or (b) all models enter a plateau where gradients are small regardless of trajectory. Weight-space trajectory divergence will distinguish these cases.

**Relationship to REQ_063 (Fourier Nucleation):** The nucleation predictor used iterative Fourier masking as a synthetic proxy for first-gradient effects. This requirement provides the actual measurement. If weight-trajectory divergence consistently emerges at epoch 100 across variant pairs, the "critical first-descent window" is a real phenomenon and REQ_063's synthetic approach was approximating something real but not measuring it directly.

**DMD connection (REQ_073):** Weight-space trajectory divergence is a scalar summary of the fork. DMD on the weight evolution would decompose the fork into its dominant modes — potentially revealing whether the divergence is broad (all weights changing) or localized (specific matrices driving the split). REQ_072 is the prerequisite measurement; REQ_073 builds the modal analysis on top of it.
