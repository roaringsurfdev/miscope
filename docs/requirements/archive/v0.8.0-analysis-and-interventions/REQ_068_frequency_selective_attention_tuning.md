# REQ_068: Frequency-Selective Attention Tuning

**Status:** Active
**Priority:** High — depends on REQ_067
**Related:** REQ_067 (Intervention Model Family Spec), REQ_055 (Attention Head Phase Analysis), REQ_056 (Frequency Specialization Sequencing), REQ_066 (Multi-Stream Specialization)
**Last Updated:** 2026-03-11

## Problem Statement

After first descent (train loss near zero), the quality of attention head frequency selection appears to have downstream consequences for whether and how fast a model grokks. Three distinct failure modes are hypothesized:

1. **Weak signal** — frequency selection is present but low-amplitude. The attention heads are routing toward the right frequencies, but the signal is too faint to give sufficient boost for MLP mass accumulation. Result: slow grokking or prolonged plateaus.
2. **Frequency competition** — too many frequencies receive symmetric or near-equal attention. The MLP cannot settle on a small winning set because it is receiving distributed signal across too many candidates. Result: long competition windows, MLP thrashing.
3. **Head imbalance** — attention heads are unevenly committed across the winning frequency set. Slower/lower frequencies may need more accumulation time and may not receive adequate early routing. Result: differential die-out, some frequencies failing to accumulate before second descent.

The hypothesis is that a targeted, gentle intervention on attention head outputs during the plateau — after gradients have found a direction but before MLP mass and thresholds have locked in — can steer the model toward healthier routing without disrupting the natural learning dynamics.

The Thrasher variant (p=59, seed=485) is the first target: it shows a missing frequency band and a long plateau, consistent with failure modes 1 or 2. Characterizing its attention Fourier profile at first descent will determine which intervention is appropriate.

## Conditions of Satisfaction

### Intervention Mechanism

- [ ] A hook is registered on `hook_attn_out` during training that applies a per-frequency gain function to the attention output
- [ ] The frequency projection uses W_E-based Fourier directions: frequency-aligned directions in d_model space are derived from `F @ W_E` where F is the Fourier basis (prime-parameterized, provided by the family) — not from runtime activation patterns
- [ ] The gain function is a per-frequency scalar vector: frequencies to amplify have gain > 1.0, frequencies to dampen have gain ∈ (0, 1), frequencies to leave unchanged have gain = 1.0
- [ ] No hard zeroing — minimum gain is bounded above 0.0 (a model default of 0.1 is a reasonable floor; exact value is a Claude decision)
- [ ] The hook is inactive outside the intervention window; training before and after the window is unmodified

### Intervention Window and Ramp

- [ ] The intervention is applied over a configurable window `[epoch_start, epoch_end]` read from the variant's intervention config (per REQ_067)
- [ ] Within the window, the gain ramps linearly from 1.0 (no intervention) to the target gain over `ramp_epochs` at window start, and back to 1.0 over `ramp_epochs` at window end
- [ ] The ramp prevents discontinuous jumps in the attention output at window boundaries
- [ ] `ramp_epochs` is configurable from the intervention config; default of ~50 epochs is acceptable

### Frequency Projection

- [ ] Given the family's Fourier basis F (shape: `(n_freqs, prime)`) and the embedding matrix W_E (shape: `(prime, d_model)`), the intervention computes frequency-aligned directions in d_model space: `D = F @ W_E`, normalized per row
- [ ] For each token's attention output vector `a ∈ R^{d_model}`, the intervention:
  1. Projects onto D to get per-frequency amplitudes
  2. Scales each amplitude by the corresponding gain
  3. Reconstructs the modified component and adds the delta back to `a`
- [ ] The reconstruction preserves the component of `a` orthogonal to the frequency directions — only the frequency-aligned component is modified

### Integration with REQ_067

- [ ] The intervention hook reads all parameters from the variant's intervention config — no hardcoded values
- [ ] A no-op intervention (all gains = 1.0) produces training results statistically indistinguishable from baseline — confirmed against at least one healthy variant
- [ ] Intervention artifacts are stored in the intervention family's results directory (per REQ_067) with the same per-epoch artifact structure as baseline variants

### First Test: Thrasher (p=59, seed=485)

- [ ] Before running the intervention, the Thrasher's attention Fourier profile is characterized at the plateau onset epoch to determine which failure mode applies (weak signal, competition, or imbalance)
- [ ] An intervention config is constructed based on that characterization (amplify underrepresented winning frequency, or dampen competing frequencies, as appropriate)
- [ ] The intervention variant is trained and the multi-stream specialization view (REQ_066) is used to compare the Thrasher's baseline trajectory against the intervention trajectory
- [ ] Qualitative assessment: does the intervention trajectory show earlier or stronger MLP mass accumulation in the target frequency compared to baseline?

## Constraints

**Must:**
- Use W_E-based frequency projection (Option 1) — not activation-derived directions. This keeps the projection family-defined and prime-parameterized, which generalizes across all variants within a family and across future model families
- Preserve the component of attention output orthogonal to frequency directions — the intervention must not collapse all structure in the attention output, only modify the frequency-aligned component
- Be configurable entirely from the intervention spec — no changes to training code per new experiment
- Soft dampening only — no hard zeroing

**Must not:**
- Modify the baseline training pipeline or any baseline family code
- Apply outside the configured intervention window
- Use activation patterns at inference time to determine frequency directions — this creates a circular dependency where the intervention reacts to what the model is currently doing rather than applying a principled external prior

**Decision authority:**
- **Claude decides:** Minimum gain floor (suggested 0.1), default ramp_epochs, exact reconstruction formula (full projection-and-delta vs. in-place rescaling of frequency components)
- **Resolved:** Intervention applies to all attention heads uniformly. A gain multiplier on all heads preserves the inter-head distribution of signal — a head contributing 80% of the signal for frequency k and one contributing 20% remain at 4:1 ratio after the gain is applied. This is the non-destructive property that makes the uniform approach safe, and avoids the risk of distorting the workload balance across heads.
- **Resolved:** "Healthy model as template" tooling is deferred — tracked as a downstream nice-to-have (see REQ_069). The config schema already supports it (`gain` as a dict), but the tooling to derive gain from a template model is out of scope for this requirement. First test uses manually specified gain values.

## Context

The research motivation is the temporal ordering observed in the multi-stream view (REQ_066): attention head frequency commitment precedes MLP neuron mass accumulation. If attention is routing toward the right frequencies before the MLP locks in, then the quality of that routing during the plateau may function as a gate on which frequencies can accumulate sufficient mass before second descent closes the window.

The Thrasher (p=59, seed=485) is the target because it has a documented anomaly: a missing frequency band and a long plateau. If the missing band is absent because attention was not routing toward it early enough, a gentle gain boost on that frequency during the plateau should accelerate or enable mass accumulation. If the long plateau is caused by frequency competition, dampening the competing frequencies should shorten the competition window. The two interventions are different enough that pre-characterization of the attention profile is required before choosing which to apply.

The choice of `hook_attn_out` over weight patching or attention pattern modification is deliberate: it is as close as possible to "what the MLP sees" without touching the learned Q/K/V weights. The Q/K weights continue to learn normally — only their downstream effect on the MLP input is modified during the window. This preserves the natural feedback loop between attention and the rest of the model.

The choice of W_E-based projection over activation-derived directions is deliberately principled over reactive. Activation-derived directions would adapt to what the model is currently doing at the moment of intervention, which creates a dependency on the model's transient state. W_E-based directions are stable, family-provided, and identical across variants with the same prime — which is a requirement if the intervention methodology is to generalize beyond single variants or single model families.

## Notes

- The "tuning" metaphor is intentional and load-bearing: the intervention is an EQ (equalizer) on the attention routing signal, not a transplant or a reset. The goal is to adjust the balance of an existing signal, not to introduce foreign structure.
- The Thrasher's missing frequency band is documented in research findings. Whether it is absent because attention never committed to it, or because MLP mass failed to accumulate despite attention routing, is a key interpretive question. The intervention result will help distinguish these.
- Intervention variants will produce their own parameter_snapshot and neuron_dynamics artifacts. The multi-stream specialization view should render correctly for intervention variants — this is a useful check on the artifact compatibility guaranteed by REQ_067.
- Future intervention types (e.g., frequency pruning during thrashing, head imbalance correction favoring low frequencies) can be added as new `type` values in the intervention config schema without changes to this mechanism, provided the hook dispatch routes by type.
- The no-op validation (all gains = 1.0 → statistically indistinguishable from baseline) is a critical sanity check and should be run before the first real intervention experiment.
