# Thoughts - Findings

Unstructured parking lot for ideas and observations about what the research is revealing. See Claude.md for context.

---
## 2026-02-11: PC2 vs PC3 Loop Structure and p101/999 Anomaly

### Universal Self-Intersecting Loop

Across all 11 variants with parameter_snapshot data, the PC2 vs PC3 trajectory forms a **self-intersecting loop** — the model passes through roughly the same region of PC2/PC3 space during early training and again during the grokking transition. This is consistent across all primes (97, 101, 103, 109, 113, 127) and both seeds (485, 999).

**One exception: p101/seed999.** Its trajectory never self-intersects. The early-epoch excursion goes so far (PC2 ≈ 30, PC3 ≈ -15) that the return path doesn't reach the original neighborhood. The loop stays open.

**Caveat:** Each variant's PCA is computed independently, so loop "size" isn't directly comparable across variants. To rigorously compare, would need shared PCA basis or geometry-invariant metrics (e.g., loop area normalized by path length).

### p101/999: Coherent Anomaly Across All Analyses

p101/999 deviates from the other 10 variants on every dimension examined:

| Dimension | Normal variants (10/11) | p101/999 |
|-----------|------------------------|----------|
| PC2/PC3 trajectory | Self-intersecting loop | Open loop — never self-intersects |
| Grokking speed | 3-5K epochs (turn to convergence) | >10K epochs |
| Frequency specialization | Clean neuron clusters | Many neurons still mixed |
| Frequency concentration | Balanced across frequencies | Unusually concentrated in high frequencies |
| Embedding balance | cos(k)/sin(k) roughly comparable | cos 13 = 2.2957, sin 13 = 0.3573 (~6:1 ratio) |
| Frequency evolution | Progressive specialization (acquiring structure) | Lost a low frequency (shedding structure) |
| Training duration | 25K epochs | Extended to 35K epochs |

The cos/sin imbalance is particularly significant. The Fourier algorithm for modular addition requires both cos(k) and sin(k) to compute the full rotation. A 6:1 ratio suggests a partial/degenerate solution — the model is using frequency 13 but not with full rotational structure.

**Key observation:** p101/seed485 looks normal — clean loop, normal grokking. Same prime, different seed. This rules out something inherent to p=101 as the cause. The specific random initialization sent p101/999 on a trajectory that overshoots the generalizing neighborhood.

### Hypothesis: Trajectory Geometry Determines Grokking Quality

The pattern suggests:
1. All models initially pass near a generalizing manifold during early training (first dip in PC3)
2. The optimizer's momentum carries them past it
3. Models that don't go too far eventually find their way back (self-intersecting loop) and converge to a clean Fourier solution
4. p101/999 went too far and never came back — converging instead to a degenerate partial solution

**Testable predictions:**
- **LR scheduling on p101/999**: Slowing learning rate during early excursion should keep the model closer to the generalizing neighborhood. If it groks faster and develops clean specialization, this confirms trajectory geometry drives grokking quality.
- **LR scheduling on a normal variant (e.g., 113/999)**: Slowing at epoch 200 (first dip) — does the model settle into the first basin? If so, the generalizing solution was accessible early.

### Per-Input Tracing as Diagnostic

Per-input traces through the *existing* p101/999 model could reveal what the partial/degenerate solution actually computes:
- Does it get the right answer for some inputs but not others?
- Are the errors structured (e.g., wrong for inputs that require the weak sin component)?
- How does the trace differ from a clean-grokking model at the same training stage?

This would connect the weight-space pathology (imbalanced cos/sin, poor specialization) to computational pathology (what the model actually does wrong on specific inputs). The trace becomes a bridge between the macro-level trajectory analysis and the micro-level computation.

### Hyperparameter Scheduling — Design Considerations

LR scheduling is standard practice in ML training (`torch.optim.lr_scheduler`). Supporting it in the workbench would require:
- Extending variant naming: e.g., `_schedule1` suffix (opaque label mapping to defined schedule config)
- Schedule is a *policy* (step decay, cosine, warmup+decay) with its own parameters — qualitatively different from scalar domain parameters like prime/seed
- Could be modeled as a **training config override** rather than a domain parameter
- Near-term: ad-hoc experiment (notebook + manual directory) to test the hypothesis
- Longer-term: formal support if scheduling experiments become routine

**Status**: Ad-hoc experiment first. Formalize if the hypothesis proves productive.

---

*The self-intersecting loop is universal (10/11 variants). p101/999's open loop correlates with every observed pathology. LR scheduling and per-input tracing are the natural next investigations.*

---
## 2026-02-14: p107 Variants — New Outlier and Late Grokker

### p107/seed999: Delayed Attention Diversification

Trained and analyzed via the new Dash Training/Analysis Run pages (REQ_040).

**Timeline**:
1. Attention heads converge to frequency 22 by ~12.5K epochs (after the test loss drop / grokking onset)
2. Heads begin diversifying around 15K
3. By 22K, the specialization chart looks "wild" — unstable diversification
4. By 25K: Head 0 has specialized on frequency 51, other 3 heads still prefer frequency 22

**Other observations**:
- Weak MLP specialization on frequency 45; overall MLP range between frequencies 22 and 51
- No pathological half-frequency specialization in embeddings (unlike p101/999's cos/sin imbalance)
- Does not perform particularly well by 25K compared to typical variants

**What's unusual**: In normal variants, attention heads diversify *during* grokking (simultaneous with test loss drop). Here, they converge first (all to freq 22), then diversify *after* grokking. The post-grokking diversification is unstable and slow. This suggests the initial convergence to a single frequency may be a local minimum that the model has to escape.

**Comparison with p101/999**: Both are anomalous, but differently:
- p101/999: Degenerate Fourier solution, open PC2/PC3 loop, cos/sin imbalance
- p107/999: Initial consensus (all heads → freq 22), followed by slow unstable diversification, no embedding pathology

Both share: poor final performance relative to typical variants, and seed 999 involvement. The seed=999 correlation across two different primes is worth watching.

### p107/seed485: Very Late Grokker

Doesn't grok until ~16.5K epochs — the latest observed grokking onset in the dataset. For comparison, most variants grok between 2-5K epochs.

**Open questions**:
- What does p107/485's attention head specialization timeline look like? Does it follow the normal pattern (diversify during grokking) but just later?
- Is p=107 inherently harder, or is this a seed interaction?
- Does the late grokking correlate with any particular trajectory geometry (open vs closed loop)?

### Emerging Pattern: seed=999 Anomalies

| Variant | Anomaly |
|---------|---------|
| p101/999 | Degenerate Fourier solution, open PC2/PC3 loop, 6:1 cos/sin imbalance |
| p107/999 | Post-grokking attention head diversification instability |
| p113/999 | (Reference "normal" variant — no anomaly) |

Two anomalies out of ~6 seed=999 variants could be coincidence or could indicate that this specific initialization creates trajectories that are more susceptible to unusual dynamics. Need more data points to distinguish.

---

*p107/999 shows a distinct failure mode from p101/999: consensus-then-diversification rather than degenerate solution. Both produce poor final performance. seed=999 correlation across primes is worth tracking.*

---
## 2026-02-14: Dimensionality Compression as Grokking Precondition

### Observation

Across all variants examined (p113/999, p113/485, p101/999, p107/999, p107/485), effective dimensionality (participation ratio) follows the same qualitative pattern:
- All weight matrices start at high participation ratio (~80-120) from random initialization
- Monotonic decline throughout training
- Sharp collapse in W_in, W_out, W_U around participation ratio ~25-30
- The sharp collapse coincides with or slightly precedes grokking onset

**The timing varies but the threshold value is consistent:**
- Fast grokkers (p113): sharp collapse at ~10-12K epochs
- Slow grokkers (p101/999): sharp collapse at ~15K epochs
- Late grokkers (p107/485): sharp collapse at ~18-20K epochs

### Hypothesis: Compression Threshold for Grokking

The model needs to compress its weight representations below a participation ratio threshold (~25-30) before the algorithmic (Fourier) solution becomes energetically favorable. Weight decay drives compression throughout training; grokking happens when compression crosses the threshold where a low-rank algorithmic solution is cheaper than a high-rank memorization solution.

**Predictions:**
- Stronger weight decay → faster compression → earlier grokking
- Variants that compress slowly (p107/485) grok late
- Variants that reach the threshold but with a distorted subspace (p101/999) grok poorly

### Spectral Gap as a Finer Metric

The participation ratio measures how many singular values contribute, but doesn't capture *how cleanly separated* the signal subspace is from the noise subspace. The spectral gap (ratio of dominant to non-dominant singular values, or the gap between top-k and remaining) may be a more mechanistically meaningful predictor.

**Connection to subnetwork competition (Merrill):** A growing spectral gap is literally the weight space separating into "subnetwork that matters" and "subnetwork that doesn't." The velocity spike during grokking could be the dominant subnetwork's singular values growing while the others shrink — visible as a spectral gap widening.

**Connection to graph Laplacian / connectedness:** The second eigenvalue of the graph Laplacian measures connectivity. Applied to weight matrices, the spectral gap measures how coherent/structured the learned representation is. Small gap = diffuse, weakly structured. Large gap = tight global structure (the algorithmic solution).

### Next Steps (post-restructuring)

The singular value spectrum is already captured per-epoch by `EffectiveDimensionalityAnalyzer`. Computing the spectral gap metric from stored singular values would be a lightweight notebook exploration — no new analyzers needed. Compare gap trajectory against participation ratio trajectory to see which better predicts grokking onset.

---

*Dimensionality compression to a consistent threshold (~PR 25-30) appears to be a precondition for grokking. Spectral gap may be a finer-grained predictor. Both connect to subnetwork competition through the lens of weight space separating into signal vs noise subspaces.*

---

## 2026-02-15: p59/485 — Second Open-Loop Anomaly

### Discovery

p59/seed485 is the second model (after p101/999) to show an open PC2/PC3 trajectory and anomalous grokking behavior. p59/seed999 looks normal — clean self-intersecting loop, grokked by epoch ~8K, robust specialization.

### Key Metrics

| Metric | p59/999 (normal) | p59/485 (anomalous) |
|--------|------------------|---------------------|
| Grokking onset (test_loss < 0.1) | Epoch 8,058 | Epoch 24,248 |
| Final test loss | 0.000000 | 0.020680 |
| Specialized neurons at 25K | ~460 (plateau) | ~180 (still climbing) |
| Frequency bands used | 3 (5, 15, 21) | 2 (5, 21 only) |
| PC2/PC3 loop | Self-intersecting | Open |

### What's Distinct About This Anomaly

**Missing frequency band.** p59/485 only developed neuron specialization for frequencies 5 and 21. Frequency 15 is completely absent — zero neurons specialized to it across the entire training run. p59/999 uses all three bands (5, 15, 21) with robust neuron counts in each.

**Nothing converged.** At epoch 25K, dominant Fourier norms are still climbing, neuron counts still rising, effective dimensionality still dropping, attention heads still specializing. The model was caught mid-transition — it barely crossed the grokking threshold 750 epochs before training ended.

**Low velocity throughout.** Component velocity drops to near-zero early and stays there. The model moved through weight space extremely slowly — it wasn't that it tried and failed, it was barely moving. The initialization put it on a gradient-poor path.

**Landscape flatness still noisy.** Oscillations persist past epoch 15K (p59/999 settles by ~8K). Baseline loss only recently approached zero.

### Comparison with Other Anomalies

| Property | p101/999 | p107/999 | p59/485 |
|----------|----------|----------|---------|
| PC2/PC3 loop | Open | Not yet characterized | Open |
| Grokking quality | Degenerate (cos/sin imbalance) | Post-grokking instability | Incomplete (missing freq band) |
| Seed | 999 | 999 | 485 |
| Root cause hypothesis | Overshooting generalizing manifold | Consensus-then-diversification | Gradient-poor initialization path |

**Notable: p59/485 is the first seed=485 anomaly**, breaking the pattern where only seed=999 produced pathological variants. This suggests the anomaly mechanism isn't specific to one seed — it's about the interaction between prime and seed producing an unlucky initialization.

### Implications for Cross-Variant Analysis

With three anomalous models now identified (p101/999, p107/999, p59/485), quantitative cross-variant comparison becomes increasingly valuable. Four metrics that would cleanly separate normal from anomalous:

1. **Loop closure** — does the PC2/PC3 path self-intersect?
2. **Grokking timing** — when does test loss cross threshold?
3. **Frequency coverage** — how many expected frequency bands developed specialized neurons?
4. **Convergence status** — have dimensionality/specialization/norms plateaued by training end?

These could form the basis of a "variant health dashboard" in the cross-variant analysis requirement.

---

*p59/485 is a second open-loop anomaly with distinct pathology (missing frequency band, gradient-poor path). First seed=485 anomaly — mechanism is prime×seed interaction, not seed-specific.*

---

## 2026-02-16: Competition Window and Basin Landing — Thrashing as Momentum

*Note: "thrashing" = neurons repeatedly switching dominant frequency across epochs.*

### The Triad

Three variants illustrate a spectrum of subnetwork competition dynamics, visible in both the PC2/PC3 trajectory loop and neuron frequency thrashing (REQ_042):

| Variant | Loop Geometry | Thrashing Pattern | Grokking | Interpretation |
|---------|--------------|-------------------|----------|----------------|
| p59/485 | Open (overshoot) | Thrashes throughout training, never fully resolves | Never groks cleanly | Too much momentum — overshoots the basin |
| p101/999 | Open (undershoot) | Stops thrashing early, commits prematurely | Slow, degenerate solution | Too little momentum — undershoots the basin |
| p109/485 | Self-intersecting (sweet spot) | Competition resolves productively | Early grokker (5.5K) | Right momentum — lands in the basin |

### The Basin Landing Hypothesis

There appears to be a basin in weight space that the model must reach for clean grokking. The dynamics look like a momentum problem:

1. The model needs **enough competition** (thrashing) to build momentum toward the generalizing basin
2. But **not so much** that it overshoots and can't settle

This frames grokking as a landing problem: the training trajectory must arrive at the generalizing basin with the right velocity — fast enough to reach it, slow enough to stay.

### Causal Direction: Open Question

Two plausible framings:

**A. Thrashing drives momentum.** Subnetwork competition generates the gradient signal that pushes the model through weight space. Early commitment (p101/999) means losing the gradient signal from competition too soon — the model stalls. Persistent thrashing (p59/485) means the gradient signal never coherently points toward the basin — the model wanders.

**B. Momentum drives thrashing.** The initialization determines the trajectory through weight space. Trajectories that pass near the basin at the right speed naturally resolve competition (neurons commit because the basin is attractive). Trajectories that miss the basin keep thrashing because there's no attractor to settle into.

These aren't mutually exclusive — thrashing and trajectory likely co-determine each other. But the causal priority matters for intervention design: if (A), you'd want to modulate competition directly; if (B), you'd want to modulate trajectory (e.g., LR scheduling).

### Supporting Evidence

- **p59/485** chooses frequencies and follows a "somewhat normal path" on some metrics, yet thrashes throughout. This suggests the thrashing isn't just about *finding* frequencies — it's about the dynamics of *settling* into them. The model found frequencies but couldn't commit. (Supports framing B: trajectory determines commitment.)
- **p101/999** stops thrashing early and underperforms. If thrashing were purely noise, early cessation should be beneficial. Instead, it correlates with a degenerate solution. (Supports framing A: competition is productive signal.)
- **p109/485** shows the sweet spot: enough competition to explore, decisive enough commitment to converge. The loop closes at the right moment.

### Potential Metrics

- **Competition window duration**: epoch range where >50% of neurons are actively switching
- **Resolution sharpness**: rate of commitment (sigmoid-like vs gradual)
- **Terminal commitment fraction**: what % of neurons are committed at training end
- **Commitment timing relative to grokking onset**: does commitment precede, coincide with, or follow the test loss drop?

These could be computed from existing neuron_dynamics cross-epoch data without new analyzers.

---

*The competition-resolution dynamics may be the mechanistic core of grokking. Too much competition (overshoot), too little (undershoot), or just right (basin landing). Causal direction between thrashing and momentum is the key open question.*

---

## 2026-02-16: p59/485 Extended Training — Settling With Damage

### Follow-Up Experiment

Extended p59/485 training from 25K to 35K epochs.

### Results

1. **The thrashing did settle.** The chaotic frequency switching resolved shortly after 25K. The model was not stuck in a fundamentally different attractor landscape — it just needed more damping time.
2. **The PCA loop tightened.** Recomputing parameter trajectory with 35K epochs reduced the original overshoot. The trajectory geometry moved toward the self-intersecting pattern seen in normal variants.
3. **Test loss remains underperformant.** Despite settling, the model did not achieve clean grokking performance.

### Interpretation: Arrival State Matters

Settling late is not the same as settling well. By the time p59/485 resolved its thrashing, it had already committed to an impoverished frequency allocation (2 bands instead of 3 — frequency 15 never developed). The model landed in *a* basin, just not the optimal one.

This suggests the **competition window is not just about duration but about timing**. The productive phase — where subnetwork competition shapes which frequency bands develop — has a finite window. What the model brings with it when it finally settles determines solution quality:

| Variant | Arrival state | Outcome |
|---------|--------------|---------|
| p109/485 | 3 frequency bands differentiating, right momentum | Clean grokking |
| p59/485 | Missing frequency band, late arrival after extended chaos | Suboptimal convergence |
| p101/999 | Premature commitment, degenerate cos/sin ratio | Degenerate solution |

### Refinement to Basin Landing Hypothesis

The basin exists for p59/485 — the model eventually found it. But the basin has structure: arriving with incomplete frequency coverage means settling into a suboptimal region of the basin. The competition window doesn't just determine *whether* you reach the basin, but *where in the basin* you land.

---

*Extended training confirmed p59/485's chaos was temporary but damaging. The competition window shapes what structure the model carries into convergence. Late settling = impoverished solution, even if the model eventually stabilizes.*

---
## 2026-02-16: Representational Geometry — Activation-Space Confirmation of Weight-Space Story

### Three-Variant Comparison (REQ_044 Visualizations)

The representational geometry analyzer (centroid PCA, distance heatmaps, time-series metrics) provides activation-space evidence that directly confirms and extends the weight-space observations from parameter trajectory analysis.

| Metric | p113/999 (control) | p101/485 (slow) | p101/999 (anomalous) |
|--------|-------------------|-----------------|---------------------|
| Circle formation epoch | ~9K (grokking turn) | ~16K+ (delayed) | ~18K (transient) |
| Var explained (top-2 PCs) | 76% → 57% (stable) | 30% → 39% (slowly improving) | 30% → 29% (lost) |
| Circle quality | Tight, evenly spaced | Wobbly, less crisp | Momentary then collapsed |
| Distance heatmap banding | Clean diagonal stripes | Faint then moderate | Brief structure, then muddy |
| Fisher discriminant (Resid Post) | ~13 (strong) | ~10 (moderate) | ~6.5 (weak) |
| Dimensionality collapse | Sharp at ~9K | Gradual at ~15K | Gradual, never fully resolves |
| Circularity time-series | Clean transition | Noisy oscillation 8-15K | Violent oscillation 8-20K, never settles |

### Key Insight: Variance Explained as Compression Proxy

The top-2 PCA variance explained tracks how well the model has compressed its class representations into a 2D Fourier subspace. This is the **activation-space mirror** of the weight-space dimensionality compression (participation ratio ~25-30 threshold) observed earlier:

- **p113/999**: 76% at epoch 9K → the model has already packed most class structure into 2 dimensions. This is a *compressed* representation.
- **p101/485**: Only 39% even at 25K → class structure is spread across more dimensions. The model is working but inefficiently.
- **p101/999**: 30% and declining → the model briefly found 2D structure but couldn't maintain it. The representations are *expanding* back into higher dimensions.

### Connection to Basin Landing Hypothesis

The momentary circle at epoch 18K in p101/999 is the geometric manifestation of the "transient approach to the generalizing manifold" from the basin landing hypothesis. The model's activation-space centroids briefly arrange into the Fourier circle — the model *found* the solution — but couldn't maintain it because the weight-space dynamics (degenerate cos/sin ratio, premature neuron commitment) don't support stable circular structure.

This provides a new way to think about grokking failure: it's not that p101/999 never discovers the algorithm, it's that the algorithm is **transiently expressed but not consolidated**. The weight-space substrate can't hold the activation-space geometry.

### Circularity Time-Series as Diagnostic

The circularity & Fourier alignment panel in the time-series view turns out to be an excellent diagnostic for grokking health:
- **Clean transition** (p113/999): single sharp rise, stable plateau → healthy grokking
- **Noisy oscillation** (p101/485): oscillations that eventually settle → delayed but viable
- **Violent persistent oscillation** (p101/999): never settles → pathological

This could be a strong candidate metric for the "variant health dashboard" concept from the cross-variant analysis discussion.

---

*Representational geometry provides activation-space confirmation of the weight-space grokking story. Variance explained tracks compression quality; circularity time-series tracks grokking health. The transient circle in p101/999 shows the model found but couldn't hold the solution.*

### p59/485 — Overshooter Lands Late But Sticks

p59/485 adds the critical fourth comparison point:

| Metric | p113/999 | p101/485 | p101/999 | p59/485 |
|--------|----------|----------|----------|---------|
| Circle at training end | Solid (57%) | Wobbly (39%) | Lost (29%) | Solid (56%) |
| Fisher discriminant | ~13 | ~10 | ~6.5 | ~20 (highest) |
| Center spread (Resid Post) | ~50, plateaued | ~40, plateaued | ~35 | ~120, still climbing |
| Centroid scale (PC axes) | ±50 | ±30 | ±20 | ±100 |
| Grokking onset | ~9K | ~15K | Never clean | ~24K |

**Key finding**: p59/485 arrives late to the generalizing basin but builds the strongest separation of any variant observed. Its Fisher discriminant (~20) exceeds even p113/999 (~13). The centroid scale is enormous (±100 vs ±50) — the model appears to be *overcompensating* for having only 2 frequency bands instead of 3 by pushing centroids further apart.

**Contrast with p101/999**: Both have open PC2/PC3 loops. Both are anomalous. But p59/485 eventually forms a solid circle (56% var explained, rising), while p101/999 loses its circle (29%, declining). The difference: p59/485 had enough momentum to stick when it arrived; p101/999 committed too early and bounced off.

**Refinement to basin landing hypothesis**: Arrival velocity matters more than arrival timing. Late arrivals can succeed if they carry enough momentum. Early arrivals fail if they lack the dynamical structure (frequency balance, neuron competition) to sustain the solution. The basin is sticky for well-prepared models and slippery for under-prepared ones.

---

*p59/485 proves the basin is reachable even via the overshoot path. The overshooter compensates with massive scale. The undershooter (p101/999) fails not because it can't find the basin, but because it can't stay.*

---
## 2026-02-23: Phase Scatter Voids — Dual View of Centroid Geometry

### Observation

The phase alignment scatter (ψ_m vs. φ_m at dominant frequency) shows characteristic void regions — not noise, not missing data, but structured gaps. This pattern immediately recalled the centroid PCA Lissajous structure.

### What Creates the Voids

Both φ and ψ are constrained to [-π, π] by arctan2, but the alignment relationship is ψ = 2φ. This requires ψ ∈ [-2π, 2π] — double the range available. The wrapping at ±π splits what would be a single diagonal into **three bands**:

1. **Central band**: φ ∈ [-π/2, π/2], ψ = 2φ ∈ [-π, π] — lies on the dashed ideal line
2. **Lower-right band**: φ ∈ (π/2, π], theoretical ψ = 2φ wraps to 2φ - 2π ∈ (-π, 0]
3. **Upper-left band**: φ ∈ (-π, -π/2], theoretical ψ = 2φ wraps to 2φ + 2π ∈ (0, π]

The voids are the gaps at φ ≈ ±π/2 — the wrapping discontinuities. They're structural proof the doubled-phase relationship is real.

### Connection to Centroid PCA

Both representations are projections of the same torus topology:

- **Centroid PCA**: plots output class centroids on a circle/Lissajous parameterized by e^{ikx} and e^{2ikx}. The self-intersections and figure-8 shapes come from the double-angle relationship cos(kx)cos(ky) → cos(k(x+y)).
- **Phase scatter**: plots per-neuron weight phases (φ_m, ψ_m) subject to ψ ≈ 2φ. The three-band void structure comes from the same doubling, wrapped back into [-π, π].

Both are signatures of the same circuit: the model computes cos(kx)·cos(ky) - sin(kx)·sin(ky) = cos(k(x+y)), and the double-angle relationship is the mathematical spine of that computation. Centroid PCA sees it in output-space organization; phase scatter sees it in weight-space organization.

### Diagnostic Potential

For pathological variants (p101/999, p59/485): a model with degenerate or incomplete frequency coverage should show a distorted version of this three-band structure — bands that are faint, smeared, or misaligned. The clarity of the three-band void pattern may be a visual analog to circularity in the centroid time-series: sharp bands = well-formed Fourier circuit; blurred/missing bands = degraded or incomplete alignment.

Worth comparing the phase scatter for p101/999 (degenerate cos/sin imbalance, transient circle that collapses) against p113/999 (clean). If the three-band structure is absent or smeared for p101/999, the weight-space and activation-space views are telling the same story from complementary angles.

---

*The phase scatter voids are not artifacts — they're the weight-space signature of the same doubled-phase topology that creates Lissajous figures in centroid PCA. Both views are projections of the model's ψ = 2φ constraint onto accessible phase ranges.*

---

## 2026-02-23: Initialization Implicated in p=101/999 — Lottery Ticket Non-Null Result

### Key Finding

Null lottery ticket result (neither phase margin nor magnitude ratio at init predicts winning frequency) holds for p=113/999 (healthy variant). p=101/999 shows structure in the phase margin vs magnitude scatter — initialization IS predictive for the anomalous variant.

This is diagnostic. In healthy variants, the competition plays out through gradient flow — init doesn't matter. In p=101/999, amplitude locks early (β IPR rises to ~0.9 ahead of phase alignment), and wherever the neuron started is closer to where it ends up. The competition isn't resolved by training dynamics; it's resolved by the initialization.

### p=101/485 as Control Case

p=101/485 shows canonical grokking from a healthy loss curve. Same prime, different seed. This rules out p=101 being intrinsically hard — seed=999 is a bad initialization draw for this prime. The lottery ticket non-null result is not a property of p=101; it's specific to p=101/999's initialization.

**The mechanism**: p=101/999's amplitude competition commits prematurely before phase gradient flow can establish ψ≈2φ (dissociation: β IPR ≈0.9, α IPR ≈0.4). Because amplitude competition resolved early based on initial magnitudes rather than sustained competition, initialization is more deterministic in this variant than in healthy ones.

### p=109/485 — Fast Grokker as the Ideal Case

p=109/485 locked in fast — appears to have landed on a converged representation quickly but still went through a refinement phase. This is the opposite of p=101/999's pathology: amplitude competition resolved productively AND the model continued refining phase alignment post-commitment. Fast grokking here is evidence of arriving at the right place with the right momentum, not just arriving fast.

**Open question**: Does p=109/485's phase scatter show the clean three-band structure earlier than slower grokkers? If so, it would suggest the fast landing came with intact phase alignment from the start — the initialization happened to be well-positioned for both amplitude and phase competition.

---

*p=101/999's non-null lottery ticket result is diagnostic of broken competition dynamics. p=101/485 control confirms the pathology is init-dependent, not prime-intrinsic.*

---

## 2026-03-03: Fourier Quality Score Mirrors Train/Test Loss

### Observation (REQ_052)

Across all healthy variants, the `fourier_quality_trajectory` plot closely tracks train/test loss curves. For the two anomalous variants (p101/999 and p59/485), the quality score captures a gradual slope rather than a sharp transition — consistent with their slow/degenerate grokking behavior.

### What This Means

**Quality score and loss are tracking the same underlying event.** Both drop/rise at grokking onset because grokking IS the model committing to Fourier structure. This is a positive validation: the metric is measuring something real, not noise.

**But there's an important difference in what each metric can see.** Loss can drop via memorization OR via Fourier algorithm learning. Quality score only tracks Fourier structure expression. If they're tightly coupled across all variants, it suggests:
1. In this task (mod-p addition), most of the loss reduction *is* Fourier learning — the memorization pathway doesn't persist long enough to show up as a sustained loss plateau
2. The quality score is a weight-space proxy for the same signal loss is measuring from the output side

### Does Quality Score Add Information?

For healthy variants: probably not much beyond what loss already shows. The value is in being a weight-space anchor — it tells you *why* the loss dropped, not just *that* it dropped.

For anomalous variants: the gradual slope is informative. A gradual quality score rise in p101/999 or p59/485 means the model IS accumulating Fourier structure, just slowly and incompletely. Loss alone doesn't distinguish "building the right algorithm slowly" from "memorizing slowly." Quality score does.

**The diagnostic gap**: quality score can't tell you about *which* frequencies are present, only *how much* of the energy is in whatever dominant frequencies were identified. A model that found the wrong frequency set but committed cleanly would show high quality. This is where REQ_053 (per-class accuracy) would add orthogonal information.

---

*Quality score validates the Fourier learning story already visible in loss, but adds weight-space grounding and helps distinguish algorithmic accumulation from memorization in anomalous variants.*

---

## 2026-03-08: Data Seed Controls Frequency Set Selection — Attention-MLP Handshake Failure

### Setup

p113/seed999 trained with data_seed=999 (vs. the canonical data_seed=598). All prior models used dseed=598. This is the first clean comparison of the same model identity under two different data splits.

### Key Results

| Metric | p113/seed999/dseed598 (canonical) | p113/seed999/dseed999 |
|--------|-----------------------------------|-----------------------|
| Final test loss | ~0.000000 | 0.049271 |
| Min test loss ever | ~0.000000 | 0.015372 (epoch 13,157) |
| Frequency set used | {9, 33, 38, 55} | {13, 25, 33} (after Freq 40 collapse) |
| Landscape flatness post-grokking | Dead flat (machine epsilon) | Residual oscillations remain |
| Neuron 0 activation pattern | Clean sinusoidal grid stripes | Triangular/threshold structure |

### The Frequency Set Difference

The two models don't use the same frequency basis. The canonical model settled on {9, 33, 38, 55}; the dseed=999 model attempted {13, 25, 33, 40} and then lost Freq 40. **The data split is steering which frequency set the model attempts to build** — this is a deeper effect than reduced gradient magnitude for one frequency.

### The Attention-MLP Handshake Failure

The per-band neuron specialization chart tells the story:
- Freq 40 builds to ~50 committed neurons (MLP side) between epochs 10k-15k
- The QKT Fourier spectrum at epoch 12,300 (the inflection point) shows Freqs 13, 25, 33 with clear QKT mass — Freq 40 has essentially none
- The MLP learned to specialize to Freq 40; attention never routed queries to it
- Weight decay then eliminates the unattended neurons; Freq 40 collapses to zero by epoch 20k
- Test loss, which had reached a minimum of 0.015 at epoch 13,157, climbs back to 0.049

The Freq 40 neurons were computing without being selected. The attention-MLP co-evolution needed to close the loop didn't complete.

### The Landscape Flatness Signature

The canonical model's landscape flatness drops to an absolute flat line at epoch ~10,900 — machine epsilon, no visible oscillation for the remaining 14k epochs. The dseed=999 model never achieves this. Residual oscillations remain in mean delta loss throughout the post-settling period. This is the geometric signature of the difference:
- Dead flat = the model found a crisp algorithmic minimum; perturbations do nothing because the solution is implemented exactly
- Residual oscillations = the model is in a wide shallow basin, not a sharp minimum; the partial solution is sensitive to perturbation

### The Unusual Neuron Shape

The failing model shows a triangular activation pattern over the (a,b) grid — strongly positive where a+b is small, essentially off elsewhere. Confirmed: Neuron 0 at this epoch is at 62% frac explained on Frequency 3, not on Frequency 40. Frequency 3 barely registers on the per-band specialization chart — it shows up as a faint bump near zero.

The triangular shape is mechanistically interpretable for Freq 3. A neuron partially specialized to k=3 computes something proportional to cos(3·2π·(a+b)/113). At k=3 over a range [0, 112], the cosine function completes ~3 full cycles, making its first positive lobe span roughly the lower-sum region of the grid — the upper-left triangle where a+b < p/3. This is not a broken or malformed Fourier neuron; it's what partial (62%) specialization to a very low frequency looks like over the (a,b) activation surface.

The observation that this pattern is unfamiliar reinforces that Freq 3 specialization essentially never appears in the dseed=598 model population. It's not a frequency the canonical models use. Its presence here — even weakly — is another signal that the data split is steering the dseed=999 model toward an entirely different candidate frequency basis.

### Data Seed as Frequency Steering

dseed=598 is not lucky — across all 17 models, it consistently produced splits that allowed the model to settle on a viable frequency basis and complete the attention-MLP co-evolution. The data split determines which (a,b) pairs provide gradient signal for each Fourier component. dseed=999 may underrepresent pairs that are most informative for the components the model needs — specifically those that would allow Freq 40 to build sufficient gradient momentum to secure attention routing.

This connects directly to the gradient momentum story from the 2026-03-06 findings: 50 neurons on Freq 40 vs. 80+ on Freq 33 is a meaningful gradient disadvantage. If the split provided weaker signal for Freq 40 from the start, it never builds the neuron count needed to compete for attention routing.

---

*Data split steers frequency set selection, not just gradient magnitude. Freq 40 failed the attention-MLP handshake. The landscape flatness quality gap distinguishes a crisp algorithm from a wide shallow basin. The unusual triangular neuron is a Fourier basis function that never formed.*

---

## 2026-03-06: Neuron Distribution as Grokking Health Predictor — Preliminary Observations

### Core Observation

At the >90% specialization threshold, neuron band distribution predicts grokking outcome better than embedding Fourier magnitudes alone.

**p101/999 (failure):** Three attention heads were weakly aligned to frequency 5 pre-second-descent. Higher frequencies (35, 43, 44, 41) overwhelmed frequency 5 in neuron volume — growing at disproportionate rates while frequency 5 peaked at ~62 neurons then declined. At epoch 15K: frequency 43 had 41 neurons vs frequency 5's 34. Head 2 held frequency 5 longest, abandoning it at ~17.5K; test loss rose after. Embedding Fourier magnitudes throughout showed frequency 5 as dominant — direct embedding-neuron misalignment. Gradient momentum from neuron count (not embedding signal) drove the outcome.

**p103/485 (healthy control):** Groks near canonical window, looks normal from loss. Loses frequency 20 gracefully — 3 remaining frequencies (6, 24, 40). Neuron counts across Low/Medium/High bands grew with similar slopes. By epoch 20500, QK^T firmly concentrated on {6, 24, 40}; V still retains some frequency-20 signal. QK led V in dropping the frequency — consistent with QK as active router, V as passive content carrier (healthy adaptation sequence).

### The Gradient Momentum Story

Each specialized neuron contributes gradient signal toward its frequency. 41 neurons on frequency 43 vs 34 on frequency 5 is a ~20% gradient advantage, compounded every step. Attention heads fighting this tide can hold briefly but eventually capitulate. The lottery isn't fixed at initialization — it plays out at the attention-MLP interface during the competition window, resolved by neuron count gradient momentum, not embedding structure.

### Hidden Dynamics Warning

The >90% threshold may be missing earlier dynamics in the 60-90% specialization range. There could be partial specialization activity that predates what's currently visible. Threshold was initially set at 90% because it aligns with grokking onset. Lower thresholds are unexplored. REQ_058 proposes threshold-parameterized metrics; the dashboard threshold slider enables exploration.

---

*Neuron volume (not embedding magnitude) determines gradient competition outcome. QK leads V in healthy frequency transitions. The 90% threshold may hide earlier dynamics worth exploring via threshold sweep.*

### Frequency Pairs Travel Together

Across several variants, pairs of frequencies share nearly identical growth slopes in committed neuron counts — they grow together, plateau together, and (in pathological cases) decline together. This is not random. If two frequencies have correlated slopes across the entire training run, they may be sharing a structural role in the computation (e.g., both encoding related harmonics of the same Fourier component) or have similar gradient signal levels due to similar initialization positions in the lottery.

**Implication for HHI metrics (REQ_058):** If pairs travel together, the effective number of independent frequency bands may be lower than the raw active band count suggests. A concentration index computed over all active bands could understate concentration if two of the four bands are essentially moving as one. The slope CV metric (CoS 4) should detect this — paired slopes will produce low variance between the two paired bands and high variance vs the uncoupled bands.

**Open questions:** Does the pairing persist after grokking (suggesting a stable structural relationship) or dissolve (suggesting it's a competition artifact from initialization)? Does the pairing align with mathematical properties — e.g., are the paired frequencies harmonically related (k and 2k)?

### 75%/100-Neuron Critical Mass Hypothesis

Rough inspection at 75% threshold suggests ~100 committed neurons may be the trigger for second descent (grokking). The 75% threshold appears to be a stable diagnostic window — lower than 90% (which may miss early dynamics) but high enough to exclude transient partial alignments.

**Why 75% may be special:** At 75% specialization, a neuron's dominant frequency contributes 75% of its Fourier energy. This is enough to produce meaningful gradient signal toward that frequency but below the threshold where phase alignment is complete. The 75% population likely represents neurons in the "committed but still refining" phase — counting them gives earlier visibility into the emerging frequency distribution.

**Why ~100 neurons may matter:** 100 neurons out of 512 is ~20% of the MLP. Gradient signal from 20% of neurons uniformly pushing toward 2-3 frequencies likely crosses a tipping point for attention head alignment. Below this, attention can resist the gradient momentum; above it, the pull becomes irresistible and the second descent begins.

**The pathological shape:** The distinction between healthy and anomalous may not be *when* 100 neurons is reached, but *which frequencies they're distributed across*. Healthy: 100 neurons spread across 2-3 frequencies with similar slopes → balanced gradient pull → all frequencies survive → clean grokking. Anomalous: 100 neurons concentrated in one or two frequencies with steep slope → asymmetric gradient pull → competing frequencies eliminated → degenerate solution.

*This hypothesis drives the core metrics in REQ_058: HHI (concentration), slope CV (balance), and the critical mass snapshot (frequency distribution at the 100-neuron crossing epoch).*

### Correction: Initial Computation Was Inverted

The first run used an absolute threshold of 1.0, borrowed from the bar chart renderer (a display convention). At random init all p coefficients ≈ 1.07, so all passed and quality ≈ 1. After grokking, concentrated frequencies spiked but the rest dropped, so quality went to near 0. Fixed by using a relative threshold: `coefficient > 3 × mean(coefficients)`. Now quality starts near 0 and rises around grokking.

### Quality Is Inherently Low in Absolute Terms

Analytically, quality = 2k/p. For p=113 and k=3 dominant frequency pairs, the ceiling is 6/113 ≈ 5.3%. This is a property of the metric: it measures how much of the *full* Fourier space (56 possible frequency pairs for p=113) is covered by the model's 2-3 selected pairs. A fully grokked model only needs ~3 pairs to compute the algorithm — it doesn't need to cover all 56. So low absolute quality is expected and correct; shape and timing are what carry diagnostic signal.

### Frequency Specialization Doesn't Look Like a Choice

The corrected charts confirm: frequency quality rises within the grokking window, not before it. The model doesn't "select" frequencies in advance — specialization emerges during grokking, driven by the task structure and initialization. The quality score is better read as "did frequency specialization complete?" than "did the model choose well?"

---

## 2026-03-03: Frequency Selection Precedes Generalization — A Two-Stage Picture

### Observation (REQ_052 trajectory charts)

k↓ and quality↑ are **co-moving** — they track essentially the same line. Together, this composite signal precedes the test loss drop. The structure is two stages, not three:

1. **Selection + quality buildup**: k narrows and quality rises together (the model commits to a frequency set and builds Fourier structure around it)
2. **Generalization**: test loss drops afterward

Both anomalous variants (p101/999, p59/485) show a different shape in Stage 1 — the k/quality trajectory doesn't cleanly convert to a Stage 2 test loss drop, or does so much later and less sharply.

### Mechanistic Interpretation

k and quality are co-moving because they're aspects of the same event: frequency selection determines both the count (k) and the projection quality (R²). The model doesn't select frequencies and then build quality separately — it does both simultaneously as competition resolves and neurons commit.

The precursor relationship is: **[selection + quality buildup] → [generalization]**. Not a three-stage sequence — Stage 1 is a single event with two observable faces.

### p101/999 and p59/485 as Diagnostic

For healthy variants, Stage 1 completes with a clean shape and Stage 2 follows. For p101/999, the Stage 1 trajectory has a different shape that doesn't convert to a Stage 2 transition. For p59/485, Stage 1 is slower and more gradual, with Stage 2 following — but late and weakly.

---

*k and quality co-move as two faces of frequency selection. [Selection + quality buildup] precedes [generalization] as a two-stage pattern. Anomalous variants show degraded Stage 1 that fails to trigger or delays Stage 2.*

---

## 2026-03-03: Frequency Allocation → Geometric Constraints → Irreducible Error Floor

### The Causal Chain

**Initialization → frequency allocation (lottery ticket) → geometric constraints on class separation → irreducible error floor**

The model's frequency set is largely fixed by initialization. Some allocations are structurally insufficient for perfect class separation, producing a test loss floor the model cannot cross regardless of training duration.

- **p101/999**: cos/sin imbalance at frequency 13 (6:1 ratio). The Fourier algorithm requires both sin(k) and cos(k) to compute the full rotation. Without sin, residue class pairs that differ *in the sin(k) direction* cannot be separated. This produces a structurally dense cluster in those class pairs in the distance heatmap.
- **p59/485**: missing frequency band 15. Every class pair whose natural separation lives in that frequency band is underserved. The model compensates by amplifying frequencies 5 and 21 (hence enormous centroid scale ±100), but those frequencies can only separate pairs they're geometrically equipped to separate.

### Why the Floor Is Irreducible

If errors were distributed uniformly, longer training or more capacity would reduce them. A floor that persists suggests the model *literally cannot represent the correct output* for certain inputs given its frequency basis. The frequency allocation determines which inputs are representable and which aren't.

### The Diagnostic Value of Per-Class Accuracy (REQ_053)

The test loss floor is the aggregate signature. Per-class accuracy (by residue, by input pair) localizes **which class pairs drive it**:
- **Structured errors** (concentrated on specific residue classes): implicates the frequency gap — those residues require the missing/imbalanced component
- **Uniform-but-high errors**: suggests general model weakness, not frequency-specific pathology

Fisher discriminant and centroid distance heatmaps would show the same structure from the geometry side: low discriminant / high proximity for the specific class pairs the missing frequency should separate.

This is why tying frequency quality to class geometry (Fisher, centroid distances) and per-class accuracy into a joint view is the right next step. Each lens sees the same structural limitation from a different angle.

---

*Frequency allocation constraints produce irreducible error floors. Per-class accuracy localizes which residues bear the cost. Fisher discriminant and distance heatmaps provide the geometric side of the same story.*

---

## 2026-03-09: Second Descent Survivability and Frequency Competition Dynamics

### The Survivability Reframe

The prevailing narrative is "entering second descent = generalization." The emerging picture is more specific: **a model can enter second descent and fail to survive it.** Second descent is a process with its own trajectory, not a binary milestone.

Three model archetypes identified:

| Model | Second Descent | Outcome |
|-------|---------------|---------|
| 109/485/598 | Fast, steep, no bumps | Clean grokker |
| 101/999/598 | Slow, multiple drops in MLP/Resid Post Fourier alignment | Late grokker |
| 113/999/999 | Enters descent, test loss climbs back up | Failed second descent |

113/999/999 requires a new failure mode category: `degraded_recovery` — the model did the work, descended 80%+ from peak test loss, then lost it.

### First-Mover Advantage

The frequency that achieves early neuron mass wins the competition window. **The first-mover determines the trajectory, not the final count.**

Observed:
- 109/485/598: Frequency 4 (low band) leads by thousands of epochs → clean geometry → clean descent
- 101/999/598: Frequency 40 (high band) leads first → low-band frequency tries to build mass later but too late — attention already aligned to the high-band winner
- 113/999/999: Frequency 13 (mid band) leads; frequency 3 (low) tries to build mass late and loses it

Once a frequency achieves early dominance, attention heads begin aligning to it. This creates a self-reinforcing cycle: attention alignment → stronger gradient signal for that frequency → more neurons commit → more attention alignment. Late arrivals face entrenched attention weights, not just neuron competition.

### Frequency Portfolio at Second Descent Onset

Not all frequency losses are equal. The damage depends on **what remains** after the loss:

- **Redundant loss**: loses a frequency but retains another in the same band → survived (103/485/598 case: lost a mid-range frequency, retained the winning mid-range frequency, no MLP alignment drop)
- **Anchor loss**: loses the only representative of a band, leaving the portfolio single-band → compromised geometry entering descent (101/999/598: lost its only low-band frequency, remaining portfolio entirely high-band)
- **Late-build failure**: a needed frequency tries to build neuron mass but is overpowered by already-entrenched frequencies — never achieves critical mass (101/999/598, 113/999/999)

Band diversity of the surviving portfolio matters more than total frequency count. A model entering second descent with frequencies all in the high band may be worse positioned than one spanning low and mid.

### MLP vs. Attention Alignment as Damage Indicator

Frequency losses leave different signatures depending on which layer's alignment is affected:

| Signature | Interpretation |
|-----------|---------------|
| Attention alignment drops only | Survivable — attention reorganized, MLP computation was stable (redundant loss) |
| MLP/Resid Post alignment drops | Fundamental damage — MLP's Fourier structure destabilized, computational substrate broken |
| MLP alignment drops, no attention drop | High-frequency secondary participant lost; attention never fully committed to it but MLP depended on it |

Observed:
- 103/485/598: attention drop only → survived
- 101/999/598: multiple MLP/Resid Post alignment drops → difficult, slow descent
- 113/999/999: MLP alignment drop, no attention drop → catastrophic failure mode

The timing of alignment drops against the commitment timeline (from `neuron_dynamics`) is checkable: if an alignment drop coincides with a neuron-count drop at a given frequency, the causal chain is visible.

### Circularity and High-Band Geometry

High-band frequencies (k near p/2) have tighter geometric spacing between residue classes in their 2D Fourier subspace. A model locked into high-band frequencies may show lower circularity even with adequate data coverage, because the ring representing residue classes is denser and more sensitive to interference.

Predicts: max circularity achieved before second descent onset should be lower for models whose first-mover was high-band. Worth measuring as `pre_descent_max_circularity` against `first_mover_band` across all variants.

### The Nucleation Predictor as Conditional Signal

The REQ_063 Fourier Nucleation analysis is not failing when predicted frequencies diverge from winning frequencies. **The divergence IS the signal.**

- Models where init-predicted frequency ≈ winning frequency: healthy grokkers (109/485 → frequency 4 predicted and won)
- Models where init-predicted frequencies diverge from winning frequencies: anomalous (101/999: init predicted 30, 14, 25; actual winners 35, 41, 43, 44)

The degree of divergence may predict anomaly severity. Models that had to fight their own initialization bias to reach their trained state are the ones with damaged trajectories.

### Data Compatibility: A Flatness Finding

For primes in the 59–113 range with 30% training fraction, data compatibility is near-uniform across all frequencies. ~30% of p² pairs, randomly sampled, provides approximately 0.3×p pairs per residue class — sufficient for balanced Gram matrix coverage at all frequencies by law of large numbers. Data seed does not meaningfully change the compatibility landscape at these scales.

Implication: the variation in grokking character across data seeds is not primarily driven by differential data support for specific frequencies. The cause lies upstream — in initialization (which frequency gets first-mover advantage) and in the competition dynamics that unfold before second descent.

---

*First-mover advantage determines trajectory. Second descent survivability depends on frequency portfolio composition at onset. MLP vs. attention alignment drops distinguish survivable from fundamental damage. Nucleation divergence is the anomaly signal.*

---
