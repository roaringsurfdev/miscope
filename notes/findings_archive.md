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
