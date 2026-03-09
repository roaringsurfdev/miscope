# Thoughts - Findings

Unstructured parking lot for ideas and observations about what the research is revealing. See Claude.md for context.

---

## Archived Findings (February 2026)

Full entries in [findings_archive.md](findings_archive.md).

| Date | Entry | Core Takeaway |
|------|-------|---------------|
| 2026-02-11 | PC2 vs PC3 Loop Structure and p101/999 Anomaly | All healthy variants show a self-intersecting PC2/PC3 loop; p101/999 has an open loop — its early excursion overshot the generalizing manifold and never returned. |
| 2026-02-14 | p107 Variants — New Outlier and Late Grokker | p107/seed485 is a slow, damaged grokker (low-band frequency missing); p107/seed999 groks late but cleanly. Same prime, different trajectories — seed determines the lottery. |
| 2026-02-14 | Dimensionality Compression as Grokking Precondition | Weight-space dimensionality (from PCA) compresses just before grokking. The model consolidates into a lower-dimensional attractor prior to the test loss drop. |
| 2026-02-15 | p59/485 — Second Open-Loop Anomaly | p59/485 is the second open-loop variant. An overshooter — misses a frequency band, eventually sticks but late. Open loop + missing band are co-occurring, not coincidental. |
| 2026-02-16 | Competition Window and Basin Landing | Arrival velocity into the generalizing basin matters more than timing. Thrashing during the competition window is gradient momentum at the attention-MLP interface, not noise. |
| 2026-02-16 | p59/485 Extended Training — Settling With Damage | p59/485 groks after extended training but with a structural cost: missing frequency band, elevated centroid scale. Late grokking often means grokking with a compromised frequency portfolio. |
| 2026-02-16 | Representational Geometry — Activation-Space Confirmation | Per-class centroid geometry, Fisher discriminant, and circularity all converge on the same story visible in weight-space Fourier metrics. Geometry confirms frequency allocation outcomes. |
| 2026-02-23 | Phase Scatter Voids — Dual View of Centroid Geometry | Residue classes cluster into distinct phase-scatter regions. Voids between clusters mark natural boundaries in the representational geometry and predict which residues share error structure. |
| 2026-02-23 | Initialization Implicated in p101/999 — Lottery Ticket Non-Null Result | Reinitialized p101/999 with a different seed groks normally. The original initialization is the proximate cause of the anomaly — the degenerate trajectory is not intrinsic to p=101. |

---

## 2026-03-03: Fourier Quality Score Mirrors Train/Test Loss

### Setup

REQ_052 Fourier Quality Scoring computed and visualized across all variants. The quality trajectory (fraction of weight energy in dominant Fourier components) was compared to train/test loss curves for healthy and anomalous variants.

### Key Results

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
