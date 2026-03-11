# Thoughts - Analysis Directions

Unstructured parking lot for ideas around adding new analysis capabilities. See Claude.md for context.

---

## 2026-02-02: Multi-Scale Structure in Activations and Attention

### Discovery

When viewing neuron activation heatmaps at reduced resolution (via bilinear interpolation to 28×28), distinct "blob" regions emerge that aren't visible at full resolution (113×113). This suggests activation patterns have structure at multiple spatial scales.

### Key Observations

**Neuron activations** (first 10 neurons analyzed):
- Most neurons (1-7, 9) have energy concentrated in high frequencies (FFT no-DC ~0.005)
- Neuron 8 is an outlier with genuinely coarse structure (FFT no-DC = 0.962)
- Neuron 0 has moderate low-freq content after DC removal (0.226)

**Attention heads** show distinct specialization patterns:

| Head | FFT no-DC | Gradient | Structure Type |
|------|-----------|----------|----------------|
| 0 | 0.014 | 0.324 | High-freq content, variable intensity |
| 1 | 0.349 | 0.241 | Mixed frequencies |
| 2 | 0.969 | 0.094 | Low-freq, smooth uniform regions |
| 3 | 0.013 | 0.334 | High-freq content, most edge variation |

### Nuanced Interpretation

Initial framing of "blob vs grid" was imprecise. The metrics reveal different structural properties:

- **FFT (no DC)**: Captures spatial frequency of the dominant pattern
  - High = large-scale/low-frequency structure (Head 2's big uniform squares)
  - Low = fine-grained/high-frequency structure

- **Gradient magnitude**: Captures local intensity variation / edge density
  - High = more edges, more intensity variation (Heads 0, 3)
  - Low = smoother, more uniform regions (Head 2)

Head 2's "checkerboard" pattern is actually large flat regions (low gradient) at low spatial frequency (high FFT). Heads 0 and 3 have more continuous intensity variation creating higher gradients despite lower FFT scores.

### Open Questions

1. Does this multi-scale structure emerge during training, or is it present from initialization?
2. Do low-frequency patterns (like Head 2) converge before or after high-frequency patterns?
3. What is the functional role of heads with different scale characteristics?
4. Are there neuron clusters that share similar scale profiles?

### Experimental Validation

Working prototype cells in `ModuloAdditionRefactored.py`:
- Bilinear downsampling experiments (lines 295-334)
- Coarseness metrics: FFT energy ratio, variance preservation, gradient magnitude (lines 340-529)

Screenshots archived in `temp_artifacts/` (session 2026-02-02).

---

## 2026-02-02: Quantitative Metrics for Structural Analysis

### Metrics Tested

**1. FFT Low-Frequency Energy Ratio**
```python
fft_2d = torch.fft.fft2(activation_map)
power = torch.abs(torch.fft.fftshift(fft_2d)) ** 2
# With circular mask for low frequencies (radius = 12.5% of spectrum)
coarseness = power[low_freq_mask].sum() / power.sum()
```
- **Issue**: DC component (mean) dominates the ratio
- **Solution**: Zero out center pixel before computing ratio

**2. Variance Preservation**
```python
downsampled = F.interpolate(map, size=(14, 14), mode='bilinear')
upsampled = F.interpolate(downsampled, size=(p, p), mode='bilinear')
score = upsampled.var() / original_var
```
- Captures how much structure survives aggressive downsampling
- Correlates with FFT metric but provides different perspective

**3. Gradient Magnitude**
```python
dx = activation_map[:, 1:] - activation_map[:, :-1]
dy = activation_map[1:, :] - activation_map[:-1, :]
grad = (dx.abs().mean() + dy.abs().mean()) / 2
```
- Best at distinguishing "smooth uniform regions" vs "variable intensity"
- Complements FFT metric - captures edge density rather than frequency content

### Recommended Combination

For training dynamics analysis, track both:
- **FFT no-DC**: Spatial frequency of dominant patterns
- **Gradient magnitude**: Local intensity variation / smoothness

These capture orthogonal properties and together give a richer picture of structural evolution.

### Turn Detection (Near Requirement-Ready)

**Problem**: Grokking onset ("the turn") is currently identified by visual inspection. Quantitative detection would enable:
- Automated identification of when grokking begins
- Dynamic checkpoint scheduling (uniform spacing by default, then retroactively densify around the turn)
- Downstream analytics that key off the turn epoch

**Observations**:
- The turn appears to precede all other meaningful structural events (frequency specialization, attention head differentiation, trajectory curvature)
- The existing system already handles non-uniform checkpoint schedules gracefully — adding dense checkpoints to a variant and re-running analysis "just worked"
- Candidate signals: test loss inflection, parameter velocity spike, trajectory curvature change

**Scoping notes**:
- User suggests keeping initial scope isolated: detect the turn, report the epoch, done
- Dynamic checkpoint scheduling and downstream analytics would be separate follow-on requirements
- Could start with test loss curvature (simplest signal, doesn't require new analyzers)
### Current Coverage Gaps

The analysis pipeline currently covers: model weights, embeddings, attention heads, and neurons. Missing coverage: **logits** and **unembeddings**.

### 1. Unembedding Fourier Analysis (New Analyzer)

Pair with existing `dominant_frequencies` (embedding) analyzer. Both are weight-based — extract `W_U`, compute Fourier coefficients, store them. Side-by-side visualization of embedding vs unembedding Fourier structure across training would show how these complementary representations co-evolve during grokking.

**Status**: Ready to scope as a requirement.

### 2. Unembedding PCA (Front-End Add)

The `parameter_pca` analyzer already saves out weight trajectories. Adding unembedding to PCA visualization is likely a front-end/renderer change — the data may already be available depending on which weight matrices are currently tracked. Needs verification of what `parameter_pca` currently captures.

**Status**: Verify current PCA coverage, then likely a small front-end task.

### 3. Logit Lens / Per-Input Analysis

Logit lens is most meaningful per-input (or small batch) rather than across the full p² grid. For p=113, the full grid produces ~1.4M logit values per intermediate position per epoch — too much to store as bulk artifacts, and not directly visualizable.

This connects to a broader design question: **support for additional/custom probes**. The current architecture enforces one canonical probe (full input grid) per variant. Per-input investigation is currently best served by notebook exploration (load checkpoint, `run_with_cache`, inspect).

Adding first-class support for custom probes would enable:
- Logit lens on specific inputs or small batches
- Targeted activation tracing (e.g., "how does the model handle 4+29 across training?")
- Probe subsets (e.g., only inputs where a < b)

**Status**: Brewing. UI design pressure points and probe architecture need further thought before scoping.

---

*Unembedding Fourier is ready to scope. PCA coverage needs verification. Logits/probes need more design thinking.*

---
## 2026-03-02: Grokking Window Definition — The Core Methodological Gap

### Why This Matters

Multiple claims in the findings depend on "before grokking onset" or "during the grokking window." Without a rigorous, cross-variant-stable definition of onset, these claims are unanchored. The definition of onset also determines whether Claim 8 (geometry precedes grokking) is testable at all — "how long before" is meaningless if onset is fluid.

### The Two-Problem Split

**Performance-based definition (literature standard):** Test loss drop onset. The inflection point (maximum d²acc/dt²) is the most well-defined candidate — doesn't require knowing the final plateau value, is mathematically deterministic given sufficient smoothing. But this measures the *consequence* of grokking, not the cause.

**Mechanistic definition (appropriate for scaffolding hypothesis):** For claims about what precedes grokking, a representation-level marker is the right reference point. Circularity & Fourier Alignment exceeding a threshold (current candidate: >0.40) is promising because: (a) it's already computed, (b) it measures a property that should precede performance, (c) its timing relative to the test loss drop is itself evidence for the scaffolding hypothesis.

### DMD Residual as Third Candidate

The DMD pipeline (REQ_050/051) will produce a per-epoch residual norm measuring where the linear approximation of centroid dynamics breaks down. This is a dynamics-based onset marker — expect low residual during memorization, a spike during grokking transition, and settling post-grokking. If this aligns with both the performance marker and the Fourier Alignment marker, it strengthens all three simultaneously.

### Validation Approach

Pick two or three candidate definitions, compute them for all variants, check whether the relative ordering (fast vs. slow grokkers, successful vs. failed) is stable across definitions. If ordering is preserved, the choice is less critical than it appears. If not, the instability is informative about what's actually varying.

### Downstream Connections

- **Claim 8 resolution**: The Fourier Alignment marker, if it consistently precedes the performance marker, resolves the OR (synchronization vs. alignment) and anchors the timing claim.
- **Cross-variant table in findings.md**: Grokking window column currently based on visual inspection — a stable metric definition turns this into a computable, reproducible table.
- **PC3 Circularity/Fourier revisit**: Current circularity metric uses only top-2 PCA components (built before PC3 significance was recognized). Once global PCA exists (REQ_050), revisiting this calculation to include PC3 is a small but potentially important correction. A 3-component circularity metric may be more predictive than the current 2-component version.

---

*Grokking window definition is the methodological linchpin. DMD residual, Fourier Alignment threshold, and test loss inflection point are the three candidates. Cross-variant ordering stability is the validation test. PC3 circularity revisit is a natural follow-on once global PCA is in place.*

---

## 2026-03-08: Two New Lenses for Initialization and Data Split Mechanics

Motivated by the data_seed=999 experiment: the model attempted a different frequency set than the canonical, and one frequency (Freq 40) failed the attention-MLP handshake. Two new analytical lenses would directly address the mechanism.

### Lens 1: Initialization Frequency Bias

**What it shows:** Which frequencies the model is already biased toward at epoch 0 — before any gradient signal. This includes frequencies that later lose the competition, not just surviving ones.

**Why it matters:** The lottery ticket findings (2026-02-23) showed that initialization IS predictive for anomalous variants. But that analysis looked at what happened — this lens would look at the cause. If an initialization creates systematic energy in frequencies {13, 25, 33, 40} at epoch 0, that predicts which frequency set the model will attempt to build. If {40} has weaker initial energy, it's already disadvantaged entering the competition.

**What to compute:** Fourier analysis of weight matrices at epoch 0 (and perhaps the first few hundred epochs before memorization). The same Fourier projections already used for dominant_frequencies and neuron_fourier, but applied at initialization and in a per-frequency energy view rather than a "what won" view. Show energy across all frequencies, not just the dominant ones — the losers are the point.

**Visualization:** Per-head/per-matrix frequency energy at epoch 0 as a bar chart or heatmap. Cross-epoch view of early-epoch (0-1k) frequency energy trajectory showing which frequencies build early momentum. Overlay with the eventual competition outcome (which frequencies won) to show prediction quality.

**Scope note:** This is primarily a new visualization of data already captured in parameter_snapshot (epoch 0 is always checkpointed). No new analyzer needed — it's a new view over existing data.

---

### Lens 2: Training Data Split Composition

**What it shows:** The Fourier structure of the actual (a,b) pairs in the training set — which frequencies are well-represented and which are underrepresented by the specific data split.

**Why it matters:** The data split steers frequency set selection (2026-03-08 findings). To understand *why* dseed=999 steers toward {13, 25, 33, 40} rather than {9, 33, 38, 55}, we need to see the Fourier content of what's in the training set. If the split systematically underrepresents the pairs most informative for certain frequencies, those frequencies start with less gradient signal from step one.

**What to compute:** For each (a,b) pair in the training set, compute its contribution to each Fourier frequency. Aggregate per-frequency to produce a "coverage profile" — how much Fourier signal the training data carries at each frequency k. Compare against the full grid (uniform coverage) to show gaps. This is essentially applying the Fourier analysis to the training set as a dataset, not to model weights.

**Two sub-views:**
1. **Split visualization:** Scatter plot of the p×p grid with training vs test pairs colored differently. Organized by row/column sums (or (a+b) mod p) to reveal any structural holes.
2. **Frequency coverage profile:** Per-frequency k, compute the projection of the training set indicator function onto the Fourier basis. Show as a bar chart. A flat profile = uniform coverage across all frequencies; gaps = frequencies the training data underserves.

**Cross-variant utility:** For a fixed prime, compare the coverage profile across multiple data seeds. This directly tests whether dseed=598's consistent success is explained by better Fourier coverage, or whether it's subtler (e.g., particular (a+b) residues covered vs. missed).

**Scope note:** This is a one-time analysis tool, not a per-epoch analyzer. It operates on the training set index arrays already stored in metadata, or can be recomputed from data_seed. Lightweight to implement as a notebook or standalone view.

---

*Together these lenses close the causal chain: initialization bias → frequency candidates → data split coverage → which candidates receive sustained gradient → attention-MLP handshake success or failure.*

---
