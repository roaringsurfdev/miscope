# Experiments

Proposed training runs and interventions. Distinct from `analysis_ideas.md` (new instruments/views to build) — entries here require retraining or modified training procedures, not just new analysis over existing artifacts.

Each entry: hypothesis, proposed setup, expected outcome, and what a positive/negative result would mean.

---

## Attention Head Ablation — Breaking the Competition Window

### Hypothesis

Sustained attention head competition (multiple frequencies maintaining similar aggregate commitment) is the proximate cause of slow grokking in anomalous variants. Ablating heads mid-training forces the remaining heads into higher-stakes routing decisions, breaking the symmetric competition state and compressing the resolution window.

### Motivation

p59/seed485 and p101/seed999 both show fragmented, unresolved attention competition across their full training runs. p113 (healthy) resolves attention competition quickly via first-mover advantage. The multi-stream view makes the pattern legible: attention thrashing → MLP fragmentation → slow or failed grokking. The MLP would commit if given clean routing signal. The ablation tests whether removing routing capacity forces commitment.

### Proposed Setup

Target variant: **p59/seed485/dseed598** (clearest sustained attention competition, longest competition window ~30k epochs).

Three ablation conditions, all from the same initialization:
1. **Early ablation**: ablate 2 of 4 heads at epoch ~2k (during first descent, before any frequency has emerged as candidate)
2. **Mid ablation**: ablate 2 of 4 heads at epoch ~5k (post first descent, competition just forming — the hypothesis sweet spot)
3. **Late ablation**: ablate 2 of 4 heads at epoch ~15k (competition fully established, test whether stalemate can be broken)

Ablation method: zero the weights of the selected heads and freeze them (no gradient). Select heads by lowest attention entropy or random — worth trying both.

Control: unmodified p59/seed485/dseed598 full run (already in results).

### Expected Outcomes

- **Mid ablation succeeds**: competition window compresses, grokking onset moves earlier, multi-stream view shows faster attention lock. Confirms heads as the bottleneck.
- **Early ablation disrupts**: interferes with embedding organization phase, model groks later or fails. Establishes that heads are needed during first descent.
- **Late ablation has limited effect**: entrenched competition doesn't break from ablation alone; the state is stable, not just indeterminate. Changes interpretation of the competition window.

### What It Would Mean

A positive mid-ablation result would suggest attention head count is a sensitive hyperparameter for grokking dynamics — not just capacity, but competition resolution. Implications for training strategies: scheduled head dropout, early head specialization regularization, or head count as a curriculum variable.

---

## 2-Layer Model — Layer Specialization and Competition Propagation

### Hypothesis

In a 2-layer transformer, the competition window either (a) compresses because layer specialization naturally assigns frequency extraction to L1 and computation to L2, giving L2 cleaner signal, or (b) compounds because unresolved L1 attention competition feeds corrupted signal to L2, producing cascaded indeterminate routing.

### Motivation

The 1-layer circuit for modular arithmetic is well-characterized (Nanda et al.): attention computes Fourier products across positions, MLP applies cosine readout. A 2-layer model can decompose this differently — L1 handling extraction, L2 handling computation. If this decomposition emerges cleanly, the multi-stream timing ordering (Embedding → Attention → MLP) might become a per-layer relay: Embedding → L1 Attn → L1 MLP → L2 Attn → L2 MLP.

Whether the anomalous variants' competition dynamics become worse or better in 2 layers is an open question.

### Proposed Setup

Train 2-layer variants matching existing 1-layer configurations:
- Same primes: p59, p101, p113 (covering healthy + both anomaly types)
- Same seeds where possible for direct comparison
- Same training duration (35k epochs) and hyperparameters

The multi-stream view would need per-layer extension: separate panels for L1 attention, L2 attention, L1 MLP, L2 MLP. This requires the view to be architecture-aware (n_layers parameter). A prerequisite before running the experiments.

### Expected Outcomes

- **Layer specialization hypothesis**: L1 attention locks earlier than L2, providing cleaner signal downstream. Healthy variants show faster resolution. Anomalous variants may still struggle but the competition window might compress.
- **Compounding hypothesis**: Anomalous variants show cascaded competition — L1 never resolves, L2 inherits noise. Grokking delay compounds rather than compresses.
- **Mixed**: healthy variants benefit from layer decomposition; anomalous variants are unchanged or worse (the failure mode is initialization-driven, not capacity-driven).

### What It Would Mean

If layer specialization naturally helps, it suggests architectural depth is a partial solution to competition window problems. If it compounds, it strengthens the ablation hypothesis — the competition window is a fundamental problem that more capacity doesn't solve, only head commitment does.

This experiment also opens the model family construct to 2-layer variants, which exercises the architecture-parameterized family design.

---

## DMD Applied to Parameter Weight Trajectories

### Hypothesis

Dynamic Mode Decomposition of the stacked weight matrix snapshots will recover the Fourier frequencies of the learned algorithm as the dominant growing modes during grokking, and reveal oscillatory modes during the competition window that are not visible in PCA or per-epoch Fourier analysis.

### Motivation

PCA of the parameter trajectory shows the shape of the path through weight space. Fourier analysis shows the frequency structure at each epoch. Neither captures *dynamical structure* — which patterns are growing, decaying, or oscillating at what rates. DMD decomposes the trajectory into spatial modes and their associated eigenvalues (growth rate + oscillation frequency).

For grokking: the competition window in p59/485 and p101/999 may be a genuine oscillatory dynamical state (complex eigenvalues near the unit circle), not just gradient noise. DMD would distinguish these. For the healthy variants, the algorithm modes should appear as clean growing modes during second descent.

### Proposed Setup

No new training required — all parameter snapshots already exist. Apply DMD to:
1. **W_E trajectory**: embedding matrix over all epochs (most directly connected to the known Fourier structure)
2. **W_in/W_out trajectory**: MLP weight dynamics
3. **Stacked QKV**: attention weight dynamics

Compare DMD spectra across healthy vs. anomalous variants. Key question: do anomalous variants show eigenvalues near the unit circle (sustained oscillation) during the competition window, while healthy variants show eigenvalues rapidly moving to the positive real axis (convergent growth)?

### Expected Outcomes

- **Healthy variant**: dominant DMD modes during second descent have eigenvalues on the positive real axis, with spatial structure matching the Fourier basis. Competition window (brief) shows transient complex eigenvalues that quickly resolve.
- **Anomalous variant**: competition window shows persistent complex eigenvalues near the unit circle — a genuine oscillatory attractor, not noise. Grokking corresponds to eigenvalues moving off the unit circle toward the real axis.

### Prerequisites

DMD analyzer implementation (new, operating on stacked weight matrices). Possibly requires downsampling — 35k checkpoints at full resolution may be computationally heavy for DMD; every 100th epoch likely sufficient.

---

*This file covers interventions that require retraining or new training runs. Analysis ideas that don't require retraining live in `analysis_ideas.md`.*
