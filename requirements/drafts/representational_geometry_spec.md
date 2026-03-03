# Representational Geometry Tracker — Specification

## Purpose

This module adds a new analysis capability to the Training Dynamics Workbench that tracks **how the geometry of the model's internal representations evolves during training**. Where existing analyses track parameter-space trajectories (weight PCA, Fourier coefficients, neuron frequency assignments), this tracks what happens in **activation space** — the representations the model actually builds of its inputs.

The core question: **When and how do the model's representations of different output classes become separable during training, and what geometric mechanisms drive that separability?**

This connects directly to the GLUE framework (Chou et al., 2024) which decomposes representational "untangling" into specific geometric components. In a small model like modular addition, we can observe both the parameter-level causes (neuron frequency commitment, weight trajectory geometry) and the representation-level effects (manifold separation) simultaneously.

---

## Conceptual Overview

### What are "manifolds" in this context?

For a model computing `(a + b) mod p`, there are `p` possible output classes (residues 0 through p-1). For each output class `r`, there is a set of input pairs `{(a, b) : (a + b) mod p = r}`. When we feed all these inputs through the model and collect activations at a given layer, we get a cloud of activation vectors — one per input pair — that all share the same label. That cloud is the **manifold** for class `r`.

Early in training (before grokking), these manifolds are likely tangled — overlapping, high-dimensional, hard to separate. After grokking, they should be well-separated with specific geometric structure (likely reflecting the circular/Fourier structure the model has learned).

### What do we measure?

For each training epoch and each activation site, we compute:

1. **Class centroids**: The mean activation vector for each output class
2. **Effective radius**: How spread out each class manifold is (average distance from centroid)
3. **Effective dimensionality**: How many dimensions each manifold actually uses (PCA participation ratio)
4. **Manifold capacity** (simplified): A proxy for how many manifolds could be linearly separated in this space
5. **Center geometry**: How the centroids of all p classes are arranged relative to each other — particularly whether they form circular structure and when that emerges

---

## Implementation Specification

### New Analyzer: `RepresentationalGeometryAnalyzer`

This follows the existing Analyzer pattern. It loads checkpoints, runs the full input grid through the model, collects activations, and computes geometric measures.

#### Input Requirements

- Model checkpoint (safetensors file) for a given epoch
- The full input grid: all `p × p` input pairs `(a, b)` for `a, b ∈ {0, ..., p-1}`
- Model loaded via TransformerLens (HookedTransformer)

#### Activation Sites to Probe

For the 1-layer model, the key sites are:

| Site Name | TransformerLens Hook | Description |
|-----------|---------------------|-------------|
| `embed` | `hook_embed` (or equivalent post-embed) | Raw token embeddings before any processing |
| `post_attn` | `blocks.0.hook_attn_out` | After attention (before MLP) — residual stream state |
| `post_mlp` | `blocks.0.hook_mlp_out` | MLP output (before adding to residual stream) |
| `resid_post` | `blocks.0.hook_resid_post` | Final residual stream (after MLP addition) |

**Note**: The exact hook names should be verified against the model's `hook_dict()`. TransformerLens naming can vary slightly depending on model configuration.

For the embeddings specifically: the model embeds two tokens (a and b). The relevant representation may be the sum/concatenation of the two embeddings, or the residual stream at a specific position. Verify how the existing activation cache handles this — the `neuron_freq_norm` analyzer likely already solves this problem.

#### Computation Steps (per epoch)

```
For each checkpoint epoch:
    1. Load model from safetensors
    2. Generate full input grid: all (a, b) pairs, shape (p*p, 2)
    3. Run forward pass with activation caching at all hook sites
    4. For each activation site:
        a. Extract activations, shape (p*p, d_model) or (p*p, d_mlp)
           - For residual stream sites: use the "output position" token
           - This should match how existing analyzers extract activations
        b. Compute output labels: y = (a + b) mod p, shape (p*p,)
        c. Group activations by output class (p groups, each with p samples)
        d. Compute geometric measures (see below)
    5. Store results as npz artifact
```

#### Geometric Measures to Compute

**Per-class measures** (arrays of shape `(p,)` per epoch):

1. **Centroid**: Mean activation vector per class. Shape: `(p, d)` where `d` is the activation dimension.

2. **Effective radius**: For class `r` with activations `X_r` (shape `(p, d)`) and centroid `μ_r`:
   ```
   radius_r = sqrt(mean(||x - μ_r||² for x in X_r))
   ```
   This is just the RMS distance from centroid.

3. **Effective dimensionality** (participation ratio): Run PCA on the centered class activations. If eigenvalues are `λ_1, λ_2, ...`:
   ```
   D_eff_r = (Σ λ_i)² / Σ(λ_i²)
   ```
   This equals 1 if all variance is on one axis, and equals `d` if variance is uniform across `d` dimensions.

**Global measures** (scalars per epoch):

4. **Mean radius**: Average of per-class radii.

5. **Mean dimensionality**: Average of per-class effective dimensionalities.

6. **Center spread**: The spread of centroids in activation space.
   ```
   global_centroid = mean of all centroids
   center_spread = sqrt(mean(||μ_r - global_centroid||² for all r))
   ```

7. **Center circularity score**: How well the centroids form a regular polygon / lie on a circle. Approach:
   - Compute PCA on the `(p, d)` matrix of centroids
   - Project centroids into the top-2 PC plane
   - Fit a circle to the projected points (least-squares)
   - Score = 1 - (residual variance / total variance in top-2 plane)
   - Additionally: compute the angular ordering of centroids on the fitted circle and compare to the natural ordering of residue classes. If the model has learned Fourier structure, residue `r` should map to angle `2πkr/p` for some frequency `k`.

8. **Capacity proxy**: A simplified measure of linear separability.
   - Approach: For each pair of classes, compute the Fisher discriminant ratio:
     ```
     J(r, s) = ||μ_r - μ_s||² / (σ_r² + σ_s²)
     ```
     where `σ_r²` is the mean within-class variance for class `r`.
   - Report: mean Fisher ratio across all class pairs, and minimum Fisher ratio (the bottleneck pair).
   - A more principled alternative: use the SVM margin. Train a linear SVM on random binary dichotomies (class r vs class s) and report average margin. But the Fisher ratio is simpler and faster to start with.

9. **Signal-to-noise ratio (SNR)**:
   ```
   SNR = between-class variance / within-class variance
        = center_spread² / mean_radius²
   ```
   This single number captures how "untangled" the representations are overall.

#### Output Artifact Format

Store as npz with the following arrays:

```python
{
    'epochs': epochs,                          # (n_epochs,)
    # Per-class measures at each site
    'embed_radii': ...,                        # (n_epochs, p)
    'embed_dimensionality': ...,               # (n_epochs, p)
    'embed_centroids': ...,                    # (n_epochs, p, d_embed)
    'post_attn_radii': ...,                    # (n_epochs, p)
    'post_attn_dimensionality': ...,           # (n_epochs, p)
    'post_attn_centroids': ...,                # (n_epochs, p, d_model)
    'post_mlp_radii': ...,                     # (n_epochs, p)
    'post_mlp_dimensionality': ...,            # (n_epochs, p)
    'post_mlp_centroids': ...,                 # (n_epochs, p, d_model)
    'resid_post_radii': ...,                   # (n_epochs, p)
    'resid_post_dimensionality': ...,          # (n_epochs, p)
    'resid_post_centroids': ...,               # (n_epochs, p, d_model)
    # Global measures at each site
    'embed_mean_radius': ...,                  # (n_epochs,)
    'embed_mean_dim': ...,                     # (n_epochs,)
    'embed_center_spread': ...,                # (n_epochs,)
    'embed_circularity': ...,                  # (n_epochs,)
    'embed_snr': ...,                          # (n_epochs,)
    'embed_fisher_mean': ...,                  # (n_epochs,)
    'embed_fisher_min': ...,                   # (n_epochs,)
    # ... same pattern for post_attn, post_mlp, resid_post
}
```

**Practical note on storage**: The centroid arrays could be large for high-dimensional activations across many epochs. If storage is a concern, centroids can be stored in a separate artifact file, or only the top-k PCA projections of centroids can be stored (which is what's needed for the circularity analysis anyway). The scalar summary measures are small regardless.

---

## Visualization Specifications

### Visualization 1: Representational Geometry Dashboard

A multi-panel time series view, aligned on the epoch axis with the existing training loss curve. This is the primary "what happened" view.

**Panels (top to bottom, sharing x-axis = epoch):**

1. **Training loss** (existing — replicate or reference for alignment)
2. **SNR over training** — one line per activation site (embed, post_attn, post_mlp, resid_post), showing how signal-to-noise evolves. Expect: flat/low pre-grokking, rapid increase during grokking, plateau post-grokking.
3. **Mean radius over training** — one line per site. Expect: radius should decrease as manifolds tighten.
4. **Mean dimensionality over training** — one line per site. Expect: dimensionality should decrease as representations concentrate on task-relevant subspace.
5. **Center circularity over training** — one line per site. This is the "when does circular structure emerge" indicator.
6. **Fisher capacity (mean and min)** over training — shows separability. The min is particularly interesting because it identifies the hardest-to-separate class pair.

**Interactivity**: Epoch slider (synchronized with existing dashboard controls) + hover for exact values. Vertical reference line at the grokking epoch (where test loss drops).

### Visualization 2: Centroid Geometry Snapshots

At a selected epoch, show how class centroids are arranged.

**Panel A**: PCA projection of centroids into top-2 PCs. Points colored by residue class (using a cyclic colormap so nearby residues have similar colors). If the model has learned Fourier structure, this should show a circle (or regular polygon) with residues ordered by their value mod p.

**Panel B**: Same, but animated or slider-controlled across epochs, so you can watch the centroids organize from random scatter into structured arrangement.

**Panel C**: Pairwise distance matrix of centroids (p × p heatmap). For a model that has learned modular structure, this should show a circulant pattern (distance depends only on |r - s| mod p).

### Visualization 3: Correlation with Parameter Dynamics

This is the key novel view — overlaying representational geometry with parameter-space observations.

**Option A — Dual-axis time series**: Plot SNR (or circularity) on one y-axis and a parameter-space measure (e.g., fraction of committed neurons, or PC2 variance) on the other, sharing the epoch x-axis. Highlight correlations or phase transitions.

**Option B — Scatter plot**: For each epoch, plot a parameter-space measure (x) vs a representation-space measure (y), colored by epoch. This shows whether the two are tightly coupled or have complex/hysteretic relationships (loops in this space would indicate the parameter and representation measures lead/lag each other).

### Visualization 4: Per-Class Detail View

For a selected epoch, show the per-class radius and dimensionality as bar charts (or a heatmap over epochs × classes). This reveals whether all classes untangle simultaneously or whether some classes lead. In modular arithmetic, symmetry suggests they should be roughly simultaneous, but broken symmetry (specific seeds, specific frequencies being learned first) could produce interesting patterns.

---

## Integration Points with Existing Analyses

### Connecting to Neuron Frequency Dynamics

The `neuron_freq_trajectory` analysis tracks when individual MLP neurons commit to specific frequencies. The representational geometry analysis tracks when class manifolds become separable. The connection:

- **Hypothesis**: Each neuron's frequency commitment contributes to manifold separation for the output classes that depend on that frequency. So neuron commitment events should correlate with jumps in capacity/SNR.
- **Test**: Overlay the neuron commitment timeline (from `compute_commitment_epoch`) with the SNR trajectory. Mark each neuron's commitment epoch on the SNR curve. Do clusters of neuron commitments precede or coincide with SNR jumps?

### Connecting to Parameter Trajectory Geometry

The PC2 vs PC3 loop structure in parameter space shows the model exploring and then settling. The representational geometry trajectories show the *consequences* of that exploration for the representations.

- **Hypothesis**: The inflection points in the PC2/PC3 trajectory (where the path curves sharply) correspond to phase transitions in representational geometry.
- **Test**: Compute the curvature of the parameter trajectory at each epoch. Plot curvature alongside SNR rate-of-change. Do peaks align?

### Connecting to Embedding Fourier Structure

The embedding Fourier coefficients show which frequencies are present in the embedding layer. The centroid circularity analysis shows when circular structure emerges in activation space.

- **Hypothesis**: Circularity in representational space follows (possibly with a lag) the emergence of dominant Fourier frequencies in the embeddings.
- **Test**: Plot embedding Fourier coefficient magnitudes alongside centroid circularity score over training.

---

## Implementation Notes

### Matching Existing Code Patterns

Based on the `neuron_freq_trajectory.py` notebook:

```python
# Loading pattern:
family = load_family(FAMILY_NAME)
variant = family.get_variant(prime=PRIME, seed=SEED)
loader = ArtifactLoader(variant.artifacts_dir)

# Loading pre-computed artifacts:
stacked = loader.load_epochs("artifact_name")
epochs = stacked["epochs"]
data = stacked["key_name"]

# The analyzer should produce artifacts consumable via this same pattern.
```

### Batch Processing Considerations

- The full input grid is `p × p` pairs. For p=59, that's 3,481 forward passes; for p=113, that's 12,769. With a 1-layer transformer this is fast even on CPU.
- Geometric computations (PCA, distances) on `(p, d_model)` matrices are trivial.
- The main cost is iterating over all checkpoint epochs. If there are ~80 checkpoints per variant, total computation per variant should be under a minute.

### Dependencies

- numpy (already in use)
- torch (already in use)
- transformer_lens (already in use)
- scipy (for circle fitting in circularity score, if not already available)
- sklearn (optional, for SVM-based capacity if desired later)

---

## Future Extensions

Once the basic tracker is in place and producing interpretable results, natural extensions include:

1. **Frequency-decomposed analysis**: Instead of grouping by output class, group by Fourier frequency component and track when frequency-specific manifolds separate. This connects more directly to the neuron frequency commitment analysis.

2. **Layer-wise decomposition**: For the 1-layer model this is limited, but the framework generalizes to deeper models where you can track untangling across layers.

3. **Full manifold capacity computation**: Implement the actual manifold capacity theory from Chung & Abbott (2021) rather than the Fisher proxy. This involves solving a constrained optimization for each manifold. The `mftma` Python package exists for this.

4. **Cross-variant comparison**: Compare representational geometry trajectories across different seeds and primes to identify universal patterns vs. seed-specific variations.

5. **Connection to the dimensionality crossing phenomenon**: You've observed that output weight dimensionality falls below input weight dimensionality at the moment specialization emerges. The representational geometry tracker can show what this crossing looks like from the activation side — likely a transition from high-dimensional, dispersed representations to low-dimensional, structured ones.
