# REQ_049: Neuron Fourier Decomposition

**Status:** Draft
**Priority:** High (foundation for phase alignment, IPR, and lottery ticket analyses)
**Dependencies:** REQ_048 (Secondary Analysis Tier), REQ_021f (Per-Epoch Artifacts)
**Last Updated:** 2026-02-23

## Problem Statement

The existing `dominant_frequencies` analyzer applies Fourier analysis to the embedding matrix W_E only. This captures which frequencies are represented in the input embedding space, but misses the MLP layers entirely.

He et al. (2026) establish that the mechanistic story of modular addition is told primarily by the MLP weights: each neuron learns to specialize in a single frequency, with its input weight vector (θ_m) and output weight vector (ξ_m) both organized as sinusoids at that frequency. The relationship between the two phases — specifically that ψ_m ≈ 2φ_m — is the key structural signature of a correctly-learned neuron.

The following analyses from the catalog all require per-neuron Fourier decomposition of W_in and W_out:

- **Phase Alignment Tracking** (#2): D^k_m = 2φ^k_m − ψ^k_m; requires φ_m and ψ_m per neuron per frequency
- **IPR / Frequency Sparsity** (#3): requires the full Fourier magnitude spectra α^k_m and β^k_m
- **Lottery Ticket Initial Condition** (#4): requires magnitudes and phase misalignment at epoch 0
- **Neuron Specialization Tracking** (#9): requires α^k_m(t) to compute dominant frequency and specialization ratio

None of these can be computed from the existing artifacts. `parameter_snapshot` stores the raw weight matrices, but the Fourier decomposition has never been performed. Computing it at render time (in notebooks or the dashboard) would repeat the same mistake REQ_038 corrected for PCA trajectory.

This requirement introduces `neuron_fourier`, a secondary analyzer (REQ_048) that performs this decomposition once per epoch and stores the results.

### Architectural Note: He et al. vs Transformer Architecture

He et al. analyze a 2-layer FC network where inputs are one-hot vectors of size p. Each neuron m has:
- θ_m ∈ ℝ^p: the weight vector connecting input tokens directly to neuron m
- ξ_m ∈ ℝ^p: the weight vector connecting neuron m to output logits

The transformer used in this platform has an intermediate embedding space (d_model ≠ p). Direct application of the Fourier basis (size p) to W_in[:, m] ∈ ℝ^{d_model} is not meaningful — the basis dimension doesn't match.

To recover token-space vectors of length p, we compose:
- **Effective input weight**: θ_m = W_E[:p] @ W_in[:, m] ∈ ℝ^p
  (projects the neuron's MLP input sensitivity back through the embedding to token space)
- **Effective output weight**: ξ_m = W_out[m, :] @ W_U ∈ ℝ^p
  (projects the neuron's MLP output contribution forward through the unembedding to logit space)

This composition maps our transformer weights into He et al.'s formulation, enabling direct comparison with their theory. The Fourier basis from family context (size p) applies to these composed vectors.

Note: `W_E` from `parameter_snapshot` excludes the equals token (shape `(p, d_model)` after dropping the last row), matching the existing `dominant_frequencies` convention.

## Design

### Analyzer: NeuronFourierAnalyzer

A `SecondaryAnalyzer` named `neuron_fourier`:

```python
class NeuronFourierAnalyzer:
    name = "neuron_fourier"
    depends_on = "parameter_snapshot"

    def analyze(
        self,
        artifact: dict[str, np.ndarray],
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        ...
```

**Input**: one epoch's `parameter_snapshot` artifact, which contains W_E, W_in, W_out, W_U (among others).

**Computation**:

```
For each neuron m ∈ [0, M):
    θ_m = W_E[:p] @ W_in[:, m]          # effective input weight in token space, shape (p,)
    ξ_m = W_out[m, :] @ W_U             # effective output weight in logit space, shape (p,)

    g_m = B_p^T · θ_m                   # Fourier coefficients of input weights, shape (p,)
    r_m = B_p^T · ξ_m                   # Fourier coefficients of output weights, shape (p,)

    For each frequency pair k ∈ [1, (p-1)/2]:
        # He et al. convention: paired cos/sin components
        α^k_m = √(2/p) · ||(g_m[2k], g_m[2k+1])||    # input magnitude at frequency k
        φ^k_m = atan2(-g_m[2k+1], g_m[2k])             # input phase at frequency k

        β^k_m = √(2/p) · ||(r_m[2k], r_m[2k+1])||    # output magnitude at frequency k
        ψ^k_m = atan2(-r_m[2k+1], r_m[2k])             # output phase at frequency k
```

The Fourier basis `B_p` comes from `context["fourier_basis"]` — the same basis already used by `dominant_frequencies`.

**Output arrays** (all shape `(n_neurons, n_freq_pairs)` where `n_freq_pairs = (p-1)//2`):

| Key | Shape | Description |
|-----|-------|-------------|
| `alpha_mk` | (M, K) | Input layer Fourier magnitudes α^k_m |
| `phi_mk` | (M, K) | Input layer Fourier phases φ^k_m ∈ (-π, π] |
| `beta_mk` | (M, K) | Output layer Fourier magnitudes β^k_m |
| `psi_mk` | (M, K) | Output layer Fourier phases ψ^k_m ∈ (-π, π] |
| `freq_indices` | (K,) | Integer frequency indices k = 1 ... (p-1)/2 |

Where M = n_neurons (d_mlp), K = (p-1)/2 frequency pairs.

The DC component (k=0) is excluded from the output — it is not meaningful for the analyses this data will support (phase alignment, lottery ticket, specialization), which all focus on non-DC frequency pairs.

### Storage

```
artifacts/
  parameter_snapshot/
    epoch_00000.npz    # primary: W_E, W_in, W_out, W_U, ...
    epoch_00100.npz
    ...
  neuron_fourier/
    epoch_00000.npz    # secondary: alpha_mk, phi_mk, beta_mk, psi_mk, freq_indices
    epoch_00100.npz
    ...
```

### Family Registration

`ModuloAdditionFamily.get_secondary_analyzers()` returns `[NeuronFourierAnalyzer()]`.

### Analysis Library

The core Fourier operations (basis projection, polar extraction) should be implemented in `analysis/library/fourier.py` as reusable functions, not embedded in the analyzer class. The `dominant_frequencies` analyzer already uses `project_onto_fourier_basis()` from this library. Add:

- `extract_frequency_pairs(fourier_coeffs, prime) → (magnitudes, phases)`: extracts per-frequency-pair magnitudes and phases from a flat Fourier coefficient vector

Keeping computation in the library keeps the analyzer class thin (per CLAUDE.md function length guidance) and allows the same functions to be called from notebooks.

### View Registration

Register two views in `miscope/views/universal.py` alongside existing views:

**`neuron_fourier_heatmap`**: Per-epoch per-neuron heatmap of input Fourier magnitudes (α^k_m). Visual analog to the DFT heatmap in He et al. Figure 2. Shows which frequency each neuron has specialized into at a given epoch.

**`neuron_fourier_heatmap_output`**: Same for output Fourier magnitudes (β^k_m). Paired with the input heatmap, reveals whether input and output layers have co-specialized.

These are both `per_epoch` views (use `epoch_source_analyzer="neuron_fourier"`).

Renderers live in `visualization/` following existing patterns.

## Scope

This requirement covers:
1. `NeuronFourierAnalyzer` secondary analyzer
2. `extract_frequency_pairs()` library function in `fourier.py`
3. `ModuloAdditionFamily.get_secondary_analyzers()` returning the analyzer
4. Two view registrations: `neuron_fourier_heatmap` and `neuron_fourier_heatmap_output`
5. Tests for the analyzer and library function

This requirement does **not** cover:
- Phase alignment computation (future requirement)
- IPR computation (future requirement)
- Lottery ticket initial condition analysis (future requirement)
- Neuron specialization tracking (future requirement)
- Summary statistics for `neuron_fourier` (deferred — define when downstream requirements clarify what scalar summaries are useful)

## Conditions of Satisfaction

### Analyzer
- [ ] `NeuronFourierAnalyzer` implements `SecondaryAnalyzer` protocol
- [ ] `depends_on = "parameter_snapshot"`
- [ ] Computes θ_m via W_E[:p] @ W_in[:, m] for all neurons
- [ ] Computes ξ_m via W_out[m, :] @ W_U for all neurons
- [ ] Projects onto Fourier basis from context
- [ ] Extracts per-frequency-pair magnitudes and phases using He et al. convention
- [ ] Outputs `alpha_mk`, `phi_mk`, `beta_mk`, `psi_mk`, `freq_indices`
- [ ] All output arrays have correct shapes: (M, K) for matrices, (K,) for freq_indices

### Library
- [ ] `extract_frequency_pairs(fourier_coeffs, prime)` in `fourier.py`
- [ ] Returns magnitudes and phases for all (p-1)/2 frequency pairs
- [ ] Consistent with He et al. phase convention: φ = atan2(-sin_coeff, cos_coeff)

### Family
- [ ] `ModuloAdditionFamily.get_secondary_analyzers()` returns `[NeuronFourierAnalyzer()]`

### Views
- [ ] `neuron_fourier_heatmap` view registered in `universal.py`
- [ ] `neuron_fourier_heatmap_output` view registered in `universal.py`
- [ ] Both views renderable via `variant.at(epoch).view("neuron_fourier_heatmap").show()`
- [ ] Heatmap axes: neurons (rows) × frequencies (columns), value = magnitude

### Tests
- [ ] Analyzer output shapes are correct for a known (prime, d_mlp) configuration
- [ ] Magnitudes are non-negative
- [ ] Phases are in (-π, π]
- [ ] `freq_indices` matches expected range [1, (p-1)/2]
- [ ] Numerical consistency: a neuron fully specialized at frequency k should show α^k_m ≈ total magnitude, all others ≈ 0
- [ ] Views render without error for a real variant epoch

## Constraints

**Must have:**
- Effective weight composition (W_E @ W_in, W_out @ W_U) — not raw W_in/W_out slices, which are in the wrong space
- He et al. phase convention for phase extraction (documents the choice for reproducibility)
- Library function for frequency pair extraction — not inlined in the analyzer

**Must avoid:**
- Loading the model or checkpoint files — this is a secondary analyzer; artifact data only
- Storing DC component — it's not used by downstream analyses and wastes space
- Recomputing the Fourier basis — use context["fourier_basis"]

**Flexible:**
- Whether to store the raw complex Fourier coefficients alongside magnitudes/phases (probably not needed, but would be a simple addition)
- Exact renderer styling (color scale, axis labels) — standard Plotly heatmap is acceptable

## Decision Log

| Date | Question | Decision | Rationale |
|------|----------|----------|-----------|
| 2026-02-23 | Raw MLP weights vs composed (W_E @ W_in)? | Composed | He et al.'s formulation requires vectors in token space (length p). Raw W_in[:, m] has length d_model and the Fourier basis doesn't apply directly. Composition recovers the token-space vector. |
| 2026-02-23 | Store DC component? | No | No downstream analysis uses it. Phase alignment and lottery ticket both focus on non-DC frequency pairs. Omitting keeps array shapes clean. |
| 2026-02-23 | Magnitudes + phases vs complex coefficients? | Magnitudes + phases | Downstream analyses (phase alignment: atan2 operations, IPR: power ratios) need magnitudes and phases directly. Storing in polar form eliminates re-computation. |
| 2026-02-23 | Summary statistics in this requirement? | Deferred | Useful summaries (mean specialization, top frequency per neuron) depend on how Phase Alignment and IPR requirements shape their data needs. Better to define them once that context exists. |

## Notes

**2026-02-23:** This requirement addresses the gap identified in the discussion that led to REQ_048. The `parameter_snapshot` data for all 11 variants already exists. Once REQ_048 and this requirement are implemented, running `pipeline.run(force=False)` on existing variants will automatically populate `neuron_fourier/epoch_*.npz` artifacts without re-training or re-running primary analysis on the model.

**2026-02-23:** The two views defined here (`neuron_fourier_heatmap` and `neuron_fourier_heatmap_output`) are the immediate visual payoff — the per-epoch heatmap showing neuron × frequency specialization at any training moment. This is the "Figure 2" equivalent from He et al. and is useful for exploration before the downstream derived analyses (phase alignment, IPR) are implemented.

**Sources:**
- He, Wang, Chen & Yang (2026). *On the Mechanism and Dynamics of Modular Addition.* arXiv:2602.16849. §3.1, §5.1
- See `articles/techniques/analysis_techniques_catalog.md` entries #1 (DFT), #2 (Phase Alignment), #3 (IPR), #4 (Lottery Ticket)
