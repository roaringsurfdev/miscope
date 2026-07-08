# REQ_023: Coarseness Analyzer

**Status:** Draft
**Priority:** High (enables grokking research direction)
**Dependencies:** REQ_021b (Analysis Library), REQ_022 (Summary Statistics)
**Last Updated:** 2026-02-06

## Problem Statement

Observations across eight trained Modulo Addition variants reveal that grokking models develop "blob" neurons (large, coherent activation regions) while the sole non-grokking model (p=101, seed=999) retains only "plaid" neurons (fine-grained checkerboard patterns). There is currently no analyzer to quantify this distinction.

A **coarseness metric** — the ratio of low-frequency to total power in a neuron's activation pattern — would allow the researcher to:
- Track blob neuron emergence across training
- Compare coarseness trajectories between grokking and non-grokking models
- Test whether early coarseness predicts grokking timing

This is the foundational data generation step. Visualizations of coarseness data are deferred to a separate requirement.

## Design

### Coarseness Metric

Coarseness quantifies how much of a neuron's activation variance is explained by low modular frequencies. It composes existing library functions:

```
ActivationCache
  → extract_mlp_activations()           # (p², d_mlp)
  → reshape_to_grid()                   # (d_mlp, p, p)
  → compute_2d_fourier_transform()      # (d_mlp, n_comp, n_comp)
  → compute_frequency_variance_fractions()  # (n_freq, d_mlp)
  → sum first k frequencies per neuron   # (d_mlp,)  ← coarseness
```

All functions except the final summation already exist in `analysis/library/`. The summation is a new library function.

**Definition:** For each neuron, coarseness is the sum of variance fractions for frequencies 1 through `k_cutoff`:

```
coarseness[neuron] = sum(freq_fractions[0:k_cutoff, neuron])
```

Where `freq_fractions` is the output of `compute_frequency_variance_fractions` (DC-excluded, normalized per neuron). The result is a value in [0, 1] where:
- High coarseness (>= 0.7): "Blob" neurons with large coherent activation regions
- Low coarseness (< 0.5): "Plaid" neurons with fine-grained checkerboard patterns

### Approach: Modular Fourier vs Standard FFT

The exploratory notebook (`ModuloAdditionRefactored.py`) implements coarseness via standard 2D FFT with a circular low-frequency mask. The analysis library uses a modular Fourier basis aligned with the mathematical structure of the task.

**Recommended: Modular Fourier approach.** Reasons:
- Consistent with the existing analysis model (same basis used by `NeuronFreqClustersAnalyzer`)
- Composes directly from tested library functions
- Frequency indices (k=1, 2, 3...) have domain meaning in modular arithmetic
- No new FFT infrastructure needed

### Library Addition

A new function in `analysis/library/fourier.py`:

```python
def compute_neuron_coarseness(
    freq_fractions: torch.Tensor,  # (n_frequencies, d_mlp)
    n_low_freqs: int = 3,
) -> torch.Tensor:
    """Compute coarseness (low-frequency energy ratio) per neuron.

    Args:
        freq_fractions: Per-neuron variance fractions from
            compute_frequency_variance_fractions()
        n_low_freqs: Number of lowest frequencies to consider "low".
            Default 3 captures frequencies k=1,2,3.

    Returns:
        Tensor of shape (d_mlp,) with coarseness values in [0, 1].
    """
```

### Analyzer

A new `CoarsenessAnalyzer` in `analysis/analyzers/coarseness.py`:

**Per-epoch artifact:**
```
coarseness/epoch_{NNNNN}.npz
  coarseness: shape (d_mlp,)  — per-neuron coarseness values
```

This enables future visualizations like "Coarseness vs Neuron Index" heatmaps (recommendation #3 from coarseness research notes).

**Summary statistics** (via REQ_022 infrastructure):

| Key | Shape | Description |
|-----|-------|-------------|
| `mean_coarseness` | scalar | Mean coarseness across all neurons |
| `std_coarseness` | scalar | Standard deviation |
| `median_coarseness` | scalar | Median coarseness |
| `p25_coarseness` | scalar | 25th percentile |
| `p75_coarseness` | scalar | 75th percentile |
| `blob_count` | scalar | Count of neurons with coarseness >= 0.7 |
| `coarseness_hist` | (20,) | Histogram of coarseness values (20 bins, range [0, 1]) |

These summary stats directly enable the top research analyses:
- **Mean coarseness vs epoch** (recommendation #2) — uses `mean_coarseness`
- **Blob neuron count over time** (recommendation #5) — uses `blob_count`
- **Coarseness distribution over time** (recommendation #1) — uses `coarseness_hist`

### Family Registration

The coarseness analyzer should be added to the Modulo Addition 1-Layer family's analyzer list in `family.json`. The metric generalizes in principle, but the current implementation relies on the modular Fourier basis and 2D activation grid, making it family-specific for now.

### Parameters

**`n_low_freqs`** (default: 3): The number of lowest frequencies to include in the coarseness ratio. This is a tunable parameter. The default of 3 captures the first three harmonics (k=1, 2, 3), which should distinguish blob patterns (dominated by k=1 or k=2) from plaid patterns (energy spread across many frequencies). This may need adjustment based on initial results.

**`blob_threshold`** (default: 0.7): The coarseness threshold above which a neuron is counted as a "blob" neuron in the `blob_count` summary stat.

## Scope

This requirement covers:
1. Library function: `compute_neuron_coarseness()` in `analysis/library/fourier.py`
2. Analyzer: `CoarsenessAnalyzer` with per-epoch artifacts and summary statistics
3. Registration in the Modulo Addition 1-Layer family's analyzer list
4. Tests for the library function and analyzer

This requirement does **not** cover:
- Visualizations of coarseness data (separate requirement)
- Dashboard integration for coarseness views (separate requirement)
- Cross-variant coarseness comparison (future work)
- Alternative coarseness definitions (standard FFT approach)

## Conditions of Satisfaction

### Library
- [ ] `compute_neuron_coarseness()` added to `analysis/library/fourier.py`
- [ ] Function accepts output of `compute_frequency_variance_fractions()` and `n_low_freqs` parameter
- [ ] Returns tensor of shape `(d_mlp,)` with values in [0, 1]
- [ ] Function is exported from `analysis/library/__init__.py`

### Analyzer
- [ ] `CoarsenessAnalyzer` class in `analysis/analyzers/coarseness.py`
- [ ] Composes existing library functions (no duplicated FFT logic)
- [ ] Per-epoch artifact: `{"coarseness": ndarray(d_mlp,)}`
- [ ] Implements `get_summary_keys()` returning summary stat keys
- [ ] Implements `compute_summary()` returning all summary statistics
- [ ] Registered in `AnalyzerRegistry` via `register_default_analyzers()`

### Family Integration
- [ ] `"coarseness"` added to Modulo Addition 1-Layer family's analyzers list in `family.json`

### Tests
- [ ] Library function: coarseness values in [0, 1] for valid inputs
- [ ] Library function: known frequency patterns produce expected coarseness (high for low-freq, low for high-freq)
- [ ] Analyzer: conforms to Analyzer protocol
- [ ] Analyzer: produces correct artifact keys
- [ ] Analyzer: summary keys match `compute_summary()` output keys
- [ ] Analyzer: summary values have expected shapes
- [ ] Integration: pipeline produces both per-epoch artifacts and summary.npz

## Constraints

**Must have:**
- Uses modular Fourier approach (consistent with existing analysis model)
- Composes existing library functions — does not reimplement Fourier transform logic
- Per-epoch artifact contains per-neuron values (not just summary statistics)
- Summary statistics computed inline via REQ_022 infrastructure

**Must avoid:**
- Duplicating activation extraction or Fourier transform code from other analyzers
- Hardcoding modulus or architecture assumptions (use context and probe)

**Flexible:**
- Default `n_low_freqs` value (3 is a starting point, may be tuned)
- Default `blob_threshold` value (0.7 is a starting point)
- Exact set of summary statistics (core set above, can add more)
- Whether parameters are constructor args or derived from context

## Decision Log

| Date | Question | Decision | Rationale |
|------|----------|----------|-----------|
| 2026-02-06 | Modular Fourier vs standard FFT? | Modular Fourier | Consistent with existing analysis model; composes from tested library functions |
| 2026-02-06 | Separate analyzer vs extend freq_clusters? | Separate analyzer | Different output semantics; coarseness is a reduction, not a decomposition |
| 2026-02-06 | Per-epoch artifact content? | Per-neuron coarseness vector | Enables neuron-index heatmap visualization; small enough to store per-epoch |

## Notes

**2026-02-06:** The `NeuronFreqClustersAnalyzer` already computes the full frequency variance decomposition that coarseness is derived from. In principle, coarseness could be computed from freq_clusters artifacts via post-processing. However, computing it as a separate analyzer with summary statistics is cleaner: (1) it avoids loading freq_clusters artifacts to derive a secondary metric, (2) it produces summary stats inline, and (3) the per-epoch artifact has a clear, self-describing shape (`(d_mlp,)` vs extracting and summing from a `(n_freq, d_mlp)` matrix).

**Relationship to exploratory code:** `ModuloAdditionRefactored.py` (lines 370–552) contains four coarseness-related functions using standard FFT. These served as exploration; this requirement formalizes the concept using the modular Fourier approach for consistency with the analysis framework.
