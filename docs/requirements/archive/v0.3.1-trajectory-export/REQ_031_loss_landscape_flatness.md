# REQ_031: Loss Landscape Flatness

**Status:** Draft
**Priority:** Medium (most expensive analyzer; benefits from REQ_029/030 context)
**Dependencies:** REQ_021a (Model Family), REQ_021f (Per-Epoch Artifacts), REQ_022 (Summary Statistics)
**Last Updated:** 2026-02-09

## Problem Statement

A central argument in the grokking literature (Millidge et al., among others) is that SGD has an implicit bias toward flat regions of the loss landscape. The generalizing solution — the Fourier circuit — occupies a large-volume, flat basin of attraction, while the memorizing solution sits in a sharper basin. Grokking occurs when the model escapes the memorizing basin and finds the generalizing one.

This analyzer directly measures the **local flatness** around each checkpoint by randomly perturbing model weights and observing how much the loss changes. Flat regions tolerate large perturbations; sharp regions don't.

Combined with trajectory projections (REQ_029) and effective dimensionality (REQ_030), this completes the geometric picture:
- **Trajectory** shows where the model goes
- **Dimensionality** shows how constrained the solution is
- **Flatness** shows how stable (broadly optimal) the solution is

### Computational cost

This is the most expensive analyzer in the pipeline. Each epoch requires N additional forward passes (one per perturbation direction). For N=50 directions and 300 checkpoints, that's 15,000 forward passes. For the Modulo Addition model (~12K probe inputs, small architecture), each forward pass is fast (~10ms on GPU), so total wall time is ~2.5 minutes. This is tractable but worth being explicit about.

## Design

### Flatness Metric: Random Perturbation Sampling

For each checkpoint:
1. Load model at checkpoint, compute baseline loss on the probe dataset
2. Sample N random unit-norm direction vectors in parameter space
3. For each direction d:
   - Perturb weights: theta' = theta + epsilon * d
   - Compute loss on probe dataset
   - Record delta_loss = loss(theta') - loss(theta)
4. Store the distribution of delta_loss values and summary statistics

**Epsilon (perturbation magnitude):** The scale at which we probe flatness. Too small and everything looks flat (linear regime). Too large and we're not measuring local geometry. A reasonable default for the Modulo Addition model is epsilon = 0.1 (relative to typical parameter norms), but this should be tunable.

**Direction sampling:** Directions are sampled from a standard normal distribution and normalized to unit L2 norm. This gives uniform coverage of the parameter space hypersphere.

**Loss function:** The probe dataset loss (cross-entropy on the full (a, b) grid). This is the same loss the model is evaluated on during training, ensuring consistency.

### Analyzer: `LandscapeFlatnessAnalyzer`

A new analyzer in `analysis/analyzers/landscape_flatness.py`.

**Per-epoch artifact:**
```
landscape_flatness/epoch_{NNNNN}.npz
  baseline_loss:  scalar              — loss at unperturbed checkpoint
  delta_losses:   shape (n_directions,) — loss change per perturbation direction
  epsilon:        scalar              — perturbation magnitude used
```

Storing the full `delta_losses` array (not just summary statistics) per epoch enables:
- Distribution visualization (histogram of how loss changes)
- Percentile-based flatness measures
- Future: directional analysis (which directions are flat vs sharp)

Storage per epoch: ~50 floats x 4 bytes = ~200 bytes. Negligible.

**Summary statistics** (via REQ_022 infrastructure):

| Key | Shape | Description |
|-----|-------|-------------|
| `mean_delta_loss` | scalar | Mean loss change across perturbation directions |
| `median_delta_loss` | scalar | Median loss change (robust to outliers) |
| `max_delta_loss` | scalar | Worst-case loss change |
| `std_delta_loss` | scalar | Standard deviation of loss changes |
| `p90_delta_loss` | scalar | 90th percentile loss change |
| `flatness_ratio` | scalar | Fraction of directions where loss increases by < 10% of baseline |
| `baseline_loss` | scalar | Loss at unperturbed checkpoint |

The `flatness_ratio` is the most directly interpretable summary: "what fraction of random directions are approximately flat?" A value of 0.9 means 90% of directions tolerate the perturbation without significant loss increase.

### Library Functions

New file: `analysis/library/landscape.py`

```python
def compute_landscape_flatness(
    model: HookedTransformer,
    probe: torch.Tensor,
    loss_fn: Callable,
    n_directions: int = 50,
    epsilon: float = 0.1,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Measure local loss landscape flatness via random perturbation.

    Samples random unit directions in parameter space, perturbs the
    model weights by epsilon along each direction, and measures the
    resulting loss change.

    Args:
        model: Model at a specific checkpoint.
        probe: Input tensor for loss computation.
        loss_fn: Function(model, probe) -> scalar loss.
        n_directions: Number of random perturbation directions.
        epsilon: Perturbation magnitude (L2 norm of perturbation vector).
        seed: Optional RNG seed for reproducibility.

    Returns:
        Dict with "baseline_loss", "delta_losses", "epsilon".
    """
```

**Implementation notes:**
- The function must restore original weights after each perturbation (clone + restore pattern)
- Perturbation vectors are sampled in full parameter space, not per-matrix
- The loss function should be provided by the model family (via context or as a standard interface) to ensure the correct loss metric is used
- Seed parameter enables reproducible flatness measurements across runs

### Renderers

New file: `visualization/renderers/landscape_flatness.py`

```python
def render_flatness_trajectory(
    summary_data: dict[str, np.ndarray],
    current_epoch: int,
    metric: str = "mean_delta_loss",
    title: str | None = None,
) -> go.Figure:
    """Flatness metric over epochs.

    Primary line: selected metric (mean_delta_loss by default).
    Optional band: p10-p90 range if available.
    Epoch indicator at current_epoch.

    Lower values = flatter landscape.
    """


def render_perturbation_distribution(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    title: str | None = None,
) -> go.Figure:
    """Histogram of loss changes from random perturbations at one epoch.

    Shows the distribution of delta_loss values, annotated with
    mean, median, and flatness_ratio. Reveals whether flatness is
    uniform or directional (e.g., flat in most directions but sharp
    in a few).
    """
```

### Dashboard Integration

Two panels:

1. **Flatness Trajectory** — flatness metric over epochs with metric selector dropdown (mean, median, max, flatness_ratio). Synced with epoch slider. Overlaid with baseline_loss on secondary y-axis to show relationship between loss and flatness.
2. **Perturbation Distribution** — per-epoch histogram. Updates with epoch slider.

Both panels conditional on `landscape_flatness` artifacts existing.

### Family Registration

Add `"landscape_flatness"` to the Modulo Addition 1-Layer family's analyzers list in `family.json`.

### Loss Function

The analyzer needs a loss function to evaluate perturbed models. For Modulo Addition, this is cross-entropy on the probe dataset. The model family should provide this via the analysis context:

```python
# In ModuloAddition1LayerFamily.prepare_analysis_context():
context["loss_fn"] = cross_entropy_loss_fn  # or similar
```

If the loss function is not in context, the analyzer should fall back to a standard cross-entropy computation using the model's forward pass and the probe dataset labels.

**Design decision:** The loss function dependency is the one place where this analyzer differs meaningfully from others. Existing analyzers only need the model, probe, and cache. This one needs to evaluate loss, which requires knowing the correct output labels and loss metric. For Modulo Addition, the probe dataset contains both inputs and expected outputs (the (a, b) grid with known (a+b) mod p targets). The family's `prepare_analysis_context()` is the right place to provide this.

## Scope

**This requirement covers:**
1. Library function: `compute_landscape_flatness()` in `analysis/library/landscape.py`
2. Analyzer: `LandscapeFlatnessAnalyzer` with per-epoch artifacts and summary statistics
3. Renderers: flatness trajectory, perturbation distribution
4. Dashboard panels: trajectory + distribution
5. Loss function integration via analysis context
6. Registration in Modulo Addition 1-Layer family
7. Tests

**This requirement does not cover:**
- Directional flatness analysis (measuring flatness along specific parameter directions)
- Multi-epsilon profiling (measuring flatness at multiple scales in one run)
- Hessian trace estimation (more principled but more expensive flatness measure)
- Cross-variant flatness comparison
- Adaptive perturbation magnitude (scaling epsilon by parameter norm)

## Conditions of Satisfaction

### Library
- [ ] `compute_landscape_flatness()` returns baseline_loss, delta_losses, and epsilon
- [ ] Model weights are restored after all perturbations (no side effects)
- [ ] Perturbation directions are unit-norm in full parameter space
- [ ] Optional seed produces reproducible results
- [ ] Function exported from `analysis/library/__init__.py`

### Analyzer
- [ ] `LandscapeFlatnessAnalyzer` in `analysis/analyzers/landscape_flatness.py`
- [ ] Conforms to Analyzer protocol
- [ ] Per-epoch artifact contains baseline_loss, delta_losses, epsilon
- [ ] Implements `get_summary_keys()` returning all summary stat keys
- [ ] Implements `compute_summary()` returning all summary statistics
- [ ] Registered in `AnalyzerRegistry`
- [ ] Configurable n_directions and epsilon via constructor

### Renderers
- [ ] `render_flatness_trajectory()` plots flatness metric over epochs with metric selector
- [ ] Current epoch visually indicated
- [ ] `render_perturbation_distribution()` shows histogram with mean/median annotations
- [ ] All renderers return `go.Figure` objects

### Dashboard
- [ ] Flatness Trajectory panel with metric selector
- [ ] Perturbation Distribution panel with epoch slider sync
- [ ] Both conditional on `landscape_flatness` artifacts existing

### Family Integration
- [ ] `"landscape_flatness"` added to Modulo Addition 1-Layer `family.json`
- [ ] Loss function accessible via analysis context or reasonable default

### Tests
- [ ] Library: model weights unchanged after function call
- [ ] Library: delta_losses has length n_directions
- [ ] Library: seed produces identical results on repeated calls
- [ ] Library: perturbation of a model at a known minimum produces small delta_losses
- [ ] Analyzer: conforms to Analyzer protocol
- [ ] Analyzer: produces correct artifact keys
- [ ] Analyzer: summary keys match compute_summary output

## Constraints

**Must have:**
- Model weights restored after perturbation (no side effects on the model object)
- Perturbation applied in full parameter space (not per-matrix)
- Reproducible via seed parameter
- Per-epoch artifacts store full delta_losses distribution

**Must avoid:**
- Leaving model in a perturbed state after analysis
- Hardcoding loss function (must work with family-provided loss)
- Making n_directions or epsilon non-configurable

**Flexible:**
- Default n_directions (50 is a starting point; could be 20 for faster exploration or 100 for more precision)
- Default epsilon (0.1 is a starting point; may need tuning per model scale)
- Flatness ratio threshold (10% of baseline is a starting point)
- Whether to normalize epsilon by parameter norm
- Whether to use train loss, test loss, or probe loss for evaluation

## Decision Log

| Date | Question | Decision | Rationale |
|------|----------|----------|-----------|
| 2026-02-09 | Random perturbation vs Hessian trace? | Random perturbation | Simpler, cheaper, directly interpretable; Hessian is future work |
| 2026-02-09 | Single epsilon vs multi-epsilon? | Single (configurable) | Keep first version simple; multi-epsilon profiling is a future enhancement |
| 2026-02-09 | Full parameter space vs per-matrix perturbation? | Full parameter space | Measures global flatness; per-matrix is a different (directional) question |
| 2026-02-09 | Store distribution or just summary? | Full distribution | Small storage cost; enables richer visualization and future analysis |

## Notes

**Computational budget:** At 50 directions x 300 checkpoints x ~10ms per forward pass, expect ~2.5 minutes total on GPU for a full analysis run of the Modulo Addition model. This is significantly more expensive than other analyzers (which do 1 forward pass per checkpoint) but still tractable. The dashboard should indicate that this analyzer is computationally heavier, or the documentation should note expected runtime.

**Epsilon calibration:** The "right" epsilon depends on the scale of the parameters. For the Modulo Addition model, parameter norms are typically O(1) after training, so epsilon=0.1 represents a ~10% relative perturbation. For larger models or different architectures, epsilon may need to be scaled by the parameter norm. A future enhancement could add automatic epsilon calibration (e.g., set epsilon such that the median delta_loss equals some target value).

**Connection to theory:** The mean delta_loss under random perturbation is related to the trace of the Hessian (via a first-order Taylor expansion): E[delta_loss] ~ (epsilon^2 / 2) * Tr(H) / d, where d is the parameter count and H is the Hessian. This means the flatness trajectory is an empirical proxy for Hessian trace evolution, without computing the Hessian. If more precise Hessian information is needed later, Lanczos iteration or Hutchinson's trace estimator could be added as separate analyzers building on the same infrastructure.

**Loss function design:** This is the first analyzer that needs to evaluate loss (not just extract activations or weights). The `prepare_analysis_context()` pattern in ModelFamily is the right extension point. For Modulo Addition, the loss is cross-entropy on (a+b) mod p. If a family doesn't provide a loss function in context, the analyzer should raise a clear error rather than silently computing something wrong.
