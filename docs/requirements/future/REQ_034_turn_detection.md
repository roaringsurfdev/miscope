# REQ_034: Grokking Turn Detection

**Status:** Draft
**Priority:** High (unlocks downstream requirements: dynamic checkpointing, phase-aligned comparison)
**Dependencies:** None (uses data already stored by training pipeline)
**Last Updated:** 2026-02-10

## Problem Statement

The "turn" — the onset of grokking, where the model transitions from memorization to generalization — is the most important structural event in training. Currently it is identified by visual inspection of the test loss curve. This is imprecise, variant-dependent (different models turn at different epochs), and cannot be used programmatically.

Quantitative turn detection enables:
- Annotating visualizations with the turn epoch (vertical line on trajectories, color boundary on heatmaps)
- Phase-aligned cross-variant comparison (align variants by turn epoch instead of raw epoch number)
- Future: dynamic checkpoint scheduling (densify checkpoints around the turn)

### Scope boundary

This requirement covers **detection and reporting only**. Dynamic checkpoint scheduling, phase-aligned comparison, and other downstream uses of the turn epoch are separate future requirements. The goal is: given a variant's training data, return the estimated turn epoch and a confidence indicator.

## Design

### What is "the turn"?

The turn is the epoch at which test loss begins sustained, rapid decrease after a plateau or increase phase. Operationally, it is the point of maximum negative curvature (steepest acceleration of descent) in the test loss curve.

This is distinct from:
- **Convergence**: when loss reaches its final value (much later)
- **Memorization onset**: when train loss drops (much earlier)
- **First test loss improvement**: may be noisy and premature

### Detection Signal: Test Loss Curve

The test loss curve stored in `metadata.json` has one value per training epoch (e.g., 35,000 points for the Modulo Addition models). This is the highest-resolution signal available and shows the clearest turn signature.

**Algorithm:**
1. Smooth the test loss curve with a Gaussian filter to suppress noise
2. Compute the first derivative (rate of change)
3. Find the epoch of the most negative first derivative (steepest descent)
4. The turn epoch is the onset of this descent — back-track from the steepest point to where the derivative first crosses a threshold (sustained negative slope begins)

**Parameters:**
- `smoothing_sigma`: Gaussian smoothing window (default: proportional to total epochs, e.g., 1% of training length)
- `threshold_fraction`: What fraction of peak descent rate counts as "onset" (default: 0.1 — the turn starts when descent rate reaches 10% of its peak value)

**Why not curvature (second derivative)?** Second derivatives amplify noise and are harder to interpret. First derivative with onset detection is more robust and directly answers "when does test loss start dropping fast?"

### Library Function

New file: `analysis/library/turn_detection.py`

```python
def detect_grokking_turn(
    test_losses: list[float] | np.ndarray,
    smoothing_sigma: float | None = None,
    threshold_fraction: float = 0.1,
) -> dict[str, Any]:
    """Detect the grokking turn from a test loss curve.

    Args:
        test_losses: Per-epoch test loss values.
        smoothing_sigma: Gaussian smoothing sigma in epochs.
            None = auto (1% of total epochs).
        threshold_fraction: Fraction of peak descent rate
            that defines turn onset. Lower = earlier detection.

    Returns:
        Dict with:
            "turn_epoch": int — estimated epoch of grokking onset
            "steepest_epoch": int — epoch of maximum descent rate
            "confidence": float — 0-1 indicator of detection quality
            "descent_rate": np.ndarray — smoothed first derivative
    """
```

**Confidence indicator:**
A turn is "confident" when:
- There is a clear plateau-then-descent pattern (high contrast between pre-turn and post-turn derivative)
- The peak descent rate is significantly larger than typical noise in the derivative
- The descent is sustained (not a transient dip)

Confidence = 0.0 means "no turn detected" (e.g., model didn't grok, or loss is monotonically decreasing). Confidence = 1.0 means unambiguous sharp turn. This is a heuristic, not a statistical test.

**Edge cases:**
- **No grokking**: If test loss never drops significantly, return confidence=0.0 and turn_epoch=None
- **Immediate generalization**: If test loss drops from the start with no plateau, return confidence=0.0 (no distinct turn)
- **Multiple drops**: Use the largest/steepest drop as the primary turn

### Variant Integration

The turn epoch should be stored as part of the variant's derived metadata, not as an analyzer artifact. It is a property of the training run itself, not of a specific analysis.

```python
# New method on Variant or standalone function
def compute_turn_epoch(variant: Variant) -> dict[str, Any]:
    """Load variant's test losses and detect the turn."""
    metadata = variant.load_metadata()
    return detect_grokking_turn(metadata["test_losses"])
```

Storage: The result can be cached in `metadata.json` under a `"turn_detection"` key, or computed on demand. Given the computation is fast (smoothing + derivative of a 35k array), on-demand is fine for MVP. Caching is a future optimization.

### Visualization: Turn Annotation

Rather than creating new plots, the turn epoch should be annotatable on existing visualizations. The initial implementation adds a utility function that renderers can optionally use:

```python
def add_turn_annotation(
    fig: go.Figure,
    turn_epoch: int,
    confidence: float,
    y_range: tuple[float, float] | None = None,
) -> go.Figure:
    """Add a vertical line and label marking the turn epoch.

    Draws a dashed vertical line at turn_epoch with a "Turn" label.
    Line opacity scales with confidence (invisible at 0, solid at 1).
    """
```

This function modifies a figure in place. Any renderer can call it if the turn epoch is available. Dashboard integration (which renderers get annotations) is a separate concern — for MVP, the function exists and can be called manually or from notebooks.

### Dashboard Integration

Minimal: Add the turn epoch as a display element in the Analysis tab header (e.g., "Turn detected at epoch 26,400 (confidence: 0.92)"). No new panels or visualizations required for this requirement.

Optional enhancement: If straightforward, add the turn annotation line to the train/test loss curve plot. This is the most natural place to show it.

## Scope

**This requirement covers:**
1. Library function: `detect_grokking_turn()` in `analysis/library/turn_detection.py`
2. Turn annotation utility: `add_turn_annotation()` for adding vertical markers to figures
3. Variant-level convenience: function to detect turn from a Variant object
4. Dashboard: display detected turn epoch in the Analysis tab
5. Tests

**This requirement does not cover:**
- Dynamic checkpoint scheduling based on turn epoch
- Phase-aligned cross-variant comparison
- Multi-signal turn detection (combining test loss + velocity + dimensionality)
- Turn detection for non-grokking training dynamics
- Automatic re-detection when new checkpoints are added
- Annotation of all existing visualizations (just the utility function + loss curve)

## Conditions of Satisfaction

### Library
- [ ] `detect_grokking_turn()` returns turn_epoch, steepest_epoch, confidence, and descent_rate
- [ ] Auto-smoothing scales with input length
- [ ] Returns confidence=0.0 and turn_epoch=None when no turn is detected
- [ ] Handles edge cases: constant loss, monotonic decrease, very short sequences
- [ ] Pure numpy — no model loading, no artifact dependencies
- [ ] Function exported from `analysis/library/__init__.py`

### Annotation
- [ ] `add_turn_annotation()` adds vertical line at turn epoch
- [ ] Line opacity reflects confidence
- [ ] Works on any go.Figure (not coupled to specific renderers)

### Dashboard
- [ ] Turn epoch displayed in Analysis tab when detected
- [ ] Graceful when no turn detected (no error, just no display)

### Tests
- [ ] Synthetic test: sigmoid-like loss curve → detects turn at inflection
- [ ] Synthetic test: constant loss → returns confidence=0.0, turn_epoch=None
- [ ] Synthetic test: monotonically decreasing loss → returns confidence=0.0
- [ ] Synthetic test: noisy curve with clear turn → detects correct epoch despite noise
- [ ] Synthetic test: smoothing_sigma parameter affects detection
- [ ] Annotation test: figure gains vertical line shape after annotation
- [ ] Integration test: detect turn from real variant metadata (if available in test fixtures)

## Constraints

**Must have:**
- Pure function operating on loss arrays (no side effects, no model loading)
- Confidence indicator distinguishing "clear turn" from "unclear/no turn"
- Works on the existing 35k-point loss curves without performance issues

**Must avoid:**
- Hardcoding epoch ranges or loss thresholds specific to Modulo Addition
- Requiring new analysis pipeline runs (uses existing metadata.json)
- Modifying existing renderer signatures (annotation is additive)

**Flexible:**
- Default smoothing parameters (will likely need tuning across model families)
- Confidence calculation method (heuristic is fine for MVP)
- Whether to cache turn detection results in metadata.json
- Which existing visualizations get turn annotations in the dashboard (can start with just loss curve)

## Decision Log

| Date | Question | Decision | Rationale |
|------|----------|----------|-----------|
| 2026-02-10 | Which signal for turn detection? | Test loss curve (first derivative) | Highest resolution (35k points), clearest signal, already stored, no new computation |
| 2026-02-10 | First derivative vs second derivative? | First derivative with onset backtracking | Less noise amplification, more interpretable, directly answers "when does descent start" |
| 2026-02-10 | Store result or compute on demand? | On-demand for MVP | Computation is fast (~ms for 35k array); caching is premature optimization |
| 2026-02-10 | New visualization vs annotation on existing? | Annotation utility + display in header | Turn epoch is metadata about training, not a new analysis dimension; belongs as context on existing views |

## Notes

**Relationship to other signals:** Parameter velocity spikes, dimensionality changes, and frequency specialization onset all correlate with the turn but typically lag it. The test loss turn appears to precede all other meaningful structural events. A future multi-signal detector (combining test loss + velocity) could improve robustness, but single-signal detection from test loss is the right starting point.

**Cross-variant alignment:** Once turn epochs are detected for all variants, cross-variant comparison can align on "epochs since turn" rather than absolute epoch number. This reframes comparison from "what does variant X look like at epoch 25,000?" to "what does variant X look like 2,000 epochs after its turn?" This is the primary downstream use case, but it belongs in a separate requirement.

**Generality:** The algorithm should work for any training curve that exhibits a plateau-then-descent pattern. It is not specific to modular arithmetic or grokking — delayed generalization in any domain would produce a similar signature. However, the default parameters are tuned for the Modulo Addition training curves we have, and may need adjustment for other model families.
