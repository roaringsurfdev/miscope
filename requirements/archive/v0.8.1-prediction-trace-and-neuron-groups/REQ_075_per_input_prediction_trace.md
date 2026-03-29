# REQ_075: Per-Input Prediction Trace

**Status:** Complete (v0.8.1)
**Priority:** Medium
**Related:** REQ_074 (Variant Outcome Registry), REQ_052 (Frequency Quality Scoring)
**Last Updated:** 2026-03-15

---

## Problem Statement

All current analysis lenses operate on aggregate statistics — test loss, circularity, neuron specialization fractions, DMD modes. These answer "how well is the model doing overall" but not "which specific training pairs has the model learned, and in what order?"

The attractor dynamics hypothesis predicts that pairs sharing the same residue class (a+b mod p = c) should be learned together: if the model commits to a frequency set that represents residue c, all pairs in that class should become predictable at roughly the same epoch. If pairs graduate in isolated clusters rather than residue-class clusters, the frequency commitment story needs revision.

This is not answerable from aggregate metrics. It requires a trace: for each training pair (a, b), at each checkpoint, what does the model predict?

A first-pass implementation capturing top-1 predictions is sufficient to test the residue-class clustering hypothesis and reveal the temporal structure of pair acquisition.

---

## Conditions of Satisfaction

### 1. `input_trace` analyzer

A new analyzer, `InputTraceAnalyzer`, that runs per-checkpoint and stores per-pair predictions.

Per checkpoint output (`.npz`):
- `predictions`: int16 array of shape `(n_pairs,)` — argmax of output logits for each training pair
- `correct`: bool array of shape `(n_pairs,)` — `predictions == true_labels`
- `confidence`: float16 array of shape `(n_pairs,)` — max softmax probability for each training pair
- `pair_indices`: int16 array of shape `(n_pairs, 2)` — `(a, b)` for each pair (stored once in epoch 0 artifact; subsequent epochs may omit if retrieval handles this)

`n_pairs` is the number of training pairs for the variant (not p², since the training set is a fraction of all pairs).

### 2. Cross-epoch summary artifact

A cross-epoch summary, computed from the per-checkpoint artifacts, that answers residue-class level questions without loading every checkpoint:

- `graduation_epochs`: int32 array of shape `(n_pairs,)` — first epoch at which each pair becomes correct and stays correct for at least `min_stable_window` subsequent checkpoints (default: 3 checkpoints). `-1` if never graduated.
- `residue_class_accuracy`: float32 array of shape `(n_checkpoints, p)` — for each checkpoint and each residue value c, fraction of training pairs with `(a+b)%p == c` that are predicted correctly
- `overall_accuracy_by_epoch`: float32 array of shape `(n_checkpoints,)` — fraction of training pairs correct at each checkpoint

### 3. Views

**`input_trace_accuracy_grid`** (per-epoch)
- `(p × p)` heatmap colored by correct/incorrect at the selected epoch
- Axes: a (x), b (y); cell color: correct=blue, incorrect=light gray
- Family context provides `prime` to size the grid

**`residue_class_accuracy_timeline`** (cross-epoch)
- Line plot: one trace per residue class c (0 to p-1), x=epoch, y=fraction correct for that class
- Color by residue value using circular colormap (same as token geometry PCA)
- Exposes whether residue classes graduate together (staircase-by-class) or interleaved

**`pair_graduation_heatmap`** (summary)
- `(p × p)` heatmap where cell color encodes `graduation_epoch` for that pair
- Gray for never-graduated pairs
- Reveals spatial structure: do adjacent pairs (similar a or b) graduate together, or do residue-class diagonals graduate together?

### 4. Tests

- Unit: `InputTraceAnalyzer` produces correct shape output for a minimal model (p=11, 20 pairs)
- Unit: `graduation_epochs` is -1 for a pair that is never stable-correct across the window
- Unit: `graduation_epochs` is the first epoch of sustained correctness, not the first isolated correct prediction
- Unit: `residue_class_accuracy` sums correctly — each pair contributes to exactly one residue class
- Integration: analyzer runs on at least one real variant and artifacts load cleanly
- Integration: all three views render without error on a real variant

---

## Constraints

- The analyzer runs on training pairs only, not test pairs. Test accuracy is already tracked by existing metrics.
- `pair_indices` must be stored in the artifact (not recomputed) because training set composition depends on data seed.
- `confidence` is float16 to manage artifact size. Full float32 precision is not required for visualization.
- The cross-epoch summary is computed as a post-processing step, not per-checkpoint. It should be regenerable from the per-checkpoint artifacts.
- Views are universal (no family-specific logic in renderers). The `prime` and training pair layout are injected by the family's view loader.
- The analyzer must not store the full `(n_checkpoints, n_pairs)` predictions matrix in a single artifact — this is too large to load at once. Per-checkpoint files are the correct granularity.

---

## Notes

- **Why top-1 only for first pass?** The key question — do residue classes graduate together? — is answered by correctness alone. Top-1 confidence is included as a low-cost addition that later enables "confident but wrong" analysis, which may reveal the wrong-attractor signature (model is confident about the wrong residue class).
- **The residue-class graduation hypothesis:** If the model commits to frequency k at epoch T, and frequency k is sufficient to represent residue class c, then all pairs in class c should graduate around epoch T. A staircase pattern in `residue_class_accuracy_timeline` where classes graduate in discrete steps — rather than a smooth sigmoid — is the predicted attractor signature.
- **Pair graduation heatmap orientation:** The (p × p) grid has a natural diagonal structure: pairs with the same sum (a+b = constant) lie on anti-diagonals. If the model learns by residue class, the anti-diagonals should be uniformly colored. Deviations (spots, stripes by row or column) would indicate a different learning structure.
- **Artifact size estimate:** p=113, ~60% training fraction = ~7,700 pairs. At 300 checkpoints: predictions (int16) = ~4.6MB, confidence (float16) = ~4.6MB total across all checkpoints. Well within tractable range.
- **Future extensions:** Once this is in place, the per-input view could support "trace a single pair through training" — showing confidence evolution and intermediate activations for one (a, b) pair. This is the logit-lens-adjacent direction the researcher noted, deferred to a follow-on.
