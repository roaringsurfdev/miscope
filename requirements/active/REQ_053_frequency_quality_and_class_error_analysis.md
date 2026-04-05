# REQ_053: Frequency Quality and Per-Class Error Analysis

**Status:** Complete
**Priority:** High
**Related:** REQ_052 (Fourier Frequency Quality Scoring — required dependency), REQ_045 (Fisher Minimum Pair)
**Last Updated:** 2026-03-03

## Problem Statement

Even with a frequency quality score (REQ_052) and overall accuracy curves, there is no way to see *which specific classes* the model is failing on, or to relate class-level failure patterns to the quality of the model's frequency selection.

Two research questions motivate this requirement:

1. **When a model has a poor frequency subset, which residue classes does it fail on?** A model with low frequency quality might fail uniformly across all classes, or it might fail specifically on classes whose signal depends heavily on the missing frequencies. Distinguishing these tells us something about how the model's learned representation breaks down.

2. **Does frequency quality predict per-class failure over training?** If a model's quality score is low early in training, is there a corresponding pattern in which classes are poorly predicted, and does recovery in quality correlate with accuracy recovery by class?

The Fisher minimum (REQ_045) already measures the worst-case class separation in representational geometry. This requirement closes the loop by connecting frequency quality → class accuracy → representational separation in a single set of views.

## Conditions of Satisfaction

### Per-Class Accuracy Artifact

- [x] A new analyzer `class_accuracy` computes per-epoch per-class accuracy for modulo addition
  - Implemented as part of the `input_trace` analyzer (REQ_075); separate `class_accuracy` analyzer not needed
- [x] "Class" is the output residue `(a + b) % p` — accuracy is measured per target residue (p classes total)
- [x] Per-epoch artifact contains per-residue accuracy, per-pair accuracy, and overall accuracy
  - Fields: `test_residue_class_accuracy (n_epochs, p)`, `test_overall_accuracy (n_epochs,)`, `train_*` equivalents
  - Stored in `artifacts/input_trace/` per-epoch and summary
- [x] Accuracy is computed over the full evaluation set (all p² input pairs), not a sample

### Views

- [x] A per-epoch view `input_trace.accuracy_grid` shows a p×p grid of (a, b) pairs colored by train/test correctness
- [x] A cross-epoch view `input_trace.residue_class_timeline` shows per-class accuracy over training epochs (test split)
- [x] A cross-epoch view `input_trace.frequency_quality_vs_accuracy` overlays test accuracy and frequency quality score on a shared [0,1] y-axis
- [x] All views are accessible via `variant.at(epoch).view(name)` through the catalog
- Note: Fisher minimum overlay deferred; single-epoch bar chart by residue superseded by the more informative timeline view

### Validation

- [x] Views validated against known variants including p101/s485/ds999 (rebounder with aliasing failure) and p113/s485/ds999 (mild rebounder, viable geometry) — per-class accuracy drop signatures are visible and interpretable against VC analysis

## Constraints

**Must:**
- The `class_accuracy` analyzer is a family-contributed concept — it requires a ground-truth evaluation set that the modulo addition family provides
- Views that render class accuracy data are universal instruments — they accept the artifact regardless of which family produced it
- This requirement depends on REQ_052 for the quality trajectory overlay; the per-class accuracy artifact and its views can be implemented independently of REQ_052

**Must not:**
- Implement per-input trace analysis (probing through the network) — that is the scope of a future requirement (REQ_054 or similar)
- Require cross-variant comparison infrastructure — all views operate on a single selected variant

**Decision authority:**
- **Resolved:** `accuracy_by_pair` is stored in the artifact. At p=113 this is ~12,800 floats per epoch (~100KB uncompressed) — manageable, and avoids requiring model checkpoints at render time.
- **Claude decides:** Color scales, axis formatting, and whether to normalize the Fisher minimum to the [0, 1] range for co-plotting

## Context & Assumptions

Per-class accuracy for modulo addition is computed by running the trained model on all p² input pairs, taking the argmax of the output logits, and comparing to the ground-truth residue. This requires the model weights (from the checkpoint) and the input pairs (generated from task parameters).

The relationship between frequency quality and per-class accuracy is the central empirical question. If a model lacks frequency k, then residues whose representation depends heavily on frequency k's components should be predicted poorly. This is a testable structural claim — this requirement creates the tools to test it.

The Fisher minimum (REQ_045) measures the worst-case class pair separation in the representation geometry. Including it in the overlay view allows visual inspection of whether geometric separation failure co-occurs with accuracy failure, and whether both correlate with frequency quality.

## Notes

- The `accuracy_by_pair` artifact may be large (p² floats per epoch × number of epochs). For p=113, that is ~12,800 floats per epoch — manageable but worth noting. If storage becomes a concern, this could be stored at coarser epoch resolution.
- The notebook proof-of-concept mentioned in the roadmap ("understanding failure modes across the network") is explicitly deferred. This requirement stops at per-class accuracy and its relationship to frequency quality. Per-input network tracing is a larger infrastructure effort.
- The research question "do models that start with high-quality frequency subsets converge faster?" is best answered by combining REQ_052's quality trajectory with the cross-variant analysis infrastructure (a future requirement). This requirement lays the groundwork within single-variant analysis.
