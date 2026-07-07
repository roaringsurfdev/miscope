# REQ_089: 2L MLP Geometry Analysis

**Status:** Active
**Priority:** Medium
**Branch:** feature/req-089-2l-mlp-geometry-analysis
**Depends on:** REQ_088 complete and variant trained to grokking
**Attribution:** Engineering Claude

---

## Problem Statement

With the 2L MLP family trained and analysis artifacts computed (REQ_088), the scientific
question can be addressed: does the phase tiling and saddle geometry observed in the
transformer's neuron frequency groups also appear in the MLP's hidden layer?

This requirement covers the notebook exploration and any view/renderer additions needed
to support side-by-side comparison between the transformer and MLP variants.

---

## Conditions of Satisfaction

### Notebook Analysis
- [ ] Notebook `notebooks/compare_transformer_vs_mlp_geometry.ipynb` created
- [ ] For both the transformer (p113/s999/ds598) and the MLP equivalent:
  - Fourier phase scatter (`neuron_group.fourier_phase`) rendered at final epoch for each frequency group
  - Neuron group 3D PCA scatter (`neuron_group.scatter_3d`) rendered at final epoch
  - Saddle check (`neuron_group.saddle_check`) rendered with radius_percentile=0 and 50
- [ ] Comparison documented in the notebook with explicit observation notes:
  - Does the MLP show a ring, a saddle, or neither in the group PCA scatter?
  - Does the Fourier phase scatter show continuous ring coverage or spokes?
  - Do R² values from the saddle check differ systematically between architectures?

### Finding Documentation
- [ ] Key findings from the comparison added to `notes/findings_ring_geometry.md`
- [ ] If the result warrants a fieldnotes entry, one is drafted

### View Additions (if needed)
- [ ] Any new views or renderer changes required to support the comparison are implemented
  and registered in `universal.py`
- [ ] No new views are added speculatively — only what the comparison actually requires

---

## Constraints

**Must:**
- Use the existing `neuron_group.fourier_phase`, `neuron_group.scatter_3d`, and
  `neuron_group.saddle_check` views — do not duplicate rendering logic
- Both architectures must be analyzed at equivalent training stages
  (final epoch, or matched by test loss if training lengths differ)

**Flexible:**
- Whether the comparison extends to additional variants (multiple primes or seeds) in v1
- Whether a dashboard page is created for cross-architecture comparison (stretch goal)

---

## Architecture Notes

The existing view catalog is architecture-agnostic by design. If REQ_087 and REQ_088 are
complete, `variant.at(epoch).view("neuron_group.scatter_3d")` should work identically for
both transformer and MLP variants. The notebook just calls the same views on two different
variant objects.

The main risk is that `parameter_snapshot` may not store MLP weights under the same key
names as the transformer (W_in exists for both; W_E, W_pos, etc. do not). Views that
require transformer-specific weights will need to handle missing keys gracefully or be
excluded from the MLP variant's view list. The family's `analyzers` list already controls
which analyzers run; the view catalog's `required_analyzers` field controls which views
are available. If the MLP variant didn't run `attention_freq`, views that require it
simply won't be available — no error, just not listed.

---

## Notes

- The primary deliverable is the comparison finding, not the infrastructure. If the result
  is "MLP shows a plain 2D ring, no saddle," that is a complete and valuable finding that
  directly advances the core research question.
- If the result is "MLP shows the same saddle," the next question becomes: is it the same
  in training dynamics (does the saddle appear at the same training stage relative to
  grokking?) or only at the final epoch? That would motivate epoch-sweeping the comparison,
  which is a follow-on notebook cell rather than a new requirement.
- The fieldnotes entry for this finding, if warranted, would be a natural candidate for a
  first published entry on the platform's research blog — it's a clean, self-contained
  experiment with a binary outcome and clear implications.
