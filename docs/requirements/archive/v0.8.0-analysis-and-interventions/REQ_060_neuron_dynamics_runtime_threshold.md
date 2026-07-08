# REQ_060: Runtime Threshold for Existing Neuron Dynamics Views

**Status:** Active
**Depends on:** REQ_042 (neuron dynamics views), REQ_058 (band concentration, threshold slider pattern)
**Attribution:** Engineering Claude

---

## Problem

The neuron dynamics page has two tiers of views:

- **Tier 1 (new, threshold-aware):** `per_band_specialization`, `neuron_frequency_range`,
  `band_concentration.trajectory` — these load raw `neuron_dynamics` cross-epoch data and
  recompute at any threshold at runtime.
- **Tier 2 (original, threshold-static):** `neuron_freq_trajectory`, `switch_count_distribution`,
  `commitment_timeline` — these consume `commitment_epochs` and threshold-derived fields that
  were baked into the artifact at analysis time.

The threshold slider on the neuron dynamics page controls Tier 1 but has no effect on Tier 2.
This creates an inconsistent and potentially misleading interface: adjusting the threshold changes
some views but not the heatmap directly above them.

The goal is to bring Tier 2 views into the same runtime-threshold model as Tier 1, so the slider
controls all views on the page coherently.

---

## Scope

Three views to rewire:

### `neuron_freq_trajectory` (the main heatmap)

Currently: loads `cross_epoch["dominant_freq"]` and renders all neurons regardless of
specialization state.

**With runtime threshold:** Filter visible rows to neurons that reach the threshold at any point
during training (committed at any epoch). Neurons that never cross the threshold at the given
threshold level are excluded from the heatmap. This makes the heatmap denser at high thresholds
and broader at low ones — directly showing the effect of the threshold on which neurons
"count" as part of the Fourier solution.

### `switch_count_distribution`

Currently: uses pre-baked `commitment_epochs` array; switch counts were derived at analysis
threshold.

**With runtime threshold:** Recompute switch counts from raw `dominant_freq` and `max_frac`
at the given threshold. A neuron "switches" when its dominant_freq changes between two
consecutive epochs where max_frac >= threshold. Epochs where max_frac < threshold are treated
as uncommitted (not counted as switches).

### `commitment_timeline`

Currently: uses pre-baked `commitment_epochs` (first epoch where max_frac >= analysis threshold).

**With runtime threshold:** Recompute `commitment_epochs` from raw data as the first epoch
where max_frac >= runtime threshold. At lower thresholds, commitment happens earlier; at higher
thresholds, later or never.

---

## Conditions of Satisfaction

### CoS 1: Runtime recomputation functions
Add helper functions to `band_concentration.py` or a new `neuron_dynamics_metrics.py`:
- `compute_committed_mask(cross_epoch, threshold)` → `(n_epochs, d_mlp)` bool array
- `compute_commitment_epochs(cross_epoch, threshold, epochs)` → `(d_mlp,)` float array (nan if never committed)
- `compute_switch_counts(cross_epoch, threshold)` → `(d_mlp,)` int array

### CoS 2: View loaders updated
The three view loaders in `universal.py` pass raw `cross_epoch` data (dominant_freq, max_frac)
alongside the pre-baked fields, so renderers can recompute at any threshold.

### CoS 3: Renderers accept threshold kwarg
`render_neuron_freq_trajectory`, `render_switch_count_distribution`, `render_commitment_timeline`
each accept an optional `threshold` kwarg. When provided, they use the runtime-recomputed values
instead of the pre-baked ones. When absent, they fall back to existing behavior (backward compat).

### CoS 4: Dashboard wiring
All three views move to the `nd_threshold` filter set in `neuron_dynamics.py`, so the existing
threshold slider controls them alongside the Tier 1 views.

### CoS 5: Tests
- `compute_commitment_epochs`: returns correct first-crossing epoch; returns nan when never crossed
- `compute_switch_counts`: counts correctly for a known sequence; 0 switches for a monotonically
  committed neuron; uncommitted epochs (max_frac < threshold) not counted as switches
- All three renderers return `go.Figure` with threshold kwarg provided

---

## Constraints

- Backward compatibility: existing callers that don't pass `threshold` must continue to work.
  Pre-baked artifact fields (`commitment_epochs`, etc.) remain in the artifact — they're still
  used as the default when no runtime threshold is provided.
- The runtime recomputation must operate on the already-loaded `cross_epoch` data — no additional
  artifact loads.

---

## Notes

- The switch count recomputation is the subtlest: need to decide whether "switches" count only
  transitions between committed epochs, or whether going from committed → uncommitted → committed
  (on a different frequency) counts as two switches. The simpler definition: count the number of
  unique dominant frequencies visited while max_frac >= threshold across all epochs.
- At threshold = analysis_threshold, results should be identical to the pre-baked values (good
  regression test).
