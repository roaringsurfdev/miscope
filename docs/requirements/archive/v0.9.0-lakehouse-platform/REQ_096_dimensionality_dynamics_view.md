# REQ_096: Dimensionality Dynamics View

**Status:** Draft
**Priority:** High
**Branch:** feature/req-096-dimensionality-dynamics-view
**Dependencies:** REQ_090 (freq_group_weight_geometry artifacts), REQ_047 (view catalog)
**Attribution:** Engineering Claude

---

## Problem Statement

The dimensionality dynamics notebook prototype (`notebooks/dimensionality_dynamics.ipynb`)
established that three distinct layers of dimensionality signal — parameter trajectory shape,
activation-space class centroid geometry, and weight-space frequency group geometry — all exhibit
coordinated compression/expansion events around grokking. This multi-layer view distinguishes
healthy from degraded variants more reliably than any single lens alone.

The prototype is notebook-only. The goal is to promote it to a first-class dashboard view with the
same standing as the MultiStream view: own page, loss curve anchor, shared variant-selector, and a
view catalog entry that makes it available programmatically via `variant.at(epoch).view(...)`.

---

## Conditions of Satisfaction

### 1. Analyzer Extension — `FreqGroupWeightGeometryAnalyzer`

The existing cross-epoch artifact stores per-group effective dimensionality (`Win_dimensionality`,
full participation ratio). The dimensionality dynamics view requires a different metric — PR₃ and
f_top3 — which capture shape and concentration within the top 3 principal components only.

- [ ] Add four new fields to the `freq_group_weight_geometry` cross-epoch artifact:
  - `Win_pr3`    float32 (n_epochs, n_groups) — PR₃ of each group's W_in point cloud
  - `Win_f_top3` float32 (n_epochs, n_groups) — fraction of W_in variance in top 3 PCs
  - `Wout_pr3`   float32 (n_epochs, n_groups)
  - `Wout_f_top3` float32 (n_epochs, n_groups)
- [ ] PR₃ formula: `(f1+f2+f3)² / (f1²+f2²+f3²)` where `fi = λi / Σλ` and λ are eigenvalues
  from SVD of the mean-centered group weight matrix
- [ ] f_top3 formula: `(λ1+λ2+λ3) / Σλ` — fraction of total variance in top 3 PCs
  (denominator sums ALL eigenvalues, not just top 3)
- [ ] Groups with fewer than 3 neurons store NaN for both fields
- [ ] `_compute_group_geometry` extended in-place; no structural changes to analyzer interface
- [ ] `_empty_result` updated to include the four new fields with matching empty shapes
- [ ] The existing `Win_dimensionality` / `Wout_dimensionality` fields are preserved unchanged
- [ ] Re-run `freq_group_weight_geometry` analyzer on all variants after the code change
- [ ] Docstring in `freq_group_weight_geometry.py` updated to document the four new keys

### 2. View — `dimensionality.timeseries`

A new cross-epoch view registered in the view catalog.

**Data sources (all loaded in `load_data`):**
- `parameter_trajectory` cross-epoch: `{site}__projections` (n_epochs, 10),
  `{site}__explained_variance_ratio` (10,) for sites `embedding`, `attention`, `mlp`, `all`
- `repr_geometry` cross-epoch summary: `{site}_centroids` (n_epochs, n_classes, d) for sites
  `attn_out`, `mlp_out`, `resid_post`
- `freq_group_weight_geometry` cross-epoch: `Win_pr3`, `Win_f_top3`, `group_freqs`,
  `group_sizes`, `epochs`
- `variant_summary.json`: `grokking_epoch` (onset marker), `first_descent_window.end_epoch`
  (first-descent marker), `effective_crossover_epoch` (eff_xover marker)

**Computed in `load_data` (not pre-stored):**
- Rolling trajectory PR₃ and f_top3 per site: window=10 snapshots;
  `f_top3 = var[:3].sum() / var.sum()` where var comes from the rolling window of projections
- Class centroid PR₃ and f_top3 per site: SVD of mean-centered (n_classes, d) centroid matrix
  per epoch; same PR₃ and f_top3 formulas

**Renderer (`build_dimensionality_timeseries`):**
- [ ] Three-panel stacked figure with shared x-axis (epoch)
  - Panel 1 — **Trajectory**: rolling PR₃ (solid) + f_top3 (dashed) per site
    (embedding, attention, mlp, all); one color per site
  - Panel 2 — **Class Centroids**: PR₃ (solid) + f_top3 (dashed) per activation site
    (attn_out, mlp_out, resid_post); one color per site
  - Panel 3 — **Within-Group W_in**: PR₃ (solid) + f_top3 (dashed) per frequency group;
    one color per group, labeled by dominant frequency
- [ ] Y-axis range [0, 3.2] on all three panels
- [ ] Timing markers on all panels:
  - Orange dotted vertical line: `first_descent_window.end_epoch`
  - Black dashed vertical line: `grokking_epoch`
  - Blue dotted vertical line: `effective_crossover_epoch`
  - Each marker omitted if the value is absent or ≤ 0 in variant_summary
- [ ] Legend: vertical layout, one entry per series (PR₃ and f_top3 share color, differ by line style)
- [ ] View is `epoch_selector` kind (epoch parameter accepted but ignored — always cross-epoch)

**Catalog registration:**
- [ ] View registered as `"dimensionality.timeseries"` in `src/miscope/views/universal.py`
- [ ] `requires` reflects all three analyzer dependencies:
  `["parameter_trajectory", "repr_geometry", "freq_group_weight_geometry"]`
- [ ] `variant.at(epoch).view("dimensionality.timeseries").figure()` works end-to-end

### 3. View — `dimensionality.state_space`

A companion view showing the trajectory through `(f_top3, PR₃)` state space rather than
the time evolution of each metric separately. Time is encoded as point color (early=blue,
late=red); the axes are `f_top3` (x) and `PR₃` (y). One panel per activation site.

This view makes sequencing visible: if attention reaches `(high f_top3, PR₃ ≈ 2)` before MLP
and resid_post, the three trajectories are visibly offset in state space — a relationship that
is invisible in separate timeseries because time is the x-axis there, not the color.

**Data sources (loaded in `load_data`):**
- `repr_geometry` cross-epoch summary: PR₃ and f_top3 per site per epoch
  (same intermediate data as `dimensionality.timeseries` Panel 2 — share a loader helper)
- `variant_summary.json`: `grokking_epoch`, `effective_crossover_epoch`

**Renderer (`build_dimensionality_state_space`):**
- [ ] One row, three panels: one per activation site (`attn_out`, `mlp_out`, `resid_post`)
- [ ] X-axis: `f_top3` [0, 1.05]; Y-axis: `PR₃` [0.9, 3.1]
- [ ] Trajectory: `lines+markers`, markers colored by normalized epoch (0=blue, 1=red,
  colorscale `RdYlBu_r`), line in light grey so color carries the epoch signal
- [ ] Epoch 0 marker: open circle, site color
- [ ] Final epoch marker: filled circle, site color
- [ ] Onset marker: black diamond with white outline, labeled "onset"
- [ ] `eff_xover` marker: blue triangle-up, labeled "eff_xover"
- [ ] Horizontal dotted reference line at `PR₃ = 2.0` ("ring") on each panel
- [ ] Onset and eff_xover markers omitted if absent or ≤ 0 in variant_summary
- [ ] View is `epoch_selector` kind (epoch ignored — always cross-epoch)
- [ ] Hover shows: site, epoch, f_top3, PR₃

**Catalog registration:**
- [ ] View registered as `"dimensionality.state_space"` in `src/miscope/views/universal.py`
- [ ] `requires = ["repr_geometry"]`
- [ ] `variant.at(epoch).view("dimensionality.state_space").figure()` works end-to-end

### 4. Dashboard Page — `/dimensionality-dynamics`

- [ ] New page `dashboard/pages/dimensionality_dynamics.py` following the `AnalysisPageGraphManager`
  pattern used by `multistream.py`
- [ ] `_VIEW_LIST` contains three entries in order:
  1. `"training-loss-curves"` → `training.metadata.loss_curves` (loss curve anchor at top)
  2. `"dimensionality-timeseries"` → `dimensionality.timeseries`
  3. `"dimensionality-state-space"` → `dimensionality.state_space`
- [ ] Page exposes: `create_dimensionality_dynamics_page_nav()`,
  `create_dimensionality_dynamics_page_layout()`,
  `register_dimensionality_dynamics_page_callbacks()`
- [ ] Registered in `dashboard/sitenav.py` as `/dimensionality-dynamics` with label
  `"Dimensionality Dynamics"` alongside the existing MultiStream entry
- [ ] No page-specific controls required in the nav at launch (no threshold sliders);
  the nav panel is minimal (label only, consistent with other simple pages)

---

## Notes

**PR₃ vs effective dimensionality:** `Win_dimensionality` (already in the artifact) uses the full
participation ratio `(Σλ)² / Σλ²`. PR₃ uses only the top 3 eigenvalues. They answer different
questions — full PR₃ measures total spread across all dimensions; top-3 PR₃ measures shape within
the leading subspace. Both are worth keeping.

**f_top3 denominator:** For trajectory rolling windows, the denominator must sum ALL available PC
variances (up to 10), not just the top 3. This detects diffusion into higher PCs that PR₃ alone
cannot see. Same rule applies to class centroid and within-group panels for consistency.

**n_groups confound in Panel 3:** With 2 groups, max PR₃ = 1.0 trivially; with 3 groups max = 2.0;
with 4+ groups max = 3.0. The panel is per-group so individual traces are still meaningful, but
cross-variant min-PR₃ comparisons are not directly comparable. This is a known limitation; address
in a follow-on REQ if cross-variant normalization becomes important.

**Prototype reference:** `notebooks/dimensionality_dynamics.ipynb` contains the working
implementation of all views. Key prototype functions: `compute_group_win_pr3`,
`build_dimensionality_view`, `compute_rolling_trajectory_metrics`, `_add_pr3_f_top3_traces`
(timeseries); Cell 11 (`cell-11`) renders the single-variant state space with epoch-colored
markers and Cell 13 (`780158ea`) renders the cross-variant overlay. The dashboard renderer
should port these to `src/miscope/visualization/renderers/dimensionality_dynamics.py`. The
`load_data` functions for both views can share a helper that computes PR₃ and f_top3 from
`repr_geometry` centroid arrays, since both views use the same intermediate data.
