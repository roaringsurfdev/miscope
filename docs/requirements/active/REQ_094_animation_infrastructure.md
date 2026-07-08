# REQ_094: Animation Infrastructure — Dashboard Page + Fieldnotes Export

**Status:** Draft  
**Priority:** High  
**Dependencies:** REQ_047 (view catalog), neuron_group_pca artifact, parameter_snapshot artifact

---

## Problem

Animated views of weight geometry reveal training dynamics that are invisible in per-epoch snapshots:

- The plateau is not idle — frequency groups actively organize their global subspace positions before second descent begins
- The saddle forms during second descent — within-group neurons break symmetry from a flat disk into saddle wings, with timing that tracks the sigmoidal loss curve
- Anomalous models show distinct failure signatures (disk→spoke collapse in p59, entangled failure to separate in p101) that are only legible in motion

These views exist as notebook prototypes but are inaccessible from the dashboard. More broadly, any cross-epoch artifact is a candidate for animation — Centroid Class PCA, parameter trajectory, multi-stream specialization — and each of these currently requires notebook work to animate. The fieldnotes path for animated figures is similarly notebook-dependent.

This requirement establishes animation as a first-class capability: a dashboard page that hosts animated views, an infrastructure pattern that new animated views can plug into without structural changes, and an HTML export path that feeds directly into fieldnotes.

---

## Conditions of Satisfaction

### Page
- [ ] A dedicated `/animations` page exists in the dashboard, accessible from sitenav
- [ ] The page integrates the existing variant-selector-store (variant selection is shared with other pages)
- [ ] Loading is on-demand — animations are not built on variant change, only when explicitly triggered
- [ ] A visible loading state is shown while frames are computed
- [ ] An n_frames control (slider or numeric input, default 60) is exposed per animation

### Initial animated views (two required at launch)
- [ ] **W_in global PCA animation** — whole-matrix PCA with fixed final-epoch basis, neurons colored by frequency group, epoch scrubber + play/pause. Reference implementation: `load_epoch_projections` + `build_win_animation` in `notebooks/parameter_space_pca.ipynb`
- [ ] **In-group PCA animation** — per-group centroid-centered PCA, pre-computed projections from `neuron_group_pca` artifact, epoch scrubber + play/pause. Reference implementation: `build_ingroup_animation` in `notebooks/parameter_space_pca.ipynb`

### Export
- [ ] Each animation has an "Export HTML" button that writes `fig.write_html()` to `fieldnotes/public/figures/`
- [ ] Exported filename encodes variant and view name (e.g. `p109_s485_ds598_win_global_pca.html`)
- [ ] Exported figure is self-contained and embeds correctly as an iframe in MDX

### Architecture
- [ ] Animated views are defined by a simple pattern: `load_data(variant) -> dict` + `build_figure(data, n_frames) -> go.Figure`
- [ ] This pattern lives in a discoverable location (e.g. `src/miscope/views/animated.py` or `dashboard/animations/`) so future views can be added without touching page structure
- [ ] The page dispatches to view builders by name — adding a new animated view means registering it, not modifying page layout logic

---

## Constraints

- Animations are inherently multi-epoch and do not fit the per-epoch `EpochContext` view pattern. This page operates outside the view catalog and should not force the catalog to accommodate multi-epoch views.
- The Plotly frames + slider pattern (`go.Figure(frames=[...])`) is the animation mechanism — no Dash-specific animation dependencies.
- Axis ranges must be locked across frames (computed from all epochs before building frames) so the camera does not jump during scrubbing.
- PCA bases must be fixed at the final epoch for animations where cross-epoch comparability matters. This is a correctness requirement, not a preference.

---

## Architecture Notes

**Suggested pattern for an animated view:**

```python
class AnimatedView(Protocol):
    name: str
    label: str

    def load_data(self, variant: Variant) -> dict: ...
    def build_figure(self, data: dict, n_frames: int) -> go.Figure: ...
```

**Candidate location:** `src/miscope/views/animated.py` — parallel to `universal.py` in the view catalog, but not part of the per-epoch dispatch.

**Page structure:** single Dash page with a view selector (dropdown), n_frames input, trigger button, loading spinner, and a single `dcc.Graph` that receives the built figure. Export button appears after figure is built.

**Fieldnotes export:** `fig.write_html(path, include_plotlyjs="cdn")` keeps file size manageable. The iframe pattern from existing fieldnotes figure exports applies directly.

---

## Stretch Goals (not required at launch)

- W_out global PCA animation (symmetric to W_in)
- Centroid Class PCA animation (centroid positions across epochs — per-site, per-prime)
- Per-group isolation mode (show only one group at a time, for detailed saddle formation study)
- Side-by-side view (two variants, synchronized scrubber) for direct anomalous vs. healthy comparison

---

## Notes

The notebook prototypes in `notebooks/parameter_space_pca.ipynb` are the reference implementation for the two launch views. The functions `load_epoch_projections`, `build_win_animation`, and `build_ingroup_animation` should be ported to the registered pattern with minimal changes — they are already close to the `load_data` / `build_figure` split.

The fieldnotes export path closes a loop: currently getting an animation into a fieldnotes entry requires notebook work. With export built into the dashboard page, the path from observation to published figure becomes a single action.

This requirement was motivated by findings from `fieldnotes/src/content/drafts/weight_space_global_pca.mdx` — the animations that produced those findings should be reproducible and sharable from the dashboard without requiring notebook access.
