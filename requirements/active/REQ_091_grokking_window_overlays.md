# REQ_091: Grokking Window Overlays for Timeseries Views

**Status:** Active
**Priority:** Medium
**Branch:** feature/req-091-grokking-window-overlays
**Attribution:** Engineering Claude

---

## Problem Statement

Every timeseries view in the platform has epoch on the x-axis, but none of them show
where the grokking phases are. Users must mentally cross-reference epoch numbers against
the variant summary (first descent, plateau, cascade, second descent) to answer any
timing question. This is friction at exactly the moment when timing is the question —
does SNR rise before second descent? Does circularity cross over at cascade onset?

The epoch window boundaries are already computed and stored in `variant_summary.json`
for every variant. The data is there; it just isn't surfaced on any plot.

---

## Conditions of Satisfaction

### Core utility

- [ ] Utility function `add_grokking_windows(fig, variant)` in
  `src/miscope/visualization/grokking_markers.py`
- [ ] Reads window boundaries from `variant.metadata` (or `variant_summary.json`)
- [ ] Adds vertical lines or light shaded regions for each phase boundary present:
  first descent end, plateau start, plateau end, cascade start, second descent start
- [ ] Labels each region (e.g., "plateau", "cascade") as hover text or axis annotation
- [ ] Handles variants with no second descent gracefully — marks only the windows
  that exist
- [ ] Works on any single-axis Plotly `go.Figure` with epoch on the x-axis
- [ ] Works on multi-panel figures (applies markers to all subplots sharing the x-axis)

### Integration

- [ ] Applied in the two views where timing questions are most pressing:
  `weight_geometry.timeseries` and `geometry.timeseries`
- [ ] Controlled by a `show_grokking_windows` kwarg (default `True`) so it can be
  suppressed when variant context is unavailable

### Dashboard

- [ ] Where a timeseries view is shown alongside the epoch slider, the current phase
  window is shown as a label (e.g., "cascade") near the slider or in the view title

---

## Constraints

**Must:**
- Be post-hoc: the utility adds markers to an already-rendered figure, so renderers
  don't need to know about variant context
- Degrade silently if window data is missing for a variant

**Must not:**
- Change any renderer signature in a breaking way
- Add variant as a parameter to renderers (renderers are artifact-data-only)

**Flexible:**
- Whether boundaries are shown as vertical lines, shaded bands, or both
- Exact color and opacity of the overlay

---

## Architecture Notes

The utility sits between the renderer (which produces a figure from artifact data) and
the view/dashboard layer (which has the variant). This is cleanly post-hoc:

```python
fig = render_weight_geometry_timeseries(data, matrix="Win")
fig = add_grokking_windows(fig, variant)
fig.show()
```

The variant_summary epoch windows are stored in `variant_summary.json` under keys like
`grokking_epoch`, `cascade_start`, `second_descent_start`, etc. The exact field names
should be confirmed against the actual file format before implementation.

---

## Notes

- Motivated by the observation (2026-04-06) that SNR and Fisher discriminant in
  `weight_geometry.timeseries` clearly track second descent, but without window markers
  it's impossible to say whether they lead, coincide, or lag.
- This is a universal platform need — essentially every timeseries view benefits from
  this treatment.
