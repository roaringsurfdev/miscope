# REQ_150: Dashboard Export Ignores Left-Nav View Parameters

**Status:** Draft (stub — bug)
**Priority:** Medium — exports the wrong figure (default parameterization), forcing screenshot workarounds during analysis.
**Branch:** authored directly on `develop`; implementation branch TBD.
**Attribution:** Engineering Claude (under user direction)

---

## Problem Statement

Plot export — both the single-graph **Export** button and **Batch Export** —
renders each view with its **default** parameters, ignoring the view parameters
selected in the left nav (activation `site`, `prime`, sort order, etc.). A user
looking at *"Fisher Heatmap — Attn Out"* clicks Export and gets the
**resid_post** (default-site) figure instead; the exported title is mislabeled
to match the wrong parameterization. The current workaround is to screenshot the
live graph.

## Root Cause (located)

`apps/dashboard/src/dashboard/components/export_panel.py`, `_run_export`:

```python
path = variant_server_state.context.view(view_name).export("png")
```

`.view(view_name)` is called with **no kwargs**. The export callbacks resolve
only the `view_name` (`get_view_name(graph_id)`) and **discard the associated
`view_parameter`** that the live graph was rendered with. The dashboard's
`_VIEW_LIST` / `_GRAPH_REGISTRY` carries `{view_name, view_type,
view_parameter?}`, but only `view_name` reaches the export path — so every export
uses `BoundView` defaults.

## Scope

- **Single-graph export** (`on_single_export`) and **batch export**
  (`on_batch_export`) are both affected.
- **Batch export compounds it:** the checklist is keyed by `view_name` only, so a
  parameterized view is indistinguishable from its default in the batch list —
  there is no place to carry the parameter at all.

## Conditions of Satisfaction (to flesh out at scoping)

- [ ] Single-graph export reproduces the **on-screen** figure: same view
      parameters (`site`, `prime`, sort, epoch) as the live graph it was
      triggered from.
- [ ] The parameter value comes from the **same source the live graph reads**
      (thread `view_parameter` from the graph registry / left-nav store into
      `_run_export`) — not a second copy that can drift.
- [ ] Exported filename/title reflect the parameters actually rendered.
- [ ] Batch-export semantics decided: export each selected view at its current
      parameters, or surface parameter choice in the batch list. (Discussion.)

## Notes

- Surfaced 2026-06-10 during the p109 deep dive: an *Attn Out* Fisher /
  centroid-distances export silently produced the `resid_post` default; the user
  screenshotted instead. The site distinction was load-bearing — the analysis
  turned on `attn_out` carrying 2D operand structure that `resid_post` does not
  (research `p109_late_reorganization` notebook §6).
- Likely a small fix for the single-graph path (pass the parameter kwargs
  through), but the batch-export keying needs a design decision — hence a stub,
  not a drive-by patch.
