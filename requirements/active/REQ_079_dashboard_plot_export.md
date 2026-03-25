# REQ_079: Dashboard Plot Export

**Status:** Active
**Priority:** Medium
**Related:** REQ_047 (View Catalog), REQ_075 (Per-Input Prediction Trace)
**Last Updated:** 2026-03-24

---

## Problem Statement

Plots on the dashboard are the primary lens for research findings, but sharing or archiving a specific view requires navigating outside the tool. The browser's native Plotly camera icon captures only a low-resolution screenshot with no standard naming — it doesn't know which variant is loaded, what epoch is selected, or what the view is called.

The `BoundView.export()` method already generates high-quality server-side PNGs with canonical filenames (`{variant}__{view}__epoch{NNNNN}.png` under `results/exports/`). The gap is a dashboard-level trigger to call it: a round-trip that maps a selected graph to its view, runs the export, and reports back the saved path.

A secondary need is batch export: when studying a model at a specific epoch, the researcher may want to snapshot 3–4 related plots simultaneously rather than repeating the action per graph.

---

## Conditions of Satisfaction

### 1. Global graph registry

A module-level registry that maps any dashboard graph ID to its `view_name`. Populated automatically when `AnalysisPageGraphManager` instances are created (which happens at import time via each page's module-level `_graph_manager = AnalysisPageGraphManager(...)`).

Registry is queryable by graph ID: `graph_registry.get_view_name(graph_id) → str | None`.

No page-specific knowledge required in the export callback.

### 2. Export trigger — per-graph

Each graph on the dashboard has a small **Export** button (or icon link) visually associated with it. Clicking it:
- Sends the graph ID to the server (single-item export)
- Server looks up the view name via the registry
- Calls `variant_server_state.context.view(view_name).export("png")`
- Returns the saved file path
- Path is shown in a toast/notification visible to the user

If no variant is loaded or the view is unavailable, the button is disabled or shows an appropriate message.

### 3. Batch export

An **Export selected** panel, accessible from a per-page button or the nav, that:
- Lists all graphs currently rendered on the active page (populated dynamically from the registry for that page)
- Allows multi-select via checkboxes
- Has a single "Export" action that triggers all selected graphs
- Shows a single notification listing all saved paths on completion

The batch panel does not need to be a modal — a collapsible sidebar section or popover is sufficient.

### 4. File naming

Files are saved to `results/exports/` using the existing `BoundView._default_export_path()` convention:
```
{variant_name}__{view_name}__epoch{NNNNN}.png
```
If kwargs are relevant (e.g., `site=mlp_out`), they are included per the existing convention.

No changes to the export path logic are required.

### 5. Notification display

The notification component must be globally available (not page-scoped) since export can be triggered from any page. A single `dcc.Store` or component in the root layout carries the last export result. Notification auto-dismisses after a reasonable interval (e.g., 8 seconds).

---

## Constraints

- Export is **server-side only** — no client-side canvas capture. PNG quality is determined by the Kaleido renderer, not the browser zoom level.
- The export callback must be compatible with the existing `variant-selector-store` architecture — it reads from `variant_server_state.context`, not from callback state.
- Export should **not block** page re-renders. If Kaleido rendering is slow (~1–3s per figure), a loading indicator on the export button is acceptable. Async execution is out of scope for this requirement.
- Batch export runs exports **sequentially** server-side (not in parallel) to avoid Kaleido contention.
- The per-graph export button must not interfere with the existing Plotly toolbar (camera, zoom, pan). It should be placed outside the figure container.

---

## Architecture Notes

**Global graph registry:**
The most natural integration point is `AnalysisPageGraphManager.__init__`, which already receives both the `_VIEW_LIST` dict and the `page_prefix`. At construction time, it registers `{prefixed_graph_id → view_name}` entries in a module-level dict in `analysis_page.py`. Since page modules are imported at app startup, the registry is fully populated before any callback fires.

**Export callback:**
A single ALL-pattern callback on the export button component IDs (or a `dcc.Store` for batch selection) can handle export for any graph across any page without page-specific logic.

**Batch panel placement:**
The simplest implementation: a `dcc.Store` holds the list of selected graph IDs; a single server callback runs all exports and returns a formatted path list. The panel itself can be a simple `dbc.Collapse` section at the top of each page, populated from the registry.

---

## Notes

- The export feature emerged from the neuron group trajectory view for p101/s999/ds598, which produced a research-quality plot worth archiving with standard provenance (variant, view, epoch).
- The batch use case is: epoch cursor is set to a key moment (e.g., first-mover frequency emergence), and the researcher wants to snapshot the scatter, trajectory, purity, and polar histogram simultaneously.
- This requirement intentionally does not address export of non-view content (e.g., loss curves from metadata, intervention check plots). Those can be added as follow-on work once the core mechanism is in place.
- If SVG or HTML export formats are useful in the future, `BoundView.export()` already supports them — the dashboard trigger just needs a format selector.
