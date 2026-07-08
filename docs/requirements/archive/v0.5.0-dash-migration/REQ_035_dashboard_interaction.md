# REQ_035: Dashboard Interaction Improvements + Dash Migration

**Status:** Draft
**Priority:** High (performance and usability are limiting analysis productivity)
**Dependencies:** None
**Last Updated:** 2026-02-10

## Problem Statement

The dashboard was designed for 4-5 visualizations and now has 18. Three friction points have emerged from actual usage:

### 1. Epoch discovery is indirect

When the user sees an interesting point on a summary visualization (e.g., a spike in the flatness trajectory), they must:
1. Mouse over the summary plot to read the epoch
2. Scroll to the top of the page
3. Mouse over the train/test loss curve to find the corresponding checkpoint index
4. Type the index into the epoch slider
5. Scroll back down to see the per-epoch detail

The summary plot already knows the epoch of interest. Click-to-navigate would eliminate steps 2-4.

### 2. All 18 plots re-render on epoch change

Every epoch slider change triggers `generate_all_plots()`, which re-renders all 18 visualizations even though the user is typically focused on 1-3. This causes noticeable delay. Some callbacks already demonstrate selective rendering (neuron slider only updates heatmap, trajectory group radio only updates 4 trajectory plots), so the pattern exists — it just needs to be extended to the epoch slider.

### 3. Top controls consume vertical space

The variant selector, epoch slider, and neuron index are positioned at the top of the Analysis tab. This pushes all visualizations below the fold, requiring constant scrolling between controls and content.

### Framework assessment

These friction points expose fundamental limitations in Gradio:
- **No sidebar layout**: Gradio's linear layout model (rows/columns, top-to-bottom flow) cannot pin controls to the side while content scrolls
- **No figure patching**: Every plot update serializes the entire figure to the browser; no way to update just a marker
- **Limited click interaction**: `.select()` returns basic coordinates; no trace/point identity for distinguishing click intent
- **No client-side callbacks**: Every interaction round-trips to the Python server

Dash (Plotly's own dashboard framework) provides native solutions for all four: sidebar layouts, `Patch()` for partial figure updates, `clickData` with full trace/point identity, and client-side callbacks for zero-latency marker updates.

## Design

### Framework Migration Strategy

**Approach:** Build a new Dash-based dashboard (`dashboard_v2/`) alongside the existing Gradio dashboard (`dashboard/`). The Gradio app is frozen — no new features — but remains fully functional for ongoing analysis. The Dash app is built incrementally, starting with a spike that proves the key interaction patterns before committing to full migration.

**Why bundle migration with interaction improvements:** REQ_035 requires substantial restructuring of `dashboard/app.py` (callback architecture, layout, state management). Doing this work in Gradio only to redo it in Dash later would be double-churn. By building the interaction improvements directly in Dash, we get the improvements AND land on a framework that supports the Phase 2 roadmap (summary/detail modes, cross-variant comparison).

**Risk mitigation:** The existing Gradio dashboard stays available throughout. If Dash proves problematic, we can fall back to implementing the improvements in Gradio. The analysis layer (renderers, ArtifactLoader, analyzers) is untouched — renderers return `go.Figure` objects that work identically in both frameworks.

**What migrates:**
- `dashboard/app.py` (~1,370 lines of layout + callbacks) → `dashboard_v2/`
- `dashboard/state.py` (~120 lines) → `dashboard_v2/state.py`

**What doesn't change:**
- All renderers (`visualization/renderers/`) — they return `go.Figure`, framework-agnostic
- `ArtifactLoader`, analysis library, training pipeline — completely untouched
- `dashboard/components/` — rendering functions that return figures, reusable as-is

### Implementation Phases

#### Phase 1: Spike (prove the patterns)

Build a minimal Dash app with 5-6 visualizations — enough to validate the four key patterns:

1. **Sidebar layout** — Variant selector, epoch slider, neuron index in a collapsible left panel (`dbc.Offcanvas` or CSS-based sidebar). Validates that controls persist during scroll.
2. **A representative set of plots** — Loss curves, specialization trajectory, parameter trajectory (2D), flatness trajectory, neuron heatmap, freq clusters. Mix of summary and per-epoch, enough to test the interaction model.
3. **Click-to-navigate** — Click a data point on any summary plot → epoch slider updates → per-epoch plots refresh. Uses Dash's `clickData` callback property. Validates the interaction model.
4. **Selective rendering** — Epoch change triggers only the relevant callbacks, not all plots. Dash's callback architecture naturally supports this (one callback per output component).

**Spike exit criteria:** User can load a variant, browse epochs via sidebar slider or click-to-navigate, and see responsive updates. If the experience is better than Gradio for these 5-6 plots, proceed to Phase 2. If not, document what went wrong and reassess.

#### Phase 2: Full visualization migration

Port all 18 visualizations to the Dash app, organized into rendering groups:

| Group | Plots | Epoch change behavior |
|-------|-------|----------------------|
| Loss | Loss curves (1) | Marker update only |
| Frequency Analysis | Dominant frequencies, freq clusters (2) | Full re-render |
| Neuron Activation | Neuron heatmap (1) | Full re-render |
| Neuron Specialization | Specialization trajectory, spec by freq (2) | Marker update only |
| Attention | Attention heads, attention freq heatmap (2) | Full re-render |
| Attention Specialization | Attn spec trajectory, attn dominant freq (2) | Marker update only |
| Trajectory | 4 trajectory plots + velocity (5) | Marker update only |
| Dimensionality | Dimensionality trajectory, SV spectrum (2) | Marker update (trajectory), full re-render (spectrum) |
| Flatness | Flatness trajectory, perturbation distribution (2) | Marker update (trajectory), full re-render (distribution) |

**Key optimization:** Summary/trajectory plots cache their base figure (all traces except epoch marker) and update only the marker on epoch change. Per-epoch plots re-render from artifacts. This reduces epoch-change work from 18 full renders to ~8 full renders + ~10 marker-only patches.

#### Phase 3: Training tab migration (deferred)

The Training tab (model training UI) stays in Gradio for now. It's used infrequently and has no interaction problems. Migrate when/if needed.

### A. Click-to-Navigate

**Goal:** Clicking a data point on any summary/trajectory plot navigates the epoch slider to that epoch.

**Mechanism:** Dash's `dcc.Graph` component exposes `clickData` as a callback input. The callback extracts the clicked point's x-coordinate (epoch), finds the nearest checkpoint index, and updates the epoch slider.

**Which plots support click-to-navigate:**

| Plot | Click action |
|------|-------------|
| Loss curves | Set epoch slider to clicked epoch |
| Specialization trajectory | Set epoch slider to clicked epoch |
| Specialization by frequency | Set epoch slider to clicked epoch |
| Attention specialization trajectory | Set epoch slider to clicked epoch |
| Attention dominant frequencies | Set epoch slider to clicked epoch |
| Parameter trajectory (2D) | Set epoch slider to clicked point's epoch |
| Dimensionality trajectory | Set epoch slider to clicked epoch |
| Flatness trajectory | Set epoch slider to clicked epoch |

**Neuron click-to-navigate (stretch goal):** Clicking a neuron block on the frequency clusters plot sets the neuron index. Dash's `clickData` includes the trace name and point index, making this feasible. Include if the clusters heatmap's click data provides clean neuron identification.

**Visual feedback:** Plots with click-to-navigate use `hovermode='closest'` and the cursor indicates clickability. The current epoch indicator (already present on summary plots) provides immediate feedback.

### B. Selective Re-rendering

**Goal:** Epoch slider changes only re-render affected plots.

**Mechanism:** Dash's callback architecture naturally supports this — each `@callback` targets specific outputs. Unlike Gradio's monolithic `generate_all_plots()`, Dash callbacks fire independently.

**Epoch marker updates** use Dash's `Patch()` to modify only the epoch indicator trace/shape without re-sending the entire figure. This is a client-side optimization that Gradio cannot do.

### C. Sidebar Layout

**Goal:** Global controls in a persistent left sidebar instead of top-of-page.

**Layout:**

```
┌──────────┬──────────────────────────────────────────┐
│ Controls │  Loss Curves                             │
│          │  ┌──────────────────────────────────────┐ │
│ Family   │  │                                      │ │
│ [......] │  └──────────────────────────────────────┘ │
│          │                                           │
│ Variant  │  Dominant Frequencies                     │
│ [......] │  ┌──────────────────────────────────────┐ │
│          │  │                                      │ │
│ Epoch    │  └──────────────────────────────────────┘ │
│ [==☐===] │                                           │
│ #26400   │  Neuron Activation Heatmap                │
│          │  ┌──────────────────────────────────────┐ │
│ Neuron   │  │                                      │ │
│ [==☐===] │  └──────────────────────────────────────┘ │
│ N=42     │  ...                                      │
│          │                                           │
│ [Analyze]│                                           │
│          │                                           │
│ ──────── │                                           │
│ Traj grp │                                           │
│ (o) All  │                                           │
│ ( ) Emb  │                                           │
│ ( ) Attn │                                           │
│ ( ) MLP  │                                           │
│          │                                           │
│ SV Matrix│                                           │
│ [......] │                                           │
│          │                                           │
│ Flatness │                                           │
│ [......] │                                           │
└──────────┴──────────────────────────────────────────┘
```

The sidebar is collapsible (toggle button or `dbc.Offcanvas`) to maximize plot area when controls aren't needed. When collapsed, a compact status bar shows current epoch and neuron index.

**All visualization-specific controls** (trajectory group radio, SV matrix selector, flatness metric dropdown, attention position pair) move to the sidebar. This puts every control in one place.

## Scope

**This requirement covers:**
1. Dash-based dashboard in `dashboard_v2/` with sidebar layout
2. Click-to-navigate on summary/trajectory plots (epoch navigation)
3. Selective re-rendering with figure patching for epoch marker updates
4. Migration of all 18 Analysis tab visualizations
5. Collapsible sidebar with status display
6. Dependencies: `dash`, `dash-bootstrap-components`

**This requirement does not cover:**
- Training tab migration (stays in Gradio)
- Neuron click-to-navigate from clusters (stretch goal, include if feasible)
- Summary vs detail mode switching (Phase 2 roadmap)
- Cross-variant comparison UI (separate future requirement)
- Removal of the Gradio dashboard (it stays available)
- Any changes to renderers or analysis layer

## Conditions of Satisfaction

### Spike (Phase 1)
- [ ] Dash app launches and displays 5-6 visualizations with sidebar controls
- [ ] Click-to-navigate works on at least 3 summary plots
- [ ] Epoch change only re-renders affected plots (not all)
- [ ] Sidebar is collapsible with status display
- [ ] Qualitative assessment: interaction feels faster than Gradio dashboard

### Full Migration (Phase 2)
- [ ] All 18 Analysis tab visualizations render correctly in Dash
- [ ] Click-to-navigate on at least 5 of the 8 summary/trajectory plots
- [ ] Epoch marker updates use `Patch()` — no full re-render for summary plots
- [ ] All visualization-specific controls (trajectory group, SV matrix, flatness metric, position pair) work in sidebar
- [ ] No regression in visualization content or behavior vs Gradio dashboard
- [ ] New tests for click-to-navigate callbacks and selective rendering

### General
- [ ] Gradio dashboard remains functional (frozen, not broken)
- [ ] All existing renderer tests pass (renderers unchanged)
- [ ] `dash` and `dash-bootstrap-components` added to dependencies

## Constraints

**Must have:**
- Sidebar layout for controls
- Click-to-navigate on loss curves and trajectory plots (minimum)
- Selective rendering — epoch change must not re-render all 18 plots
- Gradio dashboard stays available during and after migration

**Must avoid:**
- Modifying any renderers or analysis code (migration is presentation-layer only)
- Breaking the Gradio dashboard
- Over-building Phase 1 — spike should be minimal to validate patterns
- Removing the Gradio dashboard before the Dash app has full feature parity

**Flexible:**
- Exact sidebar component (`dbc.Offcanvas` vs CSS sidebar vs `dbc.Col` with toggle)
- Exact set of plots in the spike (5-6 representative ones, specific choices flexible)
- Whether visualization-specific controls go in sidebar vs inline with their plots
- Visual styling (can iterate after functionality is proven)
- Whether neuron click-to-navigate is included

## Decision Log

| Date | Question | Decision | Rationale |
|------|----------|----------|-----------|
| 2026-02-10 | Stay with Gradio or migrate? | Migrate to Dash | Gradio lacks sidebar layouts, figure patching, rich click data, and client-side callbacks — all needed for the interaction improvements. Migration cost is bounded (only app.py and state.py change) |
| 2026-02-10 | Migrate all at once or incrementally? | Spike first, then full migration | Proves patterns before committing. Resolves unknowns early. User keeps working Gradio dashboard throughout |
| 2026-02-10 | Category tabs for visualization groups? | No | User explicitly flagged that requiring tab clicks to scan summaries would feel like regression. Scroll-based layout preserved |
| 2026-02-10 | Sidebar vs top controls? | Sidebar | Persistent sidebar keeps controls accessible without stealing vertical space. Dash supports this natively |
| 2026-02-10 | Migrate Training tab? | Defer | Training tab works fine in Gradio, used infrequently, no interaction problems |

## Notes

**Migration safety:** The key architectural decision that makes this migration low-risk is the separation between renderers and dashboard. Every renderer is a pure function that takes data and returns a `go.Figure`. The dashboard is just the presentation layer that calls these functions and displays the results. This separation was an intentional design choice from the beginning, and it's paying off now.

**Dash `clickData` format:** When a user clicks a Plotly scatter/line trace in a `dcc.Graph`, Dash provides:
```python
{
    "points": [{
        "curveNumber": 0,       # which trace
        "pointNumber": 42,      # which point in the trace
        "x": 26400,             # x coordinate (epoch)
        "y": 0.023,             # y coordinate (metric value)
        "customdata": [...],    # optional extra data attached to points
    }]
}
```
This is rich enough to determine both the epoch (from `x`) and, for trajectory plots, which component group was clicked (from `curveNumber` or `customdata`). Renderers can attach `customdata` to traces to pass additional context for click handling.

**Dash `Patch()` for markers:** To update an epoch marker without re-rendering the whole figure:
```python
from dash import Patch
patched_fig = Patch()
patched_fig["layout"]["shapes"][0]["x0"] = new_epoch
patched_fig["layout"]["shapes"][0]["x1"] = new_epoch
return patched_fig
```
This sends only the changed properties to the browser, not the entire figure. For summary plots with large datasets (e.g., 35k-point loss curves), this is a major performance improvement.

**Phase 2+ roadmap:** The Dash migration de-risks several future improvements:
- **Summary vs detail modes**: Dash's multi-page app pattern or dynamic layout switching could support mode toggling
- **Cross-variant comparison**: Dash can render two sets of plots side by side with independent or linked epoch sliders
- **Turn epoch annotations** (REQ_034): Once turn detection is implemented, the annotation can be added to summary plots with a single `Patch()` call
- **Paired summary-detail groups**: Dash layouts can group related plots (clusters + heatmap) with their own sub-controls
