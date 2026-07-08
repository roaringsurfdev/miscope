# v0.5.0 — Dash Dashboard Migration

## Summary

Migrated the dashboard from Gradio to Dash (Plotly's own framework) to resolve fundamental interaction limitations. The Gradio dashboard is frozen but remains functional.

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| New `dashboard_v2/` alongside existing `dashboard/` | Risk mitigation — Gradio stays available as fallback |
| Spike-first approach (Phase 1: 5-6 plots, Phase 2: all 18) | Validates key patterns before full commitment |
| Sidebar layout for all controls | Persistent controls without stealing vertical space |
| `Patch()` for epoch marker updates | Avoids re-serializing full figures (e.g., 35k-point loss curves) |
| Training tab stays in Gradio | Used infrequently, no interaction problems |

## What Changed

- `dashboard_v2/app.py` — Dash application factory
- `dashboard_v2/layout.py` — Sidebar + main content layout with `dbc`
- `dashboard_v2/callbacks.py` — Per-visualization callbacks, click-to-navigate, selective rendering
- `dashboard_v2/state.py` — DashboardState for variant/epoch/artifact management

## What Didn't Change

- All renderers (`visualization/renderers/`) — return `go.Figure`, framework-agnostic
- `ArtifactLoader`, analysis library, training pipeline — untouched
- `dashboard/` — frozen, not deleted

## Key File Locations

- Dashboard v2: `dashboard_v2/`
- Original dashboard: `dashboard/` (frozen)
- Renderers: `visualization/renderers/` (unchanged)
