# REQ_046: Dashboard Navigation UX

**Status:** Complete
**Priority:** High (quality-of-life prerequisite for write-up phase)
**Dependencies:** REQ_040 (navbar + URL routing), REQ_041 (Summary page), REQ_042 (Neuron Dynamics page), REQ_044 (Repr Geometry page)
**Last Updated:** 2026-02-18

## Problem Statement

During the research write-up phase, navigating between dashboard pages requires repeatedly re-selecting family, variant, and epoch on every page. Each analysis page (Summary, Neuron Dynamics, Repr Geometry) has an independent top-bar control row with no shared state, so switching pages resets all selections. Additionally, two inconsistencies in the dashboard make navigation harder than it needs to be: not all pages support click-to-navigate on time-series plots, and the control layout differs between pages (left sidebar on Visualization, top bar on analysis pages).

These three issues compound: a researcher jumping between pages to compare visualizations must manually re-select selections on every navigation, and the inconsistent layouts add cognitive friction on top.

## Design

### Sticky Cross-Page Selections

Use `dcc.Store(id="selection-store", storage_type="session")` in the outer shell (`create_layout()`), which persists across page navigations within a browser session. The routing callback reads the store and passes `initial` state (family, variant, epoch) to each page layout function. Pages use the initial values to pre-populate dropdowns, triggering the normal cascade.

Two Patch-based store sync callbacks per page write updates back on variant selection and epoch slider changes, using `allow_duplicate=True` since multiple pages write to the same store.

The `on_*_family_change` callbacks are updated to read the store and preserve the stored variant value when the incoming family matches the stored family. Without this, the family-change callback fires on page load (triggered by restoring `initial_family`) and resets the variant to `None` before the cascade can restore it.

### Consistent Left Sidebar Layout

Add `create_page_sidebar(prefix, initial_family, initial_variant, initial_epoch_idx, extra_controls)` factory in `layout.py`. Uses shared toggle component IDs (`"sidebar"`, `"sidebar-toggle"`, `"sidebar-collapsed"`, `"sidebar-expand"`, `"collapsed-status"`) so the existing `toggle_sidebar` callback works on all pages without modification. Data components use caller-provided prefix to avoid cross-page collision.

Each analysis page wraps its content in the same flex structure as the main Visualization page: `[sidebar, collapsed_sidebar, content_div]` with `display: flex`.

### Click-to-Navigate

Add `"velocity-plot"` to `_CLICK_NAV_PLOT_IDS` on the main Visualization page (was missing). Add click navigation callbacks to Summary (8 time-series plots) and Repr Geometry (timeseries only — spatial heatmaps have no epoch x-axis). Neuron Dynamics plots are cross-epoch heatmaps with no epoch x-axis — click navigation not applicable.

## Conditions of Satisfaction

### Sticky Selections
1. Selecting family + variant + epoch on any page persists when navigating to another page
2. On arrival at a new page, the stored family/variant is pre-selected and plots render automatically
3. Stored epoch is restored as the nearest available epoch index (accounting for pages with different epoch lists)
4. Family-change callbacks preserve stored variant when family matches (prevents wipe on page load)

### Consistent Sidebar Layout
5. Summary, Neuron Dynamics, and Repr Geometry pages have a left collapsible sidebar matching the Visualization page structure
6. Sidebar contains: Family dropdown, Variant dropdown, Epoch slider, page-specific extra controls, status display
7. Sidebar collapse/expand works identically on all pages (reuses existing toggle callback)
8. Epoch slider appears on Neuron Dynamics page even though ND plots are cross-epoch (consistency + store sync)

### Click Navigation
9. Clicking any time-series plot on the main Visualization page (including velocity-plot) navigates the epoch slider
10. Clicking any of the 8 time-series plots on the Summary page navigates the epoch slider
11. Clicking the timeseries plot on the Repr Geometry page navigates the epoch slider
12. Click navigation on RG page uses the repr_geometry-specific epoch list (not global checkpoint epochs)

### Tests
13. Existing test suite passes without regressions (618 tests)
14. `test_main_layout_has_url_and_page_content` updated to account for added `dcc.Store` child

## Constraints

### Must Have
- `dcc.Store` lives in the outer shell (`create_layout()`), never torn down by navigation
- Shared sidebar toggle IDs — only safe because a single page is mounted at a time
- `prevent_initial_call=True` on all store sync callbacks to avoid firing on page mount
- RG page epoch sync uses `loader.get_epochs("repr_geometry")`, not global `server_state` epoch list

### Must Avoid
- Epoch slider on ND page triggering plot re-renders (ND plots are cross-epoch, not per-epoch)
- Breaking the existing `toggle_sidebar` callback by introducing conflicting component IDs
- Storing artifact data in the session store (only family, variant, epoch identifiers)

### Flexible
- Which plots on Summary are included in click navigation (chosen: all 8 time-series)
- Initial epoch index on page arrival (always 0, then restored after variant loads)

## Decision Authority
- **Claude decides:** prefixes, layout details, which summary plots get click nav, extra_controls arrangement
- **Discuss first:** any changes to routing architecture, storage_type choice

## Files Modified

| File | Changes |
|---|---|
| `dashboard_v2/layout.py` | Added `dcc.Store`, style constants, `create_page_sidebar()`, `create_collapsed_page_sidebar()` |
| `dashboard_v2/navigation.py` | Routing callback reads store, passes `initial` to layout functions |
| `dashboard_v2/callbacks.py` | Added velocity-plot to click nav, store sync callbacks, epoch restoration, family-change fix |
| `dashboard_v2/pages/summary.py` | Sidebar layout, store sync, click navigation (8 plots) |
| `dashboard_v2/pages/neuron_dynamics.py` | Sidebar layout, store sync (no click nav) |
| `dashboard_v2/pages/repr_geometry.py` | Sidebar layout, store sync, click navigation (timeseries only) |
| `tests/test_req_040_dash_job_management.py` | Updated child count assertion (3 → 4) |

## Notes
- `storage_type="session"` persists across navigations but resets on browser close — appropriate for a single-researcher tool
- The family-change callback fix (CoS item 4) was discovered post-implementation during manual testing — the root cause was that Dash fires `on_*_family_change` when `initial_family` is set on page load, which reset variant before the cascade could restore it
- Training and Analysis Run pages are unchanged — their layouts are already appropriate for their use cases
