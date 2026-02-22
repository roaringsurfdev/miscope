"""Epoch-series component for cross-epoch training visualizations.

Encapsulates two concerns:
1. Layout: create_epoch_series_graph() — thin wrapper around create_graph()
2. Callbacks: register_epoch_series_callbacks() — the shared callback pattern for:
     - Full figure reload on variant/family change
     - Epoch marker Patch() on epoch change (no full re-render)
     - Click-to-select-epoch: clicking any registered plot updates the epoch slider

Epoch-as-selector is the default behavior. All epoch-series plots are assumed
to be interactive epoch selectors unless the page explicitly opts out by
calling register_epoch_series_callbacks with include_selector=False.
"""

from __future__ import annotations

from typing import Callable

import dash
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, set_props
from dash.exceptions import PreventUpdate

from dashboard_temp.components.visualization import create_empty_figure, create_graph
from dashboard_temp.state import server_state

# Convention: the epoch marker must be shapes[0] in any figure returned by
# load_figure_fn. Use fig.add_vline(x=epoch_value, ...) as the first shape
# added after constructing the figure.
_EPOCH_MARKER_IDX = 0


def create_epoch_series_graph(graph_id: str, height: str = "400px") -> dcc.Graph:
    """Create a dcc.Graph for epoch-series use.

    Functionally identical to create_graph(). The distinct name signals
    the epoch-series contract to the reader: the load_figure_fn supplied
    to register_epoch_series_callbacks must place the epoch marker as
    shapes[0] in the returned figure (e.g. via fig.add_vline).
    """
    return create_graph(graph_id, height)


def register_epoch_series_callbacks(
    app: Dash,
    plot_ids: list[str],
    load_figure_fn: Callable[[str, dict], go.Figure],
    include_selector: bool = True,
) -> None:
    """Register callbacks for a set of epoch-series plots on a page.

    Handles full reload vs. marker-only Patch() based on last_field_updated,
    and by default wires click-to-select-epoch for all registered plots.

    Contract for load_figure_fn:
        Signature: (plot_id: str, store_data: dict) -> go.Figure
        The returned figure must include the epoch marker as shapes[0],
        added via fig.add_vline(x=epoch_value, ...) before any other shapes.

    Args:
        app: The Dash application instance.
        plot_ids: Graph component IDs for this page's epoch-series plots.
        load_figure_fn: Page-supplied figure builder — owns data loading.
        include_selector: Register click-to-select-epoch callbacks (default True).
    """

    @app.callback(
        *[Output(pid, "figure") for pid in plot_ids],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
        prevent_initial_call=True,
    )
    def on_store_change(timestamp, store_data):
        print("on_store_change (register_epoch_series_callbacks)")
        stored = store_data or {}
        variant_name = stored.get("variant_name")
        last_updated = stored.get("last_field_updated")
        epoch_idx = stored.get("epoch") or 0

        if variant_name is None:
            empty = create_empty_figure("Select a variant")
            return [empty for _ in plot_ids]

        if last_updated == "epoch":
            # Marker-only: move shapes[0] to the new epoch position.
            epoch_value = server_state.get_epoch_at_index(epoch_idx)
            patches = []
            for _ in plot_ids:
                p = dash.Patch()
                p["layout"]["shapes"][_EPOCH_MARKER_IDX]["x0"] = epoch_value
                p["layout"]["shapes"][_EPOCH_MARKER_IDX]["x1"] = epoch_value
                patches.append(p)
            return patches

        # Full reload: variant or family changed.
        return [load_figure_fn(pid, stored) for pid in plot_ids]

    if not include_selector:
        return

    @app.callback(
        *[Input(pid, "clickData") for pid in plot_ids],
        State("variant-selector-store", "data"),
        prevent_initial_call=True,
    )
    def on_epoch_click(*args):
        print("on_epoch_click (register_epoch_series_callbacks)")
        store_data = args[-1]
        click_data_list = args[:-1]
        triggered_id = dash.ctx.triggered_id

        if triggered_id not in plot_ids:
            raise PreventUpdate

        triggered_idx = plot_ids.index(triggered_id)
        click_data = click_data_list[triggered_idx]

        if not click_data or not click_data.get("points"):
            raise PreventUpdate

        clicked_x = click_data["points"][0].get("x")
        if clicked_x is None:
            raise PreventUpdate

        new_epoch_idx = server_state.nearest_epoch_index(int(clicked_x))
        stored = store_data or {}

        # Update both the slider (visual) and the store (triggers marker updates).
        set_props("variant-selector-epoch-slider", {"value": new_epoch_idx})
        set_props("variant-selector-store", {"data": {
            **stored,
            "epoch": new_epoch_idx,
            "last_field_updated": "epoch",
        }})
