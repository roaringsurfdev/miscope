import math

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, dcc, html

from dashboard_temp.components.epoch_series import (
    create_epoch_series_graph,
    register_epoch_series_callbacks,
)
from dashboard_temp.state import server_state

_SITE_OPTIONS = [
    {"label": "All Sites", "value": "all"},
    {"label": "Post-Embed", "value": "resid_pre"},
    {"label": "Attn Out", "value": "attn_out"},
    {"label": "MLP Out", "value": "mlp_out"},
    {"label": "Resid Post", "value": "resid_post"},
]

_EPOCH_SERIES_PLOT_IDS = [
    "rg-centroids-plot",
    "rg-alignment-plot",
]


def _build_epoch_series_figure(plot_id: str, store_data: dict) -> go.Figure:
    """Build a stub epoch-series figure for repr geometry plots.

    Demonstrates the load_figure_fn contract: returns a figure with the
    epoch marker as shapes[0] via fig.add_vline(), called before any
    other shapes are added.
    """
    variant_name = store_data.get("variant_name", "Unknown")
    epoch_idx = store_data.get("epoch") or 0
    epoch_value = server_state.get_epoch_at_index(epoch_idx)
    available_epochs = server_state.available_epochs or list(range(0, 1000, 10))

    if plot_id == "rg-centroids-plot":
        y_values = [math.sin(e / 100.0) for e in available_epochs]
        title = f"Class Centroids Trajectory — {variant_name}"
        y_label = "Centroid Spread (stub)"
    else:
        y_values = [math.cos(e / 100.0) for e in available_epochs]
        title = f"PCA Alignment — {variant_name}"
        y_label = "Alignment Score (stub)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=available_epochs,
        y=y_values,
        mode="lines",
        name=y_label,
    ))

    # Epoch marker as shapes[0] — required by register_epoch_series_callbacks.
    # Must be the first shape added to the figure.
    fig.add_vline(
        x=epoch_value,
        line_color="crimson",
        line_dash="dash",
        line_width=2,
        annotation_text=f"ep {epoch_value}",
        annotation_position="top right",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title=y_label,
        template="plotly_white",
    )
    return fig


def create_repr_geometry_page_nav() -> html.Div:
    return html.Div(
        children=[
            dbc.Label("Activation Site", className="fw-bold"),
            dcc.Dropdown(
                id="rg-site-dropdown",
                options=_SITE_OPTIONS,
                value="all",
                clearable=False,
            )
        ]
    )


def create_repr_geometry_page_layout() -> html.Div:
    return html.Div(
        id="repr_geometry_content",
        children=[
            html.H4("Repr Geometry", className="mb-3"),
            dbc.Row(dbc.Col(create_epoch_series_graph("rg-centroids-plot", "350px"))),
            dbc.Row(dbc.Col(create_epoch_series_graph("rg-alignment-plot", "350px"))),
        ]
    )


def register_repr_geometry_page_callbacks(app: Dash) -> None:
    register_epoch_series_callbacks(app, _EPOCH_SERIES_PLOT_IDS, _build_epoch_series_figure)
