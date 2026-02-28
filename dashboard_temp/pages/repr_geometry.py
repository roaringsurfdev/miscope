import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from dashboard_temp.components.visualization import create_empty_figure, create_graph
from dashboard_temp.state import variant_state

# ---------------------------------------------------------------------------
# Plot IDs (prefixed "rg-" to avoid collisions)
# ---------------------------------------------------------------------------

_SITE_OPTIONS = [
    {"label": "All Sites", "value": "all"},
    {"label": "Post-Embed", "value": "resid_pre"},
    {"label": "Attn Out", "value": "attn_out"},
    {"label": "MLP Out", "value": "mlp_out"},
    {"label": "Resid Post", "value": "resid_post"},
]

# Summary view (site kwarg; epoch as cursor)
_SUMMARY_VIEW_LIST = {
    "rg-timeseries-plot": {"view_name": "geometry_timeseries", "view_type": "epoch_selector", "view_parameter": "site"},
}

# Per-epoch snapshot views (site kwarg; epoch as data slice)
_SNAPSHOT_VIEW_LIST = {
    "rg-centroid-pca-plot": {"view_name": "centroid_pca", "view_type": "default_graph", "view_parameter": "site"},
    "rg-centroid-dist-plot": {"view_name": "centroid_distances", "view_type": "default_graph", "view_parameter": "site"},
    "rg-fisher-heatmap-plot": {"view_name": "fisher_heatmap", "view_type": "default_graph", "view_parameter": "site"},
}

_VIEW_LIST = {**_SUMMARY_VIEW_LIST, **_SNAPSHOT_VIEW_LIST}


def _get_graph_output_list() -> list[dict]:
    return [
        {"view_type": meta["view_type"], "index": pid}
        for pid, meta in _VIEW_LIST.items()
    ]


def _get_graph_view_type(graph_key: str) -> str:
    return _VIEW_LIST[graph_key].get("view_type", "default_graph")


def _update_graphs(variant_data: dict | None, site_value: str | None) -> list[go.Figure]:
    stored = variant_data or {}
    variant_name = stored.get("variant_name")
    last_field_updated = stored.get("last_field_updated")

    if variant_name is None:
        empty = create_empty_figure("Select a variant")
        return [empty for _ in _VIEW_LIST]

    if last_field_updated not in ["variant_name", "epoch"]:
        raise PreventUpdate

    site = None if site_value == "all" else site_value
    snapshot_site = "resid_post" if site_value == "all" else site_value

    figures = []
    for pid, meta in _VIEW_LIST.items():
        view_name = meta.get("view_name", "")
        if view_name not in variant_state.available_views:
            figures.append(create_empty_figure("No view found"))
            continue
        if pid in _SUMMARY_VIEW_LIST:
            figures.append(variant_state.context.view(view_name).figure(site=site))
        else:
            figures.append(variant_state.context.view(view_name).figure(site=snapshot_site))

    return figures


def create_repr_geometry_page_nav() -> html.Div:
    return html.Div(
        children=[
            dbc.Label("Activation Site", className="fw-bold"),
            dcc.Dropdown(
                id="rg-site-dropdown",
                options=_SITE_OPTIONS,
                value="all",
                clearable=False,
            ),
        ]
    )


def create_repr_geometry_page_layout() -> html.Div:
    return html.Div(
        id="repr_geometry_content",
        children=[
            html.H4("Repr Geometry", className="mb-3"),
            # Time-series (full width, tall)
            dbc.Row(dbc.Col(create_graph("rg-timeseries-plot", "1400px", _get_graph_view_type("rg-timeseries-plot")))),
            # Fisher heatmap | Distance heatmap
            dbc.Row(
                [
                    dbc.Col(create_graph("rg-fisher-heatmap-plot", "500px", _get_graph_view_type("rg-fisher-heatmap-plot")), width=6),
                    dbc.Col(create_graph("rg-centroid-dist-plot", "500px", _get_graph_view_type("rg-centroid-dist-plot")), width=6),
                ]
            ),
            # Centroid PCA
            dbc.Row(dbc.Col(create_graph("rg-centroid-pca-plot", "800px", _get_graph_view_type("rg-centroid-pca-plot")), width=6)),
        ],
    )


def register_repr_geometry_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Repr Geometry page."""

    @app.callback(
        [Output(pid, "figure") for pid in _get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        Input("rg-site-dropdown", "value"),
        State("variant-selector-store", "data"),
    )
    def on_rg_data_change(
        _modified_timestamp: str | None, site_value: str | None, variant_data: dict | None
    ):
        return _update_graphs(variant_data, site_value)
