import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

from dashboard.components.analysis_page import AnalysisPageGraphManager

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
    "rg-timeseries-plot": {"view_name": "geometry_timeseries", "view_type": "epoch_selector"},
}

# Per-epoch snapshot views (site kwarg; epoch as data slice)
_SNAPSHOT_VIEW_LIST = {
    "rg-centroid-pca-plot": {"view_name": "centroid_pca", "view_type": "default_graph", "view_parameter": "site"},
    "rg-centroid-dist-plot": {"view_name": "centroid_distances", "view_type": "default_graph", "view_parameter": "site"},
    "rg-fisher-heatmap-plot": {"view_name": "fisher_heatmap", "view_type": "default_graph", "view_parameter": "site"},
}

_VIEW_LIST = {**_SUMMARY_VIEW_LIST, **_SNAPSHOT_VIEW_LIST}

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST)

def create_repr_geometry_page_nav() -> html.Div:
    return html.Div(
        children=[
            dbc.Label("Activation Site", className="fw-bold"),
            dcc.Dropdown(
                id="rg-site-dropdown",
                options=_SITE_OPTIONS,
                value="resid_post",
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
            dbc.Row(dbc.Col(_graph_manager.create_graph("rg-timeseries-plot", "1400px"))),
            # Fisher heatmap | Distance heatmap
            dbc.Row(
                [
                    dbc.Col(_graph_manager.create_graph("rg-fisher-heatmap-plot", "500px"), width=6),
                    dbc.Col(_graph_manager.create_graph("rg-centroid-dist-plot", "500px"), width=6),
                ]
            ),
            # Centroid PCA
            dbc.Row(dbc.Col(_graph_manager.create_graph("rg-centroid-pca-plot", "800px"), width=6)),
        ],
    )

def register_repr_geometry_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Repr Geometry page."""

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_rg_data_change(
        _modified_timestamp: str | None, variant_data: dict | None
    ):
        return _graph_manager.update_graphs(variant_data)

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list("site")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("rg-site-dropdown", "value"),
        State("variant-selector-store", "data"),
    )
    def on_rg_site_value_change(
        _modified_timestamp: str | None, site_value: str | None, variant_data: dict | None
    ):
        print("on_rg_site_value_change")
        parameters = {"site": site_value}
        return _graph_manager.update_graphs(variant_data, "site", parameters)
