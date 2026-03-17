import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

from dashboard.components.analysis_page import _SITE_OPTIONS, AnalysisPageGraphManager

# ---------------------------------------------------------------------------
# Plot IDs (prefixed "" to avoid collisions)
# ---------------------------------------------------------------------------

# Summary view (epoch as cursor)
_SUMMARY_VIEW_LIST = {
    "timeseries-plot": {"view_name": "geometry.timeseries", "view_type": "epoch_selector"},
}

# Per-epoch snapshot views (site kwarg; epoch as data slice)
_SNAPSHOT_VIEW_LIST = {
    "centroid-pca-plot": {
        "view_name": "geometry.centroid_pca",
        "view_type": "default_graph",
        "view_filter_set": "site",
    },
    "centroid-dist-plot": {
        "view_name": "geometry.centroid_distances",
        "view_type": "default_graph",
        "view_filter_set": "site",
    },
    "fisher-heatmap-plot": {
        "view_name": "geometry.fisher_heatmap",
        "view_type": "default_graph",
        "view_filter_set": "site",
    },
}

_VIEW_LIST = {**_SUMMARY_VIEW_LIST, **_SNAPSHOT_VIEW_LIST}

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST, "rg")


def create_repr_geometry_page_nav(app: Dash) -> html.Div:
    return html.Div(
        children=[
            dbc.Label("Activation Site", className="fw-bold"),
            dcc.Dropdown(
                id="site-dropdown",
                options=_SITE_OPTIONS,
                value="resid_post",
                clearable=False,
            ),
        ]
    )


def create_repr_geometry_page_layout(app: Dash) -> html.Div:
    return html.Div(
        id="repr_geometry_content",
        children=[
            html.H4("Repr Geometry", className="mb-3"),
            # Time-series (full width, tall)
            dbc.Row(dbc.Col(_graph_manager.create_graph("timeseries-plot", "1400px"))),
            # Fisher heatmap | Distance heatmap
            dbc.Row(
                [
                    dbc.Col(_graph_manager.create_graph("fisher-heatmap-plot", "500px"), width=6),
                    dbc.Col(_graph_manager.create_graph("centroid-dist-plot", "500px"), width=6),
                ]
            ),
            # Centroid PCA
            dbc.Row(dbc.Col(_graph_manager.create_graph("centroid-pca-plot", "800px"))),
        ],
    )


def register_repr_geometry_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Repr Geometry page."""

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_rg_data_change(_modified_timestamp: str | None, variant_data: dict | None):
        return _graph_manager.update_graphs(variant_data=variant_data)

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list("site")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("site-dropdown", "value"),
        State("variant-selector-store", "data"),
    )
    def on_rg_site_value_change(
        _modified_timestamp: str | None, site_value: str | None, variant_data: dict | None
    ):
        app.server.logger.debug("on_rg_site_value_change")
        view_kwargs = {"site": site_value}
        return _graph_manager.update_graphs(
            variant_data=variant_data, view_filter_set="site", view_kwargs=view_kwargs
        )
