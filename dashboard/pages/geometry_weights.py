import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

from dashboard.components.analysis_page import AnalysisPageGraphManager

# ---------------------------------------------------------------------------
# Plot IDs (prefixed "" to avoid collisions)
# ---------------------------------------------------------------------------

# Summary view (epoch as cursor)
_SUMMARY_VIEW_LIST = {
    "training-loss-curves": {
        "view_name": "training.metadata.loss_curves",
        "view_type": "epoch_selector",
    },
    "timeseries-plot": {
        "view_name": "weight_geometry.timeseries",
        "view_type": "epoch_selector",
        "view_filter_set": "matrix",
    },
}

# Per-epoch snapshot views (site kwarg; epoch as data slice)
_SNAPSHOT_VIEW_LIST = {
    "centroid-pca-plot": {
        "view_name": "weight_geometry.centroid_pca",
        "view_type": "default_graph",
        "view_filter_set": "matrix",
    },
    "centroid-dist-plot": {
        "view_name": "weight_geometry.group_snapshot",
        "view_type": "default_graph",
        "view_filter_set": "matrix",
    },
}

_VIEW_LIST = {**_SUMMARY_VIEW_LIST, **_SNAPSHOT_VIEW_LIST}
_MATRIX_OPTIONS = [
    {"label": "W_out", "value": "Wout"},
    {"label": "W_in", "value": "Win"},
]

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST, "gw")


def create_weight_geometry_page_nav(app: Dash) -> html.Div:
    return html.Div(
        children=[
            dbc.Label("Matrix", className="fw-bold"),
            dcc.Dropdown(
                id="matrix-dropdown",
                options=_MATRIX_OPTIONS,
                value="Wout",
                clearable=False,
            ),
        ]
    )


def create_weight_geometry_page_layout(app: Dash) -> html.Div:
    return html.Div(
        id="weight_geometry_content",
        children=[
            html.H4("Geometry - Weights", className="mb-3"),
            # Time-series (full width, tall)
            dbc.Row(dbc.Col(_graph_manager.create_graph("training-loss-curves", "350px"))),
            dbc.Row(dbc.Col(_graph_manager.create_graph("timeseries-plot", "1000px"))),
            dbc.Row(dbc.Col(_graph_manager.create_graph("centroid-dist-plot", "400px"))),
            # Centroid PCA
            dbc.Row(dbc.Col(_graph_manager.create_graph("centroid-pca-plot", "400px"))),
        ],
    )


def register_weight_geometry_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Repr Geometry page."""

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_gw_data_change(_modified_timestamp: str | None, variant_data: dict | None):
        return _graph_manager.update_graphs(variant_data=variant_data)

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list("matrix")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("matrix-dropdown", "value"),
        State("variant-selector-store", "data"),
    )
    def on_gw_matrix_value_change(
        _modified_timestamp: str | None, matrix_value: str | None, variant_data: dict | None
    ):
        app.server.logger.debug("on_rg_site_value_change")
        view_kwargs = {"matrix": matrix_value}
        return _graph_manager.update_graphs(
            variant_data=variant_data, view_filter_set="matrix", view_kwargs=view_kwargs
        )
