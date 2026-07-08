import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

from dashboard.components.analysis_page import AnalysisPageGraphManager

# miscope
from miscope.visualization.renderers.landscape_flatness import FLATNESS_METRICS

_VIEW_LIST = {
    "training-loss-curves": {
        "view_name": "training.metadata.loss_curves",
        "view_type": "epoch_selector",
    },
    "flatness-trajectory-plot": {
        "view_name": "loss_landscape.flatness_trajectory",
        "view_type": "epoch_selector",
        "view_filter_set": "flatness_metric",
    },
    "perturbation-plot": {
        "view_name": "loss_landscape.perturbation_distribution",
        "view_type": "default_graph",
    },
}

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST, "loss")


def create_loss_landscape_page_nav(app: Dash) -> html.Div:
    app.server.logger.debug("create_loss_landscape_page_nav")
    return html.Div(
        children=[
            # Flatness metric selector (REQ_031)
            dbc.Label("Flatness Metric", className="fw-bold"),
            dcc.Dropdown(
                id="flatness-metric-dropdown",
                options=[
                    {"label": display, "value": key} for key, display in FLATNESS_METRICS.items()
                ],
                value="mean_delta_loss",
                clearable=False,
            ),
            html.Br(),
            html.Hr(),
        ]
    )


def create_loss_landscape_page_layout(app: Dash) -> html.Div:
    app.server.logger.debug("create_loss_landscape_page_layout")
    return html.Div(
        children=[
            html.H4("Visualization", className="mb-3"),
            html.Div(
                [
                    # --- Loss ---
                    dbc.Row(dbc.Col(_graph_manager.create_graph("training-loss-curves", "350px"))),
                    # --- Flatness (summary + per-epoch, click-to-navigate) ---
                    dbc.Row(
                        dbc.Col(_graph_manager.create_graph("flatness-trajectory-plot", "400px"))
                    ),
                    dbc.Row(dbc.Col(_graph_manager.create_graph("perturbation-plot", "400px"))),
                ],
            ),
        ]
    )


def register_loss_landscape_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Neuron Dynamics page."""
    app.server.logger.debug("register_loss_landscape_page_callbacks")

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_vz_data_change(modified_timestamp: str | None, variant_data: dict | None):
        app.server.logger.debug("on_vz_data_change")
        return _graph_manager.update_graphs(variant_data, None)

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list("flatness_metric")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("flatness-metric-dropdown", "value"),
        State("variant-selector-store", "data"),
    )
    def on_vz_flatness_metric_change(
        modified_timestamp: str | None, metric_name: str, variant_data: dict | None
    ):
        app.server.logger.debug("on_vz_flatness_metric_change")
        view_kwargs = {"metric": metric_name}
        return _graph_manager.update_graphs(
            variant_data=variant_data, view_filter_set="flatness_metric", view_kwargs=view_kwargs
        )
