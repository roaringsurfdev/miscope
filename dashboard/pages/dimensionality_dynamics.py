"""REQ_095: Dimensionality Dynamics page.

Loss curve anchor + dimensionality timeseries (3-panel PR₃/f_top3 across
trajectory, class centroid, and within-group weight domains) + geometry
state space (parametric f_top3 vs PR₃ per activation site).
"""

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html

from dashboard.components.analysis_page import AnalysisPageGraphManager

_VIEW_LIST = {
    "training-loss-curves": {
        "view_name": "training.metadata.loss_curves",
        "view_type": "epoch_selector",
    },
    "dimensionality-timeseries": {
        "view_name": "dimensionality.timeseries",
        "view_type": "epoch_selector",
    },
    "velocity-plot": {
        "view_name": "parameters.pca.component_velocity",
        "view_type": "epoch_selector",
    },
    "dimensionality-state-space": {
        "view_name": "dimensionality.state_space",
        "view_type": "epoch_selector",
    },
}

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST, "dimdyn")


def create_dimensionality_dynamics_page_nav(app: Dash) -> html.Div:
    app.server.logger.debug("create_dimensionality_dynamics_page_nav")
    return html.Div(children=[])


def create_dimensionality_dynamics_page_layout(app: Dash) -> html.Div:
    app.server.logger.debug("create_dimensionality_dynamics_page_layout")
    return html.Div(
        children=[
            html.H4("Dimensionality Dynamics", className="mb-3"),
            html.Div(
                [
                    dbc.Row(dbc.Col(_graph_manager.create_graph("training-loss-curves", "350px"))),
                    dbc.Row(
                        dbc.Col(_graph_manager.create_graph("dimensionality-timeseries", "950px"))
                    ),
                    dbc.Row(dbc.Col(_graph_manager.create_graph("velocity-plot", "400px"))),
                    dbc.Row(
                        dbc.Col(_graph_manager.create_graph("dimensionality-state-space", "560px"))
                    ),
                ]
            ),
        ]
    )


def register_dimensionality_dynamics_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Dimensionality Dynamics page."""
    app.server.logger.debug("register_dimensionality_dynamics_page_callbacks")

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_dimdyn_data_change(modified_timestamp: str | None, variant_data: dict | None):
        app.server.logger.debug("on_dimdyn_data_change")
        return _graph_manager.update_graphs(variant_data, None)
