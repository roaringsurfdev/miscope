"""Input Trace page — REQ_075.

Three views for per-input prediction trace analysis:
- Residue class accuracy timeline (cross-epoch summary)
- Training pair accuracy grid (per-epoch snapshot)
- Pair graduation heatmap (cross-epoch graduation summary)
"""

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html

from dashboard.components.analysis_page import AnalysisPageGraphManager

_VIEW_LIST = {
    "training-loss-curves": {
        "view_name": "training.metadata.loss_curves",
        "view_type": "epoch_selector",
    },
    "residue-class-timeline": {
        "view_name": "input_trace.residue_class_timeline",
        "view_type": "epoch_selector",
    },
    "accuracy-grid": {
        "view_name": "input_trace.accuracy_grid",
        "view_type": "epoch_selector",
    },
    "graduation-heatmap": {
        "view_name": "input_trace.graduation_heatmap",
        "view_type": "default_graph",
    },
}

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST, "it")


def create_input_trace_page_nav(app: Dash) -> html.Div:
    app.server.logger.debug("create_input_trace_page_nav")
    return html.Div(children=[])


def create_input_trace_page_layout(app: Dash) -> html.Div:
    app.server.logger.debug("create_input_trace_page_layout")
    return html.Div(
        children=[
            html.H4("Per-Input Prediction Trace", className="mb-3"),
            dbc.Row(dbc.Col(_graph_manager.create_graph("training-loss-curves", "350px"))),
            dbc.Row(dbc.Col(_graph_manager.create_graph("residue-class-timeline", "500px"))),
            dbc.Row(
                [
                    dbc.Col(_graph_manager.create_graph("accuracy-grid", "600px"), width=6),
                    dbc.Col(_graph_manager.create_graph("graduation-heatmap", "600px"), width=6),
                ]
            ),
        ]
    )


def register_input_trace_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Input Trace page."""
    app.server.logger.debug("register_input_trace_page_callbacks")

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_input_trace_data_change(modified_timestamp: str | None, variant_data: dict | None):
        app.server.logger.debug("on_input_trace_data_change")
        return _graph_manager.update_graphs(variant_data, None)
