import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

from dashboard.components.analysis_page import AnalysisPageGraphManager

_VIEW_LIST = {
    "training-loss-curves": {
        "view_name": "training.metadata.loss_curves",
        "view_type": "epoch_selector",
    },
    # REQ_066: Multi-stream specialization
    "multi-stream-specialization": {
        "view_name": "multi_stream_specialization",
        "view_type": "epoch_selector",
        "view_filter_set": "multi_stream_thresholds",
    },
}

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST, "multi")


def create_multistream_page_nav(app: Dash) -> html.Div:
    app.server.logger.debug("create_multistream_page_nav")
    return html.Div(
        children=[
            # Multi-stream MLP threshold (REQ_066)
            dbc.Label("Multi-Stream: MLP Threshold", className="fw-bold"),
            dcc.Slider(
                id="multi-stream-mlp-threshold-slider",
                min=0.0,
                max=1.0,
                step=0.05,
                value=0.7,
                marks={0: "0%", 0.5: "50%", 0.9: "90%", 1.0: "100%"},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(
                id="multi-stream-mlp-threshold-display",
                children="MLP Threshold: 50%",
                className="text-muted small mb-3",
            ),
            # Multi-stream Embedding threshold (REQ_066)
            dbc.Label("Multi-Stream: Embedding Threshold", className="fw-bold"),
            dcc.Slider(
                id="multi-stream-emb-threshold-slider",
                min=0.0,
                max=1.0,
                step=0.05,
                value=0.2,
                marks={0: "0%", 0.5: "50%", 0.9: "90%", 1.0: "100%"},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(
                id="multi-stream-emb-threshold-display",
                children="Embedding Threshold: 50%",
                className="text-muted small mb-3",
            ),
            # Attention visibility floor (REQ_066)
            dbc.Label("Multi-Stream: Attention Floor", className="fw-bold"),
            dcc.Slider(
                id="multi-stream-attn-floor-slider",
                min=0.0,
                max=0.3,
                step=0.01,
                value=0.07,
                marks={0: "0%", 0.05: "5%", 0.1: "10%", 0.2: "20%", 0.3: "30%"},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(
                id="multi-stream-attn-floor-display",
                children="Attention Floor: 2%",
                className="text-muted small mb-3",
            ),
            html.Hr(),
        ]
    )


def create_multistream_page_layout(app: Dash) -> html.Div:
    app.server.logger.debug("create_multistream_page_layout")
    return html.Div(
        children=[
            html.H4("Visualization", className="mb-3"),
            html.Div(
                [
                    # --- Loss ---
                    dbc.Row(dbc.Col(_graph_manager.create_graph("training-loss-curves", "350px"))),
                    # --- Multi-stream specialization (REQ_066) ---
                    dbc.Row(
                        dbc.Col(
                            _graph_manager.create_graph("multi-stream-specialization", "1400px")
                        )
                    ),
                ],
            ),
        ]
    )


def register_multistream_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Multistream page."""
    app.server.logger.debug("register_multistream_page_callbacks")

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_multi_data_change(modified_timestamp: str | None, variant_data: dict | None):
        app.server.logger.debug("on_multi_data_change")
        return _graph_manager.update_graphs(variant_data, None)

    @app.callback(
        Output("multi-stream-mlp-threshold-display", "children"),
        Input("multi-stream-mlp-threshold-slider", "value"),
    )
    def on_multi_stream_mlp_display_update(threshold: float) -> str:
        return f"MLP Threshold: {int(threshold * 100)}%"

    @app.callback(
        Output("multi-stream-emb-threshold-display", "children"),
        Input("multi-stream-emb-threshold-slider", "value"),
    )
    def on_multi_stream_emb_display_update(threshold: float) -> str:
        return f"Embedding Threshold: {int(threshold * 100)}%"

    @app.callback(
        Output("multi-stream-attn-floor-display", "children"),
        Input("multi-stream-attn-floor-slider", "value"),
    )
    def on_multi_stream_attn_floor_display_update(floor: float) -> str:
        return f"Attention Floor: {int(floor * 100)}%"

    @app.callback(
        [
            Output(pid, "figure")
            for pid in _graph_manager.get_graph_output_list("multi_stream_thresholds")
        ],
        Input("variant-selector-store", "modified_timestamp"),
        Input("multi-stream-mlp-threshold-slider", "value"),
        Input("multi-stream-emb-threshold-slider", "value"),
        Input("multi-stream-attn-floor-slider", "value"),
        State("variant-selector-store", "data"),
    )
    def on_multi_stream_thresholds_change(
        modified_timestamp: str | None,
        threshold_mlp: float,
        threshold_emb: float,
        attn_floor: float,
        variant_data: dict | None,
    ):
        app.server.logger.debug("on_multi_stream_thresholds_change")
        view_kwargs = {
            "threshold_mlp": threshold_mlp,
            "threshold_embedding": threshold_emb,
            "attn_floor": attn_floor,
        }
        return _graph_manager.update_graphs(
            variant_data=variant_data,
            view_filter_set="multi_stream_thresholds",
            view_kwargs=view_kwargs,
        )
