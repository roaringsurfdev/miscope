import dash_bootstrap_components as dbc
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, set_props

from dashboard.components.analysis_page import AnalysisPageGraphManager

_VIEW_LIST = {
    "training-loss-curves": {
        "view_name": "training.metadata.loss_curves",
        "view_type": "epoch_selector",
    },
    "neuron-frequency-clusters": {
        "view_name": "activations.mlp.neuron_frequency_clusters",
        "view_type": "neuron_selector",
    },
    "activation-plot": {
        "view_name": "activations.mlp.neuron_heatmap",
        "view_type": "default_graph",
        "view_filter_set": "neuron_id",
    },
    "attention-plot": {
        "view_name": "activations.attention.head_heatmap",
        "view_type": "default_graph",
        "view_filter_set": "attention_pair",
    },
    "attn-freq-plot": {
        "view_name": "activations.attention.head_frequency_clusters",
        "view_type": "default_graph",
    },
    "attn-spec-plot": {
        "view_name": "activations.attention.frequency_clusters",
        "view_type": "epoch_selector",
    },
    "attn-qk-fourier": {
        "view_name": "parameters.attention.qk_fourier_heatmap",
        "view_type": "default_graph",
    },
    "attn-v-fourier": {
        "view_name": "parameters.attention.v_fourier_heatmap",
        "view_type": "epoch_selector",
    },
}

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST, "act")


def create_activation_heatmap_page_nav(app: Dash) -> html.Div:
    app.server.logger.debug("create_activation_heatmap_page_nav")
    return html.Div(
        children=[
            # Neuron slider
            dbc.Label("Neuron Index", className="fw-bold"),
            dcc.Slider(
                id="neuron-slider",
                min=0,
                max=511,
                step=1,
                value=0,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(
                id="neuron-display",
                children="Neuron 0",
                className="text-muted small mb-3",
            ),
            html.Hr(),
            # Attention position pair (REQ_025)
            dbc.Label("Attention Relationship", className="fw-bold"),
            dcc.Dropdown(
                id="position-pair-dropdown",
                options=[
                    {"label": "= attending to a", "value": "to_position:2,from_position:0"},
                    {"label": "= attending to b", "value": "to_position:2,from_position:1"},
                    {"label": "b attending to a", "value": "to_position:1,from_position:0"},
                ],
                value="to_position:2,from_position:0",
                clearable=False,
            ),
            html.Br(),
            html.Hr(),
        ]
    )


def create_activation_heatmap_page_layout(app: Dash) -> html.Div:
    app.server.logger.debug("create_visualization_page_layout")
    return html.Div(
        children=[
            html.H4("Activation Heatmaps", className="mb-3"),
            html.Div(
                [
                    # --- Loss ---
                    dbc.Row(dbc.Col(_graph_manager.create_graph("training-loss-curves", "350px"))),
                    # --- Neuron Specialization (summary, click-to-navigate) ---
                    dbc.Row(
                        dbc.Col(_graph_manager.create_graph("neuron-frequency-clusters", "450px"))
                    ),
                    # --- Neuron and Attention (per-epoch) ---
                    dbc.Row(
                        [
                            dbc.Col(
                                _graph_manager.create_graph("activation-plot", "300px"), width=6
                            ),
                            dbc.Col(
                                _graph_manager.create_graph("attn-freq-plot", "450px"), width=6
                            ),
                        ]
                    ),
                    dbc.Row(dbc.Col(_graph_manager.create_graph("attention-plot", "400px"))),
                    # --- Attention Specialization (summary, click-to-navigate) ---
                    dbc.Row(dbc.Col(_graph_manager.create_graph("attn-spec-plot", "450px"))),
                    # --- QK and V Fourier Alignment ---
                    dbc.Row(
                        [
                            dbc.Col(
                                _graph_manager.create_graph("attn-qk-fourier", "450px"), width=6
                            ),
                            dbc.Col(
                                _graph_manager.create_graph("attn-v-fourier", "450px"), width=6
                            ),
                        ]
                    ),
                ],
            ),
        ]
    )


def register_activation_heatmap_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Neuron Dynamics page."""
    app.server.logger.debug("register_visualization_page_callbacks")

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_vz_data_change(modified_timestamp: str | None, variant_data: dict | None):
        app.server.logger.debug("on_vz_data_change")
        return _graph_manager.update_graphs(variant_data, None)

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list("attention_pair")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("position-pair-dropdown", "value"),
        State("variant-selector-store", "data"),
    )
    def on_vz_attention_pair_change(
        modified_timestamp: str | None, attention_pair: str, variant_data: dict | None
    ):
        app.server.logger.debug("on_vz_attention_pair_change")
        attention_pair_value = dict(item.split(":") for item in attention_pair.split(","))
        view_kwargs = {key: int(value) for key, value in attention_pair_value.items()}
        return _graph_manager.update_graphs(
            variant_data=variant_data, view_filter_set="attention_pair", view_kwargs=view_kwargs
        )

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list("neuron_id")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("neuron-slider", "value"),
        Input({"view_type": "neuron_selector", "index": ALL}, "clickData"),
        State("variant-selector-store", "data"),
    )
    def on_vz_neuron_slider_change(
        modified_timestamp: str | None,
        neuron_id: int,
        click_data: list[dict | None],
        variant_data: dict | None,
    ):
        app.server.logger.debug("on_vz_neuron_slider_change")

        if ctx.triggered_id != "neuron-slider":
            click_data_component_id = ctx.triggered_id
            if click_data is not None:
                for click_data_item in click_data:
                    if click_data_item:
                        clicked_x = click_data_item["points"][0].get("x")
                        if clicked_x:
                            app.server.logger.debug(f"handling click event: {click_data}")
                            neuron_id = int(clicked_x)
                            # Reset clickData so that there's only one entry in click_data at a time
                            if click_data_component_id:
                                set_props(click_data_component_id, {"clickData": None})
                                set_props("neuron-slider", {"value": neuron_id})
                                set_props("neuron-display", {"children": f"Neuron {neuron_id}"})

                        break
        view_kwargs = {"neuron_idx": neuron_id}
        return _graph_manager.update_graphs(
            variant_data=variant_data, view_filter_set="neuron_id", view_kwargs=view_kwargs
        )
