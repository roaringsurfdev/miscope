import dash_bootstrap_components as dbc
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, set_props

from dashboard.components.analysis_page import AnalysisPageGraphManager

# miscope
from miscope.analysis.library.weights import WEIGHT_MATRIX_NAMES
from miscope.visualization.renderers.landscape_flatness import FLATNESS_METRICS

_VIEW_LIST = {
    "loss-plot": {"view_name": "loss_curve", "view_type": "epoch_selector"},
    "freq-plot": {"view_name": "dominant_frequencies", "view_type": "default_graph"},
    "clusters-plot": {"view_name": "freq_clusters", "view_type": "neuron_selector"},
    "spec-trajectory-plot": {
        "view_name": "specialization_trajectory",
        "view_type": "epoch_selector",
    },
    "spec-freq-plot": {"view_name": "specialization_by_frequency", "view_type": "epoch_selector"},
    "activation-plot": {
        "view_name": "neuron_heatmap",
        "view_type": "default_graph",
        "view_filter_set": "neuron_id",
    },
    "attention-plot": {
        "view_name": "attention_heads",
        "view_type": "default_graph",
        "view_filter_set": "attention_pair",
    },
    "attn-freq-plot": {"view_name": "attention_freq_heatmap", "view_type": "default_graph"},
    "attn-spec-plot": {
        "view_name": "attention_specialization_trajectory",
        "view_type": "epoch_selector",
    },
    "trajectory-3d-plot": {
        "view_name": "trajectory_3d",
        "view_type": "default_graph",
        "view_filter_set": "trajectory_group",
    },
    "trajectory-plot": {
        "view_name": "parameter_trajectory",
        "view_type": "default_graph",
        "view_filter_set": "trajectory_group",
    },
    "trajectory-pc1-pc3-plot": {
        "view_name": "trajectory_pc1_pc3",
        "view_type": "default_graph",
        "view_filter_set": "trajectory_group",
    },
    "trajectory-pc2-pc3-plot": {
        "view_name": "trajectory_pc2_pc3",
        "view_type": "default_graph",
        "view_filter_set": "trajectory_group",
    },
    "velocity-plot": {"view_name": "component_velocity", "view_type": "epoch_selector"},
    "dim-trajectory-plot": {
        "view_name": "dimensionality_trajectory",
        "view_type": "epoch_selector",
    },
    "sv-spectrum-plot": {
        "view_name": "singular_value_spectrum",
        "view_type": "default_graph",
        "view_filter_set": "matrix_name",
    },
    "flatness-trajectory-plot": {
        "view_name": "flatness_trajectory",
        "view_type": "epoch_selector",
        "view_filter_set": "flatness_metric",
    },
    "perturbation-plot": {"view_name": "perturbation_distribution", "view_type": "default_graph"},
}

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST, "viz")


def create_visualization_page_nav() -> html.Div:
    print("create_visualization_page_nav")
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
            # Trajectory component group (REQ_029, REQ_032)
            dbc.Label("Trajectory Group", className="fw-bold"),
            dcc.Dropdown(
                id="trajectory-group-dropdown",
                options=[
                    {"label": "All", "value": "all"},
                    {"label": "Embedding", "value": "embedding"},
                    {"label": "Attention", "value": "attention"},
                    {"label": "MLP", "value": "mlp"},
                ],
                value="all",
                clearable=False,
            ),
            html.Hr(),
            # SV matrix selector (REQ_030)
            dbc.Label("SV Matrix", className="fw-bold"),
            dcc.Dropdown(
                id="sv-matrix-dropdown",
                options=[{"label": name, "value": name} for name in WEIGHT_MATRIX_NAMES],
                value="W_in",
                clearable=False,
            ),
            html.Br(),
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


def create_visualization_page_layout() -> html.Div:
    print("create_visualization_page_layout")
    return html.Div(
        children=[
            html.H4("Visualization", className="mb-3"),
            html.Div(
                [
                    # --- Loss ---
                    dbc.Row(dbc.Col(_graph_manager.create_graph("loss-plot", "350px"))),
                    # --- Frequency Analysis ---
                    dbc.Row(dbc.Col(_graph_manager.create_graph("freq-plot", "400px"))),
                    # --- Neuron Specialization (summary, click-to-navigate) ---
                    dbc.Row(dbc.Col(_graph_manager.create_graph("clusters-plot", "450px"))),
                    dbc.Row(dbc.Col(_graph_manager.create_graph("spec-trajectory-plot", "350px"))),
                    dbc.Row(dbc.Col(_graph_manager.create_graph("spec-freq-plot", "450px"))),
                    # --- Neuron and Attention (per-epoch) ---
                    dbc.Row(
                        [
                            dbc.Col(
                                _graph_manager.create_graph("activation-plot", "300px"), width=3
                            ),
                            dbc.Col(
                                _graph_manager.create_graph("attention-plot", "400px"), width=9
                            ),
                        ]
                    ),
                    # --- Attention Specialization (summary, click-to-navigate) ---
                    dbc.Row(dbc.Col(_graph_manager.create_graph("attn-freq-plot", "450px"))),
                    dbc.Row(dbc.Col(_graph_manager.create_graph("attn-spec-plot", "450px"))),
                    # --- Trajectory (cross-epoch) ---
                    dbc.Row(
                        [
                            dbc.Col(
                                _graph_manager.create_graph("trajectory-3d-plot", "350px"), width=6
                            ),
                            dbc.Col(
                                _graph_manager.create_graph("trajectory-plot", "350px"), width=6
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                _graph_manager.create_graph("trajectory-pc1-pc3-plot", "350px"),
                                width=6,
                            ),
                            dbc.Col(
                                _graph_manager.create_graph("trajectory-pc2-pc3-plot", "350px"),
                                width=6,
                            ),
                        ]
                    ),
                    dbc.Row(dbc.Col(_graph_manager.create_graph("velocity-plot", "400px"))),
                    # --- Dimensionality (summary + per-epoch) ---
                    dbc.Row(dbc.Col(_graph_manager.create_graph("dim-trajectory-plot", "400px"))),
                    dbc.Row(dbc.Col(_graph_manager.create_graph("sv-spectrum-plot", "400px"))),
                    # --- Flatness (summary + per-epoch, click-to-navigate) ---
                    dbc.Row(
                        dbc.Col(_graph_manager.create_graph("flatness-trajectory-plot", "400px"))
                    ),
                    dbc.Row(dbc.Col(_graph_manager.create_graph("perturbation-plot", "400px"))),
                ],
            ),
        ]
    )


def register_visualization_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Neuron Dynamics page."""
    print("register_visualization_page_callbacks")

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_vz_data_change(modified_timestamp: str | None, variant_data: dict | None):
        print("on_vz_data_change")
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
        print("on_vz_attention_pair_change")
        attention_pair_value = dict(item.split(":") for item in attention_pair.split(","))
        view_kwargs = {key: int(value) for key, value in attention_pair_value.items()}
        return _graph_manager.update_graphs(
            variant_data=variant_data, view_filter_set="attention_pair", view_kwargs=view_kwargs
        )

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list("matrix_name")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("sv-matrix-dropdown", "value"),
        State("variant-selector-store", "data"),
    )
    def on_vz_sv_matrix_change(
        modified_timestamp: str | None, matrix_name: str, variant_data: dict | None
    ):
        print("on_vz_sv_matrix_change")
        view_kwargs = {"matrix_name": matrix_name}
        return _graph_manager.update_graphs(
            variant_data=variant_data, view_filter_set="matrix_name", view_kwargs=view_kwargs
        )

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list("trajectory_group")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("trajectory-group-dropdown", "value"),
        State("variant-selector-store", "data"),
    )
    def on_vz_trajectory_group_change(
        modified_timestamp: str | None, trajectory_group: str, variant_data: dict | None
    ):
        print("on_vz_trajectory_group_change")
        view_kwargs = {"group_label": trajectory_group, "group": trajectory_group}
        return _graph_manager.update_graphs(
            variant_data=variant_data, view_filter_set="trajectory_group", view_kwargs=view_kwargs
        )

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list("flatness_metric")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("flatness-metric-dropdown", "value"),
        State("variant-selector-store", "data"),
    )
    def on_vz_flatness_metric_change(
        modified_timestamp: str | None, metric_name: str, variant_data: dict | None
    ):
        print("on_vz_flatness_metric_change")
        view_kwargs = {"metric": metric_name}
        return _graph_manager.update_graphs(
            variant_data=variant_data, view_filter_set="flatness_metric", view_kwargs=view_kwargs
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
        print("on_vz_neuron_slider_change")

        if ctx.triggered_id != "neuron-slider":
            click_data_component_id = ctx.triggered_id
            if click_data is not None:
                for click_data_item in click_data:
                    if click_data_item:
                        clicked_x = click_data_item["points"][0].get("x")
                        if clicked_x:
                            print(f"handling click event: {click_data}")
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
