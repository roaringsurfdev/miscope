import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, set_props
from dash.exceptions import PreventUpdate

from dashboard.components.visualization import create_empty_figure, create_graph
from dashboard.state import variant_state

# miscope
from miscope.analysis.library.weights import WEIGHT_MATRIX_NAMES
from miscope.visualization.renderers.landscape_flatness import FLATNESS_METRICS

_VIEW_LIST = {
    "loss-plot": {"view_name": "loss_curve", "view_type": "epoch_selector"},
    "freq-plot": {"view_name": "dominant_frequencies", "view_type": "default_graph"},
    "clusters-plot": {"view_name": "freq_clusters", "view_type": "neuron_selector"},
    "spec-trajectory-plot": {"view_name": "specialization_trajectory", "view_type": "epoch_selector"},
    "spec-freq-plot": {"view_name": "specialization_by_frequency", "view_type": "epoch_selector"},
    "activation-plot": {"view_name": "neuron_heatmap", "view_type": "default_graph", "view_parameter": "neuron_id"},
    "attention-plot": {"view_name": "attention_heads", "view_type": "default_graph", "view_parameter": "attention_pair"},
    "attn-freq-plot": {"view_name": "attention_freq_heatmap", "view_type": "default_graph"},
    "attn-spec-plot": {"view_name": "attention_specialization_trajectory", "view_type": "epoch_selector"},

    "trajectory-3d-plot": {"view_name": "trajectory_3d", "view_type": "default_graph", "view_parameter": "trajectory_group"},
    "trajectory-plot": {"view_name": "parameter_trajectory", "view_type": "default_graph", "view_parameter": "trajectory_group"},
    "trajectory-pc1-pc3-plot": {"view_name": "trajectory_pc1_pc3", "view_type": "default_graph", "view_parameter": "trajectory_group"},
    "trajectory-pc2-pc3-plot": {"view_name": "trajectory_pc2_pc3", "view_type": "default_graph", "view_parameter": "trajectory_group"},
    "velocity-plot": {"view_name": "component_velocity", "view_type": "epoch_selector"},

    "dim-trajectory-plot": {"view_name": "dimensionality_trajectory", "view_type": "epoch_selector"},
    "sv-spectrum-plot": {"view_name": "singular_value_spectrum", "view_type": "default_graph", "view_parameter": "matrix_name"},

    "flatness-trajectory-plot": {"view_name": "flatness_trajectory", "view_type":"epoch_selector", "view_parameter": "flatness_metric"},
    "perturbation-plot": {"view_name": "perturbation_distribution", "view_type":"default_graph"},
}

def _get_graph_output_list(view_parameter: str | None = None):
    graph_list = []
    views = [key for key in _VIEW_LIST.keys() if _VIEW_LIST[key].get("view_parameter") == view_parameter]
    for view_item in views:
        view_type = _VIEW_LIST[view_item].get("view_type")
        graph_list.append({'view_type': view_type, 'index': view_item})

    return graph_list

def _get_graph_view_type(graph_key) -> str:
    view_type = _VIEW_LIST[graph_key].get("view_type")
    if not view_type:
        view_type = "default_graph"
    return view_type

def _update_graphs(variant_data: dict | None, view_parameter: str | None = None, view_parameter_value: dict | None = None) -> list[go.Figure]:
    stored = variant_data or {}
    variant_name = stored.get("variant_name")
    last_field_updated = stored.get("last_field_updated")
    figures = []

    # Clear graphs if variant_name is None
    if variant_name is None:
        no_data = create_empty_figure("Select a variant")
        figures = [no_data for pid in _VIEW_LIST.keys()]

    if last_field_updated in ["variant_name", "epoch"]:
        #Update graphs
        views = [key for key in _VIEW_LIST.keys() if _VIEW_LIST[key].get("view_parameter") == view_parameter]
        print(views)
        for view_item in views:
            view_name = _VIEW_LIST[view_item].get("view_name")
            if view_name in variant_state.available_views:
                if view_parameter_value:
                    figures.append(variant_state.context.view(view_name).figure(**view_parameter_value))
                else:
                    figures.append(variant_state.context.view(view_name).figure())                    
            else:
                figures.append(create_empty_figure("No view found"))
    else:
        raise PreventUpdate
    
    return figures

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
                        dbc.Row(dbc.Col(create_graph("loss-plot", "350px", _get_graph_view_type("loss-plot")))),
                        # --- Frequency Analysis ---
                        dbc.Row(dbc.Col(create_graph("freq-plot", "400px", _get_graph_view_type("freq-plot")))),
                        # --- Neuron Specialization (summary, click-to-navigate) ---
                        dbc.Row(dbc.Col(create_graph("clusters-plot", "450px", _get_graph_view_type("clusters-plot")))),
                        dbc.Row(dbc.Col(create_graph("spec-trajectory-plot", "350px", _get_graph_view_type("spec-trajectory-plot")))),
                        dbc.Row(dbc.Col(create_graph("spec-freq-plot", "450px", _get_graph_view_type("spec-freq-plot")))),

                        # --- Neuron and Attention (per-epoch) ---
                        dbc.Row(
                            [
                                dbc.Col(create_graph("activation-plot", "300px", _get_graph_view_type("activation-plot")), width=3),
                                dbc.Col(create_graph("attention-plot", "400px", _get_graph_view_type("attention-plot")), width=9),
                            ]
                        ),
                        # --- Attention Specialization (summary, click-to-navigate) ---
                        dbc.Row(dbc.Col(create_graph("attn-freq-plot", "450px", _get_graph_view_type("attn-freq-plot")))),
                        dbc.Row(dbc.Col(create_graph("attn-spec-plot", "450px", _get_graph_view_type("attn-spec-plot")))),
                        # --- Trajectory (cross-epoch) ---
                        dbc.Row(
                            [
                                dbc.Col(create_graph("trajectory-3d-plot", "350px", _get_graph_view_type("trajectory-3d-plot")), width=6),
                                dbc.Col(create_graph("trajectory-plot", "350px", _get_graph_view_type("trajectory-plot")), width=6),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(create_graph("trajectory-pc1-pc3-plot", "350px", _get_graph_view_type("trajectory-pc1-pc3-plot")), width=6),
                                dbc.Col(create_graph("trajectory-pc2-pc3-plot", "350px", _get_graph_view_type("trajectory-pc2-pc3-plot")), width=6),
                            ]
                        ),
                        dbc.Row(dbc.Col(create_graph("velocity-plot", "400px", _get_graph_view_type("velocity-plot")))),

                        # --- Dimensionality (summary + per-epoch) ---
                        dbc.Row(dbc.Col(create_graph("dim-trajectory-plot", "400px", _get_graph_view_type("dim-trajectory-plot")))),
                        dbc.Row(dbc.Col(create_graph("sv-spectrum-plot", "400px", _get_graph_view_type("sv-spectrum-plot")))),
                        # --- Flatness (summary + per-epoch, click-to-navigate) ---
                        dbc.Row(dbc.Col(create_graph("flatness-trajectory-plot", "400px", _get_graph_view_type("flatness-trajectory-plot")))),
                        dbc.Row(dbc.Col(create_graph("perturbation-plot", "400px", _get_graph_view_type("perturbation-plot")))),
                    ],
                )            
        ]
    )

def register_visualization_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Neuron Dynamics page."""
    print("register_visualization_page_callbacks")

    @app.callback(
        [Output(pid, "figure") for pid in _get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data")
    )
    def on_vz_data_change(modified_timestamp: str | None, variant_data: dict | None):
        print("on_vz_data_change")
        return _update_graphs(variant_data, None)


    @app.callback(
        [Output(pid, "figure") for pid in _get_graph_output_list("attention_pair")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("position-pair-dropdown", "value"),
        State("variant-selector-store", "data")
    )
    def on_vz_attention_pair_change(modified_timestamp: str | None, attention_pair: str, variant_data: dict | None):
        print("on_vz_attention_pair_change")
        attention_pair_value = dict(item.split(":") for item in attention_pair.split(","))
        parameters = {key: int(value) for key, value in attention_pair_value.items()}
        return _update_graphs(variant_data, "attention_pair", parameters)    

    @app.callback(
        [Output(pid, "figure") for pid in _get_graph_output_list("matrix_name")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("sv-matrix-dropdown", "value"),
        State("variant-selector-store", "data")
    )
    def on_vz_sv_matrix_change(modified_timestamp: str | None, matrix_name: str, variant_data: dict | None):
        print("on_vz_sv_matrix_change")
        parameters = {"matrix_name": matrix_name}
        return _update_graphs(variant_data, "matrix_name", parameters)    

    @app.callback(
        [Output(pid, "figure") for pid in _get_graph_output_list("trajectory_group")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("trajectory-group-dropdown", "value"),
        State("variant-selector-store", "data")
    )
    def on_vz_trajectory_group_change(modified_timestamp: str | None, trajectory_group: str, variant_data: dict | None):
        print("on_vz_trajectory_group_change")
        parameters = {"group_label": trajectory_group, "group": trajectory_group}
        return _update_graphs(variant_data, "trajectory_group", parameters)    

    @app.callback(
        [Output(pid, "figure") for pid in _get_graph_output_list("flatness_metric")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("flatness-metric-dropdown", "value"),
        State("variant-selector-store", "data")
    )
    def on_vz_flatness_metric_change(modified_timestamp: str | None, metric_name: str, variant_data: dict | None):
        print("on_vz_flatness_metric_change")
        parameters = {"metric": metric_name}
        return _update_graphs(variant_data, "flatness_metric", parameters)    
    
    @app.callback(
        [Output(pid, "figure") for pid in _get_graph_output_list("neuron_id")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("neuron-slider", "value"),
        Input({'view_type': 'neuron_selector', 'index': ALL}, "clickData"),
        State("variant-selector-store", "data")
    )
    def on_vz_neuron_slider_change(modified_timestamp: str | None, neuron_id: int, click_data: list[dict | None], variant_data: dict | None):
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
                                set_props(click_data_component_id, {'clickData': None})
                                set_props("neuron-slider", {'value': neuron_id})
                                set_props("neuron-display", {'children': f"Neuron {neuron_id}"})

                        break
        parameters = {"neuron_idx": neuron_id}
        return _update_graphs(variant_data, "neuron_id", parameters)    
    
