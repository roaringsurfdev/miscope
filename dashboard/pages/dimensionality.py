import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

from dashboard.components.analysis_page import AnalysisPageGraphManager

# miscope
from miscope.analysis.library.weights import WEIGHT_MATRIX_NAMES

# Page for showing dimensionality metrics in one place
_VIEW_LIST = {
    "loss-plot": {"view_name": "loss_curve", "view_type": "epoch_selector"},
    "spec-trajectory-plot": {
        "view_name": "specialization_trajectory",
        "view_type": "epoch_selector",
    },
    "spec-freq-plot": {"view_name": "specialization_by_frequency", "view_type": "epoch_selector"},
    "attn-spec-plot": {
        "view_name": "attention_specialization_trajectory",
        "view_type": "epoch_selector",
    },
    "parameter-pca-summary-plot": {
        "view_name": "trajectory_pca_variance",
        "view_type": "epoch_selector",
    },
    "centroid-class-pca-summary-plot": {
        "view_name": "centroid_pca_variance",
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
}

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST, "dim")


def create_dimensionality_page_nav() -> html.Div:
    print("create_dimensionality_page_nav")
    return html.Div(
        children=[
            # Neuron slider
            # Trajectory component group (REQ_029, REQ_032)
            dbc.Label("Trajectory Group", className="fw-bold"),
            dcc.Dropdown(
                id="dim-trajectory-group-dropdown",
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
                id="dim-sv-matrix-dropdown",
                options=[{"label": name, "value": name} for name in WEIGHT_MATRIX_NAMES],
                value="W_in",
                clearable=False,
            ),
            html.Br(),
            html.Hr(),
        ]
    )


def create_dimensionality_page_layout() -> html.Div:
    print("create_dimensionality_page_layout")
    return html.Div(
        children=[
            html.H4("Dimensionality", className="mb-3"),
            html.Div(
                [
                    # --- Loss ---
                    dbc.Row(dbc.Col(_graph_manager.create_graph("loss-plot", "350px"))),
                    # --- Neuron Specialization  ---
                    dbc.Row(dbc.Col(_graph_manager.create_graph("spec-trajectory-plot", "350px"))),
                    dbc.Row(dbc.Col(_graph_manager.create_graph("spec-freq-plot", "450px"))),
                    # --- Attention Specialization (summary, click-to-navigate) ---
                    dbc.Row(dbc.Col(_graph_manager.create_graph("attn-spec-plot", "450px"))),
                    dbc.Row(
                        children=[
                            dbc.Col("Parameter Space PCA", style={"align": "center"}),
                            dbc.Col("Centroid Class PCA", style={"align": "center"}),
                        ],
                        style={"height": "50px"},
                    ),
                    dbc.Row(
                        children=[
                            dbc.Col(
                                _graph_manager.create_graph("parameter-pca-summary-plot", "350px"),
                                width=6,
                            ),
                            dbc.Col(
                                _graph_manager.create_graph(
                                    "centroid-class-pca-summary-plot", "350px"
                                ),
                                width=6,
                            ),
                        ],
                        style={"height": "600px"},
                    ),
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
                ],
            ),
        ]
    )


def register_dimensionality_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Neuron Dynamics page."""
    print("register_dimensionality_page_callbacks")

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_vz_data_change(modified_timestamp: str | None, variant_data: dict | None):
        print("on_vz_data_change")
        return _graph_manager.update_graphs(variant_data, None)

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list("matrix_name")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("dim-sv-matrix-dropdown", "value"),
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
        Input("dim-trajectory-group-dropdown", "value"),
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
