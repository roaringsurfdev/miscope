import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

from dashboard.components.analysis_page import AnalysisPageGraphManager

# miscope
from miscope.analysis.library.weights import WEIGHT_MATRIX_NAMES

# Page for showing dimensionality metrics in one place
_VIEW_LIST = {
    "training-loss-curves": {
        "view_name": "training.metadata.loss_curves",
        "view_type": "epoch_selector",
    },
    "parameter-pca-summary-plot": {
        "view_name": "parameters.pca.variance_explained",
        "view_type": "epoch_selector",
    },
    "gradient_site_convergence": {
        "view_name": "analysis.gradient.site_convergence",
        "view_type": "epoch_selector",
    },
    "gradient_site_heatmap": {
        "view_name": "analysis.gradient.site_heatmap",
        "view_type": "epoch_selector",
    },
    "centroid-class-pca-summary-plot": {
        "view_name": "geometry.centroid_pca_variance",
        "view_type": "epoch_selector",
    },
    "parameters-pca-3d-scatter": {
        "view_name": "parameters.pca.scatter_3d",
        "view_type": "default_graph",
        "view_filter_set": "trajectory_group",
    },
    "parameters-pca-pc1-pc2": {
        "view_name": "parameters.pca.pc1_pc2",
        "view_type": "default_graph",
        "view_filter_set": "trajectory_group",
    },
    "parameters-pca-pc1-pc3": {
        "view_name": "parameters.pca.pc1_pc3",
        "view_type": "default_graph",
        "view_filter_set": "trajectory_group",
    },
    "parameters-pca-pc2-pc3": {
        "view_name": "parameters.pca.pc2_pc3",
        "view_type": "default_graph",
        "view_filter_set": "trajectory_group",
    },
    "parameters-pca-group-overlay-pc1-pc2": {
        "view_name": "parameters.pca.group_overlay",
        "view_type": "default_graph",
    },
    "parameters-pca-group-overlay-pc2-pc3": {
        "view_name": "parameters.pca.group_overlay_pc2_pc3",
        "view_type": "default_graph",
    },
    "parameters-pca-group-overlay-proximity": {
        "view_name": "parameters.pca.proximity",
        "view_type": "epoch_selector",
    },
    "velocity-plot": {
        "view_name": "parameters.pca.component_velocity",
        "view_type": "epoch_selector",
    },
    "dim-trajectory-plot": {
        "view_name": "parameters.effective_dimensionality",
        "view_type": "epoch_selector",
    },
    "sv-spectrum-plot": {
        "view_name": "parameters.singular_value_spectrum",
        "view_type": "default_graph",
        "view_filter_set": "matrix_name",
    },
}

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST, "dim")


def create_dimensionality_page_nav(app: Dash) -> html.Div:
    app.server.logger.debug("create_dimensionality_page_nav")
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


def create_dimensionality_page_layout(app: Dash) -> html.Div:
    app.server.logger.debug("create_dimensionality_page_layout")
    return html.Div(
        children=[
            html.H4("Dimensionality", className="mb-3"),
            html.Div(
                [
                    # --- Loss ---
                    dbc.Row(dbc.Col(_graph_manager.create_graph("training-loss-curves", "350px"))),
                    dbc.Row(dbc.Col(_graph_manager.create_graph("gradient_site_convergence", "600px"))),
                    dbc.Row(dbc.Col(_graph_manager.create_graph("gradient_site_heatmap", "600px"))),
                    dbc.Row(children=[dbc.Col(children=["Parameter Space"]), dbc.Col(children=["Activation Space"])]),
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
                                _graph_manager.create_graph("parameters-pca-3d-scatter", "350px"),
                                width=6,
                            ),
                            dbc.Col(
                                _graph_manager.create_graph("parameters-pca-pc1-pc2", "350px"),
                                width=6,
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                _graph_manager.create_graph("parameters-pca-pc1-pc3", "350px"),
                                width=6,
                            ),
                            dbc.Col(
                                _graph_manager.create_graph("parameters-pca-pc2-pc3", "350px"),
                                width=6,
                            ),
                        ]
                    ),
                    dbc.Row(dbc.Col(_graph_manager.create_graph("parameters-pca-group-overlay-proximity", "400px"))),
                    dbc.Row(dbc.Col(_graph_manager.create_graph("parameters-pca-group-overlay-pc1-pc2", "400px"))),
                    dbc.Row(dbc.Col(_graph_manager.create_graph("parameters-pca-group-overlay-pc2-pc3", "400px"))),
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
    app.server.logger.debug("register_dimensionality_page_callbacks")

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_vz_data_change(modified_timestamp: str | None, variant_data: dict | None):
        app.server.logger.debug("on_vz_data_change")
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
        app.server.logger.debug("on_vz_sv_matrix_change")
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
        app.server.logger.debug("on_vz_trajectory_group_change")
        view_kwargs = {"group_label": trajectory_group, "group": trajectory_group}
        return _graph_manager.update_graphs(
            variant_data=variant_data, view_filter_set="trajectory_group", view_kwargs=view_kwargs
        )
