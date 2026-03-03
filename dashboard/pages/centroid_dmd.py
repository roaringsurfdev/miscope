import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

from dashboard.components.analysis_page import _SITE_OPTIONS, AnalysisPageGraphManager

# Page for showing dimensionality metrics in one place
views = ["centroid_pca", "centroid_dmd_reconstruction", "centroid_dmd_eigenvalues"]
_VIEW_LIST = {
    "loss-plot": {"view_name": "loss_curve", "view_type": "epoch_selector"},
    "centroid-global-pca": {
        "view_name": "centroid_global_pca",
        "view_type": "default_graph",
        "view_filter_set": "site",
    },
    "centroid-dmd-residual": {
        "view_name": "centroid_dmd_residual",
        "view_type": "epoch_selector",
    },
    "centroid-dmd-reconstruction": {
        "view_name": "centroid_dmd_reconstruction",
        "view_type": "default_graph",
        "view_filter_set": "site",
    },
    "centroid-dmd-eigenvalues": {
        "view_name": "centroid_dmd_eigenvalues",
        "view_type": "default_graph",
        "view_filter_set": "site",
    },
}

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST, "dmd")


def create_centroid_dmd_nav() -> html.Div:
    return html.Div(
        children=[
            dbc.Label("Activation Site", className="fw-bold"),
            dcc.Dropdown(
                id="dmd-site-dropdown",
                options=_SITE_OPTIONS,
                value="resid_post",
                clearable=False,
            ),
        ]
    )


def create_centroid_dmd_layout() -> html.Div:
    print("create_centroid_dmd_layout")
    return html.Div(
        children=[
            html.H4("Centroid DMD", className="mb-3"),
            html.Div(
                [
                    # --- Loss ---
                    dbc.Row(dbc.Col(_graph_manager.create_graph("loss-plot", "350px"))),
                    # --- Centroid DMD  ---
                    dbc.Row(dbc.Col(_graph_manager.create_graph("centroid-global-pca", "350px"))),
                    dbc.Row(dbc.Col(_graph_manager.create_graph("centroid-dmd-residual", "450px"))),
                    dbc.Row(
                        dbc.Col(_graph_manager.create_graph("centroid-dmd-reconstruction", "450px"))
                    ),
                    dbc.Row(
                        dbc.Col(_graph_manager.create_graph("centroid-dmd-eigenvalues", "450px"))
                    ),
                ],
            ),
        ]
    )


def register_centroid_dmd_callbacks(app: Dash) -> None:
    """Register all callbacks for the Neuron Dynamics page."""
    print("register_centroid_dmd_callbacks")

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_vz_data_change(modified_timestamp: str | None, variant_data: dict | None):
        print("on_vz_data_change")
        return _graph_manager.update_graphs(variant_data, None)

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list("site")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("dmd-site-dropdown", "value"),
        State("variant-selector-store", "data"),
    )
    def on_dmd_site_value_change(
        _modified_timestamp: str | None, site_value: str | None, variant_data: dict | None
    ):
        print("on_dmd_site_value_change")
        view_kwargs = {"site": site_value}
        return _graph_manager.update_graphs(
            variant_data=variant_data, view_filter_set="site", view_kwargs=view_kwargs
        )

    """
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
    """
