import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

from dashboard.components.analysis_page import AnalysisPageGraphManager

_VIEW_LIST = {
    "neuron_freq_trajectory": {
        "view_name": "activations.mlp.neuron_freq_trajectory",
        "view_type": "epoch_selector",
        "view_filter_set": "nd_specialization_threshold",
    },
    "neuron_freq_distribution": {
        "view_name": "activations.mlp.neuron_freq_distribution",
        "view_type": "epoch_selector",
    },
    "neuron-frequency-clusters": {
        "view_name": "activations.mlp.neuron_frequency_clusters",
        "view_type": "neuron_selector",
    },
    "spec-freq-plot": {
        "view_name": "activations.mlp.neuron_frequency_specialization",
        "view_type": "epoch_selector",
        "view_filter_set": "nd_specialization_threshold",
    },
    "switch-plot": {
        "view_name": "activations.mlp.switch_count_distribution",
        "view_type": "default_graph",
    },
    "commitment-plot": {
        "view_name": "activations.mlp.commitment_timeline",
        "view_type": "epoch_selector",
    },
    # Threshold-sensitive views
    "per-band-specialization": {
        "view_name": "activations.mlp.per_band_specialization",
        "view_type": "epoch_selector",
        "view_filter_set": "nd_specialization_threshold",
    },
    "neuron-frequency-range": {
        "view_name": "activations.mlp.neuron_frequency_range",
        "view_type": "epoch_selector",
    },
    "band-concentration": {
        "view_name": "analysis.band_concentration.trajectory",
        "view_type": "epoch_selector",
    },
}

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST, "nd")


def create_neuron_dynamics_page_nav(app: Dash) -> html.Div:
    app.server.logger.debug("create_neuron_dynamics_page_nav")
    return html.Div(
        children=[
            dbc.Label("Sort Order", className="fw-bold"),
            dbc.RadioItems(
                id="sort-toggle",
                options=[
                    {"label": "Natural Order", "value": "natural"},
                    {"label": "Sorted by Final Frequency", "value": "sorted"},
                ],
                value="natural",
                className="mb-3",
            ),
            dbc.Label("Specialization Threshold", className="fw-bold"),
            dcc.Slider(
                id="nd-specialization-threshold-slider",
                min=0.0,
                max=1.0,
                step=0.05,
                value=0.75,
                marks={0: "0%", 0.5: "50%", 0.75: "75%", 1.0: "100%"},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(
                id="nd-specialization-threshold-display",
                children="Threshold: 75%",
                className="text-muted small mb-3",
            ),
        ]
    )


def create_neuron_dynamics_page_layout(app: Dash) -> html.Div:
    app.server.logger.debug("create_neuron_dynamics_page_layout")
    return html.Div(
        children=[
            html.H4("Neuron Competition", className="mb-3"),
            html.Div(
                [
                    # Trajectory heatmap (full width)
                    dbc.Row(
                        dbc.Col(_graph_manager.create_graph("neuron_freq_trajectory", "600px"))
                    ),
                    # Frequency Distribution
                    dbc.Row(
                        dbc.Col(_graph_manager.create_graph("neuron_freq_distribution", "600px"))
                    ),
                    # --- Neuron Specialization (summary, click-to-navigate) ---
                    dbc.Row(
                        dbc.Col(_graph_manager.create_graph("neuron-frequency-clusters", "600px"))
                    ),
                    dbc.Row(dbc.Col(_graph_manager.create_graph("spec-freq-plot", "350px"))),
                    # Switch distribution | Commitment timeline
                    dbc.Row(
                        [
                            dbc.Col(
                                _graph_manager.create_graph("switch-plot", "350px"),
                                width=6,
                            ),
                            dbc.Col(
                                _graph_manager.create_graph("commitment-plot", "350px"),
                                width=6,
                            ),
                        ]
                    ),
                    # Threshold-sensitive views
                    dbc.Row(
                        dbc.Col(_graph_manager.create_graph("per-band-specialization", "350px"))
                    ),
                    dbc.Row(
                        dbc.Col(_graph_manager.create_graph("neuron-frequency-range", "350px"))
                    ),
                    dbc.Row(dbc.Col(_graph_manager.create_graph("band-concentration", "350px"))),
                ],
            ),
        ]
    )


def register_neuron_dynamics_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Neuron Dynamics page."""
    app.server.logger.debug("register_neuron_dynamics_page_callbacks")

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_nd_data_change(modified_timestamp: str | None, variant_data: dict | None):
        app.server.logger.debug("on_nd_data_change")
        return _graph_manager.update_graphs(variant_data=variant_data)

    """
    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list("sort_order")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("sort-toggle", "value"),
        State("variant-selector-store", "data"),
    )
    def on_nd_sort_change(
        _modified_timestamp: str | None, sort_value: str | None, variant_data: dict | None
    ):
        sorted_by_final = sort_value == "sorted"
        return _graph_manager.update_graphs(
            variant_data=variant_data,
            view_filter_set="sort_order",
            view_kwargs={"sorted_by_final": sorted_by_final},
        )
    @app.callback(
        Output("nd-specialization-threshold-display", "children"),
        Input("nd-specialization-threshold-slider", "value"),
    )
    """

    def on_nd_threshold_display_update(threshold: float) -> str:
        return f"Threshold: {int(threshold * 100)}%"

    @app.callback(
        [
            Output(pid, "figure")
            for pid in _graph_manager.get_graph_output_list("nd_specialization_threshold")
        ],
        Input("variant-selector-store", "modified_timestamp"),
        Input("nd-specialization-threshold-slider", "value"),
        State("variant-selector-store", "data"),
    )
    def on_nd_threshold_change(
        modified_timestamp: str | None, threshold: float, variant_data: dict | None
    ):
        app.server.logger.debug("on_nd_threshold_change")
        view_kwargs = {"threshold": threshold}
        return _graph_manager.update_graphs(
            variant_data=variant_data,
            view_filter_set="nd_specialization_threshold",
            view_kwargs=view_kwargs,
        )
