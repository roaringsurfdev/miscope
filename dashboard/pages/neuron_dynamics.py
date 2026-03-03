import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html

from dashboard.components.analysis_page import AnalysisPageGraphManager

_VIEW_LIST = {
    "nd-trajectory-plot": {
        "view_name": "neuron_freq_trajectory",
        "view_type": "default_graph",
        "view_filter_set": "sort_order",
    },
    "nd-switch-plot": {"view_name": "switch_count_distribution", "view_type": "default_graph"},
    "nd-commitment-plot": {"view_name": "commitment_timeline", "view_type": "default_graph"},
}

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST)


def create_neuron_dynamics_page_nav() -> html.Div:
    print("create_neuron_dynamics_page_nav")
    return html.Div(
        children=[
            dbc.Label("Sort Order", className="fw-bold"),
            dbc.RadioItems(
                id="nd-sort-toggle",
                options=[
                    {"label": "Natural Order", "value": "natural"},
                    {"label": "Sorted by Final Frequency", "value": "sorted"},
                ],
                value="natural",
                className="mb-3",
            ),
        ]
    )


def create_neuron_dynamics_page_layout() -> html.Div:
    print("create_neuron_dynamics_page_layout")
    return html.Div(
        children=[
            html.H4("Neuron Dynamics", className="mb-3"),
            html.Div(
                [
                    # Trajectory heatmap (full width)
                    dbc.Row(dbc.Col(_graph_manager.create_graph("nd-trajectory-plot", "600px"))),
                    # Switch distribution | Commitment timeline
                    dbc.Row(
                        [
                            dbc.Col(
                                _graph_manager.create_graph(
                                    "nd-switch-plot",
                                    "350px",
                                ),
                                width=6,
                            ),
                            dbc.Col(
                                _graph_manager.create_graph(
                                    "nd-commitment-plot",
                                    "350px",
                                ),
                                width=6,
                            ),
                        ]
                    ),
                ],
            ),
        ]
    )


def register_neuron_dynamics_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Neuron Dynamics page."""
    print("register_neuron_dynamics_page_callbacks")

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_nd_data_change(modified_timestamp: str | None, variant_data: dict | None):
        print("on_nd_data_change")
        return _graph_manager.update_graphs(variant_data=variant_data)

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list("sort_order")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("nd-sort-toggle", "value"),
        State("variant-selector-store", "data"),
    )
    def on_nd_sort_change(
        _modified_timestamp: str | None, sort_value: str | None, variant_data: dict | None
    ):
        sorted_by_final = sort_value == "sorted"
        view_kwargs = {"sorted_by_final": sorted_by_final}
        return _graph_manager.update_graphs(
            variant_data=variant_data, view_filter_set="sort_order", view_kwargs=view_kwargs
        )
