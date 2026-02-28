import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, html
from dash.exceptions import PreventUpdate

from dashboard.components.visualization import create_empty_figure, create_graph
from dashboard.state import variant_state

# ---------------------------------------------------------------------------
# Plot IDs (prefixed "nd-" to avoid collisions)
# ---------------------------------------------------------------------------

_VIEW_LIST = {
    "nd-trajectory-plot": {"view_name": "neuron_freq_trajectory", "view_type": "default_graph", "view_parameter": "sort_order"},
    "nd-switch-plot": {"view_name": "switch_count_distribution", "view_type": "default_graph"},
    "nd-commitment-plot": {"view_name": "commitment_timeline", "view_type": "default_graph"},
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

def _update_graphs(
    variant_data: dict | None,
    view_parameter: str | None = None,
    view_kwargs: dict | None = None,
) -> list[go.Figure]:
    stored = variant_data or {}
    variant_name = stored.get("variant_name")
    last_field_updated = stored.get("last_field_updated")
    figures = []

    if variant_name is None:
        no_data = create_empty_figure("Select a variant")
        return [no_data for _ in _VIEW_LIST]

    if last_field_updated in ["variant_name", "epoch"]:
        views = [key for key in _VIEW_LIST if _VIEW_LIST[key].get("view_parameter") == view_parameter]
        for view_item in views:
            view_name = _VIEW_LIST[view_item].get("view_name")
            if view_name in variant_state.available_views:
                figures.append(variant_state.context.view(view_name).figure(**(view_kwargs or {})))
            else:
                figures.append(create_empty_figure("No view found"))
    else:
        raise PreventUpdate

    return figures

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
            )
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
                        dbc.Row(dbc.Col(create_graph("nd-trajectory-plot", "600px", _get_graph_view_type("nd-trajectory-plot")))),
                        # Switch distribution | Commitment timeline
                        dbc.Row(
                            [
                                dbc.Col(create_graph("nd-switch-plot", "350px", _get_graph_view_type("nd-switch-plot")), width=6),
                                dbc.Col(create_graph("nd-commitment-plot", "350px", _get_graph_view_type("nd-commitment-plot")), width=6),
                            ]
                        ),
                    ],
                )            
        ]
    )
def register_neuron_dynamics_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Neuron Dynamics page."""
    print("register_neuron_dynamics_page_callbacks")

    @app.callback(
        [Output(pid, "figure") for pid in _get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data")
    )
    def on_nd_data_change(modified_timestamp: str | None, variant_data: dict | None):
        print("on_nd_data_change")
        return _update_graphs(variant_data, None)


    @app.callback(
        [Output(pid, "figure") for pid in _get_graph_output_list("sort_order")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("nd-sort-toggle", "value"),
        State("variant-selector-store", "data"),
    )
    def on_nd_sort_change(
        _modified_timestamp: str | None, sort_value: str | None, variant_data: dict | None
    ):
        sorted_by_final = sort_value == "sorted"
        return _update_graphs(
            variant_data, "sort_order", view_kwargs={"sorted_by_final": sorted_by_final}
        )