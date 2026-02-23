import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, html
from dash.exceptions import PreventUpdate

from dashboard_temp.components.visualization import create_empty_figure, create_graph
from dashboard_temp.state import variant_state

# ---------------------------------------------------------------------------
# Plot IDs (prefixed "nd-" to avoid collisions)
# ---------------------------------------------------------------------------

_VIEW_LIST = {
    "nd-trajectory-plot": {"view_name": "neuron_freq_trajectory", "view_type":"default_graph"},
    "nd-switch-plot": {"view_name": "", "view_type":"default_graph"},
    "nd-commitment-plot": {"view_name": "", "view_type":"default_graph"},
}

def _get_graph_output_list():
    graph_list = []
    for view_item in _VIEW_LIST.keys():
        view_type = _VIEW_LIST[view_item].get("view_type")
        graph_list.append({'view_type': view_type, 'index': view_item})

    return graph_list

def _get_graph_view_type(graph_key) -> str:
    view_type = _VIEW_LIST[graph_key].get("view_type")
    if not view_type:
        view_type = "default_graph"
    return view_type

def _update_graphs(variant_data: dict | None, sort_value: str | None) -> list[go.Figure]:
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
        for view_item in _VIEW_LIST.keys():
            view_name = _VIEW_LIST[view_item].get("view_name")
            #view_type = _VIEW_LIST[view_item].get("view_type")
            if view_name in variant_state.available_views:
                figures.append(variant_state.context.view(view_name).figure())
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
                        dbc.Row(dbc.Col(create_graph("nd-trajectory-plot", "600px"))),
                        # Switch distribution | Commitment timeline
                        dbc.Row(
                            [
                                dbc.Col(create_graph("nd-switch-plot", "350px"), width=6),
                                dbc.Col(create_graph("nd-commitment-plot", "350px"), width=6),
                            ]
                        ),
                    ],
                )            
        ]
    )
def register_neuron_dynamics_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Neuron Dynamics page."""

    @app.callback(
        *[Output(pid, "figure") for pid in _get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        Input("nd-sort-toggle", "value"),
        State("variant-selector-store", "data")
    )
    def on_nd_data_change(modified_timestamp: str | None, sort_value: str | None, variant_data: dict | None):
        print("on_nd_data_change")
        return _update_graphs(variant_data, sort_value)