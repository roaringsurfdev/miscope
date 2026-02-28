import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, html
from dash.exceptions import PreventUpdate

from dashboard.components.visualization import create_empty_figure, create_graph
from dashboard.state import variant_state

# ---------------------------------------------------------------------------
# Plot IDs (all prefixed "summary-" to avoid collisions)
# ---------------------------------------------------------------------------
_lens_name = "summary" #default lens_name

# TODO: Standardize _VIEW_LIST to use a shared schema across pages
_LENS_DEF = {
    "summary": [
        "summary-loss-plot",
        "summary-freq-over-time-plot",
        "summary-spec-trajectory-plot",
        "summary-spec-freq-plot",
        "summary-attn-spec-plot",
        "summary-attn-dom-freq-plot",
        "summary-trajectory-3d-plot",
        "summary-trajectory-plot",
        "summary-trajectory-pc1-pc3-plot",
        "summary-trajectory-pc2-pc3-plot",
        "summary-velocity-plot",
        "summary-dim-trajectory-plot",
    ],
    "repr_geometry": [
        "rg-centroids-plot",
        "rg-alignment-plot",
    ],
    "neuron_dynamics": [
        "nd-trajectory-plot",
        "nd-switch-plot",
        "nd-commitment-plot",
    ],
}
_VIEW_LIST = {
    "summary-loss-plot": {"view_name": "loss_curve", "view_type": "epoch_selector"},
    "summary-freq-over-time-plot": {"view_name": "dominant_frequencies_over_time", "view_type": "epoch_selector"},
    "summary-spec-trajectory-plot": {"view_name": "specialization_trajectory", "view_type": "default_graph"},
    "summary-spec-freq-plot": {"view_name": "specialization_by_frequency", "view_type": "epoch_selector"},
    "summary-attn-spec-plot": {"view_name": "attention_specialization_trajectory", "view_type": "default_graph"},
    "summary-attn-dom-freq-plot": {"view_name": "attention_dominant_frequencies", "view_type": "default_graph"},
    "summary-trajectory-3d-plot": {"view_name": "trajectory_3d", "view_type": "default_graph"},
    "summary-trajectory-plot": {"view_name": "parameter_trajectory", "view_type": "default_graph"},
    "summary-trajectory-pc1-pc3-plot": {"view_name": "trajectory_pc1_pc3", "view_type": "default_graph"},
    "summary-trajectory-pc2-pc3-plot": {"view_name": "trajectory_pc2_pc3", "view_type": "default_graph"},
    "summary-velocity-plot": {"view_name": "parameter_velocity", "view_type": "default_graph"},
    "summary-dim-trajectory-plot": {"view_name": "dimensionality_trajectory", "view_type": "default_graph"},
    "rg-centroids-plot": {"view_name": "", "view_type":"default_graph"},
    "rg-alignment-plot": {"view_name": "", "view_type":"default_graph"},
    "nd-trajectory-plot": {"view_name": "neuron_freq_trajectory", "view_type":"epoch_selector"},
    "nd-switch-plot": {"view_name": "", "view_type":"default_graph"},
    "nd-commitment-plot": {"view_name": "", "view_type":"default_graph"},
}

# TODO: refactor to pull out common functionality across analysis pages
def _get_graph_output_list():
    print(f"_get_graph_output_list: {_lens_name}")
    graph_list = []
    for view_item in _LENS_DEF[_lens_name]:
        view_type = _VIEW_LIST[view_item].get("view_type")
        if view_type is None:
            view_type = "default_graph"
        print(f"adding graph to output list:'view_type': {view_type}, 'index': {view_item}")
        graph_list.append({'view_type': view_type, 'index': view_item})

    return graph_list

def _update_graphs(variant_data: dict | None) -> list[go.Figure]:
    print(f"_update_graphs: {_lens_name}")
    stored = variant_data or {}
    variant_name = stored.get("variant_name")
    last_field_updated = stored.get("last_field_updated")
    figures = []

    # Clear graphs if variant_name is None
    if variant_name is None:
        no_data = create_empty_figure("Select a variant")
        figures = [no_data for pid in _LENS_DEF[_lens_name]]

    if last_field_updated in ["variant_name", "epoch"]:
        #Update graphs
        for view_item in _LENS_DEF[_lens_name]:
            view_name = _VIEW_LIST[view_item].get("view_name")

            if view_name in variant_state.available_views:
                figures.append(variant_state.context.view(view_name).figure())
            else:
                figures.append(create_empty_figure("No view found"))
    else:
        raise PreventUpdate
    
    return figures

def create_variant_lens_nav(lens_name: str = "summary") -> html.Div:
    _lens_name = lens_name
    print(f"create_variant_lens_nav: {_lens_name}")
    return html.Div()

def create_variant_lens_layout(lens_name: str = "summary") -> html.Div:    
    _lens_name = lens_name
    print(f"create_variant_lens_layout: {_lens_name}")
    graph_list = []
    for view_item in _LENS_DEF[lens_name]:
        view_type = _VIEW_LIST[view_item].get("view_name")
        if view_type is None:
            view_type = "default_graph"
        graph_list.append(dbc.Col(create_graph(view_item, "300px", view_type)))

    return html.Div(children=graph_list)

def register_variant_lens_callbacks(app: Dash) -> None:
    """Register all callbacks for the Summary page."""

    @app.callback(
        *[Output(pid, "figure") for pid in _get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data")
    )
    def on_lens_data_change(modified_timestamp: str | None, variant_data: dict | None):
        print("on_lens_data_change")
        return _update_graphs(variant_data)

