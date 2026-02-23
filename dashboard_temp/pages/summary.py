import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, html
from dash.exceptions import PreventUpdate

from dashboard_temp.components.visualization import create_empty_figure, create_graph
from dashboard_temp.state import variant_state

# ---------------------------------------------------------------------------
# Plot IDs (all prefixed "summary-" to avoid collisions)
# ---------------------------------------------------------------------------

_PAGE_ID = "summary"
_PAGE_PREFIX = "summary"

_PLOT_IDS = [
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
]
_VIEW_LIST = {
    "summary-loss-plot": {"view_name": "loss_curve", "view_type": "epoch_selector"},
    "summary-freq-over-time-plot": {"view_name": "dominant_frequencies_over_time", "view_type": "epoch_selector"},
    "summary-spec-trajectory-plot": {"view_name": "specialization_trajectory", "view_type": "default_graph"},
    "summary-spec-freq-plot": {"view_name": "specialization_by_frequency", "view_type": "default_graph"},
    "summary-attn-spec-plot": {"view_name": "attention_specialization_trajectory", "view_type": "default_graph"},
    "summary-attn-dom-freq-plot": {"view_name": "attention_dominant_frequencies", "view_type": "default_graph"},
    "summary-trajectory-3d-plot": {"view_name": "trajectory_3d", "view_type": "default_graph"},
    "summary-trajectory-plot": {"view_name": "parameter_trajectory", "view_type": "default_graph"},
    "summary-trajectory-pc1-pc3-plot": {"view_name": "trajectory_pc1_pc3", "view_type": "default_graph"},
    "summary-trajectory-pc2-pc3-plot": {"view_name": "trajectory_pc2_pc3", "view_type": "default_graph"},
    "summary-velocity-plot": {"view_name": "parameter_velocity", "view_type": "default_graph"},
    "summary-dim-trajectory-plot": {"view_name": "dimensionality_trajectory", "view_type": "default_graph"},
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

def _update_graphs(variant_data: dict | None) -> list[go.Figure]:
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

def create_summary_page_nav() -> html.Div:
    print("create_summary_page_nav")
    return html.Div()

def create_summary_page_layout() -> html.Div:
    print("create_summary_page_layout")
    #set_props("variant-selector-store", {"data": {"stale_data": "1"}})
    return html.Div(
        children= [
            # Loss curve (full width)
            dbc.Row(dbc.Col(create_graph("summary-loss-plot", "300px", _get_graph_view_type("summary-loss-plot")))),
            # Embedding Fourier over time (full width)
            dbc.Row(dbc.Col(create_graph("summary-freq-over-time-plot", "350px", _get_graph_view_type("summary-freq-over-time-plot")))),
            # Neuron specialization | Attention head specialization
            dbc.Row(
                [
                    dbc.Col(create_graph("summary-spec-trajectory-plot", "350px", _get_graph_view_type("summary-spec-trajectory-plot")), width=7),
                    dbc.Col(create_graph("summary-attn-spec-plot", "350px", _get_graph_view_type("summary-attn-spec-plot")), width=5),
                ]
            ),
            # Specialized neurons by frequency (full width)
            dbc.Row(dbc.Col(create_graph("summary-spec-freq-plot", "400px", _get_graph_view_type("summary-spec-freq-plot")))),
            # Attention dominant frequencies (full width)
            dbc.Row(dbc.Col(create_graph("summary-attn-dom-freq-plot", "300px", _get_graph_view_type("summary-attn-dom-freq-plot")))),
            # Trajectory 3D (full width)
            dbc.Row(dbc.Col(create_graph("summary-trajectory-3d-plot", "550px", _get_graph_view_type("summary-trajectory-3d-plot")))),
            # PC1/PC2 | PC1/PC3 | PC2/PC3
            dbc.Row(
                [
                    dbc.Col(create_graph("summary-trajectory-plot", "400px", _get_graph_view_type("summary-trajectory-plot")), width=4),
                    dbc.Col(create_graph("summary-trajectory-pc1-pc3-plot", "400px", _get_graph_view_type("summary-trajectory-pc1-pc3-plot")), width=4),
                    dbc.Col(create_graph("summary-trajectory-pc2-pc3-plot", "400px", _get_graph_view_type("summary-trajectory-pc2-pc3-plot")), width=4),
                ]
            ),
            # Component velocity | Effective dimensionality
            dbc.Row(
                [
                    dbc.Col(create_graph("summary-velocity-plot", "350px", _get_graph_view_type("summary-velocity-plot")), width=6),
                    dbc.Col(create_graph("summary-dim-trajectory-plot", "350px", _get_graph_view_type("summary-dim-trajectory-plot")), width=6),
                ]
            ),
        ],
    )

def register_summary_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Summary page."""

    @app.callback(
        *[Output(pid, "figure") for pid in _get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data")
    )
    def on_summary_data_change(modified_timestamp: str | None, variant_data: dict | None):
        print("on_summary_data_change")
        return _update_graphs(variant_data)

