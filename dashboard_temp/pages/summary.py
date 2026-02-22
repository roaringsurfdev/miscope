import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html
from dash.exceptions import PreventUpdate

from dashboard_temp.state import variant_state
from dashboard_temp.components.visualization import create_empty_figure, create_graph

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
    "summary-loss-plot": "loss_curve",
    "summary-freq-over-time-plot": "dominant_frequencies_over_time",
    "summary-spec-trajectory-plot": "specialization_trajectory",
    "summary-spec-freq-plot": "specialization_by_frequency",
    "summary-attn-spec-plot": "attention_specialization_trajectory",
    "summary-attn-dom-freq-plot": "attention_dominant_frequencies",
    "summary-trajectory-3d-plot": "trajectory_3d",
    "summary-trajectory-plot": "parameter_trajectory",
    "summary-trajectory-pc1-pc3-plot": "trajectory_pc1_pc3",
    "summary-trajectory-pc2-pc3-plot": "trajectory_pc2_pc3",
    "summary-velocity-plot": "parameter_velocity",
    "summary-dim-trajectory-plot": "dimensionality_trajectory",
}

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
        for view_id, view_name in _VIEW_LIST.items():
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
            dbc.Row(dbc.Col(create_graph("summary-loss-plot", "300px"))),
            # Embedding Fourier over time (full width)
            dbc.Row(dbc.Col(create_graph("summary-freq-over-time-plot", "350px"))),
            # Neuron specialization | Attention head specialization
            dbc.Row(
                [
                    dbc.Col(create_graph("summary-spec-trajectory-plot", "350px"), width=7),
                    dbc.Col(create_graph("summary-attn-spec-plot", "350px"), width=5),
                ]
            ),
            # Specialized neurons by frequency (full width)
            dbc.Row(dbc.Col(create_graph("summary-spec-freq-plot", "400px"))),
            # Attention dominant frequencies (full width)
            dbc.Row(dbc.Col(create_graph("summary-attn-dom-freq-plot", "300px"))),
            # Trajectory 3D (full width)
            dbc.Row(dbc.Col(create_graph("summary-trajectory-3d-plot", "550px"))),
            # PC1/PC2 | PC1/PC3 | PC2/PC3
            dbc.Row(
                [
                    dbc.Col(create_graph("summary-trajectory-plot", "400px"), width=4),
                    dbc.Col(create_graph("summary-trajectory-pc1-pc3-plot", "400px"), width=4),
                    dbc.Col(create_graph("summary-trajectory-pc2-pc3-plot", "400px"), width=4),
                ]
            ),
            # Component velocity | Effective dimensionality
            dbc.Row(
                [
                    dbc.Col(create_graph("summary-velocity-plot", "350px"), width=6),
                    dbc.Col(create_graph("summary-dim-trajectory-plot", "350px"), width=6),
                ]
            ),
        ],
    )

def register_summary_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Summary page."""

    @app.callback(
        *[Output(pid, "figure") for pid in _VIEW_LIST.keys()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data")
    )
    def on_summary_data_change(modified_timestamp: str | None, variant_data: dict | None):
        print("on_summary_data_change")
        return _update_graphs(variant_data)

