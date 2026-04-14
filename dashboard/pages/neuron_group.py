"""Neuron Group PCA page.

Views into within-frequency-group coordination in W_in, organized in three sections:

PCA Dynamics (cross-epoch trajectories):
  - PC cohesion timeline: PC1+PC2+PC3 cumulative variance explained per group
  - Spread timeline: mean L2 distance from group centroid per group

Neuron Positions in PCA Space (epoch-parameterized scatter/trajectory):
  - PC1 vs PC2 scatter (group-colored)
  - PC1 vs PC2 scatter colored by dominant-frequency purity
  - PC1 vs PC2 vs PC3 3D scatter
  - All groups multi-panel
  - Neuron trajectory (all epochs)
  - PCA angle polar histogram

Residue Class Graduation:
  - Within-class graduation spread
  - Class cohesion heatmap
"""

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html

from dashboard.components.analysis_page import AnalysisPageGraphManager

_VIEW_LIST = {
    # PCA dynamics (cross-epoch)
    "cohesion-timeline": {
        "view_name": "neuron_group.pca_cohesion",
        "view_type": "epoch_selector",
    },
    "spread-timeline": {
        "view_name": "neuron_group.spread",
        "view_type": "epoch_selector",
    },
    # Intragroup Manifold Analysis
    "intragroup-manifold-timeline": {
        "view_name": "intragroup_manifold.timeseries",
        "view_type": "epoch_selector",
    },
    "intragroup-manifold-summary": {
        "view_name": "intragroup_manifold.summary",
        "view_type": "default_graph",
    },
    "intragroup-manifold-surface-fit": {
        "view_name": "intragroup_manifold.surface_fit",
        "view_type": "default_graph",
    },
    # Neuron positions in PCA space (epoch-parameterized)
    "pca-scatter": {
        "view_name": "neuron_group.scatter",
        "view_type": "default_graph",
    },
    "scatter-purity": {
        "view_name": "neuron_group.scatter_purity",
        "view_type": "default_graph",
    },
    "scatter-3d": {
        "view_name": "neuron_group.scatter_3d",
        "view_type": "default_graph",
    },
    "all-groups": {
        "view_name": "neuron_group.all_groups",
        "view_type": "default_graph",
    },
    "trajectory": {
        "view_name": "neuron_group.trajectory",
        "view_type": "default_graph",
    },
    "polar-histogram": {
        "view_name": "neuron_group.polar_histogram",
        "view_type": "default_graph",
    },
    # Residue class graduation
    "graduation-spread": {
        "view_name": "neuron_group.graduation_spread",
        "view_type": "epoch_selector",
    },
    "graduation-cohesion": {
        "view_name": "neuron_group.graduation_cohesion",
        "view_type": "epoch_selector",
    },
}

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST, "ng")


def create_neuron_group_page_nav(app: Dash) -> html.Div:
    return html.Div(children=[])


def create_neuron_group_page_layout(app: Dash) -> html.Div:
    return html.Div(
        children=[
            html.H4("Neuron Group PCA", className="mb-3"),
            html.H6("PCA Dynamics", className="text-muted mb-2"),
            dbc.Row(dbc.Col(_graph_manager.create_graph("cohesion-timeline", "440px"))),
            dbc.Row(dbc.Col(_graph_manager.create_graph("spread-timeline", "420px"))),
            dbc.Row(dbc.Col(_graph_manager.create_graph("intragroup-manifold-timeline", "420px"))),
            dbc.Row(dbc.Col(_graph_manager.create_graph("intragroup-manifold-summary", "600px"))),
            dbc.Row(dbc.Col(_graph_manager.create_graph("intragroup-manifold-surface-fit", "600px"))),
            html.Hr(className="my-3"),
            html.H6("Neuron Positions in PCA Space", className="text-muted mb-2"),
            dbc.Row(
                [
                    dbc.Col(_graph_manager.create_graph("pca-scatter", "540px"), width=6),
                    dbc.Col(_graph_manager.create_graph("scatter-purity", "540px"), width=6),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(_graph_manager.create_graph("scatter-3d", "560px"), width=6),
                    dbc.Col(_graph_manager.create_graph("polar-histogram", "560px"), width=6),
                ]
            ),
            dbc.Row(dbc.Col(_graph_manager.create_graph("trajectory", "580px"))),
            dbc.Row(dbc.Col(_graph_manager.create_graph("all-groups", "700px"))),
            html.Hr(className="my-3"),
            html.H6("Residue Class Graduation", className="text-muted mb-2"),
            dbc.Row(
                [
                    dbc.Col(_graph_manager.create_graph("graduation-spread", "480px"), width=6),
                    dbc.Col(_graph_manager.create_graph("graduation-cohesion", "500px"), width=6),
                ]
            ),
        ]
    )


def register_neuron_group_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Neuron Group PCA page."""

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_neuron_group_data_change(_modified_timestamp: str | None, variant_data: dict | None):
        return _graph_manager.update_graphs(variant_data, None)
