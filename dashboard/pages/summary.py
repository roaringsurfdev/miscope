import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html

from dashboard.components.analysis_page import AnalysisPageGraphManager

# TODO: Standardize _VIEW_LIST to use a shared schema across pages
_VIEW_LIST = {
    "training-loss-curves": {
        "view_name": "training.metadata.loss_curves",
        "view_type": "epoch_selector",
    },
    "freq-over-time-plot": {
        "view_name": "activations.mlp.dominant_frequencies_over_time",
        "view_type": "epoch_selector",
    },
    "neuron_frequency_range": {
        "view_name": "activations.mlp.neuron_frequency_range",
        "view_type": "default_graph",
    },
    "spec-freq-plot": {
        "view_name": "activations.mlp.neuron_frequency_specialization",
        "view_type": "epoch_selector",
    },
    "attn-spec-plot": {
        "view_name": "activations.attention.frequency_clusters",
        "view_type": "default_graph",
    },
    "attn-dom-freq-plot": {
        "view_name": "activations.attention.head_frequency_range",
        "view_type": "default_graph",
    },
    "parameters-pca-3d-scatter": {
        "view_name": "parameters.pca.scatter_3d",
        "view_type": "default_graph",
    },
    "parameters-pca-pc1-pc2": {"view_name": "parameters.pca.pc1_pc2", "view_type": "default_graph"},
    "parameters-pca-pc1-pc3": {
        "view_name": "parameters.pca.pc1_pc3",
        "view_type": "default_graph",
    },
    "parameters-pca-pc2-pc3": {
        "view_name": "parameters.pca.pc2_pc3",
        "view_type": "default_graph",
    },
    "velocity-plot": {"view_name": "parameters.pca.velocity", "view_type": "default_graph"},
    "dim-trajectory-plot": {
        "view_name": "parameters.effective_dimensionality",
        "view_type": "default_graph",
    },
}

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST, "sum")


def create_summary_page_nav(app: Dash) -> html.Div:
    app.server.logger.debug("create_summary_page_nav")
    return html.Div()


def create_summary_page_layout(app: Dash) -> html.Div:
    app.server.logger.debug("create_summary_page_layout")
    # set_props("variant-selector-store", {"data": {"stale_data": "1"}})
    return html.Div(
        children=[
            # Loss curve (full width)
            dbc.Row(dbc.Col(_graph_manager.create_graph("training-loss-curves", "300px"))),
            # Embedding Fourier over time (full width)
            dbc.Row(dbc.Col(_graph_manager.create_graph("freq-over-time-plot", "350px"))),
            # Neuron specialization | Attention head specialization
            dbc.Row(
                [
                    dbc.Col(
                        _graph_manager.create_graph("neuron_frequency_range", "350px"),
                        width=7,
                    ),
                    dbc.Col(_graph_manager.create_graph("attn-spec-plot", "350px"), width=5),
                ]
            ),
            # Specialized neurons by frequency (full width)
            dbc.Row(dbc.Col(_graph_manager.create_graph("spec-freq-plot", "400px"))),
            # Attention dominant frequencies (full width)
            dbc.Row(dbc.Col(_graph_manager.create_graph("attn-dom-freq-plot", "300px"))),
            # Trajectory 3D (full width)
            dbc.Row(dbc.Col(_graph_manager.create_graph("parameters-pca-3d-scatter", "550px"))),
            # PC1/PC2 | PC1/PC3 | PC2/PC3
            dbc.Row(
                [
                    dbc.Col(
                        _graph_manager.create_graph("parameters-pca-pc1-pc2", "400px"), width=4
                    ),
                    dbc.Col(
                        _graph_manager.create_graph("parameters-pca-pc1-pc3", "400px"),
                        width=4,
                    ),
                    dbc.Col(
                        _graph_manager.create_graph("parameters-pca-pc2-pc3", "400px"),
                        width=4,
                    ),
                ]
            ),
            # Component velocity | Effective dimensionality
            dbc.Row(
                [
                    dbc.Col(_graph_manager.create_graph("velocity-plot", "350px"), width=6),
                    dbc.Col(_graph_manager.create_graph("dim-trajectory-plot", "350px"), width=6),
                ]
            ),
        ],
    )


def register_summary_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Summary page."""
    app.server.logger.debug("register_summary_page_callbacks")

    @app.callback(
        *[Output(pid, "figure") for pid in _graph_manager.get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_summary_data_change(modified_timestamp: str | None, variant_data: dict | None):
        app.server.logger.debug("on_summary_data_change")
        return _graph_manager.update_graphs(variant_data)
