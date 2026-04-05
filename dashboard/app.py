# imports
import logging

import dash_bootstrap_components as dbc
from dash import Dash

from dashboard.components.export_panel import register_export_callbacks
from dashboard.components.leftnav import register_left_nav_callbacks
from dashboard.components.sitenav import register_sitenav_callbacks
from dashboard.components.variant_context_bar import register_variant_context_bar_callbacks
from dashboard.components.variant_selector import register_variant_selector_callbacks
from dashboard.layout import create_default_layout
from dashboard.pages.activation_heatmaps import register_activation_heatmap_page_callbacks
from dashboard.pages.analysis_run import register_analysis_run_page_callbacks
from dashboard.pages.centroid_dmd import register_centroid_dmd_callbacks
from dashboard.pages.checkpoint_schedule import register_checkpoint_schedule_page_callbacks
from dashboard.pages.dimensionality import register_dimensionality_page_callbacks
from dashboard.pages.initialization_sweep import register_initialization_sweep_page_callbacks
from dashboard.pages.input_trace import register_input_trace_page_callbacks
from dashboard.pages.intervention_check import register_intervention_check_callbacks
from dashboard.pages.loss_landscape import register_loss_landscape_page_callbacks
from dashboard.pages.multistream import register_multistream_page_callbacks
from dashboard.pages.neuron_dynamics import register_neuron_dynamics_page_callbacks
from dashboard.pages.neuron_group import register_neuron_group_page_callbacks
from dashboard.pages.peer_comparison import register_peer_comparison_page_callbacks
from dashboard.pages.repr_geometry import register_repr_geometry_page_callbacks
from dashboard.pages.summary import register_summary_page_callbacks
from dashboard.pages.training import register_training_page_callbacks
from dashboard.pages.transient_frequency import register_transient_page_callbacks
from dashboard.pages.variant_table import register_variant_table_page_callbacks
from dashboard.pages.viability_certificate import register_viability_certificate_page_callbacks
from dashboard.pages.visualization import register_visualization_page_callbacks


def create_app() -> Dash:
    """Create and configure the Dash application."""
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )
    app.server.logger.setLevel(logging.INFO)
    app.title = "MechInterp Scope"
    app.layout = create_default_layout(app)
    # core application callbacks
    register_export_callbacks(app)
    register_left_nav_callbacks(app)
    register_sitenav_callbacks(app)
    register_variant_selector_callbacks(app)
    register_variant_context_bar_callbacks(app)
    # page-specific callbacks
    register_activation_heatmap_page_callbacks(app)
    register_analysis_run_page_callbacks(app)
    register_centroid_dmd_callbacks(app)
    register_neuron_dynamics_page_callbacks(app)
    register_repr_geometry_page_callbacks(app)
    register_summary_page_callbacks(app)
    register_dimensionality_page_callbacks(app)
    register_loss_landscape_page_callbacks(app)
    register_multistream_page_callbacks(app)
    register_peer_comparison_page_callbacks(app)
    register_checkpoint_schedule_page_callbacks(app)
    register_training_page_callbacks(app)
    register_visualization_page_callbacks(app)
    register_intervention_check_callbacks(app)
    register_input_trace_page_callbacks(app)
    register_neuron_group_page_callbacks(app)
    register_transient_page_callbacks(app)
    register_variant_table_page_callbacks(app)
    register_initialization_sweep_page_callbacks(app)
    register_viability_certificate_page_callbacks(app)
    return app


def main() -> None:
    """Entry point for the Dash dashboard."""
    app = create_app()
    app.run(debug=False, host="0.0.0.0", port=8060)


if __name__ == "__main__":
    main()
