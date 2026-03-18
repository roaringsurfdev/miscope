# imports
import logging

import dash_bootstrap_components as dbc
from dash import Dash

from dashboard.components.leftnav import register_left_nav_callbacks
from dashboard.components.sitenav import register_sitenav_callbacks
from dashboard.components.variant_selector import register_variant_selector_callbacks
from dashboard.layout import create_default_layout
from dashboard.pages.analysis_run import register_analysis_run_page_callbacks
from dashboard.pages.centroid_dmd import register_centroid_dmd_callbacks
from dashboard.pages.dimensionality import register_dimensionality_page_callbacks
from dashboard.pages.intervention_check import register_intervention_check_callbacks
from dashboard.pages.multistream import register_multistream_page_callbacks
from dashboard.pages.peer_comparison import register_peer_comparison_page_callbacks
from dashboard.pages.neuron_dynamics import register_neuron_dynamics_page_callbacks
from dashboard.pages.repr_geometry import register_repr_geometry_page_callbacks
from dashboard.pages.summary import register_summary_page_callbacks
from dashboard.pages.training import register_training_page_callbacks
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
    register_left_nav_callbacks(app)
    register_sitenav_callbacks(app)
    register_variant_selector_callbacks(app)
    # page-specific callbacks
    register_analysis_run_page_callbacks(app)
    register_centroid_dmd_callbacks(app)
    register_neuron_dynamics_page_callbacks(app)
    register_repr_geometry_page_callbacks(app)
    register_summary_page_callbacks(app)
    register_dimensionality_page_callbacks(app)
    register_multistream_page_callbacks(app)
    register_peer_comparison_page_callbacks(app)
    register_training_page_callbacks(app)
    register_visualization_page_callbacks(app)
    register_intervention_check_callbacks(app)
    return app


def main() -> None:
    """Entry point for the Dash dashboard."""
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=8060)


if __name__ == "__main__":
    main()
