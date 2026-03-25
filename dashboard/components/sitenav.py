"""Site-level navigation for the Dash dashboard.

Provides a top navbar with links to all pages, plus URL-driven routing callback.
"""

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, html

from dashboard.version import __version__


def create_sitenav() -> dbc.NavbarSimple:
    """Create the top navigation bar."""
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/", active="exact")),
            dbc.DropdownMenu(
                [
                    dbc.DropdownMenuItem(
                        "Frequency Specialization", href="/frequency-specialization"
                    ),
                    dbc.DropdownMenuItem("Geometry", href="/geometry"),
                    dbc.DropdownMenuItem("Neuron Competition", href="/neuron-competition"),
                    dbc.DropdownMenuItem("PCA", href="/pca"),
                    dbc.DropdownMenuItem("Activations", href="/activations"),
                    dbc.DropdownMenuItem("Centroid DMD", href="/centroid-dmd"),
                    dbc.DropdownMenuItem("Loss Landscape", href="/loss-landscape"),
                    dbc.DropdownMenuItem("Input Trace", href="/input-trace"),
                    dbc.DropdownMenuItem("Neuron Groups", href="/neuron-group"),
                ],
                nav=True,
                in_navbar=True,
                label="Variant Analysis",
            ),
            dbc.DropdownMenu(
                [
                    dbc.DropdownMenuItem("Peer Comparison", href="/peer-comparison"),
                ],
                nav=True,
                in_navbar=True,
                label="Cross-Variant Analysis",
            ),
            dbc.DropdownMenu(
                [
                    dbc.DropdownMenuItem("Training", href="/training"),
                    dbc.DropdownMenuItem("Analysis Run", href="/analysis-run"),
                    dbc.DropdownMenuItem("Intervention Check", href="/intervention-check"),
                ],
                nav=True,
                in_navbar=True,
                label="Training",
            ),
        ],
        brand=f"MechInterp Scope v{__version__}",
        brand_href="/",
        color="dark",
        dark=True,
        sticky="fixed",
        fluid=True,
    )


def register_sitenav_callbacks(app: Dash) -> None:
    """Register URL routing callback."""
    from dashboard.pages.activation_heatmaps import (
        create_activation_heatmap_page_layout,
        create_activation_heatmap_page_nav,
    )
    from dashboard.pages.analysis_run import (
        create_analysis_run_page_layout,
        create_analysis_run_page_nav,
    )
    from dashboard.pages.centroid_dmd import (
        create_centroid_dmd_layout,
        create_centroid_dmd_nav,
    )
    from dashboard.pages.dimensionality import (
        create_dimensionality_page_layout,
        create_dimensionality_page_nav,
    )
    from dashboard.pages.intervention_check import (
        create_intervention_check_page_layout,
        create_intervention_check_page_nav,
    )
    from dashboard.pages.loss_landscape import (
        create_loss_landscape_page_layout,
        create_loss_landscape_page_nav,
    )
    from dashboard.pages.multistream import (
        create_multistream_page_layout,
        create_multistream_page_nav,
    )
    from dashboard.pages.neuron_dynamics import (
        create_neuron_dynamics_page_layout,
        create_neuron_dynamics_page_nav,
    )
    from dashboard.pages.peer_comparison import (
        create_peer_comparison_page_layout,
        create_peer_comparison_page_nav,
    )
    from dashboard.pages.repr_geometry import (
        create_repr_geometry_page_layout,
        create_repr_geometry_page_nav,
    )
    from dashboard.pages.summary import create_summary_page_layout, create_summary_page_nav
    from dashboard.pages.training import (
        create_training_page_layout,
        create_training_page_nav,
    )
    from dashboard.pages.input_trace import (
        create_input_trace_page_layout,
        create_input_trace_page_nav,
    )
    from dashboard.pages.neuron_group import (
        create_neuron_group_page_layout,
        create_neuron_group_page_nav,
    )
    from dashboard.pages.visualization import (
        create_visualization_page_layout,
        create_visualization_page_nav,
    )

    @app.callback(
        Output("page_left_nav", "children"),
        Output("page-content", "children"),
        Input("url", "pathname"),
    )
    def display_page(pathname: str | None) -> list[html.Div]:
        if pathname == "/visualization":
            return [create_visualization_page_nav(app), create_visualization_page_layout(app)]
        if pathname == "/activations":
            return [
                create_activation_heatmap_page_nav(app),
                create_activation_heatmap_page_layout(app),
            ]
        if pathname == "/frequency-specialization":
            return [create_multistream_page_nav(app), create_multistream_page_layout(app)]
        if pathname == "/loss-landscape":
            return [create_loss_landscape_page_nav(app), create_loss_landscape_page_layout(app)]
        if pathname == "/peer-comparison":
            return [create_peer_comparison_page_nav(app), create_peer_comparison_page_layout(app)]
        if pathname == "/neuron-competition":
            return [create_neuron_dynamics_page_nav(app), create_neuron_dynamics_page_layout(app)]
        elif pathname == "/geometry":
            return [create_repr_geometry_page_nav(app), create_repr_geometry_page_layout(app)]
        elif pathname == "/summary":
            return [create_summary_page_nav(app), create_summary_page_layout(app)]
        elif pathname == "/pca":
            return [create_dimensionality_page_nav(app), create_dimensionality_page_layout(app)]
        elif pathname == "/centroid-dmd":
            return [create_centroid_dmd_nav(app), create_centroid_dmd_layout(app)]
        elif pathname == "/training":
            return [create_training_page_nav(app), create_training_page_layout(app)]
        elif pathname == "/analysis-run":
            return [create_analysis_run_page_nav(app), create_analysis_run_page_layout(app)]
        elif pathname == "/intervention-check":
            return [
                create_intervention_check_page_nav(app),
                create_intervention_check_page_layout(app),
            ]
        elif pathname == "/input-trace":
            return [create_input_trace_page_nav(app), create_input_trace_page_layout(app)]
        elif pathname == "/neuron-group":
            return [create_neuron_group_page_nav(app), create_neuron_group_page_layout(app)]
        else:
            # Multistream is now the default page.
            return [create_multistream_page_nav(app), create_multistream_page_layout(app)]
