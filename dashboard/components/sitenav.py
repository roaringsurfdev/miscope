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
            dbc.NavItem(dbc.NavLink("Visualization", href="/visualization")),
            dbc.NavItem(dbc.NavLink("Multi-Stream", href="/multistream")),
            dbc.NavItem(dbc.NavLink("Summary", href="/summary")),
            dbc.NavItem(dbc.NavLink("Neuron Dynamics", href="/neuron-dynamics")),
            dbc.NavItem(dbc.NavLink("Repr Geometry", href="/repr-geometry")),
            dbc.NavItem(dbc.NavLink("Dimensionality", href="/dimensionality")),
            dbc.NavItem(dbc.NavLink("Centroid DMD", href="/centroid-dmd")),
            dbc.NavItem(dbc.NavLink("Training", href="/training")),
            dbc.NavItem(dbc.NavLink("Analysis Run", href="/analysis-run")),
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
    from dashboard.pages.multistream import (
        create_multistream_page_layout,
        create_multistream_page_nav,
    )
    from dashboard.pages.neuron_dynamics import (
        create_neuron_dynamics_page_layout,
        create_neuron_dynamics_page_nav,
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
        if pathname == "/multistream":
            return [create_multistream_page_nav(), create_multistream_page_layout()]
        if pathname == "/neuron-dynamics":
            return [create_neuron_dynamics_page_nav(), create_neuron_dynamics_page_layout()]
        elif pathname == "/repr-geometry":
            return [create_repr_geometry_page_nav(), create_repr_geometry_page_layout()]
        elif pathname == "/summary":
            return [create_summary_page_nav(), create_summary_page_layout()]
        elif pathname == "/dimensionality":
            return [create_dimensionality_page_nav(), create_dimensionality_page_layout()]
        elif pathname == "/centroid-dmd":
            return [create_centroid_dmd_nav(), create_centroid_dmd_layout()]
        elif pathname == "/training":
            return [create_training_page_nav(), create_training_page_layout()]
        elif pathname == "/analysis-run":
            return [create_analysis_run_page_nav(), create_analysis_run_page_layout()]
        else:
            return [create_visualization_page_nav(), create_visualization_page_layout()]
