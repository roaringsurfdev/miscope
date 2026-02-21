"""Site-level navigation for the Dash dashboard.

Provides a top navbar with links to Visualization, Training, and Analysis Run
pages, plus URL-driven routing callback.
"""
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, html

from dashboard_v2.version import __version__


def create_sitenav() -> dbc.NavbarSimple:
    """Create the top navigation bar."""
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Summary", href="/summary")),
            dbc.NavItem(dbc.NavLink("Neuron Dynamics", href="/neuron-dynamics")),
            dbc.NavItem(dbc.NavLink("Repr Geometry", href="/repr-geometry")),
        ],
        brand=f"MechInterp Scope v{__version__}",
        brand_href="/",
        color="dark",
        dark=True,
        sticky="fixed",
        fluid=True,
    )


def register_sitenav_callbacks(app: Dash) -> None:
    print("register_routing_callbacks")
    """Register URL routing callback."""
    from dashboard_temp.pages.neuron_dynamics import (
        create_neuron_dynamics_page_layout,
        create_neuron_dynamics_page_nav,
    )
    from dashboard_temp.pages.repr_geometry import (
        create_repr_geometry_page_layout,
        create_repr_geometry_page_nav,
    )
    from dashboard_temp.pages.summary import create_summary_page_layout, create_summary_page_nav

    @app.callback(
        Output("page_left_nav", "children"),
        Output("page-content", "children"),
        Input("url", "pathname"),
    )
    def display_page(pathname: str | None) -> list[html.Div]:
        print("display_page")
        if pathname == "/neuron-dynamics":
            return [create_neuron_dynamics_page_nav(), create_neuron_dynamics_page_layout()]
            #return create_neuron_dynamics_page_layout()
        elif pathname == "/repr-geometry":
            return [create_repr_geometry_page_nav(), create_repr_geometry_page_layout()]
        elif pathname == "/summary":
            return [create_summary_page_nav(), create_summary_page_layout()]
        else:
            return [create_summary_page_nav(), create_summary_page_layout()]
