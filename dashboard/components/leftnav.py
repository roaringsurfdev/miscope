"""
Analysis Left Navigation
    Will contain
        Primary Variant Selection (components.variant_selector)
            Family Drop-Down (Single Select)
            Variant List (Single Select) displayed as [domain parameter list]
            Epoch Slider
            Epoch Index Text Input
            Load Button (Intentional Commit of changed values)
        Visualization-Specific Selections (selectors found in specific lens pages)
            Flatness Metric
            Neuron Sort Order (Neuron Dominant Frequency Over Training)
        Common Analysis Themes
            Neuron Index
            Activation Component Group
            Activation Site
            Weight Component Group
"""

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html

from dashboard.components.variant_selector import get_variant_selector

_SIDEBAR_STYLE = {
    "width": "280px",
    "minWidth": "280px",
    "padding": "20px",
    "backgroundColor": "#f8f9fa",
    "overflowY": "auto",
    "borderRight": "1px solid #dee2e6",
    "height": "100vh",
    "position": "sticky",
    "top": "0",
}

_COLLAPSED_SIDEBAR_STYLE = {
    "width": "40px",
    "minWidth": "40px",
    "padding": "10px 5px",
    "backgroundColor": "#f8f9fa",
    "borderRight": "1px solid #dee2e6",
    "height": "100vh",
    "position": "sticky",
    "top": "0",
    "display": "none",
}


def create_sidebar() -> html.Div:
    print("create_sidebar")
    """Create the collapsible sidebar with all controls."""
    return html.Div(
        id="sidebar",
        children=[
            html.Div(
                className="sidebar-header",
                children=[
                    html.H5("Controls", className="mb-0"),
                    dbc.Button(
                        "\u2190",
                        id="sidebar-toggle",
                        size="sm",
                        color="secondary",
                        outline=True,
                        className="ms-auto",
                    ),
                ],
                style={"display": "flex", "alignItems": "center"},
            ),
            # Family selector
            get_variant_selector(),
            html.Hr(),
            # Placeholder for page-specific left nav items
            html.Div(
                id="page_left_nav",
                children=[],
            ),
        ],
        style=_SIDEBAR_STYLE,
    )


def create_collapsed_sidebar() -> html.Div:
    print("create_collapsed_sidebar")
    """Create the collapsed sidebar status bar."""
    return html.Div(
        id="sidebar-collapsed",
        children=[
            dbc.Button(
                "\u2192",
                id="sidebar-expand",
                size="sm",
                color="secondary",
                outline=True,
            ),
            html.Div(
                id="collapsed-status",
                children="",
                className="small text-muted mt-2",
                style={"writingMode": "vertical-lr", "textOrientation": "mixed"},
            ),
        ],
        style=_COLLAPSED_SIDEBAR_STYLE,
    )


def register_left_nav_callbacks(app: Dash) -> None:
    print("register_layout_callbacks")
    """Register URL routing callback."""

    # --- Sidebar toggle ---
    @app.callback(
        Output("sidebar", "style"),
        Output("sidebar-collapsed", "style"),
        Output("collapsed-status", "children"),
        Input("sidebar-toggle", "n_clicks"),
        Input("sidebar-expand", "n_clicks"),
        State("sidebar", "style"),
        prevent_initial_call=True,
    )
    def toggle_sidebar(collapse_clicks, expand_clicks, current_style):
        from dash import ctx

        sidebar_visible = {
            "width": "280px",
            "minWidth": "280px",
            "padding": "20px",
            "backgroundColor": "#f8f9fa",
            "overflowY": "auto",
            "borderRight": "1px solid #dee2e6",
            "height": "100vh",
            "position": "sticky",
            "top": "0",
        }
        sidebar_hidden = {**sidebar_visible, "display": "none"}

        collapsed_visible = {
            "width": "40px",
            "minWidth": "40px",
            "padding": "10px 5px",
            "backgroundColor": "#f8f9fa",
            "borderRight": "1px solid #dee2e6",
            "height": "100vh",
            "position": "sticky",
            "top": "0",
            "display": "block",
        }
        collapsed_hidden = {**collapsed_visible, "display": "none"}

        if ctx.triggered_id == "sidebar-toggle":
            return sidebar_hidden, collapsed_visible, ""
        else:
            return sidebar_visible, collapsed_hidden, ""
