"""Layout components for the Dash dashboard.

Defines the sidebar (controls) and content area (plots) layout.
All 18 Analysis tab visualizations organized by rendering group.

REQ_040: create_layout() now wraps pages in a navbar + URL-driven container.
The visualization layout (sidebar + plots) moved to create_visualization_layout().
"""

from dash import dcc, html

from dashboard.components.leftnav import create_collapsed_sidebar, create_sidebar
from dashboard.components.sitenav import create_sitenav

# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

_PAGE_CONTENT_STYLE = {
    "flex": "1",
    "padding": "20px",
    "overflowY": "auto",
    "height": "100vh",
}

_FLEX_WRAPPER_STYLE = {
    "display": "flex",
    "height": "calc(100vh - 56px)",
    "overflow": "hidden",
}

def create_page_content() -> html.Div:
    print("create_page_content")
    """Create the scrollable content area with all 18 plot containers.

    Organized by rendering group, matching the Gradio dashboard order.
    """
    return html.Div(
        id="page-content",
        children=[
            html.H4("Content Area", className="mb-3"),
        ],
        style=_PAGE_CONTENT_STYLE,
    )

def create_default_layout() -> html.Div:
    print("create_default_layout")
    """Create the visualization page layout (sidebar + plots)."""
    return html.Div(
        [
            dcc.Location(id="url", refresh=False),
            create_sitenav(),
            html.Div(
                id="default_page_layout",
                children=[
                    create_sidebar(),
                    create_collapsed_sidebar(),
                    create_page_content(),
                    dcc.Input(id="page-out-of-date", value="0", type="hidden"),
                ],
                style=_FLEX_WRAPPER_STYLE,)
        ],        
    )
