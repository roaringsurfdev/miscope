import dash_bootstrap_components as dbc
from dash import dcc, html

_SITE_OPTIONS = [
    {"label": "All Sites", "value": "all"},
    {"label": "Post-Embed", "value": "resid_pre"},
    {"label": "Attn Out", "value": "attn_out"},
    {"label": "MLP Out", "value": "mlp_out"},
    {"label": "Resid Post", "value": "resid_post"},
]

def create_repr_geometry_page_nav() -> html.Div:
    print("create_repr_geometry_page_layout")
    return html.Div(
        children=[
            dbc.Label("Activation Site", className="fw-bold"),
            dcc.Dropdown(
                id="rg-site-dropdown",
                options=_SITE_OPTIONS,
                value="all",
                clearable=False,
            )
        ]
    )

def create_repr_geometry_page_layout() -> html.Div:
    print("create_repr_geometry_page_layout")
    return html.Div(
        id="repr_geometry_content",
        children=[
            html.H4("Repr Geometry", className="mb-3"),
        ]
    )