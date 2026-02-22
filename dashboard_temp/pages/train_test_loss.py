import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html
from dash.exceptions import PreventUpdate

from dashboard_temp.components.visualization import create_empty_figure, create_graph

# ---------------------------------------------------------------------------
# Plot IDs
# ---------------------------------------------------------------------------

_PLOT_IDS = [
    "train-test-loss-plot",
]

def _update_graphs(variant_data: dict | None, sort_value: str | None = None):
    stored = variant_data or {}
    #family_name = stored.get("family_name")
    variant_name = stored.get("variant_name")
    epoch = stored.get("epoch")
    last_field_updated = stored.get("last_field_updated")

    # Clear graphs if variant_name is None
    if variant_name is None:
        no_data = create_empty_figure("Select a variant")
        return [no_data for pid in _PLOT_IDS]

    if last_field_updated in ["variant_name", "epoch"]:
        #Update graphs
        updated_data = create_empty_figure(f"Data for {variant_name} at epoch: {epoch}")
        return [updated_data for pid in _PLOT_IDS]
    else:
        raise PreventUpdate

def create_loss_page_nav() -> html.Div:
    print("create_loss_page_nav")
    return html.Div()    

def create_loss_page_layout() -> html.Div:
    print("create_loss_page_layout")
    return html.Div(
        children=[
            html.H4("Train/Test Loss", className="mb-3"),
            html.Div(
                    [
                        # Trajectory heatmap (full width)
                        dbc.Row(dbc.Col(create_graph("train-test-loss-plot", "600px"))),
                    ],
                )            
        ]
    )
def register_loss_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Train/Test Loss page."""


    @app.callback(
        *[Output(pid, "figure") for pid in _PLOT_IDS],
        Input("variant-selector-store", "modified_timestamp"),
        Input("nd-sort-toggle", "value"),
        State("variant-selector-store", "data"),
    )
    def on_loss_data_change(modified_timestamp: str | None, sort_value: str | None, variant_data: dict | None):
        print("on_loss_data_change")
        return _update_graphs(variant_data, sort_value)

