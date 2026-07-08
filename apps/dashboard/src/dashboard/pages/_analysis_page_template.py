import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

from dashboard.components.analysis_page import AnalysisPageGraphManager

_VIEW_LIST = {
    # example of view item that does not take parameters from left nav
    "[VIEW_CONTROL_ID1]": {
        "view_name": "[name of view from the view catalog]",
        "view_type": "[VIEW_TYPE: default_graph, epoch_selector]",
    },
    # example of view item that does take parameters from left nav
    "[VIEW_CONTROL_ID2]": {
        "view_name": "[name of view from the view catalog]",
        "view_type": "[VIEW_TYPE: default_graph or epoch_selector]",
        "view_filter_set": "[FILTER_NAME1]",
    },
    # example of view item that does take parameters from left nav
    "[VIEW_CONTROL_ID3]": {
        "view_name": "[name of view from the view catalog]",
        "view_type": "[VIEW_TYPE: default_graph or epoch_selector]",
        "view_filter_set": "[FILTER_NAME1]",
    },
    # Multiple filters from the left nav are support. This is just one example.
}

_graph_manager = AnalysisPageGraphManager(
    _VIEW_LIST, "[page abbreviation to prevent cross-page control collisions]"
)


def create_PAGE_NAME_page_nav(app: Dash) -> html.Div:
    app.server.logger.debug("create_PAGE_NAME_page_nav")
    return html.Div(
        children=[
            # list of left nav controls that allow user to change parameters in plots
            # FILTER 1
            dbc.Label("Multi-Stream: MLP Threshold", className="fw-bold"),
            dcc.Slider(
                id="multi-stream-mlp-threshold-slider",
                min=0.0,
                max=1.0,
                step=0.05,
                value=0.7,
                marks={0: "0%", 0.5: "50%", 0.9: "90%", 1.0: "100%"},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            # FILTER 2 - Can be any control type. Slider is just an example
            dbc.Label("[FILTER_LABEL_1]", className="fw-bold"),
            dcc.Slider(
                id="[FILTER_CONTROL_ID1]",
                min=0.0,
                max=1.0,
                step=0.05,
                value=0.7,
                marks={0: "0%", 0.5: "50%", 0.9: "90%", 1.0: "100%"},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            # Optional output control for the filter. Often used to display selection details.
            html.Div(
                id="[FILTER_OUTPUT_CONROL_ID1]",
                children="[Display Text]",
                className="text-muted small mb-3",
            ),
            # FILTER 2 - Can be any control type. Slider is just an example
            dbc.Label("[FILTER_LABEL_2]", className="fw-bold"),
            dcc.Slider(
                id="[FILTER_CONTROL_ID2]",
                min=0.0,
                max=1.0,
                step=0.05,
                value=0.7,
                marks={0: "0%", 0.5: "50%", 0.9: "90%", 1.0: "100%"},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            # Optional output control for the filter. Often used to display selection details.
            html.Div(
                id="[FILTER_OUTPUT_CONROL_ID2]",
                children="[Display Text]",
                className="text-muted small mb-3",
            ),
            html.Hr(),
        ]
    )


def create_PAGE_NAME_page_layout(app: Dash) -> html.Div:
    app.server.logger.debug("create_PAGE_NAME_page_layout")
    return html.Div(
        children=[
            html.H4("PLOT_LABEL", className="mb-3"),
            html.Div(
                [
                    # --- VIEW_CONTROL_ID1 ---
                    dbc.Row(dbc.Col(_graph_manager.create_graph("[VIEW_CONTROL_ID1]", "400px"))),
                    # --- VIEW_CONTROL_ID2 ---
                    dbc.Row(dbc.Col(_graph_manager.create_graph("[VIEW_CONTROL_ID2]", "400px"))),
                ],
            ),
        ]
    )


def register_PAGE_NAME_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Multistream page."""
    app.server.logger.debug("register_PAGE_NAME_callbacks")

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_PAGE_NAME_data_change(modified_timestamp: str | None, variant_data: dict | None):
        app.server.logger.debug("on_PAGE_NAME_data_change")
        return _graph_manager.update_graphs(variant_data, None)

    @app.callback(
        Output("[FILTER_OUTPUT_CONROL_ID1]", "children"),
        Input("[FILTER_CONTROL_ID1]", "value"),
    )
    def on_PAGE_NAME_FILTER_OUTPUT1_update(INPUT_VAR: float) -> str:
        return f"[DISPLAY TEXT {INPUT_VAR}]"

    @app.callback(
        Output("[FILTER_OUTPUT_CONROL_ID2]", "children"),
        Input("[FILTER_CONTROL_ID2]", "value"),
    )
    def on_PAGE_NAME_FILTER_OUTPUT2_update(INPUT_VAR: float) -> str:
        return f"[DISPLAY TEXT {INPUT_VAR}]"

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list("[FILTER_NAME1]")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("[FILTER_CONTROL_ID2]", "value"),
        Input("[FILTER_CONTROL_ID2]", "value"),
        State("variant-selector-store", "data"),
    )
    def on_PAGE_NAME_FILTER_NAME_change(
        modified_timestamp: str | None,
        FILTER_INPUT1: float,
        FILTER_INPUT2: float,
        variant_data: dict | None,
    ):
        app.server.logger.debug("on_PAGE_NAME_FILTER_NAME1_change")
        # build view_kwargs using FILTER_INPUT vars
        view_kwargs = {
            "FILTER_INPUT1": FILTER_INPUT1,
            "FILTER_INPUT2": FILTER_INPUT2,
        }
        return _graph_manager.update_graphs(
            variant_data=variant_data,
            view_filter_set="[FILTER_NAME1]",
            view_kwargs=view_kwargs,
        )
