"""REQ_117 phase 2: Parameter DMD dashboard page.

Per-variant view of the per-(group, matrix) windowed + per-regime DMD
analysis. Shows three "all groups stacked × W_in/W_out" plots
(residuals, eigenvalues, per-regime vs windowed) plus a per-(group,
matrix) tracks plot driven by left-nav dropdowns. Reference epoch
indicator surfaces which `neuron_grouping` snapshot the analyzer used.
"""

from typing import Any

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from dashboard.components.analysis_page import AnalysisPageGraphManager

_MATRIX_OPTIONS = [
    {"label": "W_in", "value": "W_in"},
    {"label": "W_out", "value": "W_out"},
]


_VIEW_LIST = {
    "training-loss-curves": {
        "view_name": "training.metadata.loss_curves",
        "view_type": "epoch_selector",
    },
    "parameter-dmd-residuals": {
        "view_name": "parameter_dmd.residuals_with_regimes",
        "view_type": "epoch_selector",
    },
    "parameter-dmd-eigenvalues": {
        "view_name": "parameter_dmd.eigenvalue_migration",
        "view_type": "default_graph",
    },
    "parameter-dmd-per-regime": {
        "view_name": "parameter_dmd.per_regime_vs_windowed",
        "view_type": "epoch_selector",
    },
    "parameter-dmd-tracks": {
        "view_name": "parameter_dmd.track_trajectories",
        "view_type": "epoch_selector",
        "view_filter_set": "group_matrix",
    },
}

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST, "parameter_dmd")


def create_parameter_dmd_nav(app: Dash) -> html.Div:
    return html.Div(
        children=[
            dbc.Label("Group", className="fw-bold"),
            dcc.Dropdown(
                id="parameter-dmd-group-dropdown",
                options=[],
                value=None,
                clearable=False,
                placeholder="select a variant first",
            ),
            dbc.Label("Matrix", className="fw-bold mt-3"),
            dcc.Dropdown(
                id="parameter-dmd-matrix-dropdown",
                options=_MATRIX_OPTIONS,
                value="W_in",
                clearable=False,
            ),
            html.Small(
                "Group + Matrix dropdowns apply to the per-track trajectories "
                "panel only. Other panels show all populated groups stacked.",
                className="text-muted d-block mt-2",
            ),
            html.Hr(),
            html.Div(
                id="parameter-dmd-reference-epoch-indicator",
                className="small text-muted",
            ),
        ]
    )


def create_parameter_dmd_layout(app: Dash) -> html.Div:
    app.server.logger.debug("create_parameter_dmd_layout")
    return html.Div(
        children=[
            html.H4("Parameter DMD", className="mb-3"),
            html.P(
                "Windowed Dynamic Mode Decomposition on per-(group, matrix) "
                "weight trajectories. Groups come from neuron_grouping at a "
                "configurable reference epoch (shown in the left nav). "
                "W_in and W_out are treated as separate matrices because they "
                "may reorganize on independent timelines.",
                className="text-muted small",
            ),
            html.Div(
                [
                    dbc.Row(dbc.Col(_graph_manager.create_graph("training-loss-curves", "350px"))),
                    dbc.Row(
                        dbc.Col(_graph_manager.create_graph("parameter-dmd-residuals", "1100px"))
                    ),
                    dbc.Row(
                        dbc.Col(_graph_manager.create_graph("parameter-dmd-eigenvalues", "1300px"))
                    ),
                    dbc.Row(
                        dbc.Col(_graph_manager.create_graph("parameter-dmd-per-regime", "1100px"))
                    ),
                    dbc.Row(dbc.Col(_graph_manager.create_graph("parameter-dmd-tracks", "600px"))),
                ],
            ),
        ]
    )


def register_parameter_dmd_callbacks(app: Dash) -> None:
    """Register all callbacks for the Parameter DMD page."""
    app.server.logger.debug("register_parameter_dmd_callbacks")

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_vz_data_change(modified_timestamp: str | None, variant_data: dict | None):
        app.server.logger.debug("on_vz_data_change (parameter_dmd)")
        return _graph_manager.update_graphs(variant_data, None)

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list("group_matrix")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("parameter-dmd-group-dropdown", "value"),
        Input("parameter-dmd-matrix-dropdown", "value"),
        State("variant-selector-store", "data"),
    )
    def on_group_matrix_change(
        _ts: str | None,
        group_value: int | None,
        matrix_value: str | None,
        variant_data: dict | None,
    ):
        app.server.logger.debug("on_group_matrix_change (parameter_dmd)")
        if group_value is None or matrix_value is None:
            raise PreventUpdate
        view_kwargs: dict[str, Any] = {
            "group_id": int(group_value),
            "matrix": matrix_value,
        }
        return _graph_manager.update_graphs(
            variant_data=variant_data,
            view_filter_set="group_matrix",
            view_kwargs=view_kwargs,
        )

    @app.callback(
        Output("parameter-dmd-group-dropdown", "options"),
        Output("parameter-dmd-group-dropdown", "value"),
        Output("parameter-dmd-reference-epoch-indicator", "children"),
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def update_group_dropdown(_ts: str | None, variant_data: dict | None) -> tuple[list, Any, str]:
        """Populate the group dropdown from the loaded variant's
        parameter_dmd artifact, and surface the reference_epoch."""
        from dashboard.state import variant_server_state

        if not variant_data or not variant_data.get("variant_name"):
            return [], None, ""

        try:
            variant = variant_server_state.variant
        except AttributeError:
            return [], None, ""

        try:
            artifact = variant.artifacts.load_cross_epoch("parameter_dmd")
        except FileNotFoundError:
            return (
                [],
                None,
                "parameter_dmd artifact not found — run the analyzer first.",
            )

        populated = [int(g) for g in artifact["populated_groups"]]
        n_per = [int(n) for n in artifact["group_n_neurons"]]
        ref_epoch = int(artifact["reference_epoch"])

        options = [
            {
                "label": f"group {g} (k={g + 1}) — {n} neurons",
                "value": g,
            }
            for g, n in zip(populated, n_per)
        ]
        default_value = populated[0] if populated else None
        indicator = f"reference_epoch = {ref_epoch}"
        return options, default_value, indicator
