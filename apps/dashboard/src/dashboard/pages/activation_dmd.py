"""REQ_117: Activation DMD dashboard page.

Per-variant view of the windowed + per-regime activation DMD analysis.
Shows residual + regime boundaries (the primary diagnostic), eigenvalue
migration at zoom (per-variant signature), per-track |lambda| and
arg(lambda) trajectories for one selected site, and a windowed-vs-
per-regime residual comparison.
"""

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

from dashboard.components.analysis_page import _SITE_OPTIONS, AnalysisPageGraphManager

_VIEW_LIST = {
    "training-loss-curves": {
        "view_name": "training.metadata.loss_curves",
        "view_type": "epoch_selector",
    },
    "activation-dmd-residuals": {
        "view_name": "activation_dmd.residuals_with_regimes",
        "view_type": "epoch_selector",
    },
    "activation-dmd-eigenvalues": {
        "view_name": "activation_dmd.eigenvalue_migration",
        "view_type": "default_graph",
    },
    "activation-dmd-per-regime": {
        "view_name": "activation_dmd.per_regime_vs_windowed",
        "view_type": "epoch_selector",
    },
    "activation-dmd-tracks": {
        "view_name": "activation_dmd.track_trajectories",
        "view_type": "epoch_selector",
        "view_filter_set": "site",
    },
}

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST, "activation_dmd")


def create_activation_dmd_nav(app: Dash) -> html.Div:
    return html.Div(
        children=[
            dbc.Label("Track Site", className="fw-bold"),
            dcc.Dropdown(
                id="activation-dmd-site-dropdown",
                options=_SITE_OPTIONS,
                value="mlp_out",
                clearable=False,
            ),
            html.Small(
                "Site selector applies to the per-track trajectories panel only. "
                "All other panels show all four sites.",
                className="text-muted d-block mt-2",
            ),
        ]
    )


def create_activation_dmd_layout(app: Dash) -> html.Div:
    app.server.logger.debug("create_activation_dmd_layout")
    return html.Div(
        children=[
            html.H4("Activation DMD", className="mb-3"),
            html.P(
                "Windowed Dynamic Mode Decomposition on per-class centroid trajectories "
                "in global PCA space, with peak-based regime detection and per-regime "
                "DMD as a recursive second pass. Boundaries mark where the linear "
                "approximation breaks down. Eigenvalues plot at auto-zoom to surface "
                "the small-radius migration patterns.",
                className="text-muted small",
            ),
            html.Div(
                [
                    dbc.Row(dbc.Col(_graph_manager.create_graph("training-loss-curves", "350px"))),
                    dbc.Row(
                        dbc.Col(_graph_manager.create_graph("activation-dmd-residuals", "900px"))
                    ),
                    dbc.Row(
                        dbc.Col(
                            _graph_manager.create_graph("activation-dmd-eigenvalues", "850px")
                        )
                    ),
                    dbc.Row(
                        dbc.Col(
                            _graph_manager.create_graph("activation-dmd-per-regime", "750px")
                        )
                    ),
                    dbc.Row(
                        dbc.Col(_graph_manager.create_graph("activation-dmd-tracks", "600px"))
                    ),
                ],
            ),
        ]
    )


def register_activation_dmd_callbacks(app: Dash) -> None:
    """Register all callbacks for the Activation DMD page."""
    app.server.logger.debug("register_activation_dmd_callbacks")

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_vz_data_change(modified_timestamp: str | None, variant_data: dict | None):
        app.server.logger.debug("on_vz_data_change (activation_dmd)")
        return _graph_manager.update_graphs(variant_data, None)

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list("site")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("activation-dmd-site-dropdown", "value"),
        State("variant-selector-store", "data"),
    )
    def on_site_change(
        _modified_timestamp: str | None,
        site_value: str | None,
        variant_data: dict | None,
    ):
        app.server.logger.debug("on_site_change (activation_dmd)")
        view_kwargs = {"site": site_value}
        return _graph_manager.update_graphs(
            variant_data=variant_data, view_filter_set="site", view_kwargs=view_kwargs
        )
