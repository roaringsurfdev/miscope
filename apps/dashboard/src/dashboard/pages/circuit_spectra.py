"""REQ_155: Circuit Spectra page.

Per-head spectral invariants of composed weight circuits over training, via the
universal ``circuits.spectra.trajectory`` view. Two left-nav dropdowns (circuit +
metric) parameterize the single trajectory graph; the renderer is plot-only, so
the page only wires control values into ``view_kwargs``.
"""

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

from dashboard.components.analysis_page import AnalysisPageGraphManager

_CIRCUIT_OPTIONS = [
    {"label": "Full OV  (W_U W_O W_V W_E)", "value": "full_ov"},
    {"label": "Full QK  (W_Eᵀ W_Qᵀ W_K W_E)", "value": "full_qk"},
    {"label": "OV  (W_O W_V, residual)", "value": "ov"},
    {"label": "QK  (W_Qᵀ W_K, residual)", "value": "qk"},
    {"label": "Direct path  (W_U W_E)", "value": "direct_path"},
]

_METRIC_OPTIONS = [
    {"label": "Copying score", "value": "copying_score"},
    {"label": "Effective rank", "value": "effective_rank"},
    {"label": "Operator norm", "value": "operator_norm"},
]

_VIEW_LIST = {
    "circuit-spectra-trajectory": {
        "view_name": "circuits.spectra.trajectory",
        "view_type": "default_graph",
        "view_filter_set": "circuit",
    },
}

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST, "cspec")


def create_circuit_spectra_page_nav(app: Dash) -> html.Div:
    app.server.logger.debug("create_circuit_spectra_page_nav")
    return html.Div(
        children=[
            dbc.Label("Circuit", className="fw-bold"),
            dcc.Dropdown(
                id="circuit-site-dropdown",
                options=_CIRCUIT_OPTIONS,
                value="full_ov",
                clearable=False,
            ),
            dbc.Label("Metric", className="fw-bold mt-3"),
            dcc.Dropdown(
                id="circuit-metric-dropdown",
                options=_METRIC_OPTIONS,
                value="copying_score",
                clearable=False,
            ),
        ]
    )


def create_circuit_spectra_page_layout(app: Dash) -> html.Div:
    app.server.logger.debug("create_circuit_spectra_page_layout")
    return html.Div(
        children=[
            html.H4("Circuit Spectra", className="mb-3"),
            html.P(
                "Per-head spectral invariants of composed weight circuits over training. "
                "Try Full OV / copying score (the copy–transform head split that opens "
                "at grok) vs Full QK / effective rank (QK sparsens to rank-1 pre-grok). "
                "The dashed green line marks grok (test-loss crossing).",
                className="text-muted",
            ),
            dbc.Row(dbc.Col(_graph_manager.create_graph("circuit-spectra-trajectory", "600px"))),
        ]
    )


def register_circuit_spectra_page_callbacks(app: Dash) -> None:
    """Register callbacks for the Circuit Spectra page."""
    app.server.logger.debug("register_circuit_spectra_page_callbacks")

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list("circuit")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("circuit-site-dropdown", "value"),
        Input("circuit-metric-dropdown", "value"),
        State("variant-selector-store", "data"),
    )
    def on_circuit_spectra_change(
        _modified_timestamp: str | None,
        site_value: str | None,
        metric_value: str | None,
        variant_data: dict | None,
    ):
        view_kwargs = {
            "site": site_value or "full_ov",
            "metric": metric_value or "copying_score",
        }
        return _graph_manager.update_graphs(
            variant_data=variant_data, view_filter_set="circuit", view_kwargs=view_kwargs
        )
