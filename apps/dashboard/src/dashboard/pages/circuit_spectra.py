"""REQ_155: Circuit Spectra page.

Per-head spectral invariants of composed weight circuits (REQ_152/154), surfaced
through universal ViewCatalog views (renderers are plot-only; the page only wires
control values into ``view_kwargs``):

- **Trajectory** (``circuits.spectra.trajectory``): a metric per head over training.
- **Ranking** (``circuits.spectra.ranking``): per-head metric at the cursor epoch,
  sorted — the copy/transform head split at a checkpoint.
- **Matrix** (``circuits.spectra.matrix``): the composed ``(p, p)`` circuit at the
  cursor epoch/head as a heatmap, resolved through the tensor catalog.
- **Cross-variant overlay**: one metric's head-mean trajectory across data-seed or
  model-seed peers on event-relative time (``views.circuit_spectra`` prep).
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate

from dashboard.components.analysis_page import AnalysisPageGraphManager
from dashboard.state import get_families, variant_server_state
from miscope.families.variant import Variant
from miscope.views.circuit_spectra import load_circuit_spectra_cross_variant
from miscope.visualization import render_circuit_spectra_cross_variant

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

_HEAD_OPTIONS = [{"label": f"Head {h}", "value": h} for h in range(4)]

_AXIS_OPTIONS = [
    {"label": "Data Seeds", "value": "data_seed"},
    {"label": "Model Seeds", "value": "seed"},
]

# Per-circuit default metric: copying_score is OV-meaningful only — for the QK
# circuits it is degenerate (rank-1, non-positive-real eigenvalues → pinned ~0),
# so they open on operator_norm where the dynamics (e.g. the late-instability
# magnitude spike) are visible. The user can still pick any metric.
_DEFAULT_METRIC = {"full_qk": "operator_norm", "qk": "operator_norm"}


def _default_metric_for(site: str | None) -> str:
    return _DEFAULT_METRIC.get(site or "full_ov", "copying_score")


# Trajectory + ranking share (site, metric) kwargs → one filter set; the matrix
# takes (site, head) → its own set, so each callback passes only the kwargs its
# renderers accept.
_VIEW_LIST = {
    "trajectory": {
        "view_name": "circuits.spectra.trajectory",
        "view_type": "default_graph",
        "view_filter_set": "spectra",
    },
    "ranking": {
        "view_name": "circuits.spectra.ranking",
        "view_type": "default_graph",
        "view_filter_set": "spectra",
    },
    "matrix": {
        "view_name": "circuits.spectra.matrix",
        "view_type": "default_graph",
        "view_filter_set": "matrix",
    },
}

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST, "cspec")


def _axis_peers(axis: str) -> list[Variant]:
    """Anchor variant + its peers along ``axis`` (shared other params)."""
    anchor = getattr(variant_server_state, "variant", None)
    if anchor is None:
        return []
    all_variants = get_families()[anchor.family.name].variants
    fixed = {k: v for k, v in anchor.params.items() if k != axis}
    peers = [v for v in all_variants if all(v.params.get(k) == val for k, val in fixed.items())]
    return sorted(peers, key=lambda v: v.params.get(axis, 0))


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
            dbc.Label("Matrix head", className="fw-bold mt-3"),
            dcc.Dropdown(
                id="circuit-head-dropdown",
                options=_HEAD_OPTIONS,
                value=0,
                clearable=False,
            ),
            html.Hr(),
            dbc.Label("Overlay across", className="fw-bold"),
            dcc.RadioItems(
                id="circuit-overlay-axis",
                options=_AXIS_OPTIONS,
                value="data_seed",
                labelStyle={"display": "block", "marginBottom": "4px"},
                className="mb-2",
            ),
            dbc.Button(
                "Load Overlay",
                id="circuit-overlay-button",
                color="primary",
                size="sm",
                className="mb-2 w-100",
            ),
            html.Div(id="circuit-overlay-status", className="text-muted small"),
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
            dbc.Row(dbc.Col(_graph_manager.create_graph("trajectory", "500px"))),
            dbc.Row(
                [
                    dbc.Col(_graph_manager.create_graph("ranking", "420px"), md=6),
                    dbc.Col(_graph_manager.create_graph("matrix", "420px"), md=6),
                ],
                className="mt-3",
            ),
            html.Hr(),
            html.H5("Cross-variant overlay", className="mt-2"),
            html.P(
                "One metric's head-mean trajectory across peers, aligned on each "
                "variant's grok epoch — does the circuit's dynamics land at a shared "
                "event-relative time? Pick the axis, then Load Overlay.",
                className="text-muted small",
            ),
            dbc.Row(dbc.Col(dcc.Graph(id="circuit-overlay-graph", style={"height": "460px"}))),
        ]
    )


def register_circuit_spectra_page_callbacks(app: Dash) -> None:
    """Register callbacks for the Circuit Spectra page."""
    app.server.logger.debug("register_circuit_spectra_page_callbacks")

    @app.callback(
        Output("circuit-metric-dropdown", "value"),
        Input("circuit-site-dropdown", "value"),
    )
    def on_circuit_change_set_default_metric(site_value: str | None) -> str:
        """Reset the metric to the circuit's sensible default when the circuit changes."""
        return _default_metric_for(site_value)

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list("spectra")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("circuit-site-dropdown", "value"),
        Input("circuit-metric-dropdown", "value"),
        State("variant-selector-store", "data"),
    )
    def on_spectra_change(
        _modified_timestamp: str | None,
        site_value: str | None,
        metric_value: str | None,
        variant_data: dict | None,
    ):
        """Trajectory + ranking — both keyed by (site, metric) at the cursor epoch."""
        view_kwargs = {
            "site": site_value or "full_ov",
            "metric": metric_value or "copying_score",
        }
        return _graph_manager.update_graphs(
            variant_data=variant_data, view_filter_set="spectra", view_kwargs=view_kwargs
        )

    @app.callback(
        [Output(pid, "figure") for pid in _graph_manager.get_graph_output_list("matrix")],
        Input("variant-selector-store", "modified_timestamp"),
        Input("circuit-site-dropdown", "value"),
        Input("circuit-head-dropdown", "value"),
        State("variant-selector-store", "data"),
    )
    def on_matrix_change(
        _modified_timestamp: str | None,
        site_value: str | None,
        head_value: int | None,
        variant_data: dict | None,
    ):
        """Composed circuit matrix heatmap at the cursor epoch for (site, head)."""
        view_kwargs = {"site": site_value or "full_ov", "head": head_value or 0}
        return _graph_manager.update_graphs(
            variant_data=variant_data, view_filter_set="matrix", view_kwargs=view_kwargs
        )

    @app.callback(
        Output("circuit-overlay-graph", "figure"),
        Output("circuit-overlay-status", "children"),
        Input("circuit-overlay-button", "n_clicks"),
        State("circuit-overlay-axis", "value"),
        State("circuit-site-dropdown", "value"),
        State("circuit-metric-dropdown", "value"),
        prevent_initial_call=True,
    )
    def on_overlay_load(
        _n_clicks: int | None,
        axis: str | None,
        site_value: str | None,
        metric_value: str | None,
    ) -> tuple[go.Figure, str]:
        """Cross-variant overlay — thin: the package prep does all the processing."""
        if ctx.triggered_id != "circuit-overlay-button":
            raise PreventUpdate
        peers = _axis_peers(axis or "data_seed")
        if len(peers) <= 1:
            return go.Figure(), f"No peers found on {axis} axis"
        data = load_circuit_spectra_cross_variant(
            peers, site=site_value or "full_ov", metric=metric_value or "copying_score"
        )
        n = len(data["series"])
        if n == 0:
            return go.Figure(), "No peers have circuit_spectra for this site"
        return render_circuit_spectra_cross_variant(data), f"Overlaid {n} variant(s)"
