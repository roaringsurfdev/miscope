"""Dashboard page for transient frequency analysis (REQ_084)."""

from __future__ import annotations

import json

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from dashboard.state import variant_server_state

_PAGE_PREFIX = "tf"


def _read_variant_summary() -> dict | None:
    ctx = variant_server_state.context
    if ctx is None:
        return None
    summary_path = variant_server_state.variant.variant_dir / "variant_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text())


def _empty_figure(message: str = "Select a variant") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=14, color="gray"),
    )
    fig.update_layout(
        template="plotly_white",
        height=300,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _make_graph(graph_id: str, height: str) -> dcc.Graph:
    return dcc.Graph(
        id=f"{_PAGE_PREFIX}-{graph_id}",
        config={"displayModeBar": True},
        style={"height": height},
    )


def create_transient_page_nav(app: Dash) -> html.Div:
    app.server.logger.debug("create_transient_page_nav")
    return html.Div([
        html.Div(id=f"{_PAGE_PREFIX}-summary", className="mb-3"),
        html.Hr(),
        dbc.Label("Frequency Group", className="fw-bold"),
        dcc.Dropdown(
            id=f"{_PAGE_PREFIX}-freq-dropdown",
            options=[],
            value=None,
            clearable=True,
            placeholder="Default (largest group)",
            className="mb-3",
        ),
        dbc.Checklist(
            id=f"{_PAGE_PREFIX}-show-persistent",
            options=[{"label": "Show persistent frequencies", "value": "show"}],
            value=["show"],
            switch=True,
            className="mb-2",
        ),
        html.Hr(),
    ])


def create_transient_page_layout(app: Dash) -> html.Div:
    app.server.logger.debug("create_transient_page_layout")
    return html.Div([
        html.H4("Transient Frequency Analysis", className="mb-3"),
        dbc.Row(dbc.Col(_make_graph("committed-counts", "450px"), className="mb-3")),
        html.Hr(),
        dbc.Row([
            dbc.Col(_make_graph("peak-scatter", "520px"), md=6),
            dbc.Col(_make_graph("pc1-cohesion", "520px"), md=6),
        ]),
    ])


def register_transient_page_callbacks(app: Dash) -> None:
    app.server.logger.debug("register_transient_page_callbacks")

    @app.callback(
        Output(f"{_PAGE_PREFIX}-summary", "children"),
        Output(f"{_PAGE_PREFIX}-freq-dropdown", "options"),
        Output(f"{_PAGE_PREFIX}-freq-dropdown", "value"),
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_variant_change_update_nav(ts: str | None, variant_data: dict | None):
        stored = variant_data or {}
        if not stored.get("variant_name"):
            return html.P("Select a variant.", className="text-muted small"), [], None
        if stored.get("last_field_updated") not in ["variant_name", "intervention_name"]:
            raise PreventUpdate

        summary = _read_variant_summary()
        if summary is None:
            return html.P("No variant summary.", className="text-muted small"), [], None

        homeless_frac = summary.get("homeless_neuron_fraction")
        transient_freqs = summary.get("transient_frequencies") or []
        transient_count = summary.get("transient_frequency_count") or 0

        homeless_str = f"{homeless_frac:.1%}" if homeless_frac is not None else "n/a"
        transient_color = "danger" if transient_count > 0 else "secondary"
        homeless_color = "warning" if (homeless_frac or 0) > 0.05 else "secondary"

        summary_div = html.Div([
            dbc.Badge(f"{transient_count} transient group{'s' if transient_count != 1 else ''}",
                      color=transient_color, className="me-1 mb-1"),
            dbc.Badge(f"{homeless_str} homeless neurons",
                      color=homeless_color, className="mb-1"),
        ])

        options = [
            {"label": f"Freq {f}", "value": f - 1}
            for f in transient_freqs
        ]
        return summary_div, options, None

    @app.callback(
        Output(f"{_PAGE_PREFIX}-committed-counts", "figure"),
        Input("variant-selector-store", "modified_timestamp"),
        Input(f"{_PAGE_PREFIX}-show-persistent", "value"),
        State("variant-selector-store", "data"),
    )
    def on_committed_counts_update(
        ts: str | None,
        show_persistent_vals: list,
        variant_data: dict | None,
    ) -> go.Figure:
        stored = variant_data or {}
        if not stored.get("variant_name"):
            return _empty_figure("Select a variant")

        ctx = variant_server_state.context
        if ctx is None:
            return _empty_figure("Select a variant")

        show = "show" in (show_persistent_vals or [])
        try:
            return ctx.view("transient.committed_counts").figure(show_persistent=show)
        except Exception:
            return _empty_figure("Transient frequency artifact not available")

    @app.callback(
        Output(f"{_PAGE_PREFIX}-peak-scatter", "figure"),
        Output(f"{_PAGE_PREFIX}-pc1-cohesion", "figure"),
        Input("variant-selector-store", "modified_timestamp"),
        Input(f"{_PAGE_PREFIX}-freq-dropdown", "value"),
        State("variant-selector-store", "data"),
    )
    def on_freq_views_update(
        ts: str | None,
        freq_0indexed: int | None,
        variant_data: dict | None,
    ) -> tuple[go.Figure, go.Figure]:
        stored = variant_data or {}
        if not stored.get("variant_name"):
            empty = _empty_figure("Select a variant")
            return empty, empty

        ctx = variant_server_state.context
        if ctx is None:
            empty = _empty_figure("Select a variant")
            return empty, empty

        kwargs = {} if freq_0indexed is None else {"freq": int(freq_0indexed)}
        try:
            scatter = ctx.view("transient.peak_scatter").figure(**kwargs)
        except Exception:
            scatter = _empty_figure("Transient frequency artifact not available")

        try:
            cohesion = ctx.view("transient.pc1_cohesion").figure(**kwargs)
        except Exception:
            cohesion = _empty_figure("Transient frequency artifact not available")

        return scatter, cohesion
