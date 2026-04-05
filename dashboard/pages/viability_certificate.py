"""Viability Certificate page (REQ_086).

Analytical tool for characterising the geometric quality of a frequency set
before (or after) training.  No model weights required — inputs are prime p,
the chosen frequencies, d_model, and the observed W_E participation ratio at
the effective-dimensionality crossover epoch.

Lives under the "Pre-Training Analysis" top-nav menu.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from miscope.analysis.viability_certificate import (
    ALIAS_FAILURE_THRESHOLD,
    ALIAS_WARNING_THRESHOLD,
    COVERAGE_GAP_THRESHOLD,
    compute_certificate,
)

_REGISTRY_PATH = Path("results/modulo_addition_1layer/variant_registry.json")

_REGIME_STYLE: dict[str, dict[str, str]] = {
    "viable": {"color": "success", "label": "Viable"},
    "aliasing_failure": {"color": "danger", "label": "Aliasing failure risk"},
    "coverage_concern": {"color": "warning", "label": "Coverage concern"},
    "compression_risk": {"color": "danger", "label": "Compression risk"},
}

# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def _load_registry() -> list[dict[str, Any]]:
    if not _REGISTRY_PATH.exists():
        return []
    try:
        with open(_REGISTRY_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def _prime_options() -> list[dict]:
    registry = _load_registry()
    primes = sorted({v["prime"] for v in registry})
    return [{"label": f"p={p}", "value": p} for p in primes]


def _variant_options(prime: int | None) -> list[dict]:
    if prime is None:
        return []
    registry = _load_registry()
    opts = []
    for v in registry:
        if v["prime"] != prime:
            continue
        freqs = v.get("learned_frequencies") or []
        pr = v.get("effective_dimensionality_crossover_W_E_pr", -1)
        perf = (v.get("performance_classification") or ["?"])[0]
        label = f"s{v['model_seed']}/d{v['data_seed']} — {perf} — f{freqs} PR={pr:.1f}"
        opts.append(
            {
                "label": label,
                "value": json.dumps(
                    {
                        "freqs": freqs,
                        "pr": pr,
                        "prime": prime,
                    }
                ),
            }
        )
    return opts


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _empty_figure(message: str = "Enter parameters and click Compute") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=15, color="gray"),
    )
    fig.update_layout(
        template="plotly_white",
        height=400,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _render_separation_profile(cert: dict[str, Any]) -> go.Figure:
    """Compression survival curve with W_E_PR marker and 2|F| cliff annotation."""
    dims: np.ndarray = cert["dims"]
    min_dists: np.ndarray = cert["min_dists"]
    W_E_PR = cert["W_E_PR"]
    subspace_dims = cert["subspace_dims"]
    prime = cert["prime"]
    freqs = cert["freqs"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dims.tolist(),
            y=min_dists.tolist(),
            mode="lines",
            name="Min pairwise distance",
            line=dict(color="#2196F3", width=2),
        )
    )

    # W_E_PR marker
    fig.add_vline(
        x=W_E_PR,
        line=dict(color="tomato", width=1.5, dash="dash"),
        annotation_text=f"W_E PR = {W_E_PR:.1f}",
        annotation_position="top right",
    )
    # 2|F| cliff marker
    fig.add_vline(
        x=subspace_dims,
        line=dict(color="gray", width=1, dash="dot"),
        annotation_text=f"2|F| = {subspace_dims}",
        annotation_position="top left",
    )

    # Annotation about whether compression is binding
    if cert["compression_margin"] >= 0:
        note = (
            f"W_E PR ({W_E_PR:.1f}) sits {cert['compression_margin']:.1f} dims "
            f"above the 2|F| cliff — compression is not binding."
        )
    else:
        note = (
            f"W_E PR ({W_E_PR:.1f}) is BELOW 2|F|={subspace_dims} — "
            "compression is actively reducing separation."
        )

    fig.update_layout(
        title=(f"Separation under compression — p={prime}, f={freqs}<br><sup>{note}</sup>"),
        xaxis_title="Retained dimensions",
        yaxis_title="Min pairwise centroid distance",
        xaxis=dict(range=[0, min(60, len(dims))]),
        template="plotly_white",
        height=420,
    )
    return fig


def _render_aliasing_risk(cert: dict[str, Any]) -> go.Figure:
    """Per-frequency aliasing risk bar chart with threshold bands."""
    alias = cert["alias_per_freq"]
    freqs = list(alias.keys())
    risks = list(alias.values())
    prime = cert["prime"]

    bar_colors = [
        "#F44336"
        if r > ALIAS_FAILURE_THRESHOLD
        else "#FF9800"
        if r > ALIAS_WARNING_THRESHOLD
        else "#4CAF50"
        for r in risks
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[str(k) for k in freqs],
            y=risks,
            marker_color=bar_colors,
            showlegend=False,
            text=[f"{r:.3f}" for r in risks],
            textposition="outside",
        )
    )
    fig.add_hline(
        y=ALIAS_FAILURE_THRESHOLD,
        line=dict(color="#F44336", dash="dash", width=1.5),
        annotation_text=f"Failure threshold ({ALIAS_FAILURE_THRESHOLD})",
        annotation_position="right",
    )
    fig.add_hline(
        y=ALIAS_WARNING_THRESHOLD,
        line=dict(color="#FF9800", dash="dot", width=1),
        annotation_text=f"Warning ({ALIAS_WARNING_THRESHOLD})",
        annotation_position="right",
    )
    fig.update_layout(
        title=f"Aliasing risk per frequency — p={prime}  [k / ((p−1)/2)]",
        xaxis_title="Frequency k",
        yaxis_title="Aliasing risk",
        yaxis=dict(range=[0, 1.1]),
        template="plotly_white",
        height=380,
    )
    return fig


def _render_ideal_comparison(cert: dict[str, Any]) -> go.Figure:
    """Side-by-side: actual vs ideal frequency set separation."""
    prime = cert["prime"]
    freqs = cert["freqs"]
    ideal_set = cert["ideal_set"]
    ambient = cert["ambient_min_dist"]
    ideal_dist = cert["ideal_dist"]
    gap_pct = cert["gap_pct"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Min pairwise distance", "Aliasing risk profile"],
    )

    # Left: bar comparison of ambient distances
    fig.add_trace(
        go.Bar(
            x=[f"Actual {freqs}", f"Ideal {ideal_set}"],
            y=[ambient, ideal_dist],
            marker_color=["#2196F3", "rgba(33,150,243,0.35)"],
            marker_line=dict(color=["#2196F3", "#2196F3"], width=2),
            text=[f"{ambient:.3f}", f"{ideal_dist:.3f}"],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Right: aliasing risk for actual vs ideal side by side
    from miscope.analysis.viability_certificate import aliasing_risk

    ideal_alias = aliasing_risk(prime, ideal_set)
    all_freqs = sorted(set(freqs) | set(ideal_set))

    actual_risks = [cert["alias_per_freq"].get(k, 0.0) for k in all_freqs]
    ideal_risks = [ideal_alias.get(k, 0.0) for k in all_freqs]

    fig.add_trace(
        go.Bar(
            name="Actual",
            x=[str(k) for k in all_freqs],
            y=actual_risks,
            marker_color="#2196F3",
            showlegend=True,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            name="Ideal",
            x=[str(k) for k in all_freqs],
            y=ideal_risks,
            marker_color="rgba(33,150,243,0.4)",
            marker_line=dict(color="#2196F3", width=1.5),
            showlegend=True,
        ),
        row=1,
        col=2,
    )
    fig.add_hline(
        y=ALIAS_FAILURE_THRESHOLD,
        line=dict(color="#F44336", dash="dash", width=1),
    )

    fig.update_yaxes(range=[0, 1.1], col=2)
    fig.update_xaxes(title_text="Frequency set", col=1)
    fig.update_xaxes(title_text="Frequency k", col=2)
    fig.update_yaxes(title_text="Min dist", col=1)
    fig.update_yaxes(title_text="Aliasing risk", col=2)
    fig.update_layout(
        title=(
            f"Ideal set comparison — gap from ideal: {gap_pct:.1f}%  "
            f"(threshold: {COVERAGE_GAP_THRESHOLD:.0f}%)"
        ),
        barmode="group",
        template="plotly_white",
        height=400,
    )
    return fig


def _regime_badge(regime: str) -> dbc.Badge:
    style = _REGIME_STYLE.get(regime, {"color": "secondary", "label": regime})
    return dbc.Badge(
        style["label"],
        color=style["color"],
        className="fs-6 px-3 py-2",
    )


def _metric_table(cert: dict[str, Any]) -> dbc.Table:
    rows = [
        ("Prime", str(cert["prime"])),
        ("Frequencies", str(cert["freqs"])),
        ("W_E participation ratio", f"{cert['W_E_PR']:.2f}"),
        ("Ambient min pairwise dist", f"{cert['ambient_min_dist']:.4f}"),
        ("Ideal set (same size)", str(cert["ideal_set"])),
        ("Ideal min dist", f"{cert['ideal_dist']:.4f}"),
        ("Gap from ideal", f"{cert['gap_pct']:.1f}%"),
        ("Max aliasing risk", f"{cert['max_alias']:.3f}"),
        ("Mean aliasing risk", f"{cert['mean_alias']:.3f}"),
        ("Centroid subspace dims (2|F|)", str(cert["subspace_dims"])),
        ("Compression margin (PR − 2|F|)", f"{cert['compression_margin']:.1f}"),
    ]
    return dbc.Table(
        [html.Tbody([html.Tr([html.Td(k, className="fw-semibold"), html.Td(v)]) for k, v in rows])],
        bordered=True,
        size="sm",
        className="mb-0",
    )


def _hard_pairs_text(cert: dict[str, Any]) -> html.Div:
    lines = []
    for k, pairs in cert["hard_pairs"].items():
        pair_str = ", ".join(f"({a},{b})" for a, b in pairs)
        lines.append(
            html.P(
                [html.Strong(f"k={k}: "), pair_str],
                className="mb-1 small",
            )
        )
    return html.Div(lines)


# ---------------------------------------------------------------------------
# Layout & nav
# ---------------------------------------------------------------------------


def create_viability_certificate_page_nav(app: Dash) -> html.Div:
    prime_opts = _prime_options()
    return html.Div(
        children=[
            dcc.Store(id="vc-store", storage_type="memory", data={}),
            # Manual inputs
            html.P("Manual input", className="fw-bold text-muted small mb-1"),
            dbc.Label("Prime (p)", className="fw-bold"),
            dbc.Input(id="vc-prime-input", type="number", value=59, min=2, step=1),
            html.Br(),
            dbc.Label("Frequencies", className="fw-bold"),
            dbc.Input(
                id="vc-freqs-input",
                type="text",
                placeholder="e.g. 5, 15, 21",
                value="5, 15, 21",
            ),
            html.Small("Comma-separated learned frequencies", className="text-muted"),
            html.Br(),
            html.Br(),
            dbc.Label("W_E participation ratio", className="fw-bold"),
            dbc.Input(
                id="vc-pr-input",
                type="number",
                value=18.3,
                min=0,
                step=0.1,
            ),
            html.Small("From variant_summary.json or enter manually", className="text-muted"),
            html.Br(),
            html.Br(),
            dbc.Button(
                "Compute",
                id="vc-compute-button",
                color="primary",
                size="sm",
                n_clicks=0,
            ),
            html.Hr(),
            # Variant loader
            html.P("Load from variant", className="fw-bold text-muted small mb-1"),
            dcc.Dropdown(
                id="vc-loader-prime",
                options=prime_opts,
                placeholder="Prime…",
                className="mb-2",
            ),
            dcc.Dropdown(
                id="vc-loader-variant",
                options=[],
                placeholder="Variant…",
                className="mb-2",
            ),
            dbc.Button(
                "Load & Compute",
                id="vc-load-button",
                color="secondary",
                size="sm",
                n_clicks=0,
                outline=True,
            ),
            html.Hr(),
            html.Div(id="vc-status", children="", className="text-muted small"),
        ]
    )


def create_viability_certificate_page_layout(app: Dash) -> html.Div:
    return html.Div(
        children=[
            dbc.Row(
                [
                    dbc.Col(html.H4("Viability Certificate", className="mb-0"), width="auto"),
                    dbc.Col(
                        html.Div(id="vc-regime-badge"),
                        width="auto",
                        className="d-flex align-items-center",
                    ),
                ],
                className="mb-3 align-items-center",
            ),
            html.P(
                "Characterises the geometric quality of a frequency set analytically. "
                "No training required — all metrics derive from the Fourier centroid structure.",
                className="text-muted mb-3",
            ),
            dbc.Tabs(
                id="vc-tabs",
                active_tab="tab-separation",
                children=[
                    dbc.Tab(
                        label="Separation Profile",
                        tab_id="tab-separation",
                        children=dcc.Graph(
                            id="vc-separation-figure",
                            figure=_empty_figure(),
                            config={"displayModeBar": True},
                        ),
                    ),
                    dbc.Tab(
                        label="Aliasing Risk",
                        tab_id="tab-aliasing",
                        children=dcc.Graph(
                            id="vc-aliasing-figure",
                            figure=_empty_figure(),
                            config={"displayModeBar": True},
                        ),
                    ),
                    dbc.Tab(
                        label="Ideal Set",
                        tab_id="tab-ideal",
                        children=dcc.Graph(
                            id="vc-ideal-figure",
                            figure=_empty_figure(),
                            config={"displayModeBar": True},
                        ),
                    ),
                    dbc.Tab(
                        label="Summary",
                        tab_id="tab-summary",
                        children=html.Div(id="vc-summary-content", className="p-3"),
                    ),
                ],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _parse_freqs(raw: str) -> list[int]:
    result = []
    for token in raw.split(","):
        token = token.strip()
        if token.isdigit():
            result.append(int(token))
    return result


def _run_and_render(
    prime: int, freqs: list[int], W_E_PR: float
) -> tuple[go.Figure, go.Figure, go.Figure, Any, Any, str, str]:
    """Run certificate computation and produce all outputs."""
    if not freqs:
        msg = "No valid frequencies parsed."
        empty = _empty_figure(msg)
        return empty, empty, empty, html.Div(), html.Div(), msg, ""

    cert = compute_certificate(prime, freqs, W_E_PR)
    if "error" in cert:
        empty = _empty_figure(cert["error"])
        return empty, empty, empty, html.Div(), html.Div(), cert["error"], ""

    sep_fig = _render_separation_profile(cert)
    alias_fig = _render_aliasing_risk(cert)
    ideal_fig = _render_ideal_comparison(cert)
    badge = _regime_badge(cert["regime"])
    summary = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(_metric_table(cert), width=6),
                    dbc.Col(
                        [
                            html.P(
                                "Predicted hard pairs (top 5 per frequency):",
                                className="fw-bold mb-1",
                            ),
                            _hard_pairs_text(cert),
                        ],
                        width=6,
                    ),
                ]
            )
        ]
    )
    status = (
        f"p={prime}, f={freqs}, PR={W_E_PR:.1f} — "
        f"regime: {cert['regime']}, "
        f"gap from ideal: {cert['gap_pct']:.1f}%, "
        f"max alias: {cert['max_alias']:.3f}"
    )
    return sep_fig, alias_fig, ideal_fig, badge, summary, status, ""


def register_viability_certificate_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Viability Certificate page."""

    @app.callback(
        Output("vc-loader-variant", "options"),
        Output("vc-loader-variant", "value"),
        Input("vc-loader-prime", "value"),
        prevent_initial_call=True,
    )
    def on_loader_prime_selected(prime: int | None):
        if prime is None:
            raise PreventUpdate
        return _variant_options(prime), None

    @app.callback(
        Output("vc-prime-input", "value"),
        Output("vc-freqs-input", "value"),
        Output("vc-pr-input", "value"),
        Input("vc-load-button", "n_clicks"),
        State("vc-loader-variant", "value"),
        prevent_initial_call=True,
    )
    def on_load_variant(n_clicks: int, variant_json: str | None):
        if not variant_json:
            raise PreventUpdate
        data = json.loads(variant_json)
        freqs_str = ", ".join(str(f) for f in data.get("freqs", []))
        return data.get("prime"), freqs_str, data.get("pr", 0.0)

    @app.callback(
        Output("vc-separation-figure", "figure"),
        Output("vc-aliasing-figure", "figure"),
        Output("vc-ideal-figure", "figure"),
        Output("vc-regime-badge", "children"),
        Output("vc-summary-content", "children"),
        Output("vc-status", "children"),
        Output("vc-store", "data"),
        Input("vc-compute-button", "n_clicks"),
        Input("vc-load-button", "n_clicks"),
        State("vc-prime-input", "value"),
        State("vc-freqs-input", "value"),
        State("vc-pr-input", "value"),
        prevent_initial_call=True,
    )
    def on_compute(
        _compute_clicks: int,
        _load_clicks: int,
        prime_raw: int | None,
        freqs_raw: str | None,
        pr_raw: float | None,
    ):
        # Load button populates inputs but doesn't immediately trigger compute
        # (the on_load_variant callback updates the inputs, and the user then
        # clicks Compute — but if they click Load & Compute we still fire here)
        if not prime_raw or not freqs_raw or pr_raw is None:
            raise PreventUpdate

        prime = int(prime_raw)
        freqs = _parse_freqs(freqs_raw)
        W_E_PR = float(pr_raw)

        sep, alias, ideal, badge, summary, status, _ = _run_and_render(prime, freqs, W_E_PR)
        return sep, alias, ideal, badge, summary, status, {"computed": True}
