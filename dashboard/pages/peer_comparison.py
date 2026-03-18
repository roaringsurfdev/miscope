"""Peer Variant Comparison page (REQ_076).

Shows loss curves, normalized weight divergence, and per-matrix L2 breakdown
for the selected anchor variant against its data-seed or model-seed peers.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from dashboard.state import get_registry, variant_server_state
from miscope.families.variant import Variant

# --- Constants ---

_WEIGHT_KEYS = ["W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "W_U"]
_DSEED_COLORS: dict[int, str] = {598: "steelblue", 42: "tomato", 999: "orange"}
_MSEED_COLORS: dict[int, str] = {485: "mediumseagreen", 999: "mediumpurple"}
_FALLBACK_COLORS = ["steelblue", "tomato", "orange", "mediumseagreen", "mediumpurple"]

_AXIS_OPTIONS = [
    {"label": "Data Seeds", "value": "data_seed"},
    {"label": "Model Seeds", "value": "seed"},
]

# --- Helpers ---


def _color_for(variant: Variant, axis: str) -> str:
    param_val = variant.params.get(axis)
    palette = _DSEED_COLORS if axis == "data_seed" else _MSEED_COLORS
    return palette.get(param_val, _FALLBACK_COLORS[0])


def _peer_label(variant: Variant, axis: str) -> str:
    short_axis = "dseed" if axis == "data_seed" else "mseed"
    return f"{short_axis}={variant.params.get(axis)}"


def _get_axis_variants(anchor: Variant, axis: str) -> list[Variant]:
    """All variants (anchor included) sharing non-axis params with anchor."""
    registry = get_registry()
    all_variants = registry.get_variants(registry.get_family(anchor.family.name))
    fixed = {k: v for k, v in anchor.params.items() if k != axis}
    siblings = [
        v for v in all_variants
        if all(v.params.get(k) == val for k, val in fixed.items())
    ]
    return sorted(siblings, key=lambda v: v.params.get(axis, 0))


def _load_weight_epochs(variant: Variant) -> tuple[list[int], dict[str, np.ndarray]]:
    """Load all parameter_snapshot epochs for a variant."""
    epochs = variant.artifacts.get_epochs("parameter_snapshot")
    stacked: dict[str, list[np.ndarray]] = {k: [] for k in _WEIGHT_KEYS}
    for epoch in epochs:
        snap = variant.artifacts.load_epoch("parameter_snapshot", epoch)
        for k in _WEIGHT_KEYS:
            stacked[k].append(snap[k])
    return epochs, {k: np.stack(v) for k, v in stacked.items()}


def _divergence_from_anchor(
    anchor_w: dict[str, np.ndarray],
    peer_w: dict[str, np.ndarray],
    n: int,
) -> dict[str, list]:
    """Compute aggregate, normalized, and per-matrix L2 divergence."""
    agg = np.zeros(n)
    ref_norm = np.zeros(n)
    per_matrix: dict[str, list[float]] = {}
    for k in _WEIGHT_KEYS:
        diff = (peer_w[k] - anchor_w[k]).reshape(n, -1)
        l2 = np.linalg.norm(diff, axis=1)
        per_matrix[k] = l2.tolist()
        agg += l2 ** 2
        ref_norm += np.linalg.norm(anchor_w[k].reshape(n, -1), axis=1) ** 2
    agg = np.sqrt(agg)
    normalized = (agg / (np.sqrt(ref_norm) + 1e-12) * 100).tolist()
    return {"aggregate": agg.tolist(), "normalized": normalized, "per_matrix": per_matrix}


def _compute_peer_divergences(
    anchor: Variant, peers: list[Variant]
) -> tuple[list[int], list[dict]]:
    """Load parameter_snapshot artifacts and compute divergence for each peer."""
    anchor_epochs, anchor_w = _load_weight_epochs(anchor)
    results = []
    for peer in peers:
        peer_epochs, peer_w = _load_weight_epochs(peer)
        shared = sorted(set(anchor_epochs) & set(peer_epochs))
        ai = [anchor_epochs.index(e) for e in shared]
        pi = [peer_epochs.index(e) for e in shared]
        a_slice = {k: anchor_w[k][ai] for k in _WEIGHT_KEYS}
        p_slice = {k: peer_w[k][pi] for k in _WEIGHT_KEYS}
        results.append((shared, _divergence_from_anchor(a_slice, p_slice, len(shared))))
    common_epochs = results[0][0] if results else []
    return common_epochs, [r[1] for r in results]


# --- Plot builders ---


def _placeholder_fig(message: str, height: int = 350) -> go.Figure:
    return go.Figure().update_layout(
        annotations=[dict(
            text=message, showarrow=False, font_size=13,
            xref="paper", yref="paper", x=0.5, y=0.5,
        )],
        height=height,
    )


def _add_cursor(fig: go.Figure, cursor_epoch: int | float | None) -> None:
    if cursor_epoch is None:
        return
    fig.add_vline(
        x=cursor_epoch, row="all", col="all",
        line_dash="dash", line_color="rgba(100,100,100,0.55)", line_width=1,
    )


def _build_loss_figure(
    all_variants: list[Variant],
    anchor_name: str,
    axis: str,
    cursor_epoch: int | float | None,
) -> go.Figure:
    fig = go.Figure()
    for v in all_variants:
        color = _color_for(v, axis)
        is_anchor = v.name == anchor_name
        label = f"anchor ({_peer_label(v, axis)})" if is_anchor else _peer_label(v, axis)
        width = 2.5 if is_anchor else 1.5
        epochs_idx = list(range(len(v.test_losses)))
        fig.add_trace(go.Scatter(
            x=epochs_idx, y=v.train_losses, mode="lines",
            name=f"{label} train",
            line=dict(color=color, dash="dash", width=width), opacity=0.7,
        ))
        fig.add_trace(go.Scatter(
            x=epochs_idx, y=v.test_losses, mode="lines",
            name=f"{label} test",
            line=dict(color=color, width=width),
        ))
    fig.update_layout(
        title="Train / Test Loss  (train=dashed · test=solid · anchor=bold)",
        xaxis_title="Epoch", yaxis_title="Loss", height=350,
    )
    _add_cursor(fig, cursor_epoch)
    return fig


def _build_divergence_figure(
    store_data: dict, cursor_epoch: int | float | None
) -> go.Figure:
    fig = go.Figure()
    for peer in store_data.get("peers", []):
        fig.add_trace(go.Scatter(
            x=store_data["epochs"], y=peer["normalized"],
            mode="lines", name=peer["label"],
            line=dict(color=peer["color"]),
        ))
    fig.update_layout(
        title="Normalized Weight Divergence from Anchor  (% of anchor norm)",
        xaxis_title="Epoch", yaxis_title="Divergence (%)", height=350,
    )
    _add_cursor(fig, cursor_epoch)
    return fig


def _build_matrix_figure(
    store_data: dict, cursor_epoch: int | float | None
) -> go.Figure:
    fig = make_subplots(rows=3, cols=3, subplot_titles=_WEIGHT_KEYS, shared_xaxes=False)
    for midx, matrix_name in enumerate(_WEIGHT_KEYS):
        mrow, mcol = divmod(midx, 3)
        for peer in store_data.get("peers", []):
            fig.add_trace(
                go.Scatter(
                    x=store_data["epochs"], y=peer["per_matrix"][matrix_name],
                    mode="lines", name=peer["label"],
                    line=dict(color=peer["color"]),
                    showlegend=(midx == 0),
                ),
                row=mrow + 1, col=mcol + 1,
            )
    fig.update_layout(title="Per-Matrix L2 Divergence from Anchor", height=700)
    fig.update_xaxes(title_text="Epoch")
    _add_cursor(fig, cursor_epoch)
    return fig


# --- Page layout ---


def create_peer_comparison_page_nav(app: Dash) -> html.Div:
    return html.Div([
        dbc.Label("Compare Axis", className="fw-bold"),
        dcc.RadioItems(
            id="peer-axis-radio",
            options=_AXIS_OPTIONS,
            value="data_seed",
            labelStyle={"display": "block", "marginBottom": "4px"},
            className="mb-3",
        ),
        dbc.Button(
            "Load Divergence",
            id="peer-load-button",
            color="primary",
            size="sm",
            className="mb-2 w-100",
        ),
        html.Div(id="peer-status-message", className="text-muted small mt-1"),
        html.Hr(),
    ])


def create_peer_comparison_page_layout(app: Dash) -> html.Div:
    return html.Div([
        html.H4("Peer Variant Comparison", className="mb-3"),
        dcc.Store(id="peer-comparison-data-store", data={"status": "empty"}),
        dcc.Store(id="peer-cursor-store", data=None),
        dbc.Row(dbc.Col(dcc.Graph(id="peer-loss-graph"))),
        dbc.Row(dbc.Col(dcc.Graph(id="peer-divergence-graph")), className="mt-3"),
        dbc.Row(dbc.Col(dcc.Graph(id="peer-matrix-graph")), className="mt-3"),
    ])


# --- Callbacks ---


def register_peer_comparison_page_callbacks(app: Dash) -> None:

    @app.callback(
        Output("peer-loss-graph", "figure"),
        Input("variant-selector-store", "modified_timestamp"),
        Input("peer-axis-radio", "value"),
        Input("peer-cursor-store", "data"),
        State("variant-selector-store", "data"),
    )
    def update_loss_plot(
        _ts: str | None,
        axis: str,
        cursor_epoch: int | float | None,
        variant_data: dict | None,
    ) -> go.Figure:
        stored = variant_data or {}
        if not stored.get("variant_name"):
            raise PreventUpdate
        anchor = getattr(variant_server_state, "variant", None)
        if anchor is None:
            return _placeholder_fig("No variant loaded")
        all_variants = _get_axis_variants(anchor, axis)
        if len(all_variants) <= 1:
            return _placeholder_fig(f"No peers found on {axis} axis")
        return _build_loss_figure(all_variants, anchor.name, axis, cursor_epoch)

    @app.callback(
        Output("peer-comparison-data-store", "data"),
        Output("peer-status-message", "children"),
        Input("peer-load-button", "n_clicks"),
        Input("variant-selector-store", "modified_timestamp"),
        Input("peer-axis-radio", "value"),
        State("variant-selector-store", "data"),
    )
    def update_divergence_data(
        _n_clicks: int | None,
        _ts: str | None,
        axis: str,
        variant_data: dict | None,
    ) -> tuple[dict, str]:
        if ctx.triggered_id != "peer-load-button":
            return {"status": "empty"}, "Click Load to compute divergence"
        stored = variant_data or {}
        if not stored.get("variant_name"):
            return {"status": "empty"}, "No variant selected"
        anchor = getattr(variant_server_state, "variant", None)
        if anchor is None:
            return {"status": "empty"}, "No variant loaded"
        peers = [v for v in _get_axis_variants(anchor, axis) if v.name != anchor.name]
        if not peers:
            return {"status": "empty"}, f"No peers found on {axis} axis"
        try:
            epochs, divergences = _compute_peer_divergences(anchor, peers)
        except Exception as exc:
            return {"status": "empty"}, f"Error: {exc}"
        peer_records = [
            {
                "name": peer.name,
                "label": _peer_label(peer, axis),
                "color": _color_for(peer, axis),
                **div,
            }
            for peer, div in zip(peers, divergences)
        ]
        store = {
            "status": "loaded",
            "axis": axis,
            "anchor_name": anchor.name,
            "epochs": epochs,
            "peers": peer_records,
        }
        return store, f"Loaded {len(peers)} peer(s) · {len(epochs)} shared epochs"

    @app.callback(
        Output("peer-divergence-graph", "figure"),
        Output("peer-matrix-graph", "figure"),
        Input("peer-comparison-data-store", "modified_timestamp"),
        Input("peer-cursor-store", "data"),
        State("peer-comparison-data-store", "data"),
    )
    def update_divergence_plots(
        _ts: str | None,
        cursor_epoch: int | float | None,
        store_data: dict | None,
    ) -> tuple[go.Figure, go.Figure]:
        data = store_data or {}
        if data.get("status") != "loaded":
            empty = _placeholder_fig("Load divergence data to view")
            return empty, _placeholder_fig("Load divergence data to view", height=700)
        return (
            _build_divergence_figure(data, cursor_epoch),
            _build_matrix_figure(data, cursor_epoch),
        )

    @app.callback(
        Output("peer-cursor-store", "data"),
        Input("peer-loss-graph", "clickData"),
        Input("peer-divergence-graph", "clickData"),
        Input("peer-matrix-graph", "clickData"),
        prevent_initial_call=True,
    )
    def update_cursor(
        loss_click: dict | None,
        div_click: dict | None,
        matrix_click: dict | None,
    ) -> int | float | None:
        click = loss_click or div_click or matrix_click
        if click and click.get("points"):
            return click["points"][0]["x"]
        return None
