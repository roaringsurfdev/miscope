"""Initialization Gradient Sweep page (REQ_085).

Computes per-frequency gradient energy at epoch 0 for arbitrary (prime,
model_seed, data_seed) combinations — no training required, no variant
objects created.  Runs fresh HookedTransformer initializations entirely
in memory and visualizes the resulting site-level energy profiles.

Accessible under the "Pre-Training Analysis" top-nav menu item.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from dashboard.state import get_registry
from miscope.analysis.analyzers.gradient_site import _fourier_gradient_by_site
from miscope.analysis.library import get_fourier_basis

_SITES = ("embedding", "attention", "mlp")
_SITE_LABELS = {"embedding": "Embedding", "attention": "Attention", "mlp": "MLP"}
_PALETTE = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]

_REGISTRY_PATH = Path("results") / "modulo_addition_1layer" / "variant_registry.json"

# ---------------------------------------------------------------------------
# Server-side page state
# ---------------------------------------------------------------------------


@dataclass
class _SweepState:
    prime: int | None = None
    candidates: list[tuple[int, int]] = field(default_factory=list)
    results: dict[tuple[int, int], dict[str, np.ndarray]] = field(default_factory=dict)
    key_frequencies: list[int] = field(default_factory=list)
    error: str = ""


_state = _SweepState()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated string of integers; skip non-numeric tokens."""
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token.lstrip("-").isdigit():
            values.append(int(token))
    return values


def _candidate_label(mseed: int, dseed: int) -> str:
    return f"s{mseed}/d{dseed}"


def _get_canonical_frequencies(prime: int) -> list[int]:
    """Return learned frequencies from any trained variant with the given prime.

    Reads variant_registry.json; returns an empty list if no match found.
    """
    if not _REGISTRY_PATH.exists():
        return []
    try:
        with open(_REGISTRY_PATH) as f:
            registry: list[dict[str, Any]] = json.load(f)
        for entry in registry:
            if entry.get("prime") != prime:
                continue
            freqs = entry.get("learned_frequencies")
            if freqs:
                return sorted(freqs)
    except Exception:
        pass
    return []


def _run_sweep(prime: int, model_seeds: list[int], data_seeds: list[int]) -> str:
    """Compute gradient profiles for all (model_seed × data_seed) combinations.

    Updates _state in-place.  Returns an empty string on success or an
    error message on failure.
    """
    import torch

    try:
        registry = get_registry()
        family = registry.get_family("modulo_addition_1layer")
    except Exception as exc:
        return f"Could not load family: {exc}"

    try:
        fourier_basis, _ = get_fourier_basis(prime)
    except Exception as exc:
        return f"Could not build Fourier basis for p={prime}: {exc}"

    n_freqs = prime // 2
    params = {"prime": prime}
    candidates = [(ms, ds) for ms in model_seeds for ds in data_seeds]
    results: dict[tuple[int, int], dict[str, np.ndarray]] = {}

    for mseed, dseed in candidates:
        try:
            model = family.create_model({**params, "seed": mseed})
            model.eval()
            td, tl, *_ = family.generate_training_dataset(params, data_seed=dseed)
            device = next(model.parameters()).device
            fb = fourier_basis.to(device)
            energies = _fourier_gradient_by_site(
                model, td.to(device), tl.to(device), prime, fb, n_freqs
            )
            model.zero_grad()
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            results[(mseed, dseed)] = energies
        except Exception as exc:
            return f"Failed for s{mseed}/d{dseed}: {exc}"

    _state.prime = prime
    _state.candidates = candidates
    _state.results = results
    _state.key_frequencies = _get_canonical_frequencies(prime)
    _state.error = ""
    return ""


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _render_site_profiles(
    results: dict[tuple[int, int], dict[str, np.ndarray]],
    candidates: list[tuple[int, int]],
    prime: int,
    key_freqs: list[int],
) -> go.Figure:
    """Three-panel overlaid frequency energy curves, one panel per site."""
    freqs = list(range(1, prime // 2 + 1))
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[_SITE_LABELS[s] for s in _SITES],
        shared_yaxes=False,
    )
    for cidx, cand in enumerate(candidates):
        color = _PALETTE[cidx % len(_PALETTE)]
        label = _candidate_label(*cand)
        for colidx, site in enumerate(_SITES):
            energy = results[cand][site].tolist()
            fig.add_trace(
                go.Scatter(
                    x=freqs,
                    y=energy,
                    mode="lines",
                    name=label,
                    line=dict(color=color),
                    showlegend=(colidx == 0),
                    legendgroup=label,
                ),
                row=1,
                col=colidx + 1,
            )
    for kf in key_freqs:
        fig.add_vline(x=kf, line_dash="dot", line_color="rgba(120,120,120,0.5)")
    fig.update_xaxes(title_text="Frequency k")
    fig.update_yaxes(title_text="RMS gradient energy", col=1)
    fig.update_layout(
        title=f"Epoch-0 gradient energy per site — p={prime}",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
    )
    return fig


def _render_difference(
    results: dict[tuple[int, int], dict[str, np.ndarray]],
    cand_a: tuple[int, int],
    cand_b: tuple[int, int],
    prime: int,
    key_freqs: list[int],
) -> go.Figure:
    """Three-panel difference view: A − B per site."""
    freqs = list(range(1, prime // 2 + 1))
    label_a = _candidate_label(*cand_a)
    label_b = _candidate_label(*cand_b)
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[_SITE_LABELS[s] for s in _SITES],
    )
    for colidx, site in enumerate(_SITES):
        diff = (results[cand_a][site] - results[cand_b][site]).tolist()
        bar_colors = ["#636EFA" if d >= 0 else "#EF553B" for d in diff]
        fig.add_trace(
            go.Bar(
                x=freqs,
                y=diff,
                marker_color=bar_colors,
                name=site,
                showlegend=False,
            ),
            row=1,
            col=colidx + 1,
        )
    fig.add_hline(y=0, line_color="black", line_width=0.8)
    for kf in key_freqs:
        fig.add_vline(x=kf, line_dash="dot", line_color="rgba(120,120,120,0.5)")
    fig.update_xaxes(title_text="Frequency k")
    fig.update_yaxes(title_text="Energy difference", col=1)
    fig.update_layout(
        title=f"{label_a} − {label_b}  [blue=A pushes harder · red=B pushes harder]",
        template="plotly_white",
        height=420,
    )
    return fig


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return float("nan")
    return float(np.dot(a / na, b / nb))


def _render_site_convergence(
    results: dict[tuple[int, int], dict[str, np.ndarray]],
    candidates: list[tuple[int, int]],
) -> go.Figure:
    """Grouped bar chart: pairwise cosine similarity between site spectra per candidate."""
    labels = [_candidate_label(*c) for c in candidates]
    emb_attn = [_cosine_sim(results[c]["embedding"], results[c]["attention"]) for c in candidates]
    emb_mlp = [_cosine_sim(results[c]["embedding"], results[c]["mlp"]) for c in candidates]
    attn_mlp = [_cosine_sim(results[c]["attention"], results[c]["mlp"]) for c in candidates]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Emb ↔ Attn", x=labels, y=emb_attn, marker_color="#636EFA"))
    fig.add_trace(go.Bar(name="Emb ↔ MLP", x=labels, y=emb_mlp, marker_color="#EF553B"))
    fig.add_trace(go.Bar(name="Attn ↔ MLP", x=labels, y=attn_mlp, marker_color="#00CC96"))
    fig.update_layout(
        title="Site convergence — pairwise cosine similarity between frequency spectra",
        yaxis_title="Cosine similarity",
        yaxis_range=[-0.1, 1.0],
        barmode="group",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _empty_figure(message: str = "Run a sweep to see results") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        template="plotly_white",
        height=420,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Layout & nav
# ---------------------------------------------------------------------------

_INFO_NOTE = (
    "At epoch 0, all variants sharing the same model_seed have identical weights. "
    "Sweeping multiple data seeds against the same model seed isolates the pure "
    "data-selection effect on gradient pressure."
)


def create_initialization_sweep_page_nav(app: Dash) -> html.Div:
    return html.Div(
        children=[
            dcc.Store(id="sweep-store", storage_type="memory", data={"candidates": []}),
            dbc.Label("Prime (p)", className="fw-bold"),
            dbc.Input(
                id="sweep-prime-input",
                type="number",
                value=113,
                min=2,
                step=1,
            ),
            html.Br(),
            dbc.Label("Model seeds", className="fw-bold"),
            dbc.Input(
                id="sweep-model-seeds-input",
                type="text",
                placeholder="e.g. 485, 999",
                value="485, 999",
            ),
            html.Small(
                "Comma-separated — each model seed × data seed becomes a candidate",
                className="text-muted",
            ),
            html.Br(),
            html.Br(),
            dbc.Label("Data seeds", className="fw-bold"),
            dbc.Input(
                id="sweep-data-seeds-input",
                type="text",
                placeholder="e.g. 598, 42",
                value="598",
            ),
            html.Small("Comma-separated", className="text-muted"),
            html.Br(),
            html.Br(),
            dbc.Button(
                "Run Sweep",
                id="sweep-run-button",
                color="primary",
                size="sm",
                n_clicks=0,
            ),
            html.Br(),
            html.Br(),
            html.Div(
                id="sweep-status",
                children="Enter parameters and click Run Sweep.",
                className="text-muted small",
            ),
            html.Hr(),
            html.P(_INFO_NOTE, className="text-muted small"),
        ]
    )


def create_initialization_sweep_page_layout(app: Dash) -> html.Div:
    candidate_options: list[dict] = []
    return html.Div(
        children=[
            html.H4("Initialization Gradient Sweep", className="mb-1"),
            html.P(
                "Computes per-frequency gradient energy at epoch 0 for arbitrary "
                "(prime, model_seed, data_seed) combinations.  No training required.",
                className="text-muted mb-3",
            ),
            dbc.Tabs(
                id="sweep-tabs",
                active_tab="tab-profiles",
                children=[
                    dbc.Tab(
                        label="Site Profiles",
                        tab_id="tab-profiles",
                        children=dcc.Graph(
                            id="sweep-profiles-figure",
                            config={"displayModeBar": True},
                            figure=_empty_figure(),
                        ),
                    ),
                    dbc.Tab(
                        label="Difference",
                        tab_id="tab-difference",
                        children=[
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Candidate A", className="fw-bold"),
                                            dcc.Dropdown(
                                                id="sweep-diff-cand-a",
                                                options=candidate_options,
                                                placeholder="Select candidate A…",
                                            ),
                                        ],
                                        width=3,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Candidate B", className="fw-bold"),
                                            dcc.Dropdown(
                                                id="sweep-diff-cand-b",
                                                options=candidate_options,
                                                placeholder="Select candidate B…",
                                            ),
                                        ],
                                        width=3,
                                    ),
                                ],
                                className="mb-3 mt-2",
                            ),
                            dcc.Graph(
                                id="sweep-difference-figure",
                                config={"displayModeBar": True},
                                figure=_empty_figure(),
                            ),
                        ],
                    ),
                    dbc.Tab(
                        label="Site Convergence",
                        tab_id="tab-convergence",
                        children=dcc.Graph(
                            id="sweep-convergence-figure",
                            config={"displayModeBar": True},
                            figure=_empty_figure(),
                        ),
                    ),
                ],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def register_initialization_sweep_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Initialization Gradient Sweep page."""

    @app.callback(
        Output("sweep-profiles-figure", "figure"),
        Output("sweep-convergence-figure", "figure"),
        Output("sweep-diff-cand-a", "options"),
        Output("sweep-diff-cand-b", "options"),
        Output("sweep-diff-cand-a", "value"),
        Output("sweep-diff-cand-b", "value"),
        Output("sweep-status", "children"),
        Output("sweep-store", "data"),
        Input("sweep-run-button", "n_clicks"),
        State("sweep-prime-input", "value"),
        State("sweep-model-seeds-input", "value"),
        State("sweep-data-seeds-input", "value"),
        prevent_initial_call=True,
    )
    def on_run_sweep(
        n_clicks: int,
        prime_raw: int | None,
        mseeds_raw: str | None,
        dseeds_raw: str | None,
    ):
        if not prime_raw or not mseeds_raw or not dseeds_raw:
            raise PreventUpdate

        prime = int(prime_raw)
        model_seeds = _parse_int_list(mseeds_raw)
        data_seeds = _parse_int_list(dseeds_raw)

        if not model_seeds or not data_seeds:
            msg = "Enter at least one model seed and one data seed."
            empty = _empty_figure()
            return empty, empty, [], [], None, None, msg, {"candidates": []}

        error = _run_sweep(prime, model_seeds, data_seeds)
        if error:
            empty = _empty_figure(error)
            return empty, empty, [], [], None, None, f"Error: {error}", {"candidates": []}

        candidates = _state.candidates
        candidate_labels = [_candidate_label(*c) for c in candidates]
        options = [{"label": lbl, "value": lbl} for lbl in candidate_labels]
        default_a = candidate_labels[0] if len(candidate_labels) >= 1 else None
        default_b = candidate_labels[1] if len(candidate_labels) >= 2 else None

        profiles_fig = _render_site_profiles(
            _state.results, candidates, prime, _state.key_frequencies
        )
        convergence_fig = _render_site_convergence(_state.results, candidates)

        n = len(candidates)
        status = f"Swept {n} candidate{'s' if n != 1 else ''} for p={prime}. " + (
            f"Key frequencies: {_state.key_frequencies}"
            if _state.key_frequencies
            else "No canonical frequencies on record for this prime."
        )
        store_data = {"candidates": candidate_labels, "prime": prime}
        return (
            profiles_fig,
            convergence_fig,
            options,
            options,
            default_a,
            default_b,
            status,
            store_data,
        )

    @app.callback(
        Output("sweep-difference-figure", "figure"),
        Input("sweep-diff-cand-a", "value"),
        Input("sweep-diff-cand-b", "value"),
        prevent_initial_call=True,
    )
    def on_diff_selection(label_a: str | None, label_b: str | None):
        if not label_a or not label_b or label_a == label_b:
            raise PreventUpdate
        if not _state.results or _state.prime is None:
            raise PreventUpdate

        cand_map = {_candidate_label(*c): c for c in _state.candidates}
        if label_a not in cand_map or label_b not in cand_map:
            raise PreventUpdate

        return _render_difference(
            _state.results,
            cand_map[label_a],
            cand_map[label_b],
            _state.prime,
            _state.key_frequencies,
        )
