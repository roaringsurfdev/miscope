"""Intervention Hook Verification page (REQ_070).

Shows per-frequency amplitude of hook_attn_out — baseline vs. hook-modified —
for a selected intervention variant and checkpoint epoch.  Lets the user
confirm that the FrequencyGainHook is applying the intended gain to the
intended frequencies before running further experiments.

Selection hierarchy: family → variant → intervention → epoch.

This page maintains its own local state (separate from the global
variant-selector-store) because it loads model checkpoints and runs
forward passes rather than reading artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, set_props
from dash.exceptions import PreventUpdate

from dashboard.components.variant_selector import get_family_choices, get_variant_choices
from dashboard.state import get_registry
from miscope.families.implementations.frequency_gain_hook import compute_hook_verification
from miscope.visualization.renderers.intervention_check import render_hook_verification_chart

# ---------------------------------------------------------------------------
# Page-local server state
# ---------------------------------------------------------------------------


@dataclass
class _InterventionCheckState:
    """Caches the last computed verification result to avoid redundant recomputes."""

    variant_name: str | None = None
    intervention_name: str | None = None
    epoch: int | None = None
    last_result: dict[str, Any] | None = None
    available_checkpoints: list[int] = field(default_factory=list)


_state = _InterventionCheckState()


def _get_intervention_options(family_name: str | None, variant_name: str | None) -> list[dict]:
    """Return dropdown options for all interventions on the selected variant."""
    if not family_name or not variant_name:
        return []
    try:
        registry = get_registry()
        family = registry.get_family(family_name)
        variants = registry.get_variants(family)
        variant = next((v for v in variants if v.name == variant_name), None)
        if variant is None:
            return []
        options = []
        for iv in variant.interventions:
            label = iv.intervention_config.get("label", iv.name)
            options.append({"label": label, "value": iv.name})
        return options
    except Exception:
        return []


def _load_intervention_checkpoints(
    family_name: str, variant_name: str, intervention_name: str
) -> list[int]:
    """Return sorted checkpoint epochs for the given intervention."""
    try:
        registry = get_registry()
        family = registry.get_family(family_name)
        variants = registry.get_variants(family)
        variant = next((v for v in variants if v.name == variant_name), None)
        if variant is None:
            return []
        iv = next((i for i in variant.interventions if i.name == intervention_name), None)
        return iv.get_available_checkpoints() if iv else []
    except Exception:
        return []


def _compute_and_cache(
    family_name: str, variant_name: str, intervention_name: str, epoch: int
) -> dict[str, Any] | None:
    """Run compute_hook_verification if inputs changed; return cached result otherwise."""
    if (
        _state.variant_name == variant_name
        and _state.intervention_name == intervention_name
        and _state.epoch == epoch
        and _state.last_result is not None
    ):
        return _state.last_result

    try:
        registry = get_registry()
        family = registry.get_family(family_name)
        variants = registry.get_variants(family)
        variant = next((v for v in variants if v.name == variant_name), None)
        if variant is None:
            return None
        iv = next((i for i in variant.interventions if i.name == intervention_name), None)
        if iv is None:
            return None
        result = compute_hook_verification(iv, epoch, device="cpu")
    except Exception as exc:
        print(f"intervention_check: compute failed [{intervention_name} epoch {epoch}]: {exc}")
        return None

    _state.variant_name = variant_name
    _state.intervention_name = intervention_name
    _state.epoch = epoch
    _state.last_result = result
    return result


def _empty_figure(message: str = "Select an intervention") -> go.Figure:
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
        height=500,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Layout & nav
# ---------------------------------------------------------------------------


def create_intervention_check_page_nav() -> html.Div:
    registry = get_registry()
    family_options = [
        {"label": display, "value": name} for display, name in get_family_choices(registry)
    ]

    return html.Div(
        children=[
            dcc.Store(
                id="intervention-check-store",
                storage_type="memory",
                data={
                    "family_name": None,
                    "variant_name": None,
                    "intervention_name": None,
                    "epoch_index": 0,
                    "max_index": 0,
                },
            ),
            dbc.Label("Family", className="fw-bold"),
            dcc.Dropdown(
                id="intervention-check-family-dropdown",
                placeholder="Select family...",
                options=family_options,
            ),
            html.Br(),
            dbc.Label("Variant", className="fw-bold"),
            dcc.Dropdown(
                id="intervention-check-variant-dropdown",
                placeholder="Select variant...",
                options=[],
            ),
            html.Br(),
            dbc.Label("Intervention", className="fw-bold"),
            dcc.Dropdown(
                id="intervention-check-intervention-dropdown",
                placeholder="Select intervention...",
                options=[],
            ),
            html.Br(),
            dbc.Label("Epoch", className="fw-bold"),
            html.Div(
                [
                    dbc.Button(
                        "<",
                        id="intervention-check-epoch-prev",
                        size="sm",
                        color="secondary",
                        outline=True,
                        n_clicks=0,
                        style={"flexShrink": "0"},
                    ),
                    dcc.Slider(
                        id="intervention-check-epoch-slider",
                        min=0,
                        max=0,
                        step=1,
                        value=0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    dbc.Button(
                        ">",
                        id="intervention-check-epoch-next",
                        size="sm",
                        color="secondary",
                        outline=True,
                        n_clicks=0,
                        style={"flexShrink": "0"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "4px"},
            ),
            html.Div(
                id="intervention-check-epoch-display",
                children="Epoch —",
                className="text-muted small mb-3",
            ),
            html.Hr(),
            html.Div(
                id="intervention-check-status",
                children="No intervention selected",
                className="text-muted small",
            ),
        ]
    )


def create_intervention_check_page_layout() -> html.Div:
    return html.Div(
        children=[
            html.H4("Intervention Hook Verification", className="mb-3"),
            html.P(
                "Verify that the FrequencyGainHook applies the intended gain "
                "to the targeted frequencies in hook_attn_out at each checkpoint epoch.",
                className="text-muted",
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        id="intervention-check-figure",
                        config={"displayModeBar": True},
                        style={"height": "500px"},
                        figure=_empty_figure(),
                    )
                )
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def register_intervention_check_callbacks(app: Dash) -> None:
    """Register all callbacks for the Intervention Check page."""

    @app.callback(
        Output("intervention-check-variant-dropdown", "options"),
        Output("intervention-check-variant-dropdown", "value"),
        Input("intervention-check-family-dropdown", "value"),
        prevent_initial_call=True,
    )
    def on_family_selected(family_name: str | None):
        if family_name is None:
            raise PreventUpdate
        registry = get_registry()
        choices = get_variant_choices(registry, family_name)
        options = [{"label": display, "value": name} for display, name in choices]
        return options, None

    @app.callback(
        Output("intervention-check-intervention-dropdown", "options"),
        Output("intervention-check-intervention-dropdown", "value"),
        Input("intervention-check-variant-dropdown", "value"),
        State("intervention-check-family-dropdown", "value"),
        prevent_initial_call=True,
    )
    def on_variant_selected(variant_name: str | None, family_name: str | None):
        if not variant_name:
            raise PreventUpdate
        options = _get_intervention_options(family_name, variant_name)
        set_props(
            "intervention-check-store",
            {"data": {"family_name": family_name, "variant_name": variant_name,
                      "intervention_name": None, "epoch_index": 0, "max_index": 0}},
        )
        return options, None

    @app.callback(
        Output("intervention-check-epoch-slider", "max"),
        Output("intervention-check-epoch-slider", "value"),
        Output("intervention-check-status", "children"),
        Input("intervention-check-intervention-dropdown", "value"),
        State("intervention-check-store", "data"),
        prevent_initial_call=True,
    )
    def on_intervention_selected(intervention_name: str | None, store_data: dict | None):
        stored = store_data or {}
        family_name = stored.get("family_name")
        variant_name = stored.get("variant_name")

        if not intervention_name or not variant_name or not family_name:
            raise PreventUpdate

        checkpoints = _load_intervention_checkpoints(family_name, variant_name, intervention_name)
        _state.available_checkpoints = checkpoints

        if not checkpoints:
            return 0, 0, f"{intervention_name} — no checkpoints found"

        max_index = len(checkpoints) - 1
        set_props(
            "intervention-check-store",
            {"data": {**stored, "intervention_name": intervention_name,
                      "epoch_index": 0, "max_index": max_index}},
        )
        return max_index, 0, f"{intervention_name} — {len(checkpoints)} checkpoints"

    @app.callback(
        Output("intervention-check-epoch-display", "children"),
        Output("intervention-check-figure", "figure"),
        Input("intervention-check-epoch-slider", "value"),
        State("intervention-check-store", "data"),
        prevent_initial_call=True,
    )
    def on_epoch_changed(epoch_index: int, store_data: dict | None):
        stored = store_data or {}
        family_name = stored.get("family_name")
        variant_name = stored.get("variant_name")
        intervention_name = stored.get("intervention_name")

        if not family_name or not variant_name or not intervention_name or not _state.available_checkpoints:
            raise PreventUpdate

        epoch_index = max(0, min(epoch_index, len(_state.available_checkpoints) - 1))
        epoch = _state.available_checkpoints[epoch_index]

        set_props(
            "intervention-check-store",
            {"data": {**stored, "epoch_index": epoch_index}},
        )

        result = _compute_and_cache(family_name, variant_name, intervention_name, epoch)
        if result is None:
            return f"Epoch {epoch}", _empty_figure(f"Failed to compute for epoch {epoch}")

        fig = render_hook_verification_chart(result)
        return f"Epoch {epoch}", fig

    @app.callback(
        Input("intervention-check-epoch-prev", "n_clicks"),
        Input("intervention-check-epoch-next", "n_clicks"),
        State("intervention-check-store", "data"),
        prevent_initial_call=True,
    )
    def on_epoch_nav(_prev: int, _next: int, store_data: dict | None) -> None:
        from dash import ctx

        stored = store_data or {}
        current_index = stored.get("epoch_index", 0)
        max_index = stored.get("max_index", 0)

        if ctx.triggered_id == "intervention-check-epoch-prev":
            new_index = max(0, current_index - 1)
        else:
            new_index = min(max_index, current_index + 1)

        if new_index != current_index:
            set_props("intervention-check-epoch-slider", {"value": new_index})
        else:
            raise PreventUpdate
