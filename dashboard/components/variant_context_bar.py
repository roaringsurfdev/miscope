"""Variant context bar — secondary horizontal strip below the top navbar.

Displays variant identity, key training milestones, committed frequencies,
and final test loss for the currently loaded variant. Updates whenever
variant-selector-store changes.

Sits at position sticky, top 56px (flush below the fixed dark navbar).
"""

import json

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html
from dash.exceptions import PreventUpdate

_CONTEXT_BAR_STYLE = {
    "position": "sticky",
    "top": "56px",
    "zIndex": "900",
    "backgroundColor": "#f1f3f5",
    "borderBottom": "1px solid #dee2e6",
    "padding": "0 16px",
    "display": "flex",
    "alignItems": "center",
    "gap": "8px",
    "height": "40px",
    "fontSize": "0.82rem",
    "overflowX": "auto",
    "flexShrink": "0",
}

_DIVIDER_STYLE = {
    "color": "#ced4da",
    "margin": "0 4px",
}

_CLASSIFICATION_COLORS = {
    "healthy": "success",
    "partial": "warning",
    "anomalous": "warning",
    "degraded": "warning",
    "failed": "danger",
    "unknown": "secondary",
}


def _format_epoch(epoch: int | None) -> str:
    if epoch is None:
        return "—"
    if epoch >= 1000:
        return f"{epoch / 1000:.1f}k"
    return str(epoch)


def _parse_classification(raw) -> str:
    """Return lowercase label string from variant_summary performance_classification."""
    if isinstance(raw, list) and raw:
        return str(raw[0]).lower()
    if isinstance(raw, str):
        return raw.lower()
    return "unknown"


def _load_variant_summary(variant) -> dict:
    try:
        path = variant.variant_dir / "variant_summary.json"
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _build_context_children(variant) -> list:
    summary = _load_variant_summary(variant)
    params = variant.params  # e.g. {"prime": 113, "seed": 999, "data_seed": 598}

    # --- Identity pills ---
    param_badges = []
    for key, val in params.items():
        param_badges.append(
            dbc.Badge(f"{key}={val}", color="secondary", className="me-1", pill=True)
        )

    # Classification badge
    classification_raw = summary.get("performance_classification")
    if classification_raw is not None:
        label = _parse_classification(classification_raw)
        color = _CLASSIFICATION_COLORS.get(label, "secondary")
        class_badge = dbc.Badge(label, color=color, className="me-1", pill=True)
    else:
        class_badge = None

    identity_section = html.Span(
        children=param_badges + ([class_badge] if class_badge else []),
        style={"whiteSpace": "nowrap"},
    )

    children = [identity_section]

    if not summary:
        return children

    children.append(html.Span("|", style=_DIVIDER_STYLE))

    # --- Training milestones ---
    grokking_epoch = summary.get("test_loss_threshold_first_epoch")
    first_mover = summary.get("first_mover_epoch")
    second_descent = summary.get("second_descent_onset_epoch")

    milestone_parts = []
    if first_mover is not None:
        milestone_parts.append(f"FM: {_format_epoch(first_mover)}")
    if second_descent is not None:
        milestone_parts.append(f"2nd↓: {_format_epoch(second_descent)}")
    if grokking_epoch is not None:
        milestone_parts.append(f"Grokked: {_format_epoch(grokking_epoch)}")

    if milestone_parts:
        children.append(
            html.Span(
                " | ".join(milestone_parts), className="text-muted", style={"whiteSpace": "nowrap"}
            )
        )
        children.append(html.Span("|", style=_DIVIDER_STYLE))

    # --- Committed frequencies (1-indexed for display) ---
    final_window = summary.get("final_window", {})
    committed_raw = final_window.get("committed_frequencies_end") or []
    if committed_raw:
        freq_str = ", ".join(str(f + 1) for f in sorted(committed_raw))
        children.append(
            html.Span(
                [html.Span("Freqs: ", className="text-muted"), freq_str],
                style={"whiteSpace": "nowrap"},
            )
        )
        children.append(html.Span("|", style=_DIVIDER_STYLE))

    # --- Final test loss ---
    test_loss = summary.get("test_loss_final")
    if test_loss is not None:
        loss_str = f"{test_loss:.2e}"
        children.append(
            html.Span(
                [html.Span("Loss: ", className="text-muted"), loss_str],
                style={"whiteSpace": "nowrap"},
            )
        )

    return children


def create_variant_context_bar() -> html.Div:
    return html.Div(
        id="variant-context-bar",
        style=_CONTEXT_BAR_STYLE,
        children=[
            html.Div(
                id="variant-context-bar-content",
                children=html.Span("No variant selected", className="text-muted"),
                style={"display": "flex", "alignItems": "center", "gap": "8px", "width": "100%"},
            )
        ],
    )


def register_variant_context_bar_callbacks(app: Dash) -> None:
    @app.callback(
        Output("variant-context-bar-content", "children"),
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def update_context_bar(_ts, store_data: dict | None):
        from dashboard.state import variant_server_state

        if not store_data or not store_data.get("variant_name"):
            raise PreventUpdate

        try:
            variant = variant_server_state.variant
        except AttributeError:
            raise PreventUpdate

        children = _build_context_children(variant)
        return children
