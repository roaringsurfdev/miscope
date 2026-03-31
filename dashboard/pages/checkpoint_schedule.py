"""Checkpoint Schedule page for the Dash dashboard.

Retrain an existing variant with a denser checkpoint schedule.
Shows the loss curve with existing checkpoint density, a range-based builder
for adding new checkpoint regions, and a preview before retraining.
"""

from __future__ import annotations

import threading
import traceback

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate

from dashboard.state import get_registry, refresh_registry, training_progress


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_empty_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title="Epoch",
        yaxis_title="Loss",
        plot_bgcolor="white",
        paper_bgcolor="white",
        annotations=[
            dict(
                text="No variant loaded",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="gray"),
            )
        ],
    )
    return fig


def _build_loss_figure(
    train_losses: list[float],
    test_losses: list[float],
    existing_checkpoints: list[int],
    new_epochs: list[int] | None = None,
) -> go.Figure:
    epochs = list(range(len(train_losses)))
    rug_y = min(min(train_losses), min(test_losses)) * 0.5

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_losses,
            mode="lines",
            name="Train Loss",
            line=dict(color="steelblue", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=test_losses,
            mode="lines",
            name="Test Loss",
            line=dict(color="darkorange", width=1.5),
        )
    )
    if existing_checkpoints:
        fig.add_trace(
            go.Scatter(
                x=existing_checkpoints,
                y=[rug_y] * len(existing_checkpoints),
                mode="markers",
                name=f"Existing ({len(existing_checkpoints)})",
                marker=dict(
                    symbol="line-ns-open",
                    size=10,
                    line=dict(width=1, color="gray"),
                ),
                opacity=0.7,
            )
        )
    if new_epochs:
        fig.add_trace(
            go.Scatter(
                x=new_epochs,
                y=[rug_y * 0.6] * len(new_epochs),
                mode="markers",
                name=f"New ({len(new_epochs)})",
                marker=dict(
                    symbol="line-ns-open",
                    size=10,
                    line=dict(width=1.5, color="tomato"),
                ),
            )
        )
    fig.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        xaxis_title="Epoch",
        yaxis_title="Loss",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#eee"),
        yaxis=dict(gridcolor="#eee"),
    )
    return fig


def _generate_range_epochs(row: dict) -> set[int]:
    """Generate epoch set from a single range table row."""
    start = row.get("start")
    end = row.get("end")
    step = row.get("step")
    if any(v is None or v == "" for v in [start, end, step]):
        return set()
    step = int(step)
    if step <= 0:
        return set()
    return set(range(int(start), int(end) + 1, step))


def _generate_merged_epochs(
    table_data: list[dict],
    existing_checkpoints: list[int],
    total_epochs: int,
) -> tuple[list[int], list[int]]:
    """Return (all_epochs, new_only_epochs) after merging ranges with existing."""
    existing_set = set(existing_checkpoints)
    added: set[int] = set()
    for row in table_data or []:
        added.update(_generate_range_epochs(row))
    new_only = sorted(e for e in added if e not in existing_set and e < total_epochs)
    all_epochs = sorted((existing_set | added) - {e for e in added if e >= total_epochs})
    return all_epochs, new_only


def _range_warning(table_data: list[dict], total_epochs: int) -> str:
    """Return a warning string if any range row extends past total_epochs."""
    for row in table_data or []:
        end = row.get("end")
        if end is not None and end != "" and int(end) >= total_epochs:
            return f"Warning: a range end ({int(end)}) meets or exceeds Total Epochs ({total_epochs}). Extend Total Epochs or reduce the range."
    return ""


def _run_retrain_thread(
    family_name: str,
    variant_name: str,
    total_epochs: int,
    merged_checkpoint_epochs: list[int],
) -> None:
    try:
        training_progress.update(0.05, "Loading variant...")
        registry = get_registry()
        family = registry.get_family(family_name)
        variants = registry.get_variants(family)
        variant = next((v for v in variants if v.name == variant_name), None)
        if variant is None:
            training_progress.finish(f"Variant not found: {variant_name}")
            return

        training_fraction = variant.model_config.get("training_fraction", 0.3)
        training_progress.update(0.1, "Starting retrain...")

        def progress_callback(pct: float, desc: str) -> None:
            training_progress.update(0.1 + pct * 0.9, desc)

        result = variant.train(
            num_epochs=total_epochs,
            checkpoint_epochs=merged_checkpoint_epochs,
            training_fraction=training_fraction,
            progress_callback=progress_callback,
        )
        refresh_registry()
        training_progress.finish(
            f"Retrain complete!\n"
            f"Variant: {variant.name}\n"
            f"Total epochs: {total_epochs}\n"
            f"Checkpoints saved: {len(result.checkpoint_epochs)}\n"
            f"Final train loss: {result.final_train_loss:.6f}\n"
            f"Final test loss: {result.final_test_loss:.6f}"
        )
    except Exception as e:
        training_progress.finish(f"Retrain failed: {e}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def create_checkpoint_schedule_page_nav(app: Dash) -> html.Div:
    return html.Div()


def create_checkpoint_schedule_page_layout(app: Dash) -> html.Div:
    return html.Div(
        children=[
            html.H4("Checkpoint Schedule", className="mb-4"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H6("Selected Variant", className="fw-bold"),
                            html.Div(
                                id="ckpt-variant-info",
                                children="No variant loaded — select one from the left panel.",
                                className="text-muted small mb-3",
                            ),
                            dbc.Label("Total Epochs", className="small mt-2 fw-semibold"),
                            dbc.Input(
                                id="ckpt-total-epochs-input",
                                type="number",
                                value=25000,
                                step=1,
                                className="mb-1",
                            ),
                            html.Div(
                                id="ckpt-epoch-warning",
                                className="text-danger small mb-2",
                            ),
                            html.H6("Additional Checkpoint Ranges", className="mt-3 mb-1"),
                            html.P(
                                "Existing checkpoints are always included. "
                                "Add ranges to increase density.",
                                className="text-muted small mb-2",
                            ),
                            dash_table.DataTable(
                                id="ckpt-range-table",
                                columns=[
                                    {
                                        "name": "Start",
                                        "id": "start",
                                        "type": "numeric",
                                        "editable": True,
                                    },
                                    {
                                        "name": "End",
                                        "id": "end",
                                        "type": "numeric",
                                        "editable": True,
                                    },
                                    {
                                        "name": "Step",
                                        "id": "step",
                                        "type": "numeric",
                                        "editable": True,
                                    },
                                ],
                                data=[],
                                row_deletable=True,
                                editable=True,
                                style_table={"marginBottom": "8px"},
                                style_cell={
                                    "textAlign": "left",
                                    "padding": "6px",
                                    "fontSize": "0.85rem",
                                },
                                style_header={"fontWeight": "bold", "fontSize": "0.85rem"},
                                style_data_conditional=[
                                    {
                                        "if": {"state": "active"},
                                        "backgroundColor": "#e8f4fd",
                                        "border": "1px solid #90c0e8",
                                    }
                                ],
                            ),
                            dbc.Button(
                                "+ Add Range",
                                id="ckpt-add-range-btn",
                                color="secondary",
                                outline=True,
                                size="sm",
                                className="mb-3",
                            ),
                            html.Div(
                                id="ckpt-new-count",
                                className="text-muted small",
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="ckpt-loss-graph",
                                figure=_build_empty_figure(),
                                style={"height": "420px"},
                                config={"displayModeBar": False},
                            ),
                            dbc.Progress(
                                id="ckpt-progress-bar",
                                value=0,
                                striped=True,
                                animated=True,
                                className="mb-2 mt-3",
                                style={"display": "none"},
                            ),
                            html.Div(
                                id="ckpt-status",
                                children="Select a variant and define additional ranges, then click Retrain.",
                                style={
                                    "whiteSpace": "pre-wrap",
                                    "fontFamily": "monospace",
                                    "fontSize": "0.85rem",
                                    "backgroundColor": "#f8f9fa",
                                    "padding": "12px",
                                    "borderRadius": "4px",
                                    "minHeight": "60px",
                                    "maxHeight": "150px",
                                    "overflowY": "auto",
                                },
                                className="mb-3",
                            ),
                            dbc.Button(
                                "Retrain",
                                id="ckpt-retrain-btn",
                                color="primary",
                                className="w-100",
                                disabled=True,
                            ),
                        ],
                        md=8,
                    ),
                ],
                className="g-4",
            ),
            dcc.Interval(id="ckpt-interval", interval=500, disabled=True),
        ],
        style={"padding": "20px", "maxWidth": "1200px"},
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def register_checkpoint_schedule_page_callbacks(app: Dash) -> None:

    @app.callback(
        Output("ckpt-variant-info", "children"),
        Output("ckpt-total-epochs-input", "value"),
        Output("ckpt-retrain-btn", "disabled"),
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_variant_loaded(timestamp, store_data: dict | None):
        store = store_data or {}
        family_name = store.get("family_name")
        variant_name = store.get("variant_name")
        if not family_name or not variant_name:
            return "No variant loaded — select one from the left panel.", 25000, True
        try:
            registry = get_registry()
            family = registry.get_family(family_name)
            variants = registry.get_variants(family)
            variant = next((v for v in variants if v.name == variant_name), None)
            if variant is None:
                return f"Variant not found: {variant_name}", 25000, True
            config = variant.model_config
            meta = variant.metadata
            existing = variant.get_available_checkpoints()
            info = html.Div(
                [
                    html.Span(f"{variant_name}", className="fw-semibold"),
                    html.Br(),
                    html.Span(
                        f"p={config.get('prime', '?')}  seed={config.get('seed', '?')}  "
                        f"dseed={config.get('data_seed', '?')}  "
                        f"train_frac={config.get('training_fraction', '?')}",
                        className="text-muted",
                    ),
                    html.Br(),
                    html.Span(
                        f"{len(existing)} existing checkpoints",
                        className="text-muted",
                    ),
                ],
                className="small",
            )
            return info, meta.get("num_epochs", 25000), False
        except Exception as e:
            return f"Error loading variant: {e}", 25000, True

    @app.callback(
        Output("ckpt-range-table", "data"),
        Input("ckpt-add-range-btn", "n_clicks"),
        State("ckpt-range-table", "data"),
        prevent_initial_call=True,
    )
    def add_range_row(n_clicks: int | None, current_data: list | None) -> list:
        rows = list(current_data or [])
        rows.append({"start": None, "end": None, "step": 100})
        return rows

    @app.callback(
        Output("ckpt-loss-graph", "figure"),
        Output("ckpt-new-count", "children"),
        Output("ckpt-epoch-warning", "children"),
        Input("variant-selector-store", "modified_timestamp"),
        Input("ckpt-range-table", "data"),
        Input("ckpt-total-epochs-input", "value"),
        State("variant-selector-store", "data"),
    )
    def update_chart(timestamp, table_data, total_epochs, store_data: dict | None):
        store = store_data or {}
        family_name = store.get("family_name")
        variant_name = store.get("variant_name")
        if not family_name or not variant_name:
            return _build_empty_figure(), "", ""
        try:
            registry = get_registry()
            family = registry.get_family(family_name)
            variants = registry.get_variants(family)
            variant = next((v for v in variants if v.name == variant_name), None)
            if variant is None:
                return _build_empty_figure(), "", ""
            train_losses = variant.train_losses
            test_losses = variant.test_losses
            existing = variant.get_available_checkpoints()
            epochs_limit = int(total_epochs) if total_epochs else len(train_losses)
            warning = _range_warning(table_data, epochs_limit)
            _, new_only = _generate_merged_epochs(table_data, existing, epochs_limit)
            count_msg = f"{len(new_only)} new checkpoint epoch(s) will be added." if new_only else ""
            fig = _build_loss_figure(train_losses, test_losses, existing, new_only or None)
            return fig, count_msg, warning
        except Exception as e:
            return _build_empty_figure(), "", f"Error: {e}"

    @app.callback(
        Output("ckpt-interval", "disabled"),
        Output("ckpt-retrain-btn", "disabled", allow_duplicate=True),
        Output("ckpt-status", "children"),
        Output("ckpt-progress-bar", "style"),
        Input("ckpt-retrain-btn", "n_clicks"),
        State("ckpt-range-table", "data"),
        State("ckpt-total-epochs-input", "value"),
        State("variant-selector-store", "data"),
        prevent_initial_call=True,
    )
    def on_retrain_click(n_clicks, table_data, total_epochs, store_data: dict | None):
        if not n_clicks:
            raise PreventUpdate
        store = store_data or {}
        family_name = store.get("family_name")
        variant_name = store.get("variant_name")
        if not family_name or not variant_name:
            return no_update, no_update, "No variant selected.", no_update
        if training_progress.get_state()["running"]:
            return no_update, no_update, "A training job is already running.", no_update
        try:
            registry = get_registry()
            family = registry.get_family(family_name)
            variants = registry.get_variants(family)
            variant = next((v for v in variants if v.name == variant_name), None)
            if variant is None:
                return no_update, no_update, f"Variant not found: {variant_name}", no_update
            existing = variant.get_available_checkpoints()
            epochs_limit = int(total_epochs) if total_epochs else variant.metadata.get("num_epochs", 25000)
            merged, new_only = _generate_merged_epochs(table_data, existing, epochs_limit)
            if not new_only:
                return no_update, no_update, "No new checkpoint epochs defined. Add at least one range.", no_update
        except Exception as e:
            return no_update, no_update, f"Error: {e}", no_update

        training_progress.start()
        thread = threading.Thread(
            target=_run_retrain_thread,
            args=(family_name, variant_name, epochs_limit, merged),
            daemon=True,
        )
        thread.start()
        return False, True, "Starting retrain...", {"display": "block"}

    @app.callback(
        Output("ckpt-progress-bar", "value"),
        Output("ckpt-progress-bar", "label"),
        Output("ckpt-status", "children", allow_duplicate=True),
        Output("ckpt-interval", "disabled", allow_duplicate=True),
        Output("ckpt-retrain-btn", "disabled", allow_duplicate=True),
        Output("ckpt-progress-bar", "style", allow_duplicate=True),
        Input("ckpt-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_retrain_progress(_n_intervals: int):
        state = training_progress.get_state()
        pct = int(state["progress"] * 100)
        if state["running"]:
            return pct, f"{pct}%", state["message"], False, True, {"display": "block"}
        return (
            100,
            "100%",
            state["result"] if state["result"] else state["message"],
            True,
            False,
            {"display": "none"},
        )
