"""Training page for the Dash dashboard.

Ports the Gradio Train tab functionality: family selection, domain/training
parameter configuration, variant preview, and training execution with
real-time progress tracking.
"""

from __future__ import annotations

import threading
import traceback

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html, no_update

from dashboard.components.variant_selector import get_family_choices
from dashboard.state import get_registry, refresh_registry, training_progress
from dashboard.utils import parse_checkpoint_epochs


def create_training_page_nav(app: Dash) -> html.Div:
    return html.Div()


def create_training_page_layout(app: Dash) -> html.Div:
    """Create the Training page layout."""
    registry = get_registry()
    family_choices = get_family_choices(registry, trainable_only=True)
    family_options = [{"label": display, "value": name} for display, name in family_choices]
    default_family = family_options[0]["value"] if family_options else None

    return html.Div(
        children=[
            html.H4("Training", className="mb-4"),
            dbc.Row(
                [
                    # Left column: parameters
                    dbc.Col(
                        [
                            dbc.Label("Model Family", className="fw-bold"),
                            dcc.Dropdown(
                                id="training-family-dropdown",
                                options=family_options,
                                value=default_family,
                                clearable=False,
                            ),
                            html.Div(
                                id="training-variant-preview",
                                children="Select a family",
                                className="text-muted small mt-1 mb-3",
                            ),
                            html.H6("Domain Parameters", className="mt-3"),
                            dbc.Label("Prime (p)", className="small"),
                            dbc.Input(
                                id="training-prime-input",
                                type="number",
                                value=113,
                                step=1,
                            ),
                            dbc.Label("Seed", className="small mt-2"),
                            dbc.Input(
                                id="training-seed-input",
                                type="number",
                                value=999,
                                step=1,
                            ),
                            html.H6("Training Parameters", className="mt-4"),
                            dbc.Label("Data Seed", className="small"),
                            dbc.Input(
                                id="training-data-seed-input",
                                type="number",
                                value=598,
                                step=1,
                            ),
                            dbc.Label("Training Fraction", className="small mt-2"),
                            dcc.Slider(
                                id="training-fraction-slider",
                                min=0.1,
                                max=0.9,
                                step=0.05,
                                value=0.75,
                                marks={0.1: "0.1", 0.3: "0.3", 0.5: "0.5", 0.7: "0.7", 0.9: "0.9"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            dbc.Label("Total Epochs", className="small mt-2"),
                            dbc.Input(
                                id="training-epochs-input",
                                type="number",
                                value=50000,
                                step=1,
                            ),
                            dbc.Label(
                                "Checkpoint Epochs (comma-separated)", className="small mt-2"
                            ),
                            dbc.Input(
                                id="training-checkpoint-input",
                                type="text",
                                value="",
                                placeholder="Leave empty for default schedule",
                            ),
                        ],
                        md=5,
                    ),
                    # Right column: status + button
                    dbc.Col(
                        [
                            html.H6("Status"),
                            dbc.Progress(
                                id="training-progress-bar",
                                value=0,
                                striped=True,
                                animated=True,
                                className="mb-2",
                                style={"display": "none"},
                            ),
                            html.Div(
                                id="training-status",
                                children="Ready to train",
                                className="mb-3",
                                style={
                                    "whiteSpace": "pre-wrap",
                                    "fontFamily": "monospace",
                                    "fontSize": "0.85rem",
                                    "backgroundColor": "#f8f9fa",
                                    "padding": "12px",
                                    "borderRadius": "4px",
                                    "minHeight": "200px",
                                    "maxHeight": "400px",
                                    "overflowY": "auto",
                                },
                            ),
                            dbc.Button(
                                "Start Training",
                                id="training-start-btn",
                                color="primary",
                                className="w-100",
                            ),
                        ],
                        md=7,
                    ),
                ],
                className="g-4",
            ),
            dcc.Interval(
                id="training-interval",
                interval=500,
                disabled=True,
            ),
        ],
        style={"padding": "20px", "maxWidth": "1000px"},
    )


def _run_training_thread(
    family_name: str,
    prime: int,
    seed: int,
    data_seed: int,
    train_fraction: float,
    num_epochs: int,
    checkpoint_str: str,
) -> None:
    """Execute training in a background thread."""
    try:
        training_progress.update(0.05, "Initializing...")

        registry = get_registry()
        family = registry.get_family(family_name)
        params = {"prime": int(prime), "seed": int(seed), "data_seed": int(data_seed)}
        variant = registry.create_variant(family, params)

        checkpoint_epochs = parse_checkpoint_epochs(checkpoint_str)
        if not checkpoint_epochs:
            checkpoint_epochs = None

        training_progress.update(0.1, "Starting training...")

        def progress_callback(pct: float, desc: str) -> None:
            training_progress.update(0.1 + (pct * 0.9), desc)

        result = variant.train(
            num_epochs=int(num_epochs),
            checkpoint_epochs=checkpoint_epochs,
            training_fraction=train_fraction,
            progress_callback=progress_callback,
        )

        refresh_registry()

        training_progress.finish(
            f"Training complete!\n"
            f"Variant: {variant.name}\n"
            f"Saved to: {result.variant_dir}\n"
            f"Checkpoints: {len(result.checkpoint_epochs)}\n"
            f"Final train loss: {result.final_train_loss:.6f}\n"
            f"Final test loss: {result.final_test_loss:.6f}"
        )

    except Exception as e:
        training_progress.finish(f"Training failed: {e}\n\n{traceback.format_exc()}")


def register_training_page_callbacks(app: Dash) -> None:
    """Register all Training page callbacks."""

    @app.callback(
        Output("training-variant-preview", "children"),
        Output("training-prime-input", "value"),
        Output("training-seed-input", "value"),
        Input("training-family-dropdown", "value"),
    )
    def on_training_family_change(family_name: str | None) -> tuple[str, int, int]:
        if not family_name:
            return "Select a family", 113, 999
        registry = get_registry()
        family = registry.get_family(family_name)
        defaults = family.get_default_params()
        prime = defaults.get("prime", 113)
        seed = defaults.get("seed", 999)
        data_seed = defaults.get("data_seed", 598)
        variant_name = family.get_variant_directory_name(
            {"prime": prime, "seed": seed, "data_seed": data_seed}
        )
        return f"Variant: {variant_name}", prime, seed

    @app.callback(
        Output("training-variant-preview", "children", allow_duplicate=True),
        Input("training-prime-input", "value"),
        Input("training-seed-input", "value"),
        Input("training-data-seed-input", "value"),
        State("training-family-dropdown", "value"),
        prevent_initial_call=True,
    )
    def on_params_change(
        prime: int | None, seed: int | None, data_seed: int | None, family_name: str | None
    ):
        if not family_name or prime is None or seed is None:
            return no_update
        try:
            registry = get_registry()
            family = registry.get_family(family_name)
            variant_name = family.get_variant_directory_name(
                {"prime": int(prime), "seed": int(seed), "data_seed": int(data_seed or 598)}
            )
            return f"Variant: {variant_name}"
        except Exception:
            return "Invalid parameters"

    @app.callback(
        Output("training-interval", "disabled"),
        Output("training-start-btn", "disabled"),
        Output("training-status", "children", allow_duplicate=True),
        Output("training-progress-bar", "style", allow_duplicate=True),
        Input("training-start-btn", "n_clicks"),
        State("training-family-dropdown", "value"),
        State("training-prime-input", "value"),
        State("training-seed-input", "value"),
        State("training-data-seed-input", "value"),
        State("training-fraction-slider", "value"),
        State("training-epochs-input", "value"),
        State("training-checkpoint-input", "value"),
        prevent_initial_call=True,
    )
    def on_start_training(
        n_clicks, family_name, prime, seed, data_seed, train_fraction, num_epochs, checkpoint_str
    ) -> tuple:
        if not n_clicks or not family_name:
            return no_update, no_update, no_update, no_update
        if training_progress.get_state()["running"]:
            return no_update, no_update, "Training already in progress...", no_update

        registry = get_registry()
        family = registry.get_family(family_name)
        params = {
            "prime": int(prime or 113),
            "seed": int(seed or 999),
            "data_seed": int(data_seed or 598),
        }
        check_variant = registry.create_variant(family, params)
        checkpoint_count = len(check_variant.get_available_checkpoints())
        if checkpoint_count > 0:
            variant_name = family.get_variant_directory_name(params)
            return (
                no_update,
                no_update,
                f"Blocked: variant '{variant_name}' already has checkpoints.\n\n"
                f"Training over an existing variant is not allowed. "
                f"Change the prime, seed, or data seed to create a new variant.",
                no_update,
            )

        training_progress.start()
        thread = threading.Thread(
            target=_run_training_thread,
            args=(
                family_name,
                prime or 113,
                seed or 999,
                data_seed or 598,
                train_fraction or 0.30,
                num_epochs or 50000,
                checkpoint_str or "",
            ),
            daemon=True,
        )
        thread.start()
        return False, True, "Starting training...", {"display": "block"}

    @app.callback(
        Output("training-progress-bar", "value"),
        Output("training-progress-bar", "label"),
        Output("training-status", "children"),
        Output("training-interval", "disabled", allow_duplicate=True),
        Output("training-start-btn", "disabled", allow_duplicate=True),
        Output("training-progress-bar", "style"),
        Input("training-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_training_progress(_n_intervals: int) -> tuple:
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
