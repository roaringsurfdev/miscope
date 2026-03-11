from __future__ import annotations

from typing import TYPE_CHECKING

import dash_bootstrap_components as dbc
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, set_props
from dash.exceptions import PreventUpdate

from dashboard.state import get_registry, variant_server_state

"""
Encapulated and reuseable component for handling standard 
    Variant Selection behavior across Analysis Lens pages
        Family Drop-Down (Single Select)
            Session Management:
                Should stay set between pages, maybe per session
            Behavior:
                on_change -> update list of Variants, save state (future: update available lenses/visualizations)
                on_load -> if selected family stored in state, update selection and execute on_change functionality
        Variant List (Single Select) displayed as [domain parameter list]
        Epoch Slider
        Epoch Index Text Input
        Load Button (Intentional Commit of changed values)
"""
"""Family and variant selection components for the dashboard.

REQ_021d: Dashboard Integration with Model Families
"""

if TYPE_CHECKING:
    from miscope.families import FamilyRegistry, Variant


def get_family_choices(registry: FamilyRegistry) -> list[tuple[str, str]]:
    """Get choices for family dropdown.

    Args:
        registry: The FamilyRegistry instance

    Returns:
        List of (display_name, family_name) tuples for gr.Dropdown
    """
    families = registry.list_families()
    return [(f.display_name, f.name) for f in families if f.ui_trainable]


def get_variant_choices(registry: FamilyRegistry, family_name: str | None) -> list[tuple[str, str]]:
    """Get choices for variant dropdown.

    Args:
        registry: The FamilyRegistry instance
        family_name: Name of the selected family

    Returns:
        List of (display_name, variant_name) tuples for gr.Dropdown
    """
    if not family_name or family_name not in registry:
        return []

    family = registry.get_family(family_name)
    variants = registry.get_variants(family)

    choices = []
    for variant in variants:
        state_indicator = get_state_indicator(variant)
        params_str = format_variant_params(variant)
        display_name = f"{state_indicator} {params_str} [{variant.state.value}]"
        choices.append((display_name, variant.name))

    # Sort alphabetically by variant name
    choices.sort(key=lambda c: c[1])

    return choices


def get_state_indicator(variant: Variant) -> str:
    """Get visual indicator for variant state.

    Args:
        variant: The Variant instance

    Returns:
        Unicode character indicating state
    """
    from miscope.families import VariantState

    indicators = {
        VariantState.UNTRAINED: "○",  # Empty circle
        VariantState.TRAINED: "●",  # Filled circle
        VariantState.ANALYZED: "◉",  # Circle with dot (analyzed)
    }
    return indicators.get(variant.state, "?")


def format_variant_params(variant: Variant) -> str:
    """Format variant parameters for display.

    Args:
        variant: The Variant instance

    Returns:
        Formatted parameter string like "p=113, seed=42"
    """
    parts = []
    for key, value in variant.params.items():
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def get_variant_selector(
    initial_family: str | None = None,
    initial_variant: str | None = None,
    initial_epoch_idx: int = 0,
) -> html.Div:
    return html.Div(
        [
            dcc.Store(
                id="variant-selector-store",
                storage_type="session",
                data={
                    "family_name": None,
                    "variant_name": None,
                    "epoch": None,
                    "epoch_index": None,
                    "max_epochs": 0,
                    "last_field_updated": None,
                    "stale_data": "0",
                },
            ),
            html.Hr(),
            dbc.Label("Family", className="fw-bold"),
            dcc.Dropdown(
                id="variant-selector-family-dropdown",
                placeholder="Select family...",
                value=initial_family,
            ),
            html.Br(),
            dbc.Label("Variant", className="fw-bold"),
            dcc.Dropdown(
                id="variant-selector-variant-dropdown",
                placeholder="Select variant...",
                value=initial_variant,
            ),
            html.Br(),
            dbc.Label("Epoch", className="fw-bold"),
            html.Div(
                [
                    dbc.Button(
                        "<",
                        id="variant-selector-epoch-prev",
                        size="sm",
                        color="secondary",
                        outline=True,
                        n_clicks=0,
                        style={"flexShrink": "0"},
                    ),
                    dcc.Slider(
                        id="variant-selector-epoch-slider",
                        min=0,
                        max=1,
                        step=1,
                        value=initial_epoch_idx,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    dbc.Button(
                        ">",
                        id="variant-selector-epoch-next",
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
                id="variant-selector-epoch-display",
                children=f"Epoch {initial_epoch_idx}",
                className="text-muted small mb-3",
            ),
            html.Hr(),
            # Status
            html.Div(
                id="variant-selector-status-display",
                children="No variant selected",
                className="text-muted small",
            ),
        ]
    )


def register_variant_selector_callbacks(app: Dash) -> None:
    """Register all callbacks for the Variant Selector."""

    @app.callback(
        Output("variant-selector-family-dropdown", "options"),
        Input("variant-selector-family-dropdown", "id"),
    )
    def populate_families(_: str) -> list[dict]:
        registry = get_registry()
        choices = get_family_choices(registry)
        return [{"label": display, "value": name} for display, name in choices]

    @app.callback(
        Output("variant-selector-variant-dropdown", "options"),
        Output("variant-selector-variant-dropdown", "value"),
        Input("variant-selector-family-dropdown", "value"),
        State("variant-selector-store", "data"),
    )
    def on_family_change(family_name: str | None, store_data: dict | None):
        stored = store_data or {}
        options = []
        stored_family_name = stored.get("family_name")

        if family_name is None and stored_family_name is None:
            print(
                "on_family_change: family_name is None and stored_family_name is None, PreventUpdate"
            )
            raise PreventUpdate

        if stored_family_name != family_name:
            print("on_family_change: family_name has changed, update dependencies")
            registry = get_registry()
            choices = get_variant_choices(registry, family_name)
            options = [{"label": display, "value": name} for display, name in choices]
            set_props(
                "variant-selector-store",
                {"data": {"family_name": family_name, "last_field_updated": "family_name"}},
            )
            print(f"on_family_changed: Store updated -> family_name={family_name}")
        else:
            # cancel operation
            print("on_family_change: family_name is unchanged, PreventUpdate")
            raise PreventUpdate

        return options, None

    @app.callback(
        [
            Output("variant-selector-epoch-slider", "value"),
            Output("variant-selector-epoch-slider", "max"),
            Output("variant-selector-status-display", "children"),
        ],
        Input("variant-selector-variant-dropdown", "value"),
        State("variant-selector-store", "data"),
        prevent_initial_call=True,
    )
    def on_variant_change(variant_name: str | None, store_data: dict | None):
        stored = store_data or {}
        epoch = stored.get("epoch")
        epoch_index = 0
        max_epochs = 0
        stored_variant_name = stored.get("variant_name")
        stored_family_name = stored.get("family_name")

        if variant_name is None and stored_variant_name is None:
            print(
                "on_variant_change: variant_name is None and stored_variant_name is None, PreventUpdate"
            )
            raise PreventUpdate

        if stored_variant_name != variant_name:
            print("on_variant_change: variant_name has changed, update dependencies")
            if variant_name is None:
                # Reset to defaults
                print("on_variant_change: Variant reset")
                epoch = 0
                epoch_index = 0
                max_epochs = 0
            else:
                # load variant-specific data
                if stored_family_name is None:
                    print(
                        "on_variant_change: variant_name changed but family_name is None, PreventUpdate"
                    )
                    raise PreventUpdate

                # load variant into server_state
                variant_server_state.load_variant(stored_family_name, str(variant_name))
                # reset epoch and epoch_index
                # get max epochs for new variant
                max_epochs = max(0, len(variant_server_state.available_epochs) - 1)
                print(
                    f"New variant selected: variant_name: {variant_name}, epoch: {epoch}, max_epochs:{max_epochs}"
                )

            # save updated variant settings to store
            set_props(
                "variant-selector-store",
                {
                    "data": {
                        "family_name": stored_family_name,
                        "variant_name": variant_name,
                        "epoch": epoch,
                        "epoch_index": epoch_index,
                        "max_epochs": max_epochs,
                        "last_field_updated": "variant_name",
                    }
                },
            )
        else:
            print("on_variant_change: variant_name is unchanged, PreventUpdate")
            raise PreventUpdate

        return epoch_index, max_epochs, variant_name

    @app.callback(
        Output("variant-selector-epoch-display", "children"),
        Input("variant-selector-epoch-slider", "value"),
        Input({"view_type": "epoch_selector", "index": ALL}, "clickData"),
        State("variant-selector-store", "data"),
        prevent_initial_call=True,
    )
    def on_epoch_change(
        epoch_index: int | None, click_data: list[dict | None] | None, store_data: dict | None
    ):
        print("on_epoch_change")
        stored = store_data or {}
        stored_variant_name = stored.get("variant_name")
        stored_family_name = stored.get("family_name")
        stored_epoch = stored.get("epoch")
        stored_epoch_index = stored.get("epoch_index")
        stored_epoch_index = stored.get("epoch_index")
        stored_max_epochs = stored.get("max_epochs")
        update_slider = False

        if stored_family_name is None and stored_variant_name is None:
            print(
                "on_epoch_changed: variant_name is None and stored_variant_name is None, PreventUpdate"
            )
            raise PreventUpdate

        if epoch_index is None:
            epoch_index = 0
            epoch = 0
        else:
            epoch = stored_epoch

        # check for click events
        if ctx.triggered_id != "variant-selector-epoch-slider":
            click_data_component_id = ctx.triggered_id
            if click_data is not None:
                for click_data_item in click_data:
                    if click_data_item:
                        clicked_x = click_data_item["points"][0].get("x")
                        if clicked_x:
                            print(f"handling click event: {click_data}")
                            epoch_index = variant_server_state.get_nearest_epoch_index(int(clicked_x))
                            epoch = variant_server_state.available_epochs[epoch_index]
                            # Will need to explicitly update the slider since the event
                            # did not come through the slider
                            update_slider = True
                            # Reset clickData so that there's only one entry in click_data at a time
                            if click_data_component_id:
                                set_props(click_data_component_id, {"clickData": None})
                        break

        # commit new epoch data if it has changed
        if stored_epoch_index != epoch_index:
            epoch = variant_server_state.available_epochs[epoch_index]

            # save updated variant settings to store
            print("on_epoch_slider_changed: epoch changed, update dependencies")

            # load the variant at the newly selected epoch
            variant_server_state.load_epoch(epoch)

            # update the store with updated selections
            set_props(
                "variant-selector-store",
                {
                    "data": {
                        "family_name": stored_family_name,
                        "variant_name": stored_variant_name,
                        "epoch": epoch,
                        "epoch_index": epoch_index,
                        "max_epochs": stored_max_epochs,
                        "last_field_updated": "epoch",
                    }
                },
            )
            # update the slider if necessary
            if update_slider:
                set_props("variant-selector-epoch-slider", {"value": epoch_index})
        else:
            print("on_epoch_slider_changed: epoch unchanged, PreventUpdate")
            raise PreventUpdate

        return f"Epoch {epoch}"

    @app.callback(
        Input("variant-selector-epoch-prev", "n_clicks"),
        Input("variant-selector-epoch-next", "n_clicks"),
        State("variant-selector-store", "data"),
        prevent_initial_call=True,
    )
    def on_epoch_nav_click(
        _prev_clicks: int, _next_clicks: int, store_data: dict | None
    ) -> None:
        stored = store_data or {}
        epoch_index = stored.get("epoch_index") or 0
        max_epochs = stored.get("max_epochs") or 0

        if ctx.triggered_id == "variant-selector-epoch-prev":
            new_index = max(0, epoch_index - 1)
        else:
            new_index = min(max_epochs, epoch_index + 1)

        if new_index != epoch_index:
            set_props("variant-selector-epoch-slider", {"value": new_index})
        else:
            raise PreventUpdate

    # --- Show last field updated: for debugging purposes ---
    @app.callback(
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
        prevent_initial_call=True,
    )
    def on_variant_selector_store_update(timestamp: str | None, store_data: dict | None):
        stored = store_data or {}
        last_field_updated = stored.get("last_field_updated")
        print(f"on_variant_selector_store_update: last_field_updated: {last_field_updated}")
