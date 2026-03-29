"""Variant Table page (REQ_082).

Displays all variants and their key metrics in a sortable, filterable table.
Clicking a row selects that variant globally via variant-selector-store.
"""

from __future__ import annotations

import json

from dash import Dash, Input, Output, State, dash_table, html, set_props
from dash.exceptions import PreventUpdate

from dashboard.state import get_registry, variant_server_state

# ---------------------------------------------------------------------------
# Classification label colours
# ---------------------------------------------------------------------------

_CLASSIFICATION_COLORS: dict[str, str] = {
    "healthy": "#d4edda",
    "late_grokker": "#fff3cd",
    "degraded": "#f8d7da",
    "ungrokked": "#f8d7da",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_table_rows() -> list[dict]:
    """Load all variants from available variant_registry.json files.

    Returns a flat list of row dicts ready for DataTable.
    """
    registry = get_registry()
    rows: list[dict] = []

    for family in registry.list_families():
        # Derive results dir for this family from an existing variant, or skip.
        variants = registry.get_variants(family)
        if not variants:
            continue

        registry_path = variants[0].variant_dir.parent / "variant_registry.json"
        if not registry_path.exists():
            continue

        with open(registry_path) as f:
            records = json.load(f)

        for rec in records:
            prime = rec.get("prime")
            seed = rec.get("model_seed")
            dseed = rec.get("data_seed")
            family_name = rec.get("family", family.name)
            variant_name = f"{family_name}_p{prime}_seed{seed}_dseed{dseed}"

            classification_raw = rec.get("performance_classification", [])
            classification = classification_raw[0] if classification_raw else "unknown"

            grokking_epoch = rec.get("second_descent_onset_epoch")

            final_window = rec.get("final_window") or {}
            committed = final_window.get("committed_frequencies_end") or []
            committed_count = len(committed)

            test_loss_final = rec.get("test_loss_final")
            loss_display = f"{test_loss_final:.2e}" if test_loss_final is not None else "—"

            rows.append(
                {
                    "_variant_name": variant_name,
                    "family": family_name,
                    "prime": prime,
                    "model_seed": seed,
                    "data_seed": dseed,
                    "classification": classification,
                    "test_loss_final": loss_display,
                    "grokking_epoch": grokking_epoch if grokking_epoch is not None else "—",
                    "committed_freqs": committed_count,
                }
            )

    rows.sort(
        key=lambda r: (r["family"], r["prime"] or 0, r["model_seed"] or 0, r["data_seed"] or 0)
    )
    return rows


# ---------------------------------------------------------------------------
# DataTable column definitions
# ---------------------------------------------------------------------------

_COLUMNS = [
    {"name": "Family", "id": "family"},
    {"name": "Prime (p)", "id": "prime", "type": "numeric"},
    {"name": "Model Seed", "id": "model_seed", "type": "numeric"},
    {"name": "Data Seed", "id": "data_seed", "type": "numeric"},
    {"name": "Classification", "id": "classification"},
    {"name": "Test Loss (final)", "id": "test_loss_final"},
    {"name": "Grokking Epoch", "id": "grokking_epoch"},
    {"name": "Committed Freqs", "id": "committed_freqs", "type": "numeric"},
]

# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------


def create_variant_table_page_nav(app: Dash) -> html.Div:
    return html.Div()


def create_variant_table_page_layout(app: Dash) -> html.Div:
    rows = _load_table_rows()

    style_data_conditional = [
        {
            "if": {"filter_query": f'{{classification}} = "{label}"'},
            "backgroundColor": color,
        }
        for label, color in _CLASSIFICATION_COLORS.items()
    ]
    style_data_conditional.append(
        {
            "if": {"state": "selected"},
            "backgroundColor": "#cce5ff",
            "border": "1px solid #004085",
        }
    )

    return html.Div(
        [
            html.H4("Variant Registry", className="mb-1"),
            html.P(
                f"{len(rows)} variants — click a row to select it as the active variant.",
                className="text-muted small mb-3",
            ),
            dash_table.DataTable(
                id="variant-table",
                columns=_COLUMNS,  # pyright: ignore[reportArgumentType]
                data=rows,
                hidden_columns=["_variant_name"],
                sort_action="native",
                filter_action="native",
                filter_options={"placeholder_text": "Filter…"},
                row_selectable="single",
                selected_rows=[],
                page_action="native",
                page_size=30,
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": "#343a40",
                    "color": "white",
                    "fontWeight": "bold",
                    "fontSize": "13px",
                },
                style_cell={
                    "fontSize": "13px",
                    "padding": "6px 10px",
                    "textAlign": "left",
                    "whiteSpace": "normal",
                },
                style_data_conditional=style_data_conditional,  # pyright: ignore[reportArgumentType]
            ),
            html.Div(id="variant-table-status", className="text-muted small mt-2"),
        ],
        className="p-3",
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def register_variant_table_page_callbacks(app: Dash) -> None:
    @app.callback(
        Output("variant-table-status", "children"),
        Input("variant-table", "selected_rows"),
        State("variant-table", "data"),
        State("variant-table", "derived_virtual_data"),
        prevent_initial_call=True,
    )
    def on_row_selected(
        selected_rows: list[int] | None,
        table_data: list[dict] | None,
        virtual_data: list[dict] | None,
    ) -> str:
        if not selected_rows or not table_data:
            raise PreventUpdate

        # Use virtual_data (post-filter/sort) if available, else fall back to full data.
        active_data = virtual_data if virtual_data is not None else table_data
        row = active_data[selected_rows[0]]

        family_name = row["family"]
        variant_name = row["_variant_name"]

        ok = variant_server_state.load_variant(family_name, variant_name)
        if not ok:
            return f"Could not load variant: {variant_name}"

        max_epochs = max(0, len(variant_server_state.available_epochs) - 1)
        epoch = (
            variant_server_state.available_epochs[0] if variant_server_state.available_epochs else 0
        )

        set_props(
            "variant-selector-store",
            {
                "data": {
                    "family_name": family_name,
                    "variant_name": variant_name,
                    "intervention_name": None,
                    "epoch": epoch,
                    "epoch_index": 0,
                    "max_epochs": max_epochs,
                    "last_field_updated": "variant_name",
                }
            },
        )
        set_props("variant-selector-family-dropdown", {"value": family_name})
        set_props("variant-selector-variant-dropdown", {"value": variant_name})

        return f"Selected: {variant_name}"

    @app.callback(
        Output("variant-table", "selected_rows"),
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
        State("variant-table", "data"),
        State("variant-table", "derived_virtual_data"),
        prevent_initial_call=True,
    )
    def sync_selection_from_store(
        _ts: str | None,
        store_data: dict | None,
        table_data: list[dict] | None,
        virtual_data: list[dict] | None,
    ) -> list[int]:
        if not store_data or not table_data:
            return []

        active_variant = store_data.get("variant_name")
        if not active_variant:
            return []

        active_data = virtual_data if virtual_data is not None else table_data
        for idx, row in enumerate(active_data):
            if row.get("_variant_name") == active_variant:
                return [idx]

        return []
